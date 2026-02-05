# hierarchical_model.py
# Hierarchical Bayesian meta-learning with embedding-conditional priors

import torch
import numpy as np
from torch import nn, func
import torchopt
import posteriors
import sys
import gc
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from optree import tree_map
from torch.func import grad_and_value
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'method/hierarchical_model')

from bnn_models import linear_nn_model, sequence_nn_model
from bnn_priors import multivariate_normal_prior
from accuracy import get_pred_table
from explainability import save_embeddings
from embedding_conditional_prior import EmbeddingConditionalPrior, load_task_embeddings
from early_stopping import EarlyStopping


def compute_class_weights(y, device='cpu'):
    classes, counts = torch.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts.float())
    return weights.to(device)


def bnn_baseline_model(train_task_loader, pred_task_loader, data, hyper_params, args, pred_for='val'):
    eval_mode = (pred_for == 'test')
    local_models = {}
    
    use_early_stopping = args.get('early_stopping', False)
    early_stopping_patience = args.get('early_stopping_patience', 10)
    early_stopping_metric = args.get('early_stopping_metric', 'nelbo')
    n_epochs = args.get('max_num_epochs', 50)
    samples_per_epoch = args['batch_size'] * 10
    bnn_update_budget = int(hyper_params.get('num_inner_updates', 100))
    
    for task in pred_task_loader:
        task_label = data.target_task_name_map[task.item()]
        print(f"\nTraining BNN: {task_label}")

        updates_done = 0
        model = _create_model(data, hyper_params, args)
        params = dict(model.named_parameters())
        
        init_data = data.sample_supportquery_data(
            task_label, samples_per_epoch, sample_seed=0,
            eval=eval_mode, as_tensor=True, device=args['device']
        )
        y_init = torch.cat([init_data[4], init_data[5]], dim=0)
        n_train = len(y_init)
        
        class_weights = None
        if args.get('use_class_weights', False):
            class_weights = compute_class_weights(y_init, device=args['device'])
        
        prior_sigma = hyper_params.get('global_prior_sigma', 1.0)
        temperature = hyper_params.get('inner_temperature', 1.0) / n_train
        lr = hyper_params.get('inner_learning_rate', 1e-3)
        init_log_sds = hyper_params.get('model_init_log_sds', -1.0)
        
        def log_posterior(params, batch):
            X_tab, X_long, y = batch
            logits = func.functional_call(model, params, (X_tab, X_long))
            log_likelihood = -nn.functional.cross_entropy(logits, y, weight=class_weights)
            log_prior = multivariate_normal_prior(params, mean=0, sd_diag=prior_sigma, normalize=False)
            return log_likelihood + log_prior / n_train, logits
        
        init_log_sd_dict = {k: torch.full_like(v, init_log_sds) for k, v in params.items()}
        optimizer = torchopt.sgd(lr=lr, momentum=0.9)
        transform = posteriors.vi.diag.build(
            log_posterior, optimizer,
            temperature=temperature,
            n_samples=args.get('num_mc_samples', 5)
        )
        state = transform.init(params, init_log_sds=init_log_sd_dict)
        
        if use_early_stopping:
            mode = 'max' if early_stopping_metric == 'auroc' else 'min'
            early_stopper = EarlyStopping(patience=early_stopping_patience, mode=mode, verbose=False)
        best_state = None
        best_metric = float('-inf') if early_stopping_metric == 'auroc' else float('inf')
        
        for epoch in range(n_epochs):
            train_data = data.sample_supportquery_data(
                task_label, samples_per_epoch, sample_seed=epoch,
                eval=eval_mode, as_tensor=True, device=args['device']
            )
            X_train = torch.cat([train_data[0], train_data[1]], dim=0)
            X_l_train = torch.cat([train_data[2], train_data[3]], dim=0)
            y_train = torch.cat([train_data[4], train_data[5]], dim=0)
            
            train_dataset = TensorDataset(X_train, X_l_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
            
            epoch_nelbos = []
            for X_batch, X_long_batch, y_batch in train_loader:
                batch = (X_batch, X_long_batch, y_batch)
                state = transform.update(state, batch, inplace=False)
                epoch_nelbos.append(state.nelbo.item())
                updates_done += 1
                if updates_done >= bnn_update_budget:
                    break

            if updates_done >= bnn_update_budget:
                print(f"  Reached update budget at epoch {epoch+1}")
                break
            
            epoch_nelbo = np.mean(epoch_nelbos)
            if np.isnan(epoch_nelbo):
                break
            
            if not eval_mode:
                with torch.no_grad():
                    val_data = data.sample_supportquery_data(
                        task_label, samples_per_epoch, sample_seed=epoch + 10000,
                        eval=eval_mode, as_tensor=True, device=args['device'], dataset='val'
                    )
                    X_val = torch.cat([val_data[0], val_data[1]], dim=0)
                    X_l_val = torch.cat([val_data[2], val_data[3]], dim=0)
                    y_val = torch.cat([val_data[4], val_data[5]], dim=0)
                    
                    val_preds = forward(model, state, X_val, X_l_val, args)
                    val_y_pred = val_preds['y_pred']
                    val_y_true = y_val.cpu().numpy()
                    
                    val_nelbo = -np.mean(
                        val_y_true * np.log(val_y_pred + 1e-8) + 
                        (1 - val_y_true) * np.log(1 - val_y_pred + 1e-8)
                    )
                    
                    val_auroc = None
                    if early_stopping_metric == 'auroc' or (epoch + 1) % 20 == 0:
                        try:
                            from sklearn.metrics import roc_auc_score
                            val_auroc = roc_auc_score(val_y_true, val_y_pred)
                        except ValueError:
                            val_auroc = 0.5
                
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    auroc_str = f", AUROC={val_auroc:.4f}" if val_auroc else ""
                    print(f"  Epoch {epoch+1}: NELBO={epoch_nelbo:.4f}{auroc_str}")
                
                if use_early_stopping:
                    metric = val_auroc if (early_stopping_metric == 'auroc' and val_auroc) else val_nelbo
                    is_better = (metric > best_metric) if early_stopping_metric == 'auroc' else (metric < best_metric)
                    
                    if is_better:
                        best_metric = metric
                        best_state = posteriors.vi.diag.VIDiagState(
                            params={k: v.clone() for k, v in state.params.items()},
                            log_sd_diag={k: v.clone() for k, v in state.log_sd_diag.items()},
                            opt_state=state.opt_state,
                            nelbo=state.nelbo
                        )
                    
                    if early_stopper(metric):
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1}: NELBO={epoch_nelbo:.4f}")
        
        if use_early_stopping and best_state is not None:
            state = best_state
        
        local_models[task_label] = {'model': model, 'state': state}
        torch.cuda.empty_cache()
        gc.collect()
    
    predictions = get_predictions_bnn(
        pred_task_loader, data.target_task_name_map, data,
        local_models, hyper_params, args, pred_for
    )
    
    if pred_for == 'test':
        _save_outputs_bnn(predictions, local_models, data, hyper_params, args)
    
    return predictions


def get_predictions_bnn(taskloader, label_map, data, local_models, hyper_params, args, pred_for='val'):
    all_preds = {}
    
    for batch in taskloader:
        for task in batch:
            task_label = label_map[task.item()]
            model = local_models[task_label]['model']
            state = local_models[task_label]['state']
            
            if pred_for == 'val':
                X_spt, X_qry, X_l_spt, X_l_qry, y_spt, y_qry, _, _ = data.sample_supportquery_data(
                    task_label, args['batch_size'] * 10, sample_seed=0,
                    eval=False, as_tensor=True, device=args['device'], dataset='val'
                )
                X_data = torch.cat((X_spt, X_qry), dim=0)
                X_long = torch.cat((X_l_spt, X_l_qry), dim=0)
                y_data = torch.cat((y_spt, y_qry), dim=0)
                data_ids = None
            else:
                task_data = data.get_taskdata(
                    task_label, eval=True, as_tensor=True, device=args['device'], return_ids=True
                )
                X_data = task_data[4]
                X_long = task_data[5]
                y_data = task_data[6]
                data_ids = task_data[7]
            
            results = {k: [] for k in ['y_pred', 'y_pred_entropy', 'y_pred_bin',
                                        'aleatoric_uncertainty', 'epistemic_uncertainty']}
            
            eval_loader = DataLoader(TensorDataset(X_data, X_long), batch_size=args['batch_size'], shuffle=False)
            
            for X_batch, X_long_batch in eval_loader:
                batch_results = forward(model, state, X_batch, X_long_batch, args)
                for key in results:
                    results[key].append(batch_results[key])
            
            for key in results:
                results[key] = np.concatenate(results[key])
            
            results['y_actual'] = y_data.cpu()
            if data_ids:
                results['data_ids'] = data_ids
            
            all_preds[task_label] = results
    
    return all_preds


def _save_outputs_bnn(predictions, local_models, data, hyper_params, args):
    pred_df = get_pred_table(predictions)
    pred_df.to_csv(f'{args["outprefix"]}_pred_uncertainties.csv', index=None)
    
    for task_label, model_dict in local_models.items():
        state = model_dict['state']
        torch.save({
            'params': state.params,
            'log_sd_diag': state.log_sd_diag
        }, f'{args["outprefix"]}_model_local_{task_label}.pth')


def hierarchical_meta_model(train_task_loader, pred_task_loader, data, hyper_params, args, pred_for='val'):
    eval_mode = (pred_for == 'test')
    num_data = hyper_params['prior_scaling']
    num_tasks = len(data.task_names)

    hyper_params = dict(hyper_params)
    inner_temp = hyper_params.get('inner_temperature', None)
    outer_temp = hyper_params.get('outer_temperature', None)
    
    if inner_temp is not None:
        inner_temp = inner_temp / num_data
    if outer_temp is not None:
        outer_temp = outer_temp / num_tasks

    model = _create_model(data, hyper_params, args)
    params = dict(model.named_parameters())

    embedding_conditional_prior = None
    task_embeddings = None
    embedding_dim = None

    if args['adaptation'] == 'adaptive':
        task_embeddings, embedding_dim = _load_embeddings(args)
        if args.get('embeddings') == 'zero':
            print("Using zero embeddings (ablation)")
            for task in task_embeddings:
                task_embeddings[task] = torch.zeros_like(task_embeddings[task])

    use_class_weights = args.get('use_class_weights', False)
    class_weights = None
    if use_class_weights:
        all_labels = []
        for task_name in data.task_names:
            task_data = data.get_taskdata(task_name, eval=False, as_tensor=True, device='cpu')
            all_labels.append(task_data[2])
        all_labels = torch.cat(all_labels)
        class_weights = compute_class_weights(all_labels, device=args['device'])

    def log_posterior_support(prior_mean, prior_log_sd, params, task):
        X_spt, _, X_long_spt, _, y_spt, _, _, _ = task
        logits = func.functional_call(model, params, (X_spt, X_long_spt))
        sd_diag = tree_map(torch.exp, prior_log_sd)
        log_prior = multivariate_normal_prior(params, mean=prior_mean, sd_diag=sd_diag, normalize=False)
        log_likelihood = -nn.functional.cross_entropy(logits, y_spt, weight=class_weights)
        return log_likelihood + log_prior / num_data, (logits, log_likelihood, log_prior)

    def log_posterior_query(prior_mean, prior_log_sd, params, task):
        _, X_qry, _, X_long_qry, _, y_qry, _, _ = task
        logits = func.functional_call(model, params, (X_qry, X_long_qry))
        sd_diag = tree_map(torch.exp, prior_log_sd)
        log_prior = multivariate_normal_prior(params, mean=prior_mean, sd_diag=sd_diag, normalize=False)
        log_likelihood = -nn.functional.cross_entropy(logits, y_qry, weight=class_weights)
        return log_likelihood + log_prior / num_data, (logits, log_likelihood, log_prior)

    def compute_nelbo(m, lsd, task, log_posterior, temperature, n_samples=1, stl=True):
        sd_diag = tree_map(torch.exp, lsd)
        nelbo, _ = posteriors.vi.diag.nelbo(m, sd_diag, task, log_posterior, temperature, n_samples, stl)
        return nelbo

    def compute_task_nelbo(task, prior_params, prior_log_sd, return_local_model=False, detach_prior=True, num_inner_updates=None):
        if num_inner_updates is None:
            num_inner_updates = hyper_params['num_inner_updates']
            
        if detach_prior:
            prior_params = {p: prior_params[p].detach() for p in prior_params}
            prior_log_sd = {p: prior_log_sd[p].detach() for p in prior_log_sd}

        inner_opt = torchopt.sgd(lr=hyper_params['inner_learning_rate'])
        partial_log_posterior = partial(log_posterior_support, prior_params, prior_log_sd)
        inner_transform = posteriors.vi.diag.build(
            partial_log_posterior, inner_opt,
            temperature=inner_temp,
            init_log_sds=prior_log_sd
        )

        inner_state = inner_transform.init(prior_params, init_log_sds=prior_log_sd)
        X_spt, X_qry, X_long_spt, X_long_qry, y_spt, y_qry, _, _ = task

        for k in range(num_inner_updates):
            start_idx = k * args['batch_size']
            batch_indices = torch.arange(start_idx, start_idx + args['batch_size']).long()
            task_batch = (
                X_spt[batch_indices], X_qry[batch_indices],
                X_long_spt[batch_indices], X_long_qry[batch_indices],
                y_spt[batch_indices], y_qry[batch_indices],
                None, None
            )
            inner_state = inner_transform.update(inner_state, task_batch, inplace=False)

        if return_local_model:
            return inner_state

        partial_log_posterior_qry = partial(log_posterior_query, prior_params, prior_log_sd)
        return compute_nelbo(inner_state.params, inner_state.log_sd_diag, task_batch, partial_log_posterior_qry, inner_temp)

    def compute_batch_nelbo(outer_params, outer_log_sd, batch_data, task_labels=None, ecp=None, embeddings=None):
        use_adaptive = (task_labels is not None and ecp is not None and embeddings is not None)

        if use_adaptive:
            task_nelbos = []
            for idx, task_label in enumerate(task_labels):
                z_t = embeddings[task_label]
                mu_adapt, log_sd_adapt = ecp.compute_adaptive_prior(
                    z_t, global_params=outer_params, adaptation_scale=hyper_params.get('adaptation_scale', 0.2)
                )
                adapted_params = {name: outer_params[name] + mu_adapt[name] for name in outer_params}
                adapted_log_sd = {name: outer_log_sd[name] + log_sd_adapt[name] for name in outer_log_sd}
                task_data = tuple(td[idx] for td in batch_data)
                task_nelbo = compute_task_nelbo(task_data, adapted_params, adapted_log_sd, detach_prior=False)
                task_nelbos.append(task_nelbo)
            batch_nelbo = torch.mean(torch.stack(task_nelbos))
        else:
            def compute_task_nelbo_nodetach(task, prior_params, prior_log_sd):
                return compute_task_nelbo(task, prior_params, prior_log_sd, return_local_model=False, detach_prior=False)
            vmapped = func.vmap(compute_task_nelbo_nodetach, in_dims=((0, 0, 0, 0, 0, 0, 0, 0), None, None), randomness='different')
            task_nelbos = vmapped(batch_data, outer_params, outer_log_sd)
            batch_nelbo = torch.mean(task_nelbos)

        if hyper_params['nn_prior_name'] == 'multivariate_normal_prior':
            log_p = multivariate_normal_prior(outer_params, mean=0, sd_diag=hyper_params["global_prior_sigma"]) / num_tasks
            sd_diag = tree_map(torch.exp, outer_log_sd)
            sampled = posteriors.diag_normal_sample(outer_params, sd_diag, (args['num_mc_samples'],))
            log_q = func.vmap(posteriors.diag_normal_log_prob, (0, None, None))(sampled, outer_params, sd_diag)
            kl_term = -(log_p - log_q).mean() * outer_temp
            batch_nelbo += kl_term

        return batch_nelbo

    def update_model(state, opt, grads, nelbo_val, inplace=False):
        updates, opt_state = opt.update(grads, state.opt_state, params=[state.params, state.log_sd_diag], inplace=inplace)
        mean, log_sd = torchopt.apply_updates((state.params, state.log_sd_diag), updates, inplace=inplace)
        return posteriors.vi.diag.VIDiagState(mean, log_sd, opt_state, nelbo_val.detach())

    def get_task_data(mini_batch_tasks, epoch, label_map):
        X_tabs, X_qrys, X_longs, X_long_qrys = [], [], [], []
        y_spts, y_qrys, imp_spts, imp_qrys = [], [], [], []
        samples_per_task = hyper_params['num_inner_updates'] * args['batch_size'] * 2

        for task in mini_batch_tasks:
            task_label = label_map[task.item()]
            X_spt, X_qry, X_l_spt, X_l_qry, y_spt, y_qry, imp_spt, imp_qry = data.sample_supportquery_data(
                task_label, samples_per_task, sample_seed=epoch, eval=eval_mode
            )
            X_tabs.append(X_spt)
            X_qrys.append(X_qry)
            X_longs.append(X_l_spt)
            X_long_qrys.append(X_l_qry)
            y_spts.append(y_spt)
            y_qrys.append(y_qry)
            imp_spts.append(imp_spt)
            imp_qrys.append(imp_qry)

        device = args['device']
        return (
            torch.Tensor(np.array(X_tabs)).to(device),
            torch.Tensor(np.array(X_qrys)).to(device),
            torch.Tensor(np.array(X_longs)).to(device),
            torch.Tensor(np.array(X_long_qrys)).to(device),
            torch.Tensor(np.array(y_spts)).long().to(device),
            torch.Tensor(np.array(y_qrys)).long().to(device),
            torch.Tensor(np.array(imp_spts)).long().to(device),
            torch.Tensor(np.array(imp_qrys)).long().to(device),
        )

    # initialize
    outer_opt = torchopt.adam(lr=hyper_params['outer_learning_rate'])
    outer_transform = posteriors.vi.diag.build(
        log_posterior_query, outer_opt,
        temperature=outer_temp,
        n_samples=args['num_mc_samples']
    )
    outer_state = outer_transform.init(params, init_log_sds=hyper_params["model_init_log_sds"])

    embedding_prior_optimizer = None
    if args['adaptation'] == 'adaptive':
        embedding_conditional_prior = EmbeddingConditionalPrior(
            embedding_dim=embedding_dim,
            global_params=outer_state.params,
            global_log_sd=outer_state.log_sd_diag,
            device=args['device']
        )
        w_lr = hyper_params.get('w_learning_rate', hyper_params['outer_learning_rate'] * 0.3)
        embedding_prior_optimizer = torch.optim.Adam(embedding_conditional_prior.parameters(), lr=w_lr)

    # training
    print("Training hierarchical model...")
    epoch_nelbos = []
    epoch_val_nelbos = []
    epoch_val_aurocs = []
    track_val_loss = args.get('track_val_loss', True)
    
    use_early_stopping = args.get('early_stopping', False)
    early_stopping_patience = args.get('early_stopping_patience', 3)
    early_stopping_metric = args.get('early_stopping_metric', 'auroc')
    
    if use_early_stopping:
        mode = 'max' if early_stopping_metric == 'auroc' else 'min'
        early_stopper = EarlyStopping(patience=early_stopping_patience, mode=mode, verbose=True)

    for epoch in range(args['max_num_epochs']):
        nelbos = []

        for mini_batch in train_task_loader:
            task_data = get_task_data(mini_batch, epoch, data.task_name_map)

            if embedding_conditional_prior is not None:
                task_labels = [data.task_name_map[t.item()] for t in mini_batch]
                embedding_prior_optimizer.zero_grad()
                
                for p in list(outer_state.params.values()) + list(outer_state.log_sd_diag.values()):
                    p.requires_grad_(True)
                    p.retain_grad()

                nelbo_val = compute_batch_nelbo(
                    outer_state.params, outer_state.log_sd_diag, task_data,
                    task_labels, embedding_conditional_prior, task_embeddings
                )

                if torch.isnan(nelbo_val):
                    continue

                for p in outer_state.params.values():
                    if p.grad is not None:
                        p.grad.zero_()
                for p in outer_state.log_sd_diag.values():
                    if p.grad is not None:
                        p.grad.zero_()

                w_reg_lambda = hyper_params.get('w_reg_lambda', 0.05)
                reg = sum((w**2).sum() for w in embedding_conditional_prior.W_mu.values())
                (nelbo_val + w_reg_lambda * reg).backward()

                torch.nn.utils.clip_grad_norm_(embedding_conditional_prior.parameters(), 1.0)
                embedding_prior_optimizer.step()

                with torch.no_grad():
                    total_norm = sum(w.pow(2).sum() for w in embedding_conditional_prior.W_mu.values()).sqrt()
                    if total_norm > 0.5:
                        for w in embedding_conditional_prior.W_mu.values():
                            w.mul_(0.5 / total_norm)

                grads_params = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n, p in outer_state.params.items()}
                grads_log_sd = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n, p in outer_state.log_sd_diag.items()}
                outer_state = update_model(outer_state, outer_opt, (grads_params, grads_log_sd), nelbo_val.detach())
                nelbos.append(nelbo_val.cpu().detach())
            else:
                grads, nelbo_val = grad_and_value(compute_batch_nelbo, argnums=(0, 1))(
                    outer_state.params, outer_state.log_sd_diag, task_data
                )
                nelbos.append(nelbo_val.cpu().detach())
                outer_state = update_model(outer_state, outer_opt, grads, nelbo_val)

        epoch_nelbo = np.mean(nelbos)
        epoch_nelbos.append(epoch_nelbo)

        if track_val_loss:
            val_nelbos = []
            val_aurocs = []
            track_val_auroc = args.get('track_val_auroc', False)
            
            with torch.no_grad():
                for task in pred_task_loader:
                    task_label = data.target_task_name_map[task.item()]
                    
                    if args['adaptation'] == 'adaptive' and task_label in task_embeddings:
                        z_t = task_embeddings[task_label]
                        mu_adapt, log_sd_adapt = embedding_conditional_prior.compute_adaptive_prior(
                            z_t, global_params=outer_state.params,
                            adaptation_scale=hyper_params.get('adaptation_scale', 0.2)
                        )
                        prior_params = {n: outer_state.params[n] + mu_adapt[n] for n in outer_state.params}
                        prior_log_sd = {n: outer_state.log_sd_diag[n] + log_sd_adapt[n] for n in outer_state.log_sd_diag}
                    else:
                        prior_params = outer_state.params
                        prior_log_sd = outer_state.log_sd_diag
                    
                    samples = hyper_params['num_inner_updates'] * args['batch_size'] * 2
                    val_task_data = data.sample_supportquery_data(
                        task_label, samples, sample_seed=epoch,
                        eval=eval_mode, as_tensor=True, device=args['device']
                    )
                    
                    val_nelbo = compute_task_nelbo(val_task_data, prior_params, prior_log_sd)
                    val_nelbos.append(val_nelbo.cpu().item())
                    
                    if track_val_auroc:
                        local_state = compute_task_nelbo(val_task_data, prior_params, prior_log_sd, return_local_model=True)
                        task_test_data = data.get_taskdata(task_label, eval=True, as_tensor=True, device=args['device'])
                        X_test, X_long_test, y_test = task_test_data[3], task_test_data[4], task_test_data[5]
                        
                        preds = forward(model, local_state, X_test, X_long_test, args)
                        from sklearn.metrics import roc_auc_score
                        try:
                            val_aurocs.append(roc_auc_score(y_test.cpu().numpy(), preds['y_pred']))
                        except ValueError:
                            pass
            
            epoch_val_nelbo = np.mean(val_nelbos)
            epoch_val_nelbos.append(epoch_val_nelbo)
            
            if track_val_auroc and val_aurocs:
                epoch_val_auroc = np.mean(val_aurocs)
                epoch_val_aurocs.append(epoch_val_auroc)
                print(f"Epoch {epoch}: NELBO={epoch_nelbo:.4f}, Val NELBO={epoch_val_nelbo:.4f}, Val AUROC={epoch_val_auroc:.4f}")
            else:
                print(f"Epoch {epoch}: NELBO={epoch_nelbo:.4f}, Val NELBO={epoch_val_nelbo:.4f}")
            
            if use_early_stopping:
                if early_stopping_metric == 'auroc' and track_val_auroc and val_aurocs:
                    if early_stopper(epoch_val_auroc):
                        print(f"Early stopping at epoch {epoch}")
                        break
                elif early_stopping_metric == 'nelbo':
                    if early_stopper(epoch_val_nelbo):
                        print(f"Early stopping at epoch {epoch}")
                        break
        else:
            print(f"Epoch {epoch}: NELBO={epoch_nelbo:.4f}")

        if np.isnan(epoch_nelbo):
            return None

        torch.cuda.empty_cache()
        gc.collect()

    # save training curve
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('NELBO', color='tab:blue')
    ax1.plot(epoch_nelbos, label='Train', color='tab:blue')
    if track_val_loss and epoch_val_nelbos:
        ax1.plot(epoch_val_nelbos, label='Val', color='tab:cyan', linestyle='--')
    ax1.legend(loc='upper left')
    
    if epoch_val_aurocs:
        ax2 = ax1.twinx()
        ax2.set_ylabel('AUROC', color='tab:red')
        ax2.plot(epoch_val_aurocs, color='tab:red', linestyle='-.')
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{args["outprefix"]}_global_elbo.png')
    plt.clf()

    # prediction
    local_models = {}

    for task in pred_task_loader:
        task_label = data.target_task_name_map[task.item()]

        if args['adaptation'] == 'adaptive' and task_label in task_embeddings:
            z_t = task_embeddings[task_label]
            mu_adapt, log_sd_adapt = embedding_conditional_prior.compute_adaptive_prior(
                z_t, global_params=outer_state.params,
                adaptation_scale=hyper_params.get('adaptation_scale', 0.2)
            )
            prior_params = {n: outer_state.params[n] + mu_adapt[n] for n in outer_state.params}
            prior_log_sd = {n: outer_state.log_sd_diag[n] + log_sd_adapt[n] for n in outer_state.log_sd_diag}
        else:
            prior_params = outer_state.params
            prior_log_sd = outer_state.log_sd_diag
            
        num_inner_test = hyper_params.get('num_inner_updates_test', hyper_params['num_inner_updates'])
        samples = num_inner_test * args['batch_size'] * 2
        task_data = data.sample_supportquery_data(
            task_label, samples, sample_seed=0, eval=eval_mode,
            as_tensor=True, device=args['device']
        )
        with torch.no_grad():
            local_model = compute_task_nelbo(task_data, prior_params, prior_log_sd, return_local_model=True, num_inner_updates=num_inner_test)
        local_models[task_label] = local_model

    predictions = get_predictions(
        pred_task_loader, data.target_task_name_map, data,
        model, {'local': local_models}, hyper_params, args, pred_for
    )

    if pred_for == 'test':
        _save_outputs(predictions, model, outer_state, local_models, data, hyper_params, args)

    return predictions


def bnn_model(train_task_loader, pred_task_loader, data, hyper_params, args, pred_for='val'):
    if args['method'] == 'bnn_baseline':
        return bnn_baseline_model(train_task_loader, pred_task_loader, data, hyper_params, args, pred_for)
    return hierarchical_meta_model(train_task_loader, pred_task_loader, data, hyper_params, args, pred_for)


def _create_model(data, hyper_params, args):
    if hyper_params.get('nn_model_name') == 'linear_nn_model':
        num_features = len(data.predictors)
        n_layers = hyper_params.get('n_layers', 2)
        hidden_sizes = [hyper_params['hidden_layer_size_tabular']] * n_layers
        model = linear_nn_model(num_features, n_layers, hidden_sizes)
    else:
        model = sequence_nn_model(
            data.total_num_endpoints, len(data.predictors),
            hyper_params['hidden_layer_size_longitudinal'],
            hyper_params['hidden_layer_size_tabular']
        )
    model.to(args['device'])
    return model


def _load_embeddings(args):
    embedding_file = args.get('embeddingfile')
    if embedding_file is None:
        raise ValueError("--embeddingfile required for adaptive adaptation")
    embeddings, dim = load_task_embeddings(embedding_file, device=args['device'])
    print(f"Loaded {len(embeddings)} embeddings, dim={dim}")
    return embeddings, dim


def _save_outputs(predictions, model, outer_state, local_models, data, hyper_params, args):
    pred_df = get_pred_table(predictions)
    pred_df.to_csv(f'{args["outprefix"]}_pred_uncertainties.csv', index=None)

    X_train, X_long_train, _, X_test, X_long_test, _ = data.get_taskdata(
        task_label=None, eval=True, as_tensor=True, device=args['device']
    )
    X_data = torch.cat((X_train, X_test))
    X_long = torch.cat((X_long_train, X_long_test))
    save_embeddings(model, outer_state, "global", X_data, X_long, None, args)

    torch.save({
        'params': outer_state.params,
        'log_sd_diag': outer_state.log_sd_diag,
        'hyper_params': hyper_params
    }, f'{args["outprefix"]}_model_global.pth')

    for task_label, state in local_models.items():
        torch.save({
            'params': state.params,
            'log_sd_diag': state.log_sd_diag
        }, f'{args["outprefix"]}_model_local_{task_label}.pth')


def to_sd_diag(state):
    return tree_map(lambda x: x.exp(), state.log_sd_diag)


def get_statistics(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.where(probs < 1e-5, 1e-5, probs)
    probs /= probs.sum(dim=-1, keepdim=True)

    expected_probs = probs.mean(dim=1)
    y_pred = expected_probs[:, 1].cpu().detach().numpy().ravel()

    total_uncertainty = -(torch.log(expected_probs) * expected_probs).mean(1)
    aleatoric_uncertainty = -(torch.log(probs) * probs).mean(2).mean(1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return {
        'y_pred': y_pred,
        'y_pred_entropy': total_uncertainty.cpu().detach().numpy().ravel(),
        'y_pred_bin': expected_probs.argmax(dim=-1).cpu().detach().numpy().ravel(),
        'aleatoric_uncertainty': aleatoric_uncertainty.cpu().detach().numpy().ravel(),
        'epistemic_uncertainty': epistemic_uncertainty.cpu().detach().numpy().ravel()
    }


def forward(model, state, x_tab, x_long, args):
    sd_diag = to_sd_diag(state)
    sampled_params = posteriors.diag_normal_sample(state.params, sd_diag, (args['num_mc_samples'],))

    def model_func(p, x):
        return torch.func.functional_call(model, p, x)

    logits = torch.vmap(model_func, in_dims=(0, None))(sampled_params, (x_tab, x_long)).transpose(0, 1)
    return get_statistics(logits)


def get_predictions(taskloader, label_map, data, model, model_states, hyper_params, args, pred_for='val'):
    all_preds = {}

    for batch in taskloader:
        for task in batch:
            task_label = label_map[task.item()]
            state = model_states['local'][task_label]

            if pred_for == 'val':
                X_spt, X_qry, X_l_spt, X_l_qry, y_spt, y_qry, _, _ = data.sample_supportquery_data(
                    task_label, args['batch_size'] * 10, sample_seed=0,
                    eval=False, as_tensor=True, device=args['device'], dataset='val'
                )
                X_data = torch.cat((X_spt, X_qry), dim=0)
                X_long = torch.cat((X_l_spt, X_l_qry), dim=0)
                y_data = torch.cat((y_spt, y_qry), dim=0)
                data_ids = None
            else:
                _, _, _, _, X_data, X_long, y_data, data_ids = data.get_taskdata(
                    task_label, eval=True, as_tensor=True, device=args['device'], return_ids=True
                )

            results = {k: [] for k in ['y_pred', 'y_pred_entropy', 'y_pred_bin',
                                        'aleatoric_uncertainty', 'epistemic_uncertainty']}

            eval_loader = DataLoader(TensorDataset(X_data, X_long), batch_size=args['batch_size'], shuffle=False)

            for X_batch, X_long_batch in eval_loader:
                batch_results = forward(model, state, X_batch, X_long_batch, args)
                for key in results:
                    results[key].append(batch_results[key])

            for key in results:
                results[key] = np.concatenate(results[key])

            results['y_actual'] = y_data.cpu()
            if data_ids:
                results['data_ids'] = data_ids

            all_preds[task_label] = results

    return all_preds


def hierarchical_objective(hyper_params, train_loader, task_loader, data, args):
    return bnn_model(train_loader, task_loader, data, hyper_params, args, pred_for='val')


def hierarchical_best_model(best_params, train_loader, task_loader, data, args):
    return bnn_model(train_loader, task_loader, data, best_params, args, pred_for='test')