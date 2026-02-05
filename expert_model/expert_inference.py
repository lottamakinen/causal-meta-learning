# expert_inference.py
# Expert-guided inference of target task embeddings using BALD

import torch
import numpy as np
import pandas as pd
import argparse
import itertools
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam


def euclidean_distance(x, y):
    return torch.norm(x - y)


def normalize_embedding(z):
    norm = torch.norm(z)
    return z / norm if norm > 1e-8 else z


def bernoulli_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def probit_preference_model(triplets, source_embeddings, responses=None, embedding_dim=None,
                           prior_mean=None, prior_std=None):
    """
    Probit preference model. P(y=1) = Phi(tau * delta_ij)
    where delta_ij = d_j - d_i (positive means i is closer).
    Larger tau = more reliable expert (sharper decisions).
    """
    tau = 1.0
    
    if prior_mean is None:
        prior_mean = torch.zeros(embedding_dim)
    if prior_std is None:
        prior_std = torch.ones(embedding_dim)
    
    z_target = pyro.sample("z_target", dist.Normal(prior_mean, prior_std).to_event(1))
    
    for n, (t, i, j) in enumerate(triplets):
        if responses is not None and n >= len(responses):
            break
        
        z_i = source_embeddings[i]
        z_j = source_embeddings[j]
        
        d_i = euclidean_distance(z_target, z_i)
        d_j = euclidean_distance(z_target, z_j)
        delta_ij = d_j - d_i
        
        # tau * delta: larger tau = sharper probit = more reliable
        prob = dist.Normal(0, 1).cdf(tau * delta_ij)
        
        obs = torch.tensor(float(responses[n])) if responses is not None and n < len(responses) else None
        pyro.sample(f"y_{n}", dist.Bernoulli(prob), obs=obs)


class ExpertInferenceSystem:
    
    def __init__(self, source_embeddings, embedding_dim):
        self.source_embeddings = source_embeddings
        self.embedding_dim = embedding_dim
        self.prior_mean = torch.zeros(embedding_dim)
        self.prior_std = torch.ones(embedding_dim)
        self.guide = None
        self.svi = None
        
    def _model(self, triplets, responses=None):
        return probit_preference_model(
            triplets, self.source_embeddings, responses, 
            self.embedding_dim, self.prior_mean, self.prior_std
        )
    
    def setup_inference(self):
        if self.svi is not None:
            return
        pyro.clear_param_store()
        self.guide = AutoDiagonalNormal(self._model)
        self.svi = SVI(self._model, self.guide, Adam({"lr": 0.01}), Trace_ELBO())
    
    def fit(self, triplets, responses, num_steps=500, verbose=True):
        self.setup_inference()
        losses = []
        for step in range(num_steps):
            loss = self.svi.step(triplets, responses)
            losses.append(loss)
            if verbose and step % 100 == 0:
                print(f"Step {step}: ELBO loss = {loss:.4f}")
        if verbose:
            print(f"Training complete. Final loss: {losses[-1]:.4f}")
        return losses
    
    def get_posterior_params(self):
        if self.guide is None:
            return self.prior_mean.clone(), self.prior_std.clone()
        
        params = dict(self.guide.named_parameters())
        if "loc" in params and "scale_unconstrained" in params:
            mean = params["loc"][:self.embedding_dim].detach().clone()
            std = torch.nn.functional.softplus(
                params["scale_unconstrained"][:self.embedding_dim]
            ).detach().clone()
        else:
            return self.prior_mean.clone(), self.prior_std.clone()
        return mean, std
    
    def sample_posterior(self, num_samples):
        mean, std = self.get_posterior_params()
        epsilon = torch.randn(num_samples, self.embedding_dim)
        return mean.unsqueeze(0) + std.unsqueeze(0) * epsilon
    
    def compute_response_prob(self, z_target, source_i, source_j):
        """P(y=1 | z_target) with tau=1 (matching inference model)."""
        tau = 1.0
        z_i = self.source_embeddings[source_i]
        z_j = self.source_embeddings[source_j]
        d_i = euclidean_distance(z_target, z_i)
        d_j = euclidean_distance(z_target, z_j)
        delta_ij = d_j - d_i
        return dist.Normal(0, 1).cdf(tau * delta_ij).item()


def compute_bald_batch(system, candidates, num_samples=500):
    z_samples = system.sample_posterior(num_samples)
    results = []
    
    for (i, j) in candidates:
        z_i = system.source_embeddings[i]
        z_j = system.source_embeddings[j]
        
        probs = []
        conditional_entropies = []
        
        for s in range(num_samples):
            z = z_samples[s]
            d_i = euclidean_distance(z, z_i).item()
            d_j = euclidean_distance(z, z_j).item()
            delta_ij = d_j - d_i
            p = dist.Normal(0, 1).cdf(torch.tensor(delta_ij)).item()
            probs.append(p)
            conditional_entropies.append(bernoulli_entropy(p))
        
        expected_cond_entropy = np.mean(conditional_entropies)
        marginal_prob = np.mean(probs)
        marginal_entropy = bernoulli_entropy(marginal_prob)
        bald_score = max(0, marginal_entropy - expected_cond_entropy)
        prob_std = np.std(probs)
        
        results.append((bald_score, prob_std, marginal_prob, i, j))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def select_query_bald(system, source_tasks, target_task, used_pairs, 
                      num_samples=500, verbose=True):
    candidates = [(i, j) for i, j in itertools.combinations(source_tasks, 2)
                  if frozenset((i, j)) not in used_pairs]
    
    if not candidates:
        return None
    
    if verbose:
        print(f"Computing BALD for {len(candidates)} pairs...")
    
    results = compute_bald_batch(system, candidates, num_samples)
    
    if verbose:
        print("\n  Top 5 pairs:")
        for rank, (score, prob_std, marginal_prob, i, j) in enumerate(results[:5]):
            print(f"    {rank+1}. {i} vs {j}: BALD={score:.4f}, P(y=1)={marginal_prob:.3f}")
        
        all_scores = [r[0] for r in results]
        print(f"\n  BALD stats: min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}\n")
    
    best = results[0]
    return (target_task, best[3], best[4])


def select_query_random(source_tasks, target_task, used_pairs):
    candidates = [(i, j) for i, j in itertools.combinations(source_tasks, 2)
                  if frozenset((i, j)) not in used_pairs]
    if not candidates:
        return None
    i, j = candidates[np.random.randint(len(candidates))]
    return (target_task, i, j)


def simulate_expert_response(target_emb, source_i_emb, source_j_emb, 
                             tau_expert=1.0, deterministic=False):
    """
    Simulate expert with probit model. P(y=1) = Phi(tau * delta).
    Larger tau = more reliable expert (less noise in decisions).
    """
    d_i = euclidean_distance(target_emb, source_i_emb).item()
    d_j = euclidean_distance(target_emb, source_j_emb).item()
    delta = d_j - d_i
    prob = dist.Normal(0, 1).cdf(torch.tensor(tau_expert * delta)).item()
    
    if deterministic:
        return 1 if prob > 0.5 else 0
    return np.random.binomial(1, prob)


def load_data(embedding_file, meta_file, normalize=False, expert_embedding_file=None):
    meta_df = pd.read_csv(meta_file)
    emb_df = pd.read_csv(embedding_file)
    
    z_cols = [col for col in emb_df.columns if col.startswith(("z_", "feature_", "PC_"))]
    
    source_embeddings = {}
    for _, row in emb_df.iterrows():
        task_name = row['task']
        z_vec = torch.tensor([row[col] for col in z_cols], dtype=torch.float32)
        if normalize:
            z_vec = normalize_embedding(z_vec)
        source_embeddings[task_name] = z_vec
    
    if expert_embedding_file is not None:
        expert_emb_df = pd.read_csv(expert_embedding_file)
        expert_z_cols = [col for col in expert_emb_df.columns if col.startswith(("z_", "feature_", "PC_"))]
        
        expert_embeddings = {}
        for _, row in expert_emb_df.iterrows():
            task_name = row['task']
            z_vec = torch.tensor([row[col] for col in expert_z_cols], dtype=torch.float32)
            if normalize:
                z_vec = normalize_embedding(z_vec)
            expert_embeddings[task_name] = z_vec
        
        print(f"Loaded {len(source_embeddings)} source embeddings, {len(expert_embeddings)} expert embeddings")
    else:
        expert_embeddings = source_embeddings
        print(f"Using same embeddings for inference and expert ({len(source_embeddings)} tasks)")
    
    task_meta = meta_df[meta_df['column_type'].isin(['task_label', 'target_task'])]
    train_tasks = task_meta[task_meta['column_type'] == 'task_label']['column_name'].tolist()
    target_tasks = task_meta[task_meta['column_type'] == 'target_task']['column_name'].tolist()
    
    train_tasks = [t for t in train_tasks if t in source_embeddings]
    
    missing_targets = [t for t in target_tasks if t not in expert_embeddings]
    if missing_targets:
        raise ValueError(f"Target tasks missing from expert embeddings: {missing_targets}")
    
    return train_tasks, target_tasks, source_embeddings, expert_embeddings


def run_experiment(args):
    train_tasks, target_tasks, source_embeddings, expert_embeddings = load_data(
        args['embedding_file'], args['meta_file'], normalize=args['normalize'],
        expert_embedding_file=args.get('expert_embedding_file')
    )
    
    source_dim = len(next(iter(source_embeddings.values())))
    expert_dim = len(next(iter(expert_embeddings.values())))
    dimension_mismatch = source_dim != expert_dim
    
    if dimension_mismatch:
        print(f"\nWARNING: Dimension mismatch (source={source_dim}, expert={expert_dim})")
        print("RMSE/cosine metrics will not be computed.\n")
    
    embedding_dim = source_dim
    
    print(f"\nSource tasks: {len(train_tasks)}, Target tasks: {len(target_tasks)}")
    print(f"Embedding dim: {embedding_dim}, Mode: {args['mode']}, Queries: {args['total_queries']}")
    print(f"Expert tau: {args['tau_expert']} (larger = more reliable)\n")
    
    all_results = []
    all_results_embeddings = {}
    
    for target_task in target_tasks:
        print(f"\n{'='*50}")
        print(f"Target: {target_task}")
        print(f"{'='*50}")
        
        pyro.clear_param_store()
        system = ExpertInferenceSystem(source_embeddings, embedding_dim)
        system.setup_inference()
        
        triplets = []
        responses = []
        used_pairs = set()
        true_target = expert_embeddings[target_task]
        
        for q in range(args['total_queries'] + 1):
            print(f"\n--- Query {q}/{args['total_queries']} ---")
            
            if triplets:
                system.fit(triplets, responses, num_steps=args['svi_steps_per_query'], verbose=False)
            
            learned_mean, learned_std = system.get_posterior_params()
            
            if dimension_mismatch:
                rmse, cosine = float('nan'), float('nan')
            else:
                rmse = torch.norm(learned_mean - true_target).item() / np.sqrt(embedding_dim)
                cosine = torch.cosine_similarity(learned_mean.unsqueeze(0), true_target.unsqueeze(0)).item()
            mean_std = learned_std.mean().item()
            
            print(f"RMSE: {rmse:.4f}, Cosine: {cosine:.4f}, Posterior std: {mean_std:.4f}")
            
            all_results.append({
                'target': target_task,
                'query': q,
                'rmse': rmse,
                'cosine': cosine,
                'mean_std': mean_std,
                'tau_expert': args['tau_expert'],
                'mode': args['mode'],
                'seed': args['seed']
            })
            
            if q == args['total_queries']:
                if args['save_embeddings']:
                    if target_task not in all_results_embeddings:
                        all_results_embeddings[target_task] = {}
                    all_results_embeddings[target_task][q] = learned_mean.numpy()
                break
            
            if args['save_embeddings'] and q > 0 and q % 5 == 0:
                if target_task not in all_results_embeddings:
                    all_results_embeddings[target_task] = {}
                all_results_embeddings[target_task][q] = learned_mean.numpy()
            
            # select next query
            if args['mode'] == 'random':
                next_triplet = select_query_random(train_tasks, target_task, used_pairs)
            elif args['mode'] == 'bald':
                next_triplet = select_query_bald(
                    system, train_tasks, target_task, used_pairs,
                    num_samples=args['bald_samples'], verbose=args['verbose']
                )
            else:
                raise ValueError(f"Unknown mode: {args['mode']}")
            
            if next_triplet is None:
                print("No more queries available")
                break
            
            _, i, j = next_triplet
            print(f"Selected: {i} vs {j}")
            
            response = simulate_expert_response(
                expert_embeddings[target_task], expert_embeddings[i], expert_embeddings[j],
                tau_expert=args['tau_expert'], 
                deterministic=args['deterministic_expert']
            )
            
            d_i = euclidean_distance(expert_embeddings[target_task], expert_embeddings[i]).item()
            d_j = euclidean_distance(expert_embeddings[target_task], expert_embeddings[j]).item()
            expected = "i" if d_i < d_j else "j"
            actual = "i" if response == 1 else "j"
            match = "✓" if expected == actual else "✗"
            print(f"Expert chose {actual} {match} (d_i={d_i:.3f}, d_j={d_j:.3f})")
            
            triplets.append(next_triplet)
            responses.append(response)
            used_pairs.add(frozenset((i, j)))
    
    # save results
    results_df = pd.DataFrame(all_results)
    output_file = f"{args['outprefix']}_results_mode_{args['mode']}_tau_{args['tau_expert']}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # save embeddings
    if args['save_embeddings']:
        checkpoints = [q for q in range(5, args['total_queries'] + 1, 5)]
        if args['total_queries'] not in checkpoints:
            checkpoints.append(args['total_queries'])
        
        for checkpoint_q in checkpoints:
            emb_records = []
            
            for task in train_tasks:
                emb = source_embeddings[task].numpy()
                record = {'task': task, 'type': 'source'}
                for i, val in enumerate(emb):
                    record[f'z_{i}'] = val
                emb_records.append(record)
            
            for target_task in target_tasks:
                emb_dict = all_results_embeddings.get(target_task, {})
                if checkpoint_q in emb_dict:
                    final_emb = emb_dict[checkpoint_q]
                    record = {'task': target_task, 'type': 'inferred'}
                    for i, val in enumerate(final_emb):
                        record[f'z_{i}'] = val
                    emb_records.append(record)
                elif target_task in source_embeddings:
                    emb = source_embeddings[target_task].numpy()
                    record = {'task': target_task, 'type': 'original'}
                    for i, val in enumerate(emb):
                        record[f'z_{i}'] = val
                    emb_records.append(record)
            
            emb_df = pd.DataFrame(emb_records)
            emb_output = f"{args['outprefix']}_inferred_embeddings_mode_{args['mode']}_tau_{args['tau_expert']}_q{checkpoint_q}.csv"
            emb_df.to_csv(emb_output, index=False)
            print(f"Embeddings (Q={checkpoint_q}) saved to {emb_output}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert-guided embedding inference with BALD")
    
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--expert_embedding_file', type=str, default=None)
    parser.add_argument('--meta_file', type=str, required=True)
    parser.add_argument('--outprefix', type=str, default='results/expert')
    parser.add_argument('--normalize', action='store_true')
    
    parser.add_argument('--tau_expert', type=float, default=1.0, 
                        help='Expert reliability (larger = more reliable)')
    parser.add_argument('--deterministic_expert', action='store_true')
    
    parser.add_argument('--total_queries', type=int, default=20)
    parser.add_argument('--mode', type=str, choices=['random', 'bald'], default='bald')
    parser.add_argument('--bald_samples', type=int, default=200)
    
    parser.add_argument('--svi_steps', type=int, default=500)
    parser.add_argument('--svi_steps_per_query', type=int, default=150)
    
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_embeddings', action='store_true')
    
    args = vars(parser.parse_args())
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    pyro.set_rng_seed(args['seed'])
    
    run_experiment(args)