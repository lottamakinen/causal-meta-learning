# main.py
# Entry point for causal meta-learning experiments

import argparse
import numpy as np
import sys
import torch

sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'method')

from loader import DataModule
from hierarchical_model import hierarchical_objective
from hierarchical_eval_utils import evaluate_best_model


TOY_PARAMS = {
    'adaptive': {
        'nn_prior_name': 'multivariate_normal_prior',
        'inner_learning_rate': 0.05,
        'inner_temperature': 0.0005,
        'global_prior_sigma': 0.05,
        'model_init_log_sds': -3.0,
        'hidden_layer_size_longitudinal': 32,
        'hidden_layer_size_tabular': 32,
        'nn_model_name': 'sequence_nn_model',
        'num_inner_updates': 4,
        'prior_scaling': 10000,
        'outer_temperature': 0.0005,
        'outer_learning_rate': 0.005,
        'w_reg_lambda': 0.1,
        'w_learning_rate': 3e-4,
        'adaptation_scale': 0.12,
    },
    'baseline': {
        'nn_prior_name': 'multivariate_normal_prior',
        'inner_learning_rate': 0.05,
        'inner_temperature': 0.0005,
        'global_prior_sigma': 0.05,
        'model_init_log_sds': -3.0,
        'hidden_layer_size_longitudinal': 32,
        'hidden_layer_size_tabular': 32,
        'nn_model_name': 'sequence_nn_model',
        'num_inner_updates': 4,
        'prior_scaling': 10000,
        'outer_temperature': 0.0005,
        'outer_learning_rate': 0.005,
    },
    'bnn': {
        'num_inner_updates': 100, 
        'inner_learning_rate': 3e-3,
        'inner_temperature': 1.0,
        'global_prior_sigma': 1.0,
        'model_init_log_sds': -1.0,
        'nn_prior_name': 'multivariate_normal_prior',
        'hidden_layer_size_longitudinal': 32,
        'hidden_layer_size_tabular': 32,
        'nn_model_name': 'sequence_nn_model',
    },
}

UKBB_PARAMS = {
    'adaptive': {
        'MR': {
            'global_prior_sigma': 0.1,
            'hidden_layer_size_longitudinal': 32,
            'hidden_layer_size_tabular': 32,
            'inner_learning_rate': 1e-3,
            'inner_temperature': 1e-4,
            'model_init_log_sds': -3.0,
            'nn_model_name': 'sequence_nn_model',
            'nn_prior_name': 'multivariate_normal_prior',
            'num_inner_updates': 4,
            'outer_learning_rate': 1e-3,
            'outer_temperature': 1e-4,
            'prior_scaling': 10000,
            'w_reg_lambda': 0.05,
            'w_learning_rate': 1e-4,
            'adaptation_scale': 0.1,
        },
        'ICP': {
            'global_prior_sigma': 0.1,
            'hidden_layer_size_longitudinal': 32,
            'hidden_layer_size_tabular': 32,
            'inner_learning_rate': 1e-3,
            'inner_temperature': 1e-4,
            'model_init_log_sds': -3.0,
            'nn_model_name': 'sequence_nn_model',
            'nn_prior_name': 'multivariate_normal_prior',
            'num_inner_updates': 4,
            'outer_learning_rate': 1e-3,
            'outer_temperature': 1e-4,
            'prior_scaling': 10000,
            'w_reg_lambda': 0.05,
            'w_learning_rate': 1e-4,
            'adaptation_scale': 0.1,
        },
        'chi2': {
            'global_prior_sigma': 0.1,
            'hidden_layer_size_longitudinal': 32,
            'hidden_layer_size_tabular': 32,
            'inner_learning_rate': 1e-3,
            'inner_temperature': 1e-4,
            'model_init_log_sds': -3.0,
            'nn_model_name': 'sequence_nn_model',
            'nn_prior_name': 'multivariate_normal_prior',
            'num_inner_updates': 4,
            'outer_learning_rate': 1e-3,
            'outer_temperature': 1e-4,
            'prior_scaling': 10000,
            'w_reg_lambda': 0.05,
            'w_learning_rate': 1e-4,
            'adaptation_scale': 0.1,
        },
    },
    'baseline': {
        'global_prior_sigma': 0.1,
        'hidden_layer_size_longitudinal': 32,
        'hidden_layer_size_tabular': 32,
        'inner_learning_rate': 1e-3,
        'inner_temperature': 1e-4,
        'model_init_log_sds': -3.0,
        'nn_model_name': 'sequence_nn_model',
        'nn_prior_name': 'multivariate_normal_prior',
        'num_inner_updates': 4,
        'outer_learning_rate': 1e-3,
        'outer_temperature': 1e-4,
        'prior_scaling': 10000,
    },
    'bnn': {
        'num_inner_updates': 100,
        'inner_learning_rate': 1e-3,
        'inner_temperature': 1e-4,
        'global_prior_sigma': 0.1,
        'model_init_log_sds': -3.0,
        'prior_scaling': 10000,
        'nn_prior_name': 'multivariate_normal_prior',
        'hidden_layer_size_longitudinal': 32,
        'hidden_layer_size_tabular': 32,
        'nn_model_name': 'sequence_nn_model',
    },
}


def get_debug_params(args):
    dataset = args.get('dataset', 'toy')
    method = args['method']
    adaptation = args['adaptation']
    
    print(f"Loading params: dataset={dataset}, method={method}, adaptation={adaptation}")
    
    dataset_params = TOY_PARAMS if dataset == 'toy' else UKBB_PARAMS
    
    if method == 'bnn_baseline':
        base_params = dataset_params['bnn'].copy()
    elif method == '2_level_hierarchical':
        if adaptation == 'baseline':
            base_params = dataset_params['baseline'].copy()
        elif adaptation == 'adaptive':
            if dataset == 'ukbb':
                causal_method = args.get('causal_method', 'chi2')
                base_params = dataset_params['adaptive'].get(causal_method, dataset_params['adaptive']['chi2']).copy()
            else:
                base_params = dataset_params['adaptive'].copy()
        else:
            base_params = dataset_params['baseline'].copy()
    else:
        base_params = dataset_params['baseline'].copy()
    
    base_params['num_inner_updates'] = base_params.get('num_inner_updates', args['num_inner_updates'])
    base_params['num_inner_updates_test'] = args.get('num_inner_updates_test') or base_params['num_inner_updates']
    
    if args['data_type'] == 'tabular':
        base_params['nn_model_name'] = 'linear_nn_model'
    elif args['data_type'] == 'sequence':
        base_params['nn_model_name'] = 'sequence_nn_model'
    
    return base_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Meta-Learning")

    # data
    parser.add_argument('--tabular_datafile', type=str, required=True)
    parser.add_argument('--longitudinal_datafile', type=str, default='none')
    parser.add_argument('--metafile', type=str, required=True)
    parser.add_argument('--embeddingfile', type=str, default=None)
    parser.add_argument('--outprefix', type=str, required=True)

    # data config
    parser.add_argument('--data_type', type=str, default='sequence', choices=['tabular', 'sequence'])
    parser.add_argument('--learning_type', type=str, default='transductive', choices=['transductive', 'inductive'])
    parser.add_argument('--test_frac', type=float, default=0.5)
    parser.add_argument('--query_frac', type=float, default=0.5)
    parser.add_argument('--case_frac', type=float, default=0.5)

    # training
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=5)
    parser.add_argument('--max_num_epochs', type=int, default=30)
    parser.add_argument('--num_mc_samples', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_kfold_splits', type=int, default=2)
    parser.add_argument('--track_val_loss', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--track_val_auroc', type=lambda x: x.lower() == 'true', default=True)
    
    # early stopping
    parser.add_argument('--early_stopping', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_metric', type=str, default='auroc', choices=['auroc', 'nelbo'])

    # method
    parser.add_argument('--method', type=str, default='bnn_baseline', choices=['bnn_baseline', '2_level_hierarchical'])
    parser.add_argument('--adaptation', type=str, default='baseline', choices=['baseline', 'adaptive'])
    parser.add_argument('--causal_method', type=str, default='toy', choices=['toy', 'chi2', 'ICP', 'MR'])
    parser.add_argument('--embeddings', type=str, default='true', choices=['true', 'zero'])
    parser.add_argument('--dataset', type=str, default='toy', choices=['toy', 'ukbb'])
    
    # loss
    parser.add_argument('--use_class_weights', type=lambda x: x.lower() == 'true', default=False)
    
    # inner loop
    parser.add_argument('--num_inner_updates', type=int, default=4)
    parser.add_argument('--num_inner_updates_test', type=int, default=None)

    args = vars(parser.parse_args())
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {args['device']}")

    data = DataModule(
        tabular_datapath=args['tabular_datafile'],
        longitudinal_datapath=args['longitudinal_datafile'],
        metapath=args['metafile'],
        method=args['method'],
        learning_type=args['learning_type'],
        test_frac=args['test_frac'],
        query_frac=args['query_frac'],
        case_frac=args['case_frac'],
        n_kfold_splits=args['n_kfold_splits'],
        random_seed=args['random_seed'],
        data_type=args['data_type']
    )

    taskloader = data.get_taskloader(batchsize=1, shuffle=False, target_tasks=True)
    trainloader = data.get_taskloader(batchsize=args['minibatch_size'], shuffle=True, target_tasks=False)

    best_params = get_debug_params(args)

    taskloader = data.get_taskloader(batchsize=1, shuffle=False, target_tasks=True)
    trainloader = data.get_taskloader(batchsize=args['minibatch_size'], shuffle=True, target_tasks=False)
    evaluate_best_model(best_params, trainloader, taskloader, data, args)