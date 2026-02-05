# Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings

Code for paper "Bayesian Meta-Learning with Expert Feedback for Task-Shift Adaptation through Causal Embeddings".

## Methods

| Method | Description |
|--------|-------------|
| `bnn_baseline` | Standard Bayesian neural network (no meta-learning) |
| `2_level_hierarchical` + `baseline` | Two-level hierarchical meta-learning with global prior |
| `2_level_hierarchical` + `adaptive` | Two-level hierarchical meta-learning with embedding-conditional prior: `μ(z_t) = μ_θ + W_μ · z_t` |

## Setup

```bash
python3 -m venv experiment_env
source experiment_env/bin/activate
pip install -r requirements.txt
```

## Data Inputs

Four CSV files are required:

### 1. Tabular Data File (`--tabular_datafile`)

Contains patient IDs, features, labels, and cohort definitions:

| pid | V0 | V1 | ... | Vk | task_0 | task_1 | ... | cohort_task_0 | cohort_task_1 | ... |
|-----|----|----|-----|----|---------|---------|----|---------------|---------------|-----|
| patient0 | 0.5 | 1.2 | ... | 0.3 | 0 | 1 | ... | 1 | 1 | ... |

### 2. Longitudinal Data File (`--longitudinal_datafile`)

Time series features in sparse format:

| PATIENT_ID | EVENT_YEAR | ENDPOINT |
|------------|------------|----------|
| patient0 | 2000 | V0 |
| patient0 | 2001 | V1 |

### 3. Metadata File (`--metafile`)

Describes columns in the tabular file:

| column_name | column_type | task_cohort |
|-------------|-------------|-------------|
| pid | patient_id | |
| V0 | predictor | |
| task_0 | task_label | |
| task_20 | target_task | |
| cohort_task_0 | cohort | task_0 |

Column types: `patient_id`, `predictor`, `task_label`, `target_task`, `cohort`

### 4. Task Embeddings File (`--embeddingfile`)

Required for `--adaptation adaptive`. Contains causal embeddings for each task:

| task | z_0 | z_1 | z_2 | z_3 |
|------|-----|-----|-----|-----|
| task_0 | 0.5 | -0.3 | 1.2 | 0.1 |
| task_1 | -0.2 | 0.7 | -0.5 | 0.8 |

Supported column prefixes: `z_`, `feature_`, `PC_`

## Usage

### Running Experiments

#### BNN Baseline (No Meta-Learning)

```bash
python method/main.py \
    --tabular_datafile data/example/latent_tabular_data.csv \
    --longitudinal_datafile data/example/latent_longitudinal_data.csv \
    --metafile data/example/latent_col_metadata.csv \
    --outprefix results/bnn_baseline \
    --data_type sequence \
    --learning_type transductive \
    --method bnn_baseline \
    --dataset toy
```

#### Two-Level Hierarchical (Global Prior)

```bash
python method/main.py \
    --tabular_datafile data/example/latent_tabular_data.csv \
    --longitudinal_datafile data/example/latent_longitudinal_data.csv \
    --metafile data/example/latent_col_metadata.csv \
    --outprefix results/2level_baseline \
    --data_type sequence \
    --learning_type inductive \
    --method 2_level_hierarchical \
    --adaptation baseline \
    --dataset toy
```

#### Two-Level Hierarchical with Embedding-Conditional Prior

```bash
python method/main.py \
    --tabular_datafile data/example/latent_tabular_data.csv \
    --longitudinal_datafile data/example/latent_longitudinal_data.csv \
    --metafile data/example/latent_col_metadata.csv \
    --embeddingfile data/example/latent_embeddings_causal_noise0.0.csv \
    --outprefix results/2level_adaptive \
    --data_type sequence \
    --learning_type inductive \
    --method 2_level_hierarchical \
    --adaptation adaptive \
    --dataset toy
```


### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tabular_datafile` | Path to tabular data CSV | Required |
| `--longitudinal_datafile` | Path to longitudinal data CSV | `none` |
| `--metafile` | Path to metadata CSV | Required |
| `--embeddingfile` | Path to task embeddings CSV | None |
| `--outprefix` | Output file prefix | Required |
| `--data_type` | `tabular` or `sequence` | `sequence` |
| `--learning_type` | `transductive` or `inductive` | `transductive` |
| `--method` | `bnn_baseline` or `2_level_hierarchical` | `bnn_baseline` |
| `--adaptation` | `baseline` or `adaptive` | `baseline` |
| `--embeddings` | `true` or `zero` (ablation) | `true` |
| `--dataset` | `toy` or `ukbb` (hyperparameter preset) | `toy` |
| `--batch_size` | Batch size for task data | `100` |
| `--max_num_epochs` | Maximum training epochs | `30` |
| `--num_mc_samples` | MC samples for inference | `5` |
| `--random_seed` | Random seed | `42` |
| `--early_stopping` | Enable early stopping | `true` |
| `--early_stopping_patience` | Patience epochs | `5` |
| `--early_stopping_metric` | `auroc` or `nelbo` | `auroc` |

### Outputs

Files saved to `{outprefix}_*`:

- `*_task_classification_metrics.csv` - Per-task metrics (AUROC, AUPRC, etc.)
- `*_pred_uncertainties.csv` - Predictions with uncertainty estimates
- `*_global_elbo.png` - Training loss curve
- `*_model_global.pth` - Global model weights
- `*_model_local_{task}.pth` - Local model weights per task

---

## Expert-Guided Inference

For target tasks where embeddings are unknown, we can infer them from expert pairwise comparisons using BALD (Bayesian Active Learning by Disagreement).

### Method

Given source task embeddings and a target task, the expert is queried: "Which source task is more similar to the target?" The system uses a probit preference model and variational inference to estimate the target embedding.

### Running Expert Inference

```bash
python expert_model/expert_inference.py \
    --embedding_file data/example/latent_embeddings_causal_noise0.0.csv \
    --expert_embedding_file data/example/latent_embeddings_causal_noise0.0.csv \
    --meta_file data/example/latent_col_metadata.csv \
    --outprefix results/expert \
    --mode bald \
    --total_queries 20 \
    --tau_expert 1.0 \
    --save_embeddings
```

### Expert Inference Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--embedding_file` | Source embeddings for inference model | Required |
| `--expert_embedding_file` | Embeddings for expert simulation (can include targets) | Same as embedding_file |
| `--meta_file` | Metadata CSV | Required |
| `--outprefix` | Output prefix | `results/expert` |
| `--mode` | `bald` or `random` | `bald` |
| `--total_queries` | Number of pairwise queries | `20` |
| `--tau_expert` | Expert reliability (larger = more reliable) | `1.0` |
| `--bald_samples` | MC samples for BALD | `200` |
| `--svi_steps_per_query` | VI steps per query | `150` |
| `--normalize` | Normalize embeddings to unit norm | False |
| `--deterministic_expert` | Use deterministic expert responses | False |
| `--save_embeddings` | Save inferred embeddings to CSV | False |
| `--seed` | Random seed | `42` |

### Expert Inference Outputs

- `*_results_mode_{mode}_tau_{tau}.csv` - RMSE and cosine similarity per query
- `*_inferred_embeddings_mode_{mode}_tau_{tau}_q{n}.csv` - Inferred embeddings at query checkpoints

### Using Inferred Embeddings

After running expert inference, use the inferred embeddings for meta-learning:

```bash
python method/main.py \
    --tabular_datafile data/example/latent_tabular_data.csv \
    --longitudinal_datafile data/example/latent_longitudinal_data.csv \
    --metafile data/example/latent_col_metadata.csv \
    --embeddingfile results/expert_inferred_embeddings_mode_bald_tau_1.0_q20.csv \
    --outprefix results/2level_adaptive_inferred \
    --method 2_level_hierarchical \
    --adaptation adaptive \
    --dataset toy
```

---

## Code Structure

```
├── method/
│   ├── main.py                        # Entry point
│   ├── hierarchical_model.py          # Training and inference
│   ├── embedding_conditional_prior.py # μ(z_t) = μ_θ + W_μ · z_t
│   └── hierarchical_eval_utils.py     # Evaluation utilities
├── expert_model/
│   └── expert_inference.py            # BALD-based expert inference
├── data_generator/
│   └── data_generator.py              # Synthetic data generation
├── metrics/
│   ├── accuracy.py                    # Classification metrics
│   └── explainability.py              # Embedding extraction
├── utils/
│   ├── bnn_models.py                  # Neural network architectures
│   ├── bnn_priors.py                  # Prior distributions
│   ├── custom_lstm.py                 # LSTM implementation
│   ├── early_stopping.py              # Early stopping
│   └── loader.py                      # Data loading
├── causal_distances/
│   ├── README.md                      # Causal embedding documentation
│   ├── long_utils.py                  # Longitudinal data utilities
│   ├── BASELINE/                      # Chi-square baseline embeddings
│   ├── ICP/                           # Invariant Causal Prediction
│   └── MR/                            # Mendelian Randomization
├── data/
│   └── example/                       # Example data directory
└── results/                           # Output directory
```      

## Acknowledgements

This repository is based on and adapts code originally developed by Sophie Wharrie, released under the MIT License.

- [Posteriors](https://github.com/normal-computing/posteriors)
- [TorchOpt](https://github.com/metaopt/torchopt)
- [Pyro](https://pyro.ai/)
- [TwoSampleMR (R package)](https://mrcieu.github.io/TwoSampleMR/articles/introduction.html)
- [InvariantCausalPrediction (R package)](https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html)
