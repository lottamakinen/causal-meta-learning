# embedding_conditional_prior.py
# Embedding-conditional prior: μ(z_t) = μ_θ + W_μ · z_t

import torch
import pandas as pd
from torch import nn


class EmbeddingConditionalPrior(nn.Module):
    
    def __init__(self, embedding_dim, global_params, global_log_sd, device='cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.param_names = list(global_params.keys())
        self.param_shapes = {name: global_params[name].shape for name in self.param_names}

        self.W_mu = nn.ParameterDict()
        for name in self.param_names:
            param_size = global_params[name].numel()
            safe_name = name.replace('.', '_')
            self.W_mu[safe_name] = nn.Parameter(
                torch.zeros(param_size, embedding_dim, device=device)
            )

        print(f"EmbeddingConditionalPrior: dim={embedding_dim}, params={len(self.param_names)}")

    def compute_adaptive_prior(self, z_t, global_params=None, adaptation_scale=0.2):
        if z_t.dim() > 1:
            z_t = z_t.squeeze(0)

        mu_adaptations = {}
        log_sd_adaptations = {}

        for name in self.param_names:
            safe_name = name.replace('.', '_')
            mu_adaptation = torch.matmul(self.W_mu[safe_name], z_t)
            mu_adapt_shaped = mu_adaptation.reshape(self.param_shapes[name])

            if global_params is not None and name in global_params:
                global_norm = global_params[name].norm()
                adapt_norm = mu_adapt_shaped.norm()
                if adapt_norm > 0:
                    target_norm = adaptation_scale * global_norm
                    mu_adapt_shaped = mu_adapt_shaped * (target_norm / adapt_norm)

            mu_adaptations[name] = mu_adapt_shaped
            log_sd_adaptations[name] = torch.zeros_like(mu_adapt_shaped)

        return mu_adaptations, log_sd_adaptations

    def forward(self, z_t, global_params=None, adaptation_scale=0.2):
        return self.compute_adaptive_prior(z_t, global_params, adaptation_scale)


def load_task_embeddings(embedding_file, device='cpu'):
    df = pd.read_csv(embedding_file)

    for prefix in ['z_', 'feature_', 'PC_']:
        if any(col.startswith(prefix) for col in df.columns):
            break
    else:
        raise ValueError("No embedding columns found")

    embedding_cols = [col for col in df.columns if col.startswith(prefix)]
    embedding_dim = len(embedding_cols)

    print(f"Loading embeddings: {len(df)} tasks, dim={embedding_dim}")

    embeddings_dict = {}
    for _, row in df.iterrows():
        task_name = row['task']
        embedding = torch.tensor(
            [row[col] for col in embedding_cols],
            dtype=torch.float32,
            device=device
        )
        embeddings_dict[task_name] = embedding

    return embeddings_dict, embedding_dim