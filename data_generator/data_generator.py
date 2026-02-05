# data_generator.py
# Hierarchical latent causal generator with interpretable embeddings

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GeneratorConfig:
    num_predictors: int = 10
    parents: Tuple[int, ...] = (0, 1, 2, 3)
    z_dim: int = 4
    
    num_train_tasks: int = 20
    samples_per_task: int = 500
    source_z_std: float = 1.0
    target_ood_levels: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    
    sigma_e: float = 0.15
    sigma_y: float = 0.6
    pos_rate: float = 0.30

    shortcut_feature: int = 9
    shortcut_strength: float = 2.5
    shortcut_in_targets: str = "soft"
    
    embedding_noise_levels: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.8)
    
    seed: int = 123
    outprefix: str = "data/toy2/latent"


def make_W_and_b(cfg: GeneratorConfig, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Create weight matrix W (D x z_dim) and bias b (D,). Only parents have non-zero weights."""
    D, dz = cfg.num_predictors, cfg.z_dim
    W = np.zeros((D, dz))
    b = np.zeros(D)

    for p in cfg.parents:
        W[p, :] = rng.normal(0.0, 0.5, size=dz)
        b[p] = rng.uniform(0.5, 1.0)

    return W, b


def sample_task_effects(z_t: np.ndarray, W: np.ndarray, b: np.ndarray,
                        sigma_e: float, rng: np.random.RandomState) -> np.ndarray:
    """Compute task coefficients: e_t = b + W @ z_t + η"""
    D = b.shape[0]
    eta = rng.normal(0.0, sigma_e, size=D)
    return b + W.dot(z_t) + eta


def generate_task_data(X: np.ndarray, e_t: np.ndarray, parents: Tuple[int, ...],
                       sigma_y: float, pos_rate: float, 
                       rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Y from X using task coefficients, then binarize."""
    y_cont = X[:, list(parents)].dot(e_t[list(parents)]) + rng.normal(0.0, sigma_y, size=X.shape[0])
    thr = np.quantile(y_cont, 1.0 - pos_rate)
    y_bin = (y_cont > thr).astype(int)
    return y_cont, y_bin


def inject_shortcut_feature(X, y_bin, cfg, is_target, rng, ood_s=None):
    """Source: shortcut correlated with y. Target: shortcut decays with OOD."""
    j = cfg.shortcut_feature
    signal = (2 * y_bin - 1).astype(float)

    if not is_target:
        X[:, j] = cfg.shortcut_strength * signal + rng.normal(0.0, 1.0, size=X.shape[0])
        return X

    mode = cfg.shortcut_in_targets
    if mode == "flip":
        X[:, j] = -cfg.shortcut_strength * signal + rng.normal(0.0, 1.0, size=X.shape[0])
    elif mode == "noise":
        X[:, j] = rng.normal(0.0, 1.0, size=X.shape[0])
    else:  # soft
        s_max = max(cfg.target_ood_levels) if len(cfg.target_ood_levels) > 0 else 1.0
        alpha = max(0.0, 1.0 - (ood_s / (s_max + 1e-12)))
        X[:, j] = (alpha * cfg.shortcut_strength) * signal + rng.normal(0.0, 1.0, size=X.shape[0])
    
    return X


def compute_correlation_embeddings(task_data: List[Tuple[np.ndarray, np.ndarray]], 
                                   z_dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Compute correlation-based embeddings via PCA on corr(X_j, Y) vectors."""
    corr_vectors = []
    for X, Y in task_data:
        corrs = []
        for j in range(X.shape[1]):
            c = np.corrcoef(X[:, j], Y)[0, 1]
            corrs.append(c if not np.isnan(c) else 0.0)
        corr_vectors.append(corrs)
    
    corr_matrix = np.array(corr_vectors)
    pca = PCA(n_components=z_dim, random_state=rng.randint(10000))
    return pca.fit_transform(corr_matrix)


def generate_dataset(cfg: GeneratorConfig):
    """Main generation function."""
    rng_structure = np.random.RandomState(cfg.seed)
    rng_data = np.random.RandomState(cfg.seed + 1000)
    
    os.makedirs(os.path.dirname(cfg.outprefix) or ".", exist_ok=True)

    W, b = make_W_and_b(cfg, rng_structure)
    D = cfg.num_predictors
    P = cfg.parents
    dz = cfg.z_dim

    all_frames: List[pd.DataFrame] = []
    task_data_for_corr: List[Tuple[np.ndarray, np.ndarray]] = []
    task_Z: List[np.ndarray] = []
    task_E: List[np.ndarray] = []
    task_roles: List[str] = []
    task_names: List[str] = []
    task_ood_levels: List[float] = []
    task_idx = 0

    # source tasks
    print(f"Generating {cfg.num_train_tasks} source tasks...")
    print(f"  Causal parents: {cfg.parents}")
    
    for t in range(cfg.num_train_tasks):
        z_t = rng_structure.normal(0.0, cfg.source_z_std, size=dz)
        e_t = sample_task_effects(z_t, W, b, cfg.sigma_e, rng_structure)
        
        X = rng_data.normal(0.0, 1.0, size=(cfg.samples_per_task, D))
        y_cont, y_bin = generate_task_data(X, e_t, P, cfg.sigma_y, cfg.pos_rate, rng_data)
        X = inject_shortcut_feature(X, y_bin, cfg, is_target=False, rng=rng_data)

        if t == 0:
            c = np.corrcoef(X[:, cfg.shortcut_feature], y_bin)[0, 1]
            print(f"  [SOURCE shortcut corr] corr(V{cfg.shortcut_feature}, y)={c:.3f}")

        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(D)])
        df[f"task_{task_idx}"] = y_bin
        df[f"cohort_task_{task_idx}"] = 1
        df["group"] = "SOURCE"
        df["_task_index_source"] = task_idx
        all_frames.append(df)

        task_data_for_corr.append((X, y_bin))
        task_Z.append(z_t)
        task_E.append(e_t)
        task_roles.append("train")
        task_names.append("SOURCE")
        task_ood_levels.append(None)
        task_idx += 1
    
    source_Z = np.vstack(task_Z)
    source_z_mean = source_Z.mean(axis=0)
    print(f"  Source z mean: {source_z_mean}")
    print(f"  Source z std:  {source_Z.std(axis=0)}")
    
    for i in range(len(task_Z)):
        task_ood_levels[i] = np.linalg.norm(task_Z[i] - source_z_mean)

    # target tasks
    print(f"Generating {len(cfg.target_ood_levels)} target tasks...")
    
    direction = rng_structure.normal(0.0, 1.0, size=dz)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    target_task_indices: List[int] = []
    
    for s in cfg.target_ood_levels:
        z_t = s * direction
        e_t = sample_task_effects(z_t, W, b, cfg.sigma_e, rng_structure)
        
        X = rng_data.normal(0.0, 1.0, size=(cfg.samples_per_task, D))
        y_cont, y_bin = generate_task_data(X, e_t, P, cfg.sigma_y, cfg.pos_rate, rng_data)
        X = inject_shortcut_feature(X, y_bin, cfg, is_target=True, rng=rng_data, ood_s=s)

        if s == cfg.target_ood_levels[0]:
            c = np.corrcoef(X[:, cfg.shortcut_feature], y_bin)[0, 1]
            print(f"  [TARGET shortcut corr] corr(V{cfg.shortcut_feature}, y)={c:.3f}")

        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(D)])
        df[f"task_{task_idx}"] = y_bin
        df[f"cohort_task_{task_idx}"] = 1
        df["group"] = f"TARGET_OOD{s:.1f}"
        df["_task_index_source"] = task_idx
        all_frames.append(df)

        task_data_for_corr.append((X, y_bin))
        task_Z.append(z_t)
        task_E.append(e_t)
        task_roles.append("target")
        task_names.append(f"TARGET_OOD{s:.1f}")
        task_ood_levels.append(np.linalg.norm(z_t - source_z_mean))
        target_task_indices.append(task_idx)
        
        print(f"  Target s={s:.1f}: ||z_t||={np.linalg.norm(z_t):.3f}, OOD={np.linalg.norm(z_t - source_z_mean):.3f}")
        task_idx += 1

    # combine data
    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined.insert(0, "pid", [f"patient{i}" for i in range(len(combined))])
    
    task_cols = [c for c in combined.columns if c.startswith("task_") or c.startswith("cohort_task_")]
    combined[task_cols] = combined[task_cols].fillna(0).astype(int)

    # metadata
    metadata = [{"column_name": "pid", "column_type": "patient_id", "task_cohort": ""}]
    for i in range(D):
        metadata.append({"column_name": f"V{i}", "column_type": "predictor", "task_cohort": ""})
    for i in range(task_idx):
        coltype = "target_task" if i in target_task_indices else "task_label"
        metadata.append({"column_name": f"task_{i}", "column_type": coltype, "task_cohort": ""})
        metadata.append({"column_name": f"cohort_task_{i}", "column_type": "cohort", "task_cohort": f"task_{i}"})
    meta_df = pd.DataFrame(metadata)

    # embeddings
    Z = np.vstack(task_Z)
    E = np.vstack(task_E)
    num_tasks = task_idx
    task_names_list = [f"task_{i}" for i in range(num_tasks)]

    print("\nGenerating embedding variants...")
    
    noise_directions = np.zeros_like(Z)
    for t in range(num_tasks):
        d = rng_data.normal(0, 1, size=dz)
        noise_directions[t] = d / (np.linalg.norm(d) + 1e-10)
    
    for noise in cfg.embedding_noise_levels:
        emb_df = pd.DataFrame({"task": task_names_list})
        if noise == 0:
            Z_noisy = Z.copy()
        else:
            Z_noisy = np.zeros_like(Z)
            for t in range(num_tasks):
                z_orig = Z[t]
                z_norm = np.linalg.norm(z_orig)
                z_perturbed = z_orig + noise * z_norm * noise_directions[t]
                Z_noisy[t] = z_perturbed * (z_norm / (np.linalg.norm(z_perturbed) + 1e-10))
        
        for d in range(dz):
            emb_df[f"z_{d}"] = Z_noisy[:, d]
        emb_df["role"] = task_roles
        emb_df["ood_level"] = task_ood_levels
        emb_df["embedding_norm"] = [np.linalg.norm(Z_noisy[i]) for i in range(num_tasks)]
        
        fname = f"{cfg.outprefix}_embeddings_causal_noise{noise}.csv"
        emb_df.to_csv(fname, index=False)
        print(f"  Saved: {fname}")

    # correlation embeddings
    Z_corr = compute_correlation_embeddings(task_data_for_corr, dz, rng_data)
    emb_corr_df = pd.DataFrame({"task": task_names_list})
    for d in range(dz):
        emb_corr_df[f"z_{d}"] = Z_corr[:, d]
    emb_corr_df["role"] = task_roles
    emb_corr_df["ood_level"] = task_ood_levels
    emb_corr_df["embedding_norm"] = [np.linalg.norm(Z_corr[i]) for i in range(num_tasks)]
    emb_corr_df.to_csv(f"{cfg.outprefix}_embeddings_correlation.csv", index=False)
    print(f"  Saved: {cfg.outprefix}_embeddings_correlation.csv")

    # random embeddings
    Z_random = rng_data.normal(0, 1, size=(num_tasks, dz))
    emb_random_df = pd.DataFrame({"task": task_names_list})
    for d in range(dz):
        emb_random_df[f"z_{d}"] = Z_random[:, d]
    emb_random_df["role"] = task_roles
    emb_random_df["ood_level"] = task_ood_levels
    emb_random_df["embedding_norm"] = [np.linalg.norm(Z_random[i]) for i in range(num_tasks)]
    emb_random_df.to_csv(f"{cfg.outprefix}_embeddings_random.csv", index=False)
    print(f"  Saved: {cfg.outprefix}_embeddings_random.csv")

    # zero embeddings
    Z_zero = np.zeros((num_tasks, dz))
    emb_zero_df = pd.DataFrame({"task": task_names_list})
    for d in range(dz):
        emb_zero_df[f"z_{d}"] = Z_zero[:, d]
    emb_zero_df["role"] = task_roles
    emb_zero_df["ood_level"] = task_ood_levels
    emb_zero_df["embedding_norm"] = [0.0] * num_tasks
    emb_zero_df.to_csv(f"{cfg.outprefix}_embeddings_zero.csv", index=False)
    print(f"  Saved: {cfg.outprefix}_embeddings_zero.csv")

    # coefficients
    coef_df = pd.DataFrame({"task": task_names_list})
    for j in range(E.shape[1]):
        coef_df[f"e_{j}"] = E[:, j]
    coef_df["role"] = task_roles
    coef_df["ood_level"] = task_ood_levels

    # longitudinal data
    print("\nGenerating longitudinal data...")
    start_year, num_years = 1990, 20
    rng_long = np.random.RandomState(cfg.seed + 7)
    long_records = []
    
    for _, row in combined.iterrows():
        pid = row["pid"]
        for p in P:
            v = row[f"V{p}"]
            if abs(v) > 0.5:
                norm = (np.tanh(v) + 1) / 2
                year = int(np.clip(norm * (num_years - 1) + rng_long.randint(0, 3), 0, num_years - 1))
                long_records.append({"PATIENT_ID": pid, "EVENT_YEAR": start_year + year, "ENDPOINT": f"V{p}"})
    long_df = pd.DataFrame(long_records)

    # save everything
    print("\nSaving files...")
    combined.to_csv(f"{cfg.outprefix}_tabular_data.csv", index=False)
    meta_df.to_csv(f"{cfg.outprefix}_col_metadata.csv", index=False)
    coef_df.to_csv(f"{cfg.outprefix}_coefficients.csv", index=False)
    long_df.to_csv(f"{cfg.outprefix}_longitudinal_data.csv", index=False)

    config_df = pd.DataFrame([{
        "num_predictors": cfg.num_predictors,
        "parents": str(cfg.parents),
        "z_dim": cfg.z_dim,
        "num_train_tasks": cfg.num_train_tasks,
        "samples_per_task": cfg.samples_per_task,
        "source_z_std": cfg.source_z_std,
        "target_ood_levels": str(cfg.target_ood_levels),
        "sigma_e": cfg.sigma_e,
        "sigma_y": cfg.sigma_y,
        "pos_rate": cfg.pos_rate,
        "seed": cfg.seed,
    }])
    config_df.to_csv(f"{cfg.outprefix}_config.csv", index=False)

    print(f"\nDone! {len(combined)} patients, {task_idx} tasks")
    print(f"  Source: {cfg.num_train_tasks}, Target: {len(cfg.target_ood_levels)}")
    
    return {
        "tabular": combined,
        "metadata": meta_df,
        "embeddings_oracle": Z,
        "coefficients": E,
        "longitudinal": long_df,
        "target_indices": target_task_indices,
        "ood_levels": task_ood_levels,
        "W": W,
        "b": b,
    }


if __name__ == "__main__":
    cfg = GeneratorConfig(
        num_predictors=10,
        parents=(0, 1, 2, 3),
        z_dim=4,
        num_train_tasks=20,
        samples_per_task=500,
        source_z_std=0.8,
        target_ood_levels=(0.05, 1.0, 2.0, 3.0, 4.0),
        sigma_e=0.15,
        sigma_y=0.6,
        pos_rate=0.30,
        shortcut_feature=9,
        shortcut_strength=0.4,
        shortcut_in_targets="soft",
        embedding_noise_levels=(0.0, 0.2, 0.5, 0.8),
        seed=123,
        outprefix="data/example/latent"
    )
    
    result = generate_dataset(cfg)