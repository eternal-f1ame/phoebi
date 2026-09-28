"""Method A — Simplex Unmixing.

Mathematical specification (presence/absence detection only — no proportion estimation).

Notation
--------
    K     : number of known prototypes (= number of bacterial species at init)
    D     : DINOv2 embedding dimension
    z_t   : L2-normalized DINOv2 embedding of tile t,             z_t in S^(D-1)
    P     : prototype matrix, rows L2-normalized,                 P in R^(K x D)
    tau   : temperature scaling cosine similarities to logits

Per-tile pipeline
-----------------
    ell_t = tau * P z_t                  in R^K          (cosine similarities * tau)
    w_t   = sparsemax(ell_t)             in Delta^(K-1)  (Euclidean projection onto simplex)
    z_hat = sum_k w_{t,k} * P_k           in R^D          (reconstruction)
    r_t   = z_t - z_hat                  in R^D          (residual)

Sparsemax produces exact zeros, so any "active" prototype k has w_{t,k} > 0 by construction.

Image-level aggregation (justified by Assumption H — spatial homogeneity)
-------------------------------------------------------------------------
    w(x)  = mean_t w_t                                            (mean weights over tiles)
    r(x)  = mean_t || r_t ||_2                                    (mean residual norm)

Mean aggregation is the unbiased estimator under H; variance shrinks as O(1/T).

Decisions
---------
    Presence:    y_hat_k(x) = 1[ w_k(x) > theta_k^A ]             (per-class threshold)
    Unknown:     u_hat(x)   = 1[ r(x)   > theta_unk^A ]           (residual threshold)

Both thresholds are calibrated on a held-out validation split, not on training data.

Training objective
------------------
    L = E_x [ (1/T) sum_t || z_t - z_hat ||_2^2 ]                 (self-supervised MSE)

There is no entropy regularizer: sparsity is enforced by sparsemax itself, and any
positive lambda * H(w) term pushes weights toward uniform and fights the projection.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    embedding_dim: int
    num_prototypes: int
    temperature: float = 10.0
    learned_temperature: bool = False  # if True, tau is K per-class nn.Parameters (log-space)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: Euclidean projection onto simplex (exact zeros)."""
    logits_sorted, _ = torch.sort(logits, dim=dim, descending=True)
    cumsum = torch.cumsum(logits_sorted, dim=dim) - 1
    r = torch.arange(1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    r = r.view(shape)
    cond = logits_sorted - cumsum / r > 0
    k = cond.sum(dim=dim, keepdim=True)
    tau = torch.gather(cumsum, dim=dim, index=k - 1) / k
    return torch.clamp(logits - tau, min=0)


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sparsemax(x, dim=self.dim)


class UnmixerModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.prototypes = nn.Parameter(torch.randn(config.num_prototypes, config.embedding_dim))
        self.activation = Sparsemax(dim=1)
        if config.learned_temperature:
            # Per-class log-temperatures initialised at log(config.temperature)
            self.log_tau = nn.Parameter(
                torch.full((config.num_prototypes,), math.log(config.temperature))
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_norm = F.normalize(z, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        sims = torch.matmul(z_norm, p_norm.t())  # (N, K)
        if self.config.learned_temperature:
            logits = sims * self.log_tau.exp().unsqueeze(0)  # per-class tau
        else:
            logits = sims * self.config.temperature
        weights = self.activation(logits)
        z_recon = torch.matmul(weights, p_norm)
        residual = z_norm - z_recon
        return z_recon, weights, residual


def initialize_prototypes(
    features: torch.Tensor,
    num_prototypes: int,
    init: str = "kmeans",
    seed: int = 1337,
) -> torch.Tensor:
    features = F.normalize(features, p=2, dim=1)
    if init == "random":
        idx = torch.randperm(features.shape[0])[:num_prototypes]
        return features[idx].clone()

    if init == "kmeans":
        try:
            from sklearn.cluster import KMeans
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for kmeans init") from exc

        kmeans = KMeans(n_clusters=num_prototypes, random_state=seed, n_init="auto")
        centers = kmeans.fit(features.cpu().numpy()).cluster_centers_
        centers = torch.tensor(centers, dtype=features.dtype)
        return F.normalize(centers, p=2, dim=1)

    raise ValueError(f"Unknown init mode: {init}")


def entropy_regularizer(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -torch.sum(weights * torch.log(weights + eps), dim=1).mean()


def greedy_cosine_clustering(
    residuals: torch.Tensor,
    threshold: float,
    min_size: int,
) -> Optional[torch.Tensor]:
    residuals = F.normalize(residuals, p=2, dim=1)
    clusters = []
    unassigned = torch.ones(residuals.shape[0], dtype=torch.bool, device=residuals.device)

    while unassigned.any():
        seed_idx = torch.where(unassigned)[0][0]
        sims = torch.matmul(residuals, residuals[seed_idx])
        neighbors = (sims > threshold) & unassigned
        if neighbors.sum() >= min_size:
            centroid = residuals[neighbors].mean(dim=0)
            clusters.append(centroid)
            unassigned[neighbors] = False
        else:
            unassigned[seed_idx] = False

    if not clusters:
        return None
    return torch.stack(clusters)
