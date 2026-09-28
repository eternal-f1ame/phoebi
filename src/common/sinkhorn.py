"""Sinkhorn-Knopp doubly-stochastic assignment for prototype discovery.

Adapted from sota_ncd/Cr-KD-NCD/model/losses.py:SinkhornKnopp (which itself
follows UNO and SwAV). The SK iteration normalizes a logit matrix into a
doubly-stochastic Q matrix whose rows sum to 1/K (uniform over prototypes)
and whose columns sum to 1/B (uniform over samples). After re-scaling, each
column is a soft assignment of one tile to one of K prototypes.

Why this matters for PHOEBI discovery: the greedy cosine-clustering primitive
in src/simplex_unmixing/model.py over-fragments (proposes ~28 prototypes per
fold for one held-out class) because each seed greedily eats its neighbors
without any global balance constraint. Sinkhorn-Knopp instead enforces
balanced cluster sizes, so a fixed K=2 or K=4 candidate set ends up well-
spread across the residual-flagged tile pool rather than clumped on one
direction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, num_iters: int = 3, epsilon: float = 0.05) -> torch.Tensor:
    """Doubly-stochastic Q from logits [B, K]. Returns Q [B, K] s.t.
    each column sums to 1 (soft assignment of each tile across prototypes).

    Lifted from Cr-KD-NCD's `SinkhornKnopp.forward` (also in NCD-IIC).
    """
    Q = torch.exp(logits / epsilon).t()  # [K, B]
    B = Q.shape[1]
    K = Q.shape[0]
    Q = Q / Q.sum()
    for _ in range(num_iters):
        Q = Q / Q.sum(dim=1, keepdim=True)
        Q = Q / K
        Q = Q / Q.sum(dim=0, keepdim=True)
        Q = Q / B
    Q = Q * B  # columns sum to 1 -> each sample is a probability over prototypes
    return Q.t()  # [B, K]


@torch.no_grad()
def sinkhorn_cluster(
    features: torch.Tensor,
    num_clusters: int,
    num_iters: int = 30,
    sk_iters: int = 3,
    sk_epsilon: float = 0.05,
    init: str = "kmeans++",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn-Knopp k-means: alternates SK soft assignment + centroid update.

    Args:
      features: [N, D] (assumed L2-normalized; if not, normalized internally)
      num_clusters: K
      num_iters: number of SK + centroid update iterations
      sk_iters: inner SK iterations per round (typically 3)
      sk_epsilon: SK temperature (typical 0.05)
      init: 'kmeans++' (probabilistic seed selection) or 'random'
      device: defaults to features.device

    Returns:
      centroids [K, D] (L2-normalized)
      hard_assignments [N] (long, in [0, K))
    """
    device = device or features.device
    f = F.normalize(features, p=2, dim=1).to(device)
    N, D = f.shape

    # ---- init centroids ----
    if init == "kmeans++":
        centroids = _kmeans_plus_plus_init(f, num_clusters)
    else:
        idx = torch.randperm(N, device=device)[:num_clusters]
        centroids = f[idx].clone()
    centroids = F.normalize(centroids, p=2, dim=1)

    # ---- SK + centroid iterations ----
    for _ in range(num_iters):
        logits = f @ centroids.t()  # [N, K] cosine sim
        Q = sinkhorn_knopp(logits, num_iters=sk_iters, epsilon=sk_epsilon)  # [N, K] soft
        # Weighted centroid update
        weights = Q  # [N, K]
        new_centroids = (weights.t() @ f) / (weights.sum(dim=0).unsqueeze(-1) + 1e-8)
        centroids = F.normalize(new_centroids, p=2, dim=1)

    # ---- hard assignment ----
    final_logits = f @ centroids.t()
    hard = final_logits.argmax(dim=1)
    return centroids, hard


def _kmeans_plus_plus_init(features: torch.Tensor, k: int) -> torch.Tensor:
    """Standard k-means++ seed selection on normalized features."""
    N, D = features.shape
    device = features.device
    centroids = torch.empty(k, D, device=device)
    # First centroid: uniform random
    idx = int(torch.randint(0, N, (1,), device=device).item())
    centroids[0] = features[idx]
    # Subsequent centroids: probability proportional to squared distance to nearest existing
    for i in range(1, k):
        sims = features @ centroids[:i].t()  # [N, i]
        max_sim = sims.max(dim=1).values  # [N]
        # Convert similarity to "distance": 1 - cosine_sim
        dist = (1.0 - max_sim).clamp(min=1e-12)
        probs = (dist ** 2) / (dist ** 2).sum()
        idx = int(torch.multinomial(probs, num_samples=1).item())
        centroids[i] = features[idx]
    return centroids
