"""Method B — Prototype Matching.

Mathematical specification (presence/absence detection only — no proportion estimation).

Notation
--------
    K     : number of known prototypes (= number of bacterial species)
    D     : DINOv2 embedding dimension
    z_t   : L2-normalized DINOv2 embedding of tile t,             z_t in S^(D-1)
    P     : prototype matrix, rows L2-normalized,                 P in R^(K x D)

Per-tile pipeline (no gradient learning needed)
-----------------------------------------------
    s_t   = P z_t                        in [-1, 1]^K    (cosine similarities to all prototypes)
    m_t   = max_k s_{t,k}                in [-1, 1]      (best-class similarity)

Each prototype scores independently — no simplex constraint, no reconstruction.

Image-level aggregation (justified by Assumption H — spatial homogeneity)
-------------------------------------------------------------------------
    s_k(x) = mean_t s_{t,k}                                       (mean per-class similarity)
    m(x)   = mean_t m_t                                           (mean best-class similarity)

Decisions
---------
    Presence:    y_hat_k(x) = 1[ s_k(x) > theta_k^B ]             (per-class threshold)
    Unknown:     u_hat(x)   = 1[ m(x)   < theta_unk^B ]           (low-similarity threshold)

Note the asymmetry with Method A: high *residual* signals unknown for A; low *similarity*
signals unknown for B. Both reduce to "the embedding does not look like anything we have
a prototype for."

Training procedure
------------------
1. Tile every frame in the training split (Assumption H makes tiles label-preserving).
2. Initialize P:
     - if pure-culture videos exist for class k, P_k = mean of tile embeddings in those videos
     - else, K-means on tile embeddings (fall-back when no pure cultures available).
3. Compute s_k(x) for each frame in the validation split.
4. Calibrate per-class thresholds theta_k^B at a low quantile of positive-image similarities.
5. Calibrate theta_unk^B = quantile_{0.05}( m(x) for x in val_known ).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProtoConfig:
    embedding_dim: int
    num_prototypes: int
    thresholds: Optional[List[float]] = None
    unknown_threshold: float = 0.5


class PrototypeMatchingModel(nn.Module):
    """Cosine similarity to learned prototypes with per-class calibrated thresholds.

    Unlike simplex unmixing, prototypes score independently via cosine similarity.
    Presence = similarity > threshold_k (calibrated per class).
    Unknown = max similarity < unknown_threshold.
    """

    def __init__(self, config: ProtoConfig) -> None:
        super().__init__()
        self.config = config
        self.prototypes = nn.Parameter(torch.randn(config.num_prototypes, config.embedding_dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (similarities, max_similarity)."""
        z_norm = F.normalize(z, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarities = torch.matmul(z_norm, p_norm.t())  # (B, K)
        max_sim = similarities.max(dim=1).values  # (B,)
        return similarities, max_sim

    def predict_presence(self, similarities: torch.Tensor) -> torch.Tensor:
        """Apply per-class thresholds. Returns (B, K) float tensor of 0/1."""
        if self.config.thresholds is None:
            raise RuntimeError("Thresholds not calibrated. Run calibration first.")
        thresholds = torch.tensor(self.config.thresholds, device=similarities.device)
        return (similarities > thresholds).float()

    def predict_unknown(self, max_sim: torch.Tensor) -> torch.Tensor:
        """Returns (B,) float tensor: 1 if unknown, 0 if known."""
        return (max_sim < self.config.unknown_threshold).float()
