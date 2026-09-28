"""Method C: Mutual Channel head over DINOv2 tile embeddings.

Multi-label adaptation of UFG-NCD's MCRegionLoss (Liu et al., CVPR 2024). The
original is single-label fine-grained: split a CNN feature map's channels into
K class-groups, force every channel to discriminate its assigned class via
per-channel CE, then randomly mask 60% of the channels per group at every
step (Channel Random Activation, CRA) so no single channel dominates.

PHOEBI adaptation: DINOv2's frozen tile embedding has dimension D=384 with no
explicit spatial axis (per-tile pooled). We split D into K class-groups of
D/K channels each (default K=6, 64 channels per species). Each species gets a
1-D linear head over its 64 channels producing a presence logit. Per-step CRA
randomly zeros half the 64 channels for each species before the linear head,
forcing the 64 channels to be redundant discriminators. Loss is BCE against
the per-tile binary presence label, mean-aggregated to image level at eval.

Why this matches PHOEBI's morphology problem: our hardest species
(`bs`/`bt`/`mx`, all thin rods) require many subtle features to distinguish.
Forcing each species to learn 64 redundant detectors lets the model exploit
the full DINOv2 feature richness rather than relying on a single
discriminative dimension that might overfit one morphology cue.

Training is end-to-end on the linear heads (DINOv2 itself is frozen, like
Methods A/B). Inference: per-tile sigmoid -> mean-aggregate per image ->
threshold per class (val-calibrated).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCConfig:
    embedding_dim: int = 384       # D, DINOv2 ViT-S/14 feature dim
    num_classes: int = 6           # K, species count
    cra_drop_prob: float = 0.5     # CRA: fraction of per-class channels to zero each step
    @property
    def channels_per_class(self) -> int:
        assert self.embedding_dim % self.num_classes == 0, (
            f"embedding_dim {self.embedding_dim} must be divisible by num_classes {self.num_classes}")
        return self.embedding_dim // self.num_classes


class MCChannelHead(nn.Module):
    """Per-class linear heads over channel-grouped DINOv2 embeddings.

    Forward in training mode: applies a per-class CRA dropout mask (each
    species independently zeros `cra_drop_prob` of its 64 channels), then
    runs each species' 64-channel linear head to produce presence logits.

    Forward in eval mode: no CRA, all channels active.
    """

    def __init__(self, cfg: MCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        cpc = cfg.channels_per_class
        # K independent linear heads, each (cpc -> 1).
        # Implemented as a single (K, cpc) weight + (K,) bias for vectorization.
        self.weight = nn.Parameter(torch.zeros(cfg.num_classes, cpc))
        self.bias = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) -> logits (B, K)."""
        B, D = z.shape
        K = self.cfg.num_classes
        cpc = self.cfg.channels_per_class
        # Reshape (B, D) -> (B, K, cpc): species k's channels are z[:, k*cpc:(k+1)*cpc]
        z_grouped = z.view(B, K, cpc)
        if self.training and self.cfg.cra_drop_prob > 0:
            # Per-species CRA: independent per (B, k) drop mask of `cra_drop_prob` channels
            keep_prob = 1.0 - self.cfg.cra_drop_prob
            mask = torch.bernoulli(torch.full((B, K, cpc), keep_prob, device=z.device))
            # Rescale so expected magnitude is preserved
            mask = mask / keep_prob
            z_grouped = z_grouped * mask
        # Per-class linear: (B, K, cpc) * (K, cpc) -> (B, K)
        logits = (z_grouped * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias
        return logits

    def predict_presence(self, logits: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        """logits (B, K), thresholds (K,) -> (B, K) {0,1}."""
        return (torch.sigmoid(logits) > thresholds).float()
