"""Closed-form scoring of units against prototypes, and image-level aggregation.

Two read-outs, both taken from the paper's decoders:

    cosine     s_uk = <z_u, P_k>                       ProtoMatch, closed form
    sparsemax  w_u  = sparsemax(tau * P z_u)           SimplexUnmix at initialisation

and three aggregations from units to an image:

    mean       the paper's O(1/T) estimator under Assumption H
    q90        90th percentile over units, per class: "is some part of the image k?"
    max        the limit of the above

Per-image results are small (K numbers per read-out and aggregation), so a whole split
is kept in memory while units stream through.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.simplex_unmixing.model import sparsemax

READOUTS = ("cosine", "sparsemax")
AGGREGATIONS = ("mean", "q90", "max")


class PrototypeAccumulator:
    """Running per-class sums of units, for the paper's hybrid init at any granularity.

    For class k the prototype is the L2-normalised mean of units from pure-culture images
    of k when the training pool has them, otherwise from every image whose label has k.
    Both sums are accumulated in one pass; the choice is made at `finalize`.
    """

    def __init__(self, class_names: List[str], dim: int, device: torch.device) -> None:
        self.class_names = list(class_names)
        K = len(class_names)
        self.pure_sum = torch.zeros((K, dim), dtype=torch.float64, device=device)
        self.pure_n = torch.zeros((K,), dtype=torch.float64, device=device)
        self.any_sum = torch.zeros((K, dim), dtype=torch.float64, device=device)
        self.any_n = torch.zeros((K,), dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, units: torch.Tensor, labels: np.ndarray, combos: List[str]) -> None:
        """units (B, N, D); labels (B, K); combos length B."""
        B, N, D = units.shape
        per_image_sum = units.sum(dim=1).double()                 # (B, D)
        lab = torch.from_numpy(labels).to(units.device).double()   # (B, K)
        self.any_sum += lab.t() @ per_image_sum
        self.any_n += lab.sum(dim=0) * N
        for b, c in enumerate(combos):
            if c in self.class_names:
                k = self.class_names.index(c)
                self.pure_sum[k] += per_image_sum[b]
                self.pure_n[k] += N

    def finalize(self) -> Dict[str, object]:
        K = len(self.class_names)
        protos = torch.empty_like(self.pure_sum)
        source = []
        for k in range(K):
            if self.pure_n[k] > 0:
                protos[k] = self.pure_sum[k] / self.pure_n[k]; source.append("pure")
            elif self.any_n[k] > 0:
                protos[k] = self.any_sum[k] / self.any_n[k]; source.append("fallback")
            else:
                raise RuntimeError(f"no units for class {self.class_names[k]}")
        return {"prototypes": F.normalize(protos.float(), p=2, dim=1), "source": source,
                "n_pure": self.pure_n.tolist(), "n_any": self.any_n.tolist()}


@torch.no_grad()
def score_units(units: torch.Tensor, prototypes: torch.Tensor, temperature: float = 10.0,
                aggregations=AGGREGATIONS) -> Dict[str, torch.Tensor]:
    """units (B, N, D) L2-normalised, prototypes (K, D) L2-normalised.

    Returns {'cosine/mean': (B, K), 'cosine/q90': ..., 'sparsemax/mean': ..., ...,
             'residual/mean': (B,)} — residual is the sparsemax reconstruction residual norm.
    """
    B, N, D = units.shape
    P = F.normalize(prototypes.to(units.device, units.dtype), p=2, dim=1)
    sims = units @ P.t()                                   # (B, N, K)
    w = sparsemax(sims.reshape(B * N, -1) * temperature, dim=1).reshape(B, N, -1)
    recon = w @ P                                          # (B, N, D)
    res = (units - recon).norm(dim=-1)                     # (B, N)
    out: Dict[str, torch.Tensor] = {}
    for name, per_unit in (("cosine", sims), ("sparsemax", w)):
        for agg in aggregations:
            if agg == "mean":
                v = per_unit.mean(dim=1)
            elif agg == "max":
                v = per_unit.max(dim=1).values
            elif agg.startswith("q"):
                q = int(agg[1:]) / 100.0
                v = torch.quantile(per_unit, q, dim=1)
            else:
                raise ValueError(agg)
            out[f"{name}/{agg}"] = v.float().cpu()
    out["residual/mean"] = res.mean(dim=1).float().cpu()
    return out


class ScoreStore:
    """Collects per-image score vectors for one split, keyed by read-out/aggregation."""

    def __init__(self, n_images: int, K: int, keys: List[str]) -> None:
        self.arrays = {k: np.zeros((n_images, K), dtype=np.float32) for k in keys}
        self.residual = np.zeros((n_images,), dtype=np.float32)
        self.filled = np.zeros((n_images,), dtype=bool)

    def put(self, idxs: torch.Tensor, scored: Dict[str, torch.Tensor]) -> None:
        i = idxs.cpu().numpy()
        for k, arr in self.arrays.items():
            arr[i] = scored[k].numpy()
        self.residual[i] = scored["residual/mean"].numpy()
        self.filled[i] = True

    def complete(self) -> bool:
        return bool(self.filled.all())


def score_keys(aggregations=AGGREGATIONS) -> List[str]:
    return [f"{r}/{a}" for r in READOUTS for a in aggregations]
