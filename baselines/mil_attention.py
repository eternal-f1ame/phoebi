#!/usr/bin/env python3
"""Attention-based MIL baseline for PHOEBI, evaluated on both the random 80/10/10
split (Table 2 / random) and the leave-combinations-out split (Table 2 / heldout
and Table 3).

The head is the gated attention pooling of Ilse et al., 2018 (eqs. 8 and 9):

    h_t = embedding for tile t, t = 1..T
    e_t = (V tanh(W h_t)) * sigmoid(U h_t)               # gated attention scores
    a_t = softmax_t(e_t)                                  # per-bag (per-image)
    z   = sum_t a_t * h_t                                  # bag embedding
    y   = sigmoid(C z)                                     # K-dim presence logits

The decoder is intentionally similar in spirit to the PHOEBI pipeline (per-tile
features mean-aggregated to image-level under Assumption H) but the aggregation
weights are now LEARNED with gradient flow into a per-image discriminative head,
which is the design our anchor-based decoders are an alternative to.

We share everything else with the PHOEBI decoders:
  - Frozen DINOv2-S/14 features (cached, read directly from
    outputs/phoebi_heldout/features/ or outputs/.../*_features_cache.pt)
  - 4x4 grid tiles at 224 px under divide-by-Gaussian illumination
  - Per-class thresholds calibrated on val by argmax-F1 (matches Method C, since
    this is also a gradient-trained discriminative head)

The result is one row per regime: random 80/10/10 (paired with Tab. 2 random
block) and leave-combinations-out (paired with Tab. 3).

Outputs:
  outputs/mil_attention/{random,heldout}/results.json (per-row test metrics).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import (
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.io import load_real_split
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import TileConfig

# Reuse the heldout split + helpers
from baselines.supervised_multilabel_heldout import (
    DEFAULT_HELDOUT_COUNTS,
    parse_heldout_counts,
    select_heldout,
)
from experiments.run_phoebi_heldout import (
    collect_entries,
    image_level_90_10,
    split_lists,
)
from tools.build_splits import discover_class_names


# ---------------------------------------------------------------------------
# MIL attention head (Ilse et al. 2018 — gated)
# ---------------------------------------------------------------------------
class MILAttentionHead(nn.Module):
    """Gated attention pooling over tile features, then a K-class linear head.

    Args:
      D: tile-feature dimension (384 for DINOv2-S/14).
      K: number of classes.
      L: hidden dim of the gated attention network (Ilse et al. use 128).
    """

    def __init__(self, D: int, K: int, L: int = 128) -> None:
        super().__init__()
        self.attn_V = nn.Linear(D, L, bias=False)
        self.attn_U = nn.Linear(D, L, bias=False)
        self.attn_w = nn.Linear(L, 1, bias=False)
        self.classifier = nn.Linear(D, K)

    def forward(
        self, tiles: torch.Tensor, image_index: torch.Tensor, n_images: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """tiles: (N*T, D); image_index: (N*T,) long; returns
        (logits (N, K), attention_weights (N*T,))."""
        # Gated attention scores per tile
        v = torch.tanh(self.attn_V(tiles))
        u = torch.sigmoid(self.attn_U(tiles))
        e = self.attn_w(v * u).squeeze(-1)  # (N*T,)

        # Softmax over tiles WITHIN each image. Standard implementation:
        # subtract per-image max for numerical stability, exponentiate, then
        # normalize by per-image sum.
        e_max = torch.full((n_images,), float("-inf"), device=tiles.device)
        e_max = e_max.scatter_reduce(0, image_index, e, reduce="amax", include_self=True)
        e_centered = e - e_max[image_index]
        exp_e = torch.exp(e_centered)
        sums = torch.zeros(n_images, device=tiles.device)
        sums = sums.scatter_add(0, image_index, exp_e)
        attn = exp_e / sums[image_index]  # (N*T,) attention weights, sum-to-1 per image

        # Weighted sum of tile features per image
        weighted = tiles * attn.unsqueeze(1)  # (N*T, D)
        bag_embed = torch.zeros(n_images, tiles.shape[1], device=tiles.device)
        bag_embed = bag_embed.scatter_add(
            0, image_index.unsqueeze(1).expand_as(weighted), weighted
        )

        logits = self.classifier(bag_embed)
        return logits, attn


# ---------------------------------------------------------------------------
# Threshold calibration (matches Method C / supervised baselines)
# ---------------------------------------------------------------------------
def calibrate_argmax_f1(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    K = scores.shape[1]
    grid = np.linspace(0.01, 0.99, 99)
    thr = np.zeros(K)
    for k in range(K):
        best = -1.0
        for t in grid:
            pred = (scores[:, k] > t).astype(np.int64)
            tp = int(((pred == 1) & (labels[:, k] == 1)).sum())
            fp = int(((pred == 1) & (labels[:, k] == 0)).sum())
            fn = int(((pred == 0) & (labels[:, k] == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best:
                best = f1
                thr[k] = float(t)
    return thr


# ---------------------------------------------------------------------------
# Training/evaluation core (shared between random and heldout protocols)
# ---------------------------------------------------------------------------
def train_and_score(
    train_features: torch.Tensor,
    train_image_index: torch.Tensor,
    train_labels: np.ndarray,
    val_features: torch.Tensor,
    val_image_index: torch.Tensor,
    val_labels: np.ndarray,
    test_features: torch.Tensor,
    test_image_index: torch.Tensor,
    test_labels: np.ndarray,
    K: int,
    D: int,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    bag_batch: int = 32,
    seed: int = 1337,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    head = MILAttentionHead(D, K).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    train_features = train_features.to(device)
    train_image_index = train_image_index.to(device)
    val_features = val_features.to(device)
    val_image_index = val_image_index.to(device)
    test_features = test_features.to(device)
    test_image_index = test_image_index.to(device)

    n_train = int(train_image_index.max().item()) + 1
    n_val = int(val_image_index.max().item()) + 1
    n_test = int(test_image_index.max().item()) + 1

    train_labels_t = torch.from_numpy(train_labels).float().to(device)
    train_image_ids = torch.arange(n_train, device=device)

    # Build per-image tile slices once. image_index is contiguous-runs.
    tile_starts = torch.zeros(n_train + 1, dtype=torch.long, device=device)
    counts = torch.bincount(train_image_index, minlength=n_train)
    tile_starts[1:] = torch.cumsum(counts, dim=0)

    if verbose:
        print(f"[MIL] training: n_train={n_train}, n_val={n_val}, n_test={n_test}, K={K}, D={D}")

    best_val_f1 = -1.0
    best_state = None
    best_thr = None

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_train, device=device)
        running, n_seen = 0.0, 0

        for i in range(0, n_train, bag_batch):
            batch_img_ids = perm[i : i + bag_batch]
            B = batch_img_ids.shape[0]

            # Gather tiles for these B images, build a contiguous batch with
            # local image_index 0..B-1.
            tile_lists = []
            local_idx_lists = []
            for local_b, img_id in enumerate(batch_img_ids.tolist()):
                start, end = tile_starts[img_id].item(), tile_starts[img_id + 1].item()
                tile_lists.append(train_features[start:end])
                local_idx_lists.append(
                    torch.full(
                        (end - start,), local_b, dtype=torch.long, device=device
                    )
                )
            batch_tiles = torch.cat(tile_lists, dim=0)
            batch_idx = torch.cat(local_idx_lists, dim=0)
            batch_labels = train_labels_t[batch_img_ids]

            opt.zero_grad()
            logits, _ = head(batch_tiles, batch_idx, n_images=B)
            loss = bce(logits, batch_labels)
            loss.backward()
            opt.step()

            running += loss.item() * B
            n_seen += B

        # Val eval (single forward over all val tiles; B = n_val, batched)
        head.eval()
        with torch.no_grad():
            v_logits, _ = head(val_features, val_image_index, n_images=n_val)
            v_sig = torch.sigmoid(v_logits).cpu().numpy()
        thr = calibrate_argmax_f1(v_sig, val_labels)
        v_pred = (v_sig > thr).astype(np.int64)
        v_f1 = float(per_sample_f1(val_labels, v_pred))

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  ep{epoch:02d}  train_loss={running / n_seen:.4f}  val F1={v_f1:.4f}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_thr = thr.copy()

    # Test under best val state
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        t_logits, _ = head(test_features, test_image_index, n_images=n_test)
        t_sig = torch.sigmoid(t_logits).cpu().numpy()
    t_pred = (t_sig > best_thr).astype(np.int64)

    # Also recompute val under best state for the val-in-distribution row
    with torch.no_grad():
        v_logits, _ = head(val_features, val_image_index, n_images=n_val)
        v_sig = torch.sigmoid(v_logits).cpu().numpy()
    v_pred = (v_sig > best_thr).astype(np.int64)

    return {
        "best_val_f1": best_val_f1,
        "thresholds": best_thr.tolist(),
        "val_per_sample_f1": float(per_sample_f1(val_labels, v_pred)),
        "val_macro_f1": {
            k: float(v) for k, v in macro_f1_per_class(val_labels, v_pred, [str(i) for i in range(K)]).items()
        },
        "val_exact_match": float(exact_match_accuracy(val_labels, v_pred)),
        "test_per_sample_f1": float(per_sample_f1(test_labels, t_pred)),
        "test_macro_f1": {
            k: float(v) for k, v in macro_f1_per_class(test_labels, t_pred, [str(i) for i in range(K)]).items()
        },
        "test_exact_match": float(exact_match_accuracy(test_labels, t_pred)),
        "test_scores": t_sig,
    }


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------
def load_cached_features(cache_path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Returns (features [N*T, D], image_index [N*T], paths)."""
    payload = torch.load(cache_path, weights_only=False)
    feats = F.normalize(payload["features"], p=2, dim=1)
    return feats, payload["image_index"], payload["paths"]


# ---------------------------------------------------------------------------
# Two protocols: random 80/10/10 (Table 2 / random) and leave-combos-out (Tables 2/3)
# ---------------------------------------------------------------------------
def run_random_split(args, device: str):
    """Random 80/10/10 image-level split. Reuses the cached features from
    Method A's run dir so we extract once per protocol."""
    sim_cache_dir = ROOT / "outputs/simplex_unmixing/default"
    train_feats, train_idx, train_paths = load_cached_features(sim_cache_dir / "train_features_cache.pt")
    val_feats, val_idx, val_paths = load_cached_features(sim_cache_dir / "val_features_cache.pt")
    test_feats, test_idx, test_paths = load_cached_features(
        sim_cache_dir / "presence_test" / "test_features_cache.pt"
    )

    # Labels from canonical splits.json
    _, train_labels, class_names, _ = load_real_split(args.splits_path, "train")
    _, val_labels, _, _ = load_real_split(args.splits_path, "val")
    _, test_labels, _, _ = load_real_split(args.splits_path, "test")
    K = len(class_names)
    D = train_feats.shape[1]

    print(f"[random] D={D} K={K} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    out = train_and_score(
        train_feats, train_idx, train_labels,
        val_feats, val_idx, val_labels,
        test_feats, test_idx, test_labels,
        K=K, D=D,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        bag_batch=args.bag_batch, seed=args.seed, device=device,
    )
    out["class_names"] = class_names
    out["protocol"] = "random_80_10_10"
    return out


def run_heldout(args, device: str):
    """Leave-9-combinations-out, identical to baselines/supervised_multilabel_heldout.py
    selection (seed 1337). Uses the cached features from outputs/phoebi_heldout/features/."""
    feats_dir = ROOT / "outputs/phoebi_heldout/features"
    train_feats, train_idx, train_paths_cached = load_cached_features(feats_dir / "train.pt")
    val_feats, val_idx, val_paths_cached = load_cached_features(feats_dir / "val.pt")
    test_feats, test_idx, test_paths_cached = load_cached_features(feats_dir / "test.pt")

    # Reproduce the SAME label assignment as run_phoebi_heldout.py
    frames_dir = ROOT / args.frames_dir
    class_names = discover_class_names(frames_dir)
    K = len(class_names)
    heldout_counts = (
        parse_heldout_counts(args.heldout_counts) if args.heldout_counts else DEFAULT_HELDOUT_COUNTS
    )
    heldout, trained, _ = select_heldout(
        frames_dir, seed=args.seed, class_names=class_names, heldout_counts=heldout_counts
    )
    trained_entries = collect_entries(frames_dir, trained, class_names)
    heldout_entries = collect_entries(frames_dir, heldout, class_names)
    train_entries, val_entries = image_level_90_10(trained_entries, seed=args.seed)
    train_paths, train_labels, _ = split_lists(train_entries)
    val_paths, val_labels, _ = split_lists(val_entries)
    test_paths, test_labels, _ = split_lists(heldout_entries)

    # Sanity check: cached path order must match the regenerated path order
    if train_paths != train_paths_cached:
        raise RuntimeError(
            "Cached heldout train paths do not match regenerated paths; cache invalid."
        )
    if val_paths != val_paths_cached:
        raise RuntimeError("Cached heldout val paths mismatch.")
    if test_paths != test_paths_cached:
        raise RuntimeError("Cached heldout test paths mismatch.")

    D = train_feats.shape[1]
    print(f"[heldout] D={D} K={K} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")
    print(f"  heldout combos: {heldout}")

    out = train_and_score(
        train_feats, train_idx, train_labels,
        val_feats, val_idx, val_labels,
        test_feats, test_idx, test_labels,
        K=K, D=D,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        bag_batch=args.bag_batch, seed=args.seed, device=device,
    )
    out["class_names"] = class_names
    out["heldout_combos"] = heldout
    out["protocol"] = "leave_combos_out"
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="both", choices=["random", "heldout", "both"])
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--frames_dir", default="data/images")
    ap.add_argument("--output_dir", default="outputs/mil_attention")
    ap.add_argument("--heldout_counts", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--bag_batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    if args.protocol in ("random", "both"):
        print("\n=== Random 80/10/10 protocol ===")
        out = run_random_split(args, device)
        sub = out_dir / "random"
        sub.mkdir(exist_ok=True)
        np.save(sub / "test_scores.npy", out.pop("test_scores"))
        (sub / "results.json").write_text(json.dumps(out, indent=2))
        print(
            f"\n[random] val F1 = {out['val_per_sample_f1']:.4f} | "
            f"test F1 = {out['test_per_sample_f1']:.4f} | "
            f"test EM = {out['test_exact_match']:.4f}"
        )

    if args.protocol in ("heldout", "both"):
        print("\n=== Leave-combinations-out protocol ===")
        out = run_heldout(args, device)
        sub = out_dir / "heldout"
        sub.mkdir(exist_ok=True)
        np.save(sub / "test_scores.npy", out.pop("test_scores"))
        (sub / "results.json").write_text(json.dumps(out, indent=2))
        delta = out["test_per_sample_f1"] - out["val_per_sample_f1"]
        print(
            f"\n[heldout] val F1 = {out['val_per_sample_f1']:.4f} | "
            f"heldout F1 = {out['test_per_sample_f1']:.4f} | "
            f"delta = {delta:+.4f}"
        )


if __name__ == "__main__":
    main()
