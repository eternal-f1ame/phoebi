#!/usr/bin/env python3
"""Boundary-tile robustness check for Spatial Homogeneity (Assumption H).

Reuses cached features from outputs/phoebi_heldout/features/ (seed 1337
canonical, 4x4 grid). For each test image (16 tiles), compares per-image
F1 under \methodB{} when using:
  (a) all 16 tiles (the full grid; canonical baseline)
  (b) only the 4 central tiles (indices 5, 6, 9, 10 in row-major order;
      the 2x2 inner sub-grid that does not touch the field-of-view boundary)

If H holds at the boundary, F1 should change by less than the variance-
reduction threshold. A null result strengthens H; a meaningful drop
flags edge effects as a legitimate caveat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.metrics import per_sample_f1, macro_f1_per_class

# In a 4x4 row-major grid (indices 0..15):
#   row 0: 0,  1,  2,  3
#   row 1: 4,  5,  6,  7
#   row 2: 8,  9, 10, 11
#   row 3: 12, 13, 14, 15
# Central 2x2 = {5, 6, 9, 10}; boundary tiles = the rest.
CENTRAL_INDICES_4x4 = [5, 6, 9, 10]


def per_image_score_method_b(features, image_index, num_images, prototypes,
                             tile_indices_to_keep):
    """Method B: cosine similarity to prototypes, mean-aggregated to image level.

    Filters per-image tiles to only those in `tile_indices_to_keep`.
    """
    z = F.normalize(features, p=2, dim=1)
    p = F.normalize(prototypes, p=2, dim=1)
    sims = (z @ p.t()).numpy()  # (N*T, K)
    image_index_np = image_index.numpy()
    n_tiles = (image_index == 0).sum().item()
    K = prototypes.shape[0]

    # Reshape to (N, T, K) assuming exactly n_tiles tiles per image (sorted)
    # Verify ordering.
    perm = np.argsort(image_index_np, kind="stable")
    sims_sorted = sims[perm]
    image_index_sorted = image_index_np[perm]
    assert image_index_sorted[0] == 0
    sims_per_image = sims_sorted.reshape(num_images, n_tiles, K)

    # Filter
    keep = np.array(tile_indices_to_keep, dtype=np.int64)
    sims_filtered = sims_per_image[:, keep, :]  # (N, len(keep), K)
    # Mean-aggregate
    return sims_filtered.mean(axis=1)  # (N, K)


def hybrid_prototype_init_from_arrays(train_features, train_image_index,
                                       train_labels, train_combos, class_names):
    """Re-implementation of run_phoebi_heldout.hybrid_prototype_init."""
    K = len(class_names)
    D = train_features.shape[1]
    protos = torch.empty((K, D), dtype=train_features.dtype)
    train_combos_arr = np.array(train_combos)
    for k, name in enumerate(class_names):
        pure_image_idxs = np.where(train_combos_arr == name)[0]
        if len(pure_image_idxs) > 0:
            idx = torch.from_numpy(pure_image_idxs).long()
        else:
            present_image_idxs = np.where(train_labels[:, k] == 1)[0]
            idx = torch.from_numpy(present_image_idxs).long()
        mask = torch.isin(train_image_index, idx)
        tiles = train_features[mask]
        proto = tiles.mean(dim=0)
        protos[k] = F.normalize(proto, p=2, dim=0)
    return protos


def quantile_thresholds(scores, labels, quantile=0.05):
    K = scores.shape[1]
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        positive = scores[labels[:, k] == 1, k]
        if len(positive) == 0:
            out[k] = float(scores[:, k].max() + 1.0)
        else:
            out[k] = float(np.quantile(positive, quantile))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="outputs/phoebi_heldout/features")
    ap.add_argument("--frames_dir", default="data/images")
    ap.add_argument("--output_dir", default="outputs/boundary_tile_check")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cached features
    train_cache = torch.load(f"{args.feature_dir}/train.pt", map_location="cpu",
                             weights_only=False)
    val_cache = torch.load(f"{args.feature_dir}/val.pt", map_location="cpu",
                           weights_only=False)
    test_cache = torch.load(f"{args.feature_dir}/test.pt", map_location="cpu",
                            weights_only=False)
    train_features = F.normalize(train_cache["features"], p=2, dim=1)
    val_features = F.normalize(val_cache["features"], p=2, dim=1)
    test_features = F.normalize(test_cache["features"], p=2, dim=1)
    train_image_index = train_cache["image_index"]
    val_image_index = val_cache["image_index"]
    test_image_index = test_cache["image_index"]

    # Reconstruct splits via the same heldout selection
    from baselines.supervised_multilabel_heldout import (
        DEFAULT_HELDOUT_COUNTS, select_heldout,
    )
    from experiments.run_phoebi_heldout import (
        collect_entries, image_level_90_10, split_lists,
    )
    from tools.build_splits import discover_class_names
    class_names = discover_class_names(Path(args.frames_dir))
    heldout, trained, _ = select_heldout(
        Path(args.frames_dir), seed=args.seed,
        class_names=class_names, heldout_counts=DEFAULT_HELDOUT_COUNTS,
    )
    print(f"Classes: {class_names}")
    print(f"Heldout: {heldout}")

    trained_entries = collect_entries(Path(args.frames_dir), trained, class_names)
    heldout_entries = collect_entries(Path(args.frames_dir), heldout, class_names)
    train_entries, val_entries = image_level_90_10(trained_entries, seed=args.seed)
    train_paths, train_labels, train_combos = split_lists(train_entries)
    val_paths, val_labels, _ = split_lists(val_entries)
    test_paths, test_labels, _ = split_lists(heldout_entries)

    n_train = len(train_paths)
    n_val = len(val_paths)
    n_test = len(test_paths)
    print(f"train={n_train} val={n_val} test={n_test}")

    # Build hybrid prototypes (closed-form)
    prototypes = hybrid_prototype_init_from_arrays(
        train_features, train_image_index, train_labels, train_combos, class_names,
    )

    # Tile sets to compare
    tile_sets = {
        "all_16_tiles": list(range(16)),
        "central_4_tiles": CENTRAL_INDICES_4x4,
    }

    results = {}
    for label, tiles in tile_sets.items():
        # Calibrate thresholds on val for this tile set
        val_scores = per_image_score_method_b(
            val_features, val_image_index, n_val, prototypes, tiles)
        thresholds = quantile_thresholds(val_scores, val_labels, quantile=0.05)
        val_pred = (val_scores > thresholds[None, :]).astype(np.int64)

        test_scores = per_image_score_method_b(
            test_features, test_image_index, n_test, prototypes, tiles)
        test_pred = (test_scores > thresholds[None, :]).astype(np.int64)

        val_f1 = float(per_sample_f1(val_labels, val_pred))
        test_f1 = float(per_sample_f1(test_labels, test_pred))
        test_macro = macro_f1_per_class(test_labels, test_pred, class_names)

        results[label] = {
            "n_tiles_per_image": len(tiles),
            "tile_indices": tiles,
            "thresholds": thresholds.tolist(),
            "val_f1": val_f1,
            "test_heldout_f1": test_f1,
            "test_macro_f1": test_macro["macro"],
        }
        print(f"\n[{label}] n_tiles={len(tiles)}")
        print(f"  val F1: {val_f1:.4f}")
        print(f"  test F1: {test_f1:.4f}")
        print(f"  test macro F1: {test_macro['macro']:.4f}")

    delta_val = results["central_4_tiles"]["val_f1"] - results["all_16_tiles"]["val_f1"]
    delta_test = results["central_4_tiles"]["test_heldout_f1"] - results["all_16_tiles"]["test_heldout_f1"]
    delta_macro = results["central_4_tiles"]["test_macro_f1"] - results["all_16_tiles"]["test_macro_f1"]
    summary = {
        "results": results,
        "delta_central_minus_all": {
            "val_f1": delta_val,
            "test_heldout_f1": delta_test,
            "test_macro_f1": delta_macro,
        },
        "interpretation": (
            f"Switching from full 4x4 grid to central 2x2 sub-grid changes "
            f"per-sample F1 by {delta_test:+.4f} on the held-out compositional "
            f"split and {delta_val:+.4f} on val. A change well below 0.01 "
            f"is a null result that strengthens H at the boundary."
        ),
    }

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  delta val F1   (central - all): {delta_val:+.4f}")
    print(f"  delta test F1  (central - all): {delta_test:+.4f}")
    print(f"  delta macro F1 (central - all): {delta_macro:+.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
