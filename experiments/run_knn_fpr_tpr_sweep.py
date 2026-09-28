#!/usr/bin/env python3
"""Compute KNN OSR FPR at multiple TPR operating points (0.80, 0.90, 0.95).

Reads cached features at outputs/openset_loocv/features/ (shared with the
existing OSR sweep), recomputes per-fold KNN scores, and reports FPR at
TPR in {0.80, 0.90, 0.95} per fold and aggregated. This is the operating-
point disclosure for §4.3 / abstract: AUROC alone overstates how usable
the score is in deployment.
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

from src.common.features import scatter_mean_by_image
from src.common.io import load_real_split
from src.common.metrics import fpr_at_tpr, open_set_auroc


def knn_score(test_features, test_image_index, num_test_images,
              train_features, k, device):
    z_test = F.normalize(test_features, p=2, dim=1).to(device)
    z_train = F.normalize(train_features, p=2, dim=1).to(device)
    distances = torch.empty(z_test.shape[0], dtype=torch.float32)
    chunk = 256
    for i in range(0, z_test.shape[0], chunk):
        sims = z_test[i:i + chunk] @ z_train.t()
        topk_sim = sims.topk(k, dim=1).values
        kth_sim = topk_sim[:, -1]
        distances[i:i + chunk] = (1.0 - kth_sim).cpu()
    per_image = scatter_mean_by_image(distances.unsqueeze(-1),
                                      test_image_index, num_test_images).squeeze(-1)
    return per_image.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--feature_dir", default="outputs/openset_loocv/features")
    ap.add_argument("--output_dir", default="outputs/osr_score_sweep")
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--tpr_targets", nargs="+", type=float,
                    default=[0.80, 0.90, 0.95])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_cache = torch.load(f"{args.feature_dir}/train_features.pt",
                             map_location="cpu", weights_only=False)
    test_cache = torch.load(f"{args.feature_dir}/test_features.pt",
                            map_location="cpu", weights_only=False)
    train_features = F.normalize(train_cache["features"], p=2, dim=1)
    test_features = F.normalize(test_cache["features"], p=2, dim=1)
    train_image_index = train_cache["image_index"]
    test_image_index = test_cache["image_index"]

    train_paths, train_labels, class_names, train_videos = load_real_split(
        args.splits_path, "train")
    _, test_labels, _, _ = load_real_split(args.splits_path, "test")

    per_fold = []
    for held_out in class_names:
        k_idx = class_names.index(held_out)
        keep_mask = train_labels[:, k_idx] == 0
        keep_idx = np.where(keep_mask)[0]
        keep_set = set(int(i) for i in keep_idx)
        tile_keep = torch.tensor(
            [int(i) in keep_set for i in train_image_index.tolist()],
            dtype=torch.bool,
        )
        filtered_train_features = train_features[tile_keep]

        num_test = test_labels.shape[0]
        y_unknown = (test_labels[:, k_idx] == 1).astype(np.int64)

        scores = knn_score(test_features, test_image_index, num_test,
                           filtered_train_features, args.knn_k, device)

        fold = {"held_out": held_out, "n_unknown": int(y_unknown.sum())}
        if y_unknown.sum() == 0 or y_unknown.sum() == num_test:
            fold["auroc"] = float("nan")
            for t in args.tpr_targets:
                fold[f"fpr@tpr{int(t * 100)}"] = float("nan")
        else:
            fold["auroc"] = open_set_auroc(y_unknown, scores)
            for t in args.tpr_targets:
                fold[f"fpr@tpr{int(t * 100)}"] = fpr_at_tpr(y_unknown, scores, tpr_target=t)
        per_fold.append(fold)
        line = f"  {held_out:>3s}: AUROC {fold['auroc']:.4f}"
        for t in args.tpr_targets:
            line += f"  FPR@{int(t * 100)} {fold[f'fpr@tpr{int(t * 100)}']:.4f}"
        print(line)

    print("\n=== Aggregate (mean +/- std across folds) ===")
    aggregate = {}
    for key in ["auroc"] + [f"fpr@tpr{int(t * 100)}" for t in args.tpr_targets]:
        vals = np.array([r[key] for r in per_fold if not np.isnan(r[key])])
        m = float(vals.mean()) if vals.size else float("nan")
        s = float(vals.std()) if vals.size else float("nan")
        aggregate[f"{key}_mean"] = m
        aggregate[f"{key}_std"] = s
        print(f"  {key:>14s}: {m:.4f} +/- {s:.4f}")

    out_path = out_dir / "knn_fpr_tpr_sweep.json"
    with open(out_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_fold": per_fold,
                   "tpr_targets": args.tpr_targets, "knn_k": args.knn_k}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
