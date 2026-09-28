#!/usr/bin/env python3
"""Sweep over open-set scoring functions on the same LOOCV protocol.

Reuses outputs/openset_loocv/features/{train,test}_features.pt and the
existing fold structure. For each fold we initialize K-1 known prototypes via
pure-culture init, then compute four open-set scores and report
AUROC / AUPR / FPR@95 against the binary known/unknown label:

  1. residual  : ||z - P^T sparsemax(tau P z)|| (Method A's native score)
  2. neg_max   : -max_k cosine(z, P_k)         (Method B's native score)
  3. energy    : -log sum_k exp(cosine(z, P_k) / T)  (Liu 2020; T=0.1)
  4. knn       : distance to K-th nearest train tile in feature space (Sun 2022)

All scores are computed at the tile level, then mean-aggregated to image
level via scatter_mean. KNN uses train tile features as the reference set.

Outputs outputs/osr_score_sweep/summary.{csv,json}.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import scatter_mean_by_image
from src.common.io import load_real_split, save_json
from src.common.metrics import fpr_at_tpr, open_set_aupr, open_set_auroc
from src.common.prototypes import init_prototypes_from_pure_cultures
from src.common.tiling import TileConfig
from src.simplex_unmixing.model import (
    ModelConfig,
    UnmixerModel,
    initialize_prototypes,
)


def residual_score(features: torch.Tensor, image_index: torch.Tensor,
                   num_images: int, prototypes: torch.Tensor,
                   temperature: float, device: torch.device) -> np.ndarray:
    cfg = ModelConfig(embedding_dim=prototypes.shape[1],
                      num_prototypes=prototypes.shape[0],
                      temperature=temperature)
    m = UnmixerModel(cfg).to(device)
    m.prototypes.data.copy_(prototypes.to(device))
    m.eval()
    with torch.no_grad():
        _, _, residuals = m(features.to(device))
    norms = residuals.cpu().norm(p=2, dim=1)
    per_image = scatter_mean_by_image(norms.unsqueeze(-1), image_index, num_images).squeeze(-1)
    return per_image.numpy()


def neg_max_score(features: torch.Tensor, image_index: torch.Tensor,
                  num_images: int, prototypes: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    z = F.normalize(features, p=2, dim=1).to(device)
    p = F.normalize(prototypes, p=2, dim=1).to(device)
    sims = z @ p.t()
    max_sim_tile = sims.max(dim=1).values.cpu()
    per_image_max = scatter_mean_by_image(max_sim_tile.unsqueeze(-1), image_index, num_images).squeeze(-1)
    return (-per_image_max).numpy()


def energy_score(features: torch.Tensor, image_index: torch.Tensor,
                 num_images: int, prototypes: torch.Tensor,
                 temperature: float, device: torch.device) -> np.ndarray:
    """Energy = -T * logsumexp(sims/T). Higher = more unknown."""
    z = F.normalize(features, p=2, dim=1).to(device)
    p = F.normalize(prototypes, p=2, dim=1).to(device)
    sims = z @ p.t()
    energy_tile = -temperature * torch.logsumexp(sims / temperature, dim=1)
    energy_tile = energy_tile.cpu()
    per_image = scatter_mean_by_image(energy_tile.unsqueeze(-1), image_index, num_images).squeeze(-1)
    return per_image.numpy()


def knn_score(test_features: torch.Tensor, test_image_index: torch.Tensor,
              num_test_images: int, train_features: torch.Tensor,
              k: int, device: torch.device) -> np.ndarray:
    """Per-tile distance to K-th nearest train tile in feature space.
    Higher distance = more unknown. Uses cosine distance = 1 - cos_sim."""
    z_test = F.normalize(test_features, p=2, dim=1).to(device)
    z_train = F.normalize(train_features, p=2, dim=1).to(device)
    # Distances in chunks to avoid OOM
    distances = torch.empty(z_test.shape[0], dtype=torch.float32)
    chunk = 256
    for i in range(0, z_test.shape[0], chunk):
        sims = z_test[i:i + chunk] @ z_train.t()  # cos similarity
        # Distance = 1 - sim. Want K-th smallest distance = K-th largest sim.
        topk_sim = sims.topk(k, dim=1).values  # (chunk, k)
        kth_sim = topk_sim[:, -1]
        kth_dist = 1.0 - kth_sim
        distances[i:i + chunk] = kth_dist.cpu()
    per_image = scatter_mean_by_image(distances.unsqueeze(-1), test_image_index, num_test_images).squeeze(-1)
    return per_image.numpy()


def run_fold(held_out, class_names, train_labels, train_videos,
             train_features, train_image_index,
             test_labels, test_features, test_image_index,
             energy_T, knn_k, temperature, device,
             ):
    k_idx = class_names.index(held_out)
    kept_class_names = [c for i, c in enumerate(class_names) if i != k_idx]
    K_new = len(kept_class_names)

    # Filter train, init known prototypes
    keep_mask = train_labels[:, k_idx] == 0
    keep_idx = np.where(keep_mask)[0]
    keep_set = set(int(i) for i in keep_idx)
    tile_keep = torch.tensor([int(i) in keep_set for i in train_image_index.tolist()], dtype=torch.bool)
    filtered_train_features = train_features[tile_keep]
    n_orig = int(train_image_index.max().item()) + 1
    remap_lut = torch.full((n_orig,), -1, dtype=torch.long)
    remap_lut[torch.as_tensor(keep_idx, dtype=torch.long)] = torch.arange(keep_idx.size, dtype=torch.long)
    filtered_train_image_index = remap_lut[train_image_index][tile_keep]
    filtered_train_videos = [train_videos[i] for i in keep_idx]

    protos = init_prototypes_from_pure_cultures(
        tile_features=filtered_train_features,
        image_index=filtered_train_image_index,
        video_ids=filtered_train_videos,
        class_names=kept_class_names,
    )
    if protos is None:
        protos = initialize_prototypes(filtered_train_features, K_new, init="kmeans")

    num_test = test_labels.shape[0]
    y_unknown = (test_labels[:, k_idx] == 1).astype(np.int64)

    scores = {
        "residual": residual_score(test_features, test_image_index, num_test, protos, temperature, device),
        "neg_max": neg_max_score(test_features, test_image_index, num_test, protos, device),
        "energy_T0.1": energy_score(test_features, test_image_index, num_test, protos, temperature=0.1, device=device),
        "energy_T1.0": energy_score(test_features, test_image_index, num_test, protos, temperature=1.0, device=device),
        f"knn_k{knn_k}": knn_score(test_features, test_image_index, num_test, filtered_train_features, knn_k, device),
    }

    out = {"held_out": held_out, "n_unknown": int(y_unknown.sum())}
    for name, s in scores.items():
        if y_unknown.sum() == 0 or y_unknown.sum() == num_test:
            out[f"{name}_auroc"] = float("nan")
            out[f"{name}_aupr"] = float("nan")
            out[f"{name}_fpr95"] = float("nan")
        else:
            out[f"{name}_auroc"] = open_set_auroc(y_unknown, s)
            out[f"{name}_aupr"] = open_set_aupr(y_unknown, s)
            out[f"{name}_fpr95"] = fpr_at_tpr(y_unknown, s, tpr_target=0.95)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--feature_dir", default="outputs/openset_loocv/features")
    ap.add_argument("--output_dir", default="outputs/osr_score_sweep")
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--knn_k", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # Load cached features
    train_cache = torch.load(f"{args.feature_dir}/train_features.pt", map_location="cpu", weights_only=False)
    test_cache = torch.load(f"{args.feature_dir}/test_features.pt", map_location="cpu", weights_only=False)
    train_features = F.normalize(train_cache["features"], p=2, dim=1)
    test_features = F.normalize(test_cache["features"], p=2, dim=1)
    train_image_index = train_cache["image_index"]
    test_image_index = test_cache["image_index"]

    # Load splits
    train_paths, train_labels, class_names, train_videos = load_real_split(args.splits_path, "train")
    _, test_labels, _, _ = load_real_split(args.splits_path, "test")
    K = len(class_names)
    print(f"Classes: {class_names}; train tiles: {tuple(train_features.shape)}; test tiles: {tuple(test_features.shape)}")

    per_fold = []
    for held_out in class_names:
        print(f"\n  Fold excl={held_out!r}")
        fold = run_fold(
            held_out, class_names, train_labels, train_videos,
            train_features, train_image_index,
            test_labels, test_features, test_image_index,
            energy_T=0.1, knn_k=args.knn_k,
            temperature=args.temperature, device=device,
        )
        per_fold.append(fold)
        for score_name in ["residual", "neg_max", "energy_T0.1", "energy_T1.0", f"knn_k{args.knn_k}"]:
            print(f"    {score_name:>14s}: AUROC {fold[f'{score_name}_auroc']:.4f}  "
                  f"AUPR {fold[f'{score_name}_aupr']:.4f}  "
                  f"FPR95 {fold[f'{score_name}_fpr95']:.4f}")

    # Aggregate
    keys = list(per_fold[0].keys())
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_fold)

    def mean_std(field):
        vals = np.array([r[field] for r in per_fold if not np.isnan(r[field])])
        return (float(vals.mean()), float(vals.std())) if vals.size else (float("nan"), float("nan"))

    score_names = ["residual", "neg_max", "energy_T0.1", "energy_T1.0", f"knn_k{args.knn_k}"]
    summary = {"score_methods": {}}
    print(f"\n=== AUROC summary across {len(per_fold)} folds ===")
    for s in score_names:
        m_auroc = mean_std(f"{s}_auroc")
        m_aupr = mean_std(f"{s}_aupr")
        m_fpr = mean_std(f"{s}_fpr95")
        summary["score_methods"][s] = {
            "auroc_mean": m_auroc[0], "auroc_std": m_auroc[1],
            "aupr_mean": m_aupr[0], "aupr_std": m_aupr[1],
            "fpr95_mean": m_fpr[0], "fpr95_std": m_fpr[1],
        }
        print(f"  {s:>14s}: AUROC {m_auroc[0]:.4f} +/- {m_auroc[1]:.4f}  "
              f"AUPR {m_aupr[0]:.4f} +/- {m_aupr[1]:.4f}  "
              f"FPR95 {m_fpr[0]:.4f} +/- {m_fpr[1]:.4f}")

    save_json(str(out_dir / "summary.json"), {**summary, "per_fold": per_fold})
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
