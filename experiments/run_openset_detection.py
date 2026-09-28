#!/usr/bin/env python3
"""Phase 6 — Leave-one-class-out open-set detection harness (PHOEBI).

For each class `k` in `class_names`:
  * Build a "known" train subset by dropping every frame with label[:, k] == 1.
    Pure-culture videos for remaining classes survive the filter, so pure-culture
    init still works.
  * Reduce the label space to K-1 classes (column k removed).
  * Initialize prototypes for the remaining K-1 classes from pure-culture videos.
  * Score the FULL test split with both methods:
      - Method A: unknown score = mean tile residual norm r(x) (high = unknown)
      - Method B: unknown score = -m(x) where m(x) = mean max-cosine-sim
                  (low similarity = unknown, so negate for the "higher = more
                  unknown" convention used by open_set_auroc)
  * Ground truth: y_unknown[i] = 1 iff test_labels[i, k] == 1.
  * Record AUROC, AUPR, FPR@95TPR per method per held-out class.

Aggregates mean ± std across the K folds to `outputs/openset_loocv/summary.{csv,json}`.

Pure-culture init is sufficient — both methods are effectively closed-form once
prototypes are placed at class centroids (Method A's gradient training barely
moves them from this initialization). Skipping gradient training keeps the
harness deterministic and fast; Method A's full training loop is exercised in
`experiments/run_ablations.py` (Phase 7).
"""

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.features import (
    extract_features_multicrop,
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.illumination import make_illumination_preprocess
from src.common.io import load_real_split, save_json
from src.common.metrics import fpr_at_tpr, open_set_aupr, open_set_auroc
from src.common.prototypes import init_prototypes_from_pure_cultures
from src.common.tiling import TileConfig
from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig
from src.simplex_unmixing.model import ModelConfig, UnmixerModel, initialize_prototypes


def extract_split(
    split: str,
    splits_path: str,
    tile_cfg: TileConfig,
    backbone: str,
    device: torch.device,
    illum_method: str,
    illum_sigma: float,
    frame_batch_size: int,
    batch_size: int,
    num_workers: int,
    cache_dir: Path,
) -> Tuple[List[str], np.ndarray, List[str], List[str], torch.Tensor, torch.Tensor]:
    """Load one split from splits.json and extract its tile features once."""
    paths, labels, class_names, videos = load_real_split(splits_path, split)
    cache_path = cache_dir / f"{split}_features.pt"
    use_gpu_illum = device.type == "cuda" and illum_method != "none"

    if use_gpu_illum:
        features, image_index = extract_features_multicrop_gpu(
            image_paths=paths,
            tile_config=tile_cfg,
            backbone=backbone,
            frame_batch_size=frame_batch_size,
            num_workers=num_workers,
            device=device,
            illum_sigma=illum_sigma,
            illum_method=illum_method,
            cache_path=str(cache_path),
        )
    else:
        preprocess_fn = make_illumination_preprocess(sigma=illum_sigma, method=illum_method)
        features, image_index = extract_features_multicrop(
            image_paths=paths,
            tile_config=tile_cfg,
            backbone=backbone,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            cache_path=str(cache_path),
            mode="eval",
            preprocess_fn=preprocess_fn,
        )
    features = F.normalize(features, p=2, dim=1)
    return paths, labels, class_names, videos, features, image_index


def subset_by_images(
    features: torch.Tensor,
    image_index: torch.Tensor,
    keep_image_idx: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep only tiles whose parent image is in `keep_image_idx`; remap indices to
    [0, len(keep_image_idx))."""
    keep = torch.as_tensor(keep_image_idx, dtype=torch.long)
    n_original = int(image_index.max().item()) + 1 if image_index.numel() else 0
    remap_lut = torch.full((max(n_original, 1),), -1, dtype=torch.long)
    remap_lut[keep] = torch.arange(keep.numel(), dtype=torch.long)
    new_idx = remap_lut[image_index]
    mask = new_idx >= 0
    return features[mask], new_idx[mask]


def simplex_unknown_scores(
    features: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
    prototypes: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> np.ndarray:
    """Returns per-image mean residual norm (high = unknown)."""
    cfg = ModelConfig(
        embedding_dim=prototypes.shape[1],
        num_prototypes=prototypes.shape[0],
        temperature=temperature,
    )
    model = UnmixerModel(cfg).to(device)
    model.prototypes.data.copy_(prototypes.to(device))
    model.eval()
    with torch.no_grad():
        _, _, residuals_tile = model(features.to(device))
    res_norm_tile = residuals_tile.cpu().norm(p=2, dim=1)
    per_image = scatter_mean_by_image(
        res_norm_tile.unsqueeze(-1), image_index, num_images
    ).squeeze(-1)
    return per_image.numpy()


def prototype_unknown_scores(
    features: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
    prototypes: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Returns -m(x) per image (so higher = more unknown)."""
    cfg = ProtoConfig(
        embedding_dim=prototypes.shape[1],
        num_prototypes=prototypes.shape[0],
    )
    model = PrototypeMatchingModel(cfg).to(device)
    model.prototypes.data.copy_(prototypes.to(device))
    model.eval()
    with torch.no_grad():
        _, max_tile = model(features.to(device))
    max_tile = max_tile.cpu()
    per_image_max = scatter_mean_by_image(
        max_tile.unsqueeze(-1), image_index, num_images
    ).squeeze(-1)
    return (-per_image_max).numpy()


def run_fold(
    held_out: str,
    class_names: List[str],
    train_labels: np.ndarray,
    train_videos: List[str],
    train_features: torch.Tensor,
    train_image_index: torch.Tensor,
    test_labels: np.ndarray,
    test_features: torch.Tensor,
    test_image_index: torch.Tensor,
    temperature: float,
    device: torch.device,
    kmeans_fallback: bool,
) -> Dict[str, float]:
    """One LOOCV fold: hold out class `held_out`, train on remaining K-1 classes,
    score test on both methods."""
    k_idx = class_names.index(held_out)
    kept_class_names = [c for i, c in enumerate(class_names) if i != k_idx]
    K_new = len(kept_class_names)

    # ---- filter train: drop frames where held-out class is present ----
    train_keep_mask = train_labels[:, k_idx] == 0
    train_keep_image_idx = np.where(train_keep_mask)[0]
    if train_keep_image_idx.size == 0:
        raise RuntimeError(
            f"No training frames remain after excluding class {held_out!r}"
        )

    filtered_train_videos = [train_videos[i] for i in train_keep_image_idx]
    filtered_train_features, filtered_train_image_index = subset_by_images(
        train_features, train_image_index, train_keep_image_idx
    )

    # ---- initialize K-1 prototypes ----
    protos = init_prototypes_from_pure_cultures(
        tile_features=filtered_train_features,
        image_index=filtered_train_image_index,
        video_ids=filtered_train_videos,
        class_names=kept_class_names,
    )
    if protos is None:
        if not kmeans_fallback:
            raise RuntimeError(
                f"Pure-culture init failed for fold excl={held_out} and "
                f"--kmeans_fallback is off"
            )
        print(
            f"  [fold excl={held_out}] pure-culture init missing a class; "
            f"falling back to K-means"
        )
        protos = initialize_prototypes(filtered_train_features, K_new, init="kmeans")

    # ---- score the FULL test split ----
    num_test = test_labels.shape[0]
    y_true_unknown = (test_labels[:, k_idx] == 1).astype(np.int64)

    simplex_scores = simplex_unknown_scores(
        test_features, test_image_index, num_test, protos, temperature, device
    )
    prototype_scores = prototype_unknown_scores(
        test_features, test_image_index, num_test, protos, device
    )

    def metrics(scores: np.ndarray) -> Dict[str, float]:
        if y_true_unknown.sum() == 0 or y_true_unknown.sum() == num_test:
            return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
        return {
            "auroc": open_set_auroc(y_true_unknown, scores),
            "aupr": open_set_aupr(y_true_unknown, scores),
            "fpr95": fpr_at_tpr(y_true_unknown, scores, tpr_target=0.95),
        }

    simplex_m = metrics(simplex_scores)
    proto_m = metrics(prototype_scores)

    return {
        "held_out": held_out,
        "num_train_kept": int(train_keep_image_idx.size),
        "num_test": int(num_test),
        "num_test_unknown": int(y_true_unknown.sum()),
        "simplex_auroc": simplex_m["auroc"],
        "simplex_aupr": simplex_m["aupr"],
        "simplex_fpr95": simplex_m["fpr95"],
        "prototype_auroc": proto_m["auroc"],
        "prototype_aupr": proto_m["aupr"],
        "prototype_fpr95": proto_m["fpr95"],
        "_per_sample": {
            "simplex_scores": simplex_scores,
            "prototype_scores": prototype_scores,
            "y_true_unknown": y_true_unknown,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PHOEBI Phase 6: LOOCV open-set harness")
    parser.add_argument("--splits_path", type=str, default="data/splits.json")
    parser.add_argument("--output_dir", type=str, default="outputs/openset_loocv")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")

    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--eval_grid_size", type=int, default=4)
    parser.add_argument("--train_tiles_per_image", type=int, default=16)

    parser.add_argument("--illumination", type=str, default="divide",
                        choices=["divide", "subtract", "none"])
    parser.add_argument("--illum_sigma", type=float, default=64.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--temperature", type=float, default=10.0,
                        help="Simplex unmixing temperature for sparsemax logits")
    parser.add_argument("--kmeans_fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="Fall back to K-means init when a fold lacks a pure-culture "
                             "video for some remaining class. Use --no-kmeans_fallback to disable.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        train_tiles_per_image=args.train_tiles_per_image,
        eval_grid_size=args.eval_grid_size,
    )
    print(f"Device: {device} | TileConfig: {tile_cfg}")
    print(f"Illumination: {args.illumination} (sigma={args.illum_sigma})")

    # ---- 1. extract all three splits ONCE ------------------------------------------
    print("\n[1/3] Extracting tile features for train/val/test (once globally)...")
    (_, train_labels, class_names, train_videos, train_features, train_image_index) = \
        extract_split(
            split="train",
            splits_path=args.splits_path,
            tile_cfg=tile_cfg,
            backbone=args.backbone,
            device=device,
            illum_method=args.illumination,
            illum_sigma=args.illum_sigma,
            frame_batch_size=args.frame_batch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=features_dir,
        )
    (_, test_labels, _, _, test_features, test_image_index) = extract_split(
        split="test",
        splits_path=args.splits_path,
        tile_cfg=tile_cfg,
        backbone=args.backbone,
        device=device,
        illum_method=args.illumination,
        illum_sigma=args.illum_sigma,
        frame_batch_size=args.frame_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=features_dir,
    )
    print(f"  train tile features: {tuple(train_features.shape)}")
    print(f"  test  tile features: {tuple(test_features.shape)}")
    print(f"  classes: {class_names}")

    # ---- 2. LOOCV over each class --------------------------------------------------
    print(f"\n[2/3] Running LOOCV across {len(class_names)} classes...")
    per_fold: List[Dict[str, float]] = []
    for held_out in class_names:
        print(f"\n  Fold excl={held_out!r}")
        fold = run_fold(
            held_out=held_out,
            class_names=class_names,
            train_labels=train_labels,
            train_videos=train_videos,
            train_features=train_features,
            train_image_index=train_image_index,
            test_labels=test_labels,
            test_features=test_features,
            test_image_index=test_image_index,
            temperature=args.temperature,
            device=device,
            kmeans_fallback=args.kmeans_fallback,
        )
        per_fold.append(fold)
        print(
            f"    simplex   | AUROC {fold['simplex_auroc']:.4f} "
            f"AUPR {fold['simplex_aupr']:.4f} "
            f"FPR95 {fold['simplex_fpr95']:.4f}"
        )
        print(
            f"    prototype | AUROC {fold['prototype_auroc']:.4f} "
            f"AUPR {fold['prototype_aupr']:.4f} "
            f"FPR95 {fold['prototype_fpr95']:.4f}"
        )

    # ---- 3. aggregate & save --------------------------------------------------------
    print("\n[3/3] Aggregating...")

    # Dump per-sample scores per fold so figures can plot residual/score histograms.
    per_fold_dir = output_dir / "per_fold"
    per_fold_dir.mkdir(parents=True, exist_ok=True)
    for fold in per_fold:
        pers = fold.pop("_per_sample")
        np.savez(
            per_fold_dir / f"{fold['held_out']}.npz",
            simplex_scores=pers["simplex_scores"],
            prototype_scores=pers["prototype_scores"],
            y_true_unknown=pers["y_true_unknown"],
        )

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        writer.writeheader()
        writer.writerows(per_fold)

    def mean_std(key: str) -> Tuple[float, float]:
        vals = np.array([r[key] for r in per_fold if not np.isnan(r[key])])
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std())

    aggregated = {
        "class_names": class_names,
        "num_folds": len(per_fold),
        "simplex": {
            "auroc_mean": mean_std("simplex_auroc")[0],
            "auroc_std": mean_std("simplex_auroc")[1],
            "aupr_mean": mean_std("simplex_aupr")[0],
            "aupr_std": mean_std("simplex_aupr")[1],
            "fpr95_mean": mean_std("simplex_fpr95")[0],
            "fpr95_std": mean_std("simplex_fpr95")[1],
        },
        "prototype": {
            "auroc_mean": mean_std("prototype_auroc")[0],
            "auroc_std": mean_std("prototype_auroc")[1],
            "aupr_mean": mean_std("prototype_aupr")[0],
            "aupr_std": mean_std("prototype_aupr")[1],
            "fpr95_mean": mean_std("prototype_fpr95")[0],
            "fpr95_std": mean_std("prototype_fpr95")[1],
        },
        "per_fold": per_fold,
        "config": {
            "splits_path": args.splits_path,
            "backbone": args.backbone,
            "tile_config": asdict(tile_cfg),
            "illumination": args.illumination,
            "illum_sigma": args.illum_sigma,
            "temperature": args.temperature,
        },
    }
    save_json(str(output_dir / "summary.json"), aggregated)

    print(f"\n=== Phase 6 LOOCV summary ({len(per_fold)} folds) ===")
    for method in ("simplex", "prototype"):
        m = aggregated[method]
        print(
            f"  {method:9s} | AUROC {m['auroc_mean']:.4f} ± {m['auroc_std']:.4f}  "
            f"AUPR {m['aupr_mean']:.4f} ± {m['aupr_std']:.4f}  "
            f"FPR95 {m['fpr95_mean']:.4f} ± {m['fpr95_std']:.4f}"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
