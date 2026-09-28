#!/usr/bin/env python3
"""Train Method B — Prototype Matching — on the real bacterial dataset.

Pipeline (mirrors Method A so the two are directly comparable)
--------------------------------------------------------------
1. Load train + val splits from `data/splits.json`.
2. Tile every frame at full resolution and extract DINOv2 features once. Optional
   illumination normalization happens on the GPU before tiling.
3. Initialize prototypes from pure-culture videos when available (mean of tile
   features per single-species video), else fall back to K-means on the tile pool.
4. No gradient learning — Method B is closed-form. Compute val tile similarities,
   aggregate to image level by mean, calibrate per-class thresholds and the unknown
   threshold on the val split.
5. Save model + config + thresholds.

Method B differs from Method A only in the decision rule:
    presence_k(x) = 1[ s_k(x) > theta_k^B ]   (mean cosine similarity > threshold)
    unknown(x)   = 1[ m(x)   < theta_unk^B ]  (mean max-similarity below threshold)
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.features import (
    extract_features_multicrop,
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.illumination import make_illumination_preprocess
from src.common.io import load_real_split, save_json
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.prototypes import init_prototypes_from_pure_cultures
from src.common.tiling import TileConfig
from src.simplex_unmixing.model import initialize_prototypes
from src.prototype_matching.model import ProtoConfig, PrototypeMatchingModel


def calibrate_per_class_thresholds(
    image_sims: np.ndarray,
    image_labels: np.ndarray,
    quantile: float,
    fallback: float = 0.5,
) -> List[float]:
    """For each class k, threshold = q-th percentile of s_k over positive images.

    Sensitivity-favoring: low quantile (e.g. 0.05) keeps most positives above the line.
    """
    K = image_labels.shape[1]
    thresholds: List[float] = []
    for k in range(K):
        positive_mask = image_labels[:, k] == 1
        if positive_mask.sum() == 0:
            thresholds.append(fallback)
            continue
        positives = image_sims[positive_mask, k]
        thresholds.append(float(np.quantile(positives, quantile)))
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Method B: Prototype Matching")

    # data
    parser.add_argument("--splits_path", type=str, default="data/splits.json")
    parser.add_argument("--output_dir", type=str, default="outputs/prototype_matching/default")
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)

    # backbone
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")

    # tiling
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--train_tiles_per_image", type=int, default=16)
    parser.add_argument("--eval_grid_size", type=int, default=4)

    # illumination
    parser.add_argument("--illumination", type=str, default="divide",
                        choices=["divide", "subtract", "none"])
    parser.add_argument("--illum_sigma", type=float, default=64.0)

    # extraction
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_batch_size", type=int, default=8,
                        help="Frames per GPU batch in the GPU illumination path.")
    parser.add_argument("--num_workers", type=int, default=4)

    # init
    parser.add_argument("--num_prototypes", type=int, default=0,
                        help="0 = use len(class_names)")
    parser.add_argument("--init", type=str, default="auto",
                        choices=["auto", "pure", "kmeans", "random"])

    # calibration
    parser.add_argument("--calibrate_quantile", type=float, default=0.05,
                        help="Per-class presence threshold = q-th percentile of positive sims")
    parser.add_argument("--unknown_quantile", type=float, default=0.05,
                        help="Unknown threshold = q-th percentile of val max-similarity")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load splits --------------------------------------------------------------
    train_paths, train_labels, class_names, train_videos = load_real_split(args.splits_path, "train")
    val_paths, val_labels, _, val_videos = load_real_split(args.splits_path, "val")

    if args.max_train_images is not None:
        train_paths = train_paths[: args.max_train_images]
        train_labels = train_labels[: args.max_train_images]
        train_videos = train_videos[: args.max_train_images]
    if args.max_val_images is not None:
        val_paths = val_paths[: args.max_val_images]
        val_labels = val_labels[: args.max_val_images]
        val_videos = val_videos[: args.max_val_images]

    K = args.num_prototypes if args.num_prototypes > 0 else len(class_names)
    print(f"Loaded {len(train_paths)} train + {len(val_paths)} val frames; "
          f"{len(class_names)} classes: {class_names}")

    # ---- 2. tile + extract features --------------------------------------------------
    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        train_tiles_per_image=args.train_tiles_per_image,
        eval_grid_size=args.eval_grid_size,
    )
    print(f"TileConfig: {tile_cfg}")
    print(f"Illumination: method={args.illumination}, sigma={args.illum_sigma}")

    use_gpu_illum = device.type == "cuda" and args.illumination != "none"

    if use_gpu_illum:
        print("Using GPU illumination path")
        print("Extracting train tile features...")
        train_features, train_image_index = extract_features_multicrop_gpu(
            image_paths=train_paths,
            tile_config=tile_cfg,
            backbone=args.backbone,
            frame_batch_size=args.frame_batch_size,
            num_workers=args.num_workers,
            device=device,
            illum_sigma=args.illum_sigma,
            illum_method=args.illumination,
            cache_path=str(output_dir / "train_features_cache.pt"),
        )
        print(f"  train tile features: {tuple(train_features.shape)}")
        print("Extracting val tile features...")
        val_features, val_image_index = extract_features_multicrop_gpu(
            image_paths=val_paths,
            tile_config=tile_cfg,
            backbone=args.backbone,
            frame_batch_size=args.frame_batch_size,
            num_workers=args.num_workers,
            device=device,
            illum_sigma=args.illum_sigma,
            illum_method=args.illumination,
            cache_path=str(output_dir / "val_features_cache.pt"),
        )
        print(f"  val tile features: {tuple(val_features.shape)}")
    else:
        preprocess_fn = make_illumination_preprocess(
            sigma=args.illum_sigma, method=args.illumination
        )
        print("Extracting train tile features...")
        train_features, train_image_index = extract_features_multicrop(
            image_paths=train_paths,
            tile_config=tile_cfg,
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            cache_path=str(output_dir / "train_features_cache.pt"),
            mode="eval",
            preprocess_fn=preprocess_fn,
        )
        print(f"  train tile features: {tuple(train_features.shape)}")
        print("Extracting val tile features...")
        val_features, val_image_index = extract_features_multicrop(
            image_paths=val_paths,
            tile_config=tile_cfg,
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            cache_path=str(output_dir / "val_features_cache.pt"),
            mode="eval",
            preprocess_fn=preprocess_fn,
        )
        print(f"  val tile features: {tuple(val_features.shape)}")

    train_features = F.normalize(train_features, p=2, dim=1)
    val_features = F.normalize(val_features, p=2, dim=1)

    # ---- 3. initialize prototypes ----------------------------------------------------
    print(f"Initializing {K} prototypes (init={args.init})...")
    proto_init = None
    if args.init in {"auto", "pure"}:
        proto_init = init_prototypes_from_pure_cultures(
            tile_features=train_features,
            image_index=train_image_index,
            video_ids=train_videos,
            class_names=class_names,
        )
        if proto_init is not None:
            print("  using pure-culture mean prototypes")
        elif args.init == "pure":
            raise SystemExit("init=pure but no pure-culture video found for every class")

    if proto_init is None:
        init_kind = "random" if args.init == "random" else "kmeans"
        print(f"  using {init_kind} init")
        proto_init = initialize_prototypes(train_features, K, init=init_kind)

    # ---- 4. build model (no training) ------------------------------------------------
    proto_cfg = ProtoConfig(
        embedding_dim=train_features.shape[1],
        num_prototypes=K,
    )
    model = PrototypeMatchingModel(proto_cfg).to(device)
    model.prototypes.data.copy_(proto_init.to(device))
    model.eval()

    # ---- 5. forward val tiles, aggregate to image level ------------------------------
    print("Computing val image-level predictions...")
    with torch.no_grad():
        val_sims_tile, val_max_tile = model(val_features.to(device))
    val_sims_tile = val_sims_tile.cpu()
    val_max_tile = val_max_tile.cpu()

    num_val = len(val_paths)
    val_sims_image = scatter_mean_by_image(val_sims_tile, val_image_index, num_val)
    val_max_image = scatter_mean_by_image(
        val_max_tile.unsqueeze(-1), val_image_index, num_val
    ).squeeze(-1)

    val_sims_np = val_sims_image.numpy()
    val_max_np = val_max_image.numpy()

    # ---- 6. calibrate per-class presence thresholds ----------------------------------
    presence_thresholds = calibrate_per_class_thresholds(
        image_sims=val_sims_np[:, :K],
        image_labels=val_labels,
        quantile=args.calibrate_quantile,
    )
    print(f"Per-class presence thresholds (q={args.calibrate_quantile}):")
    for name, thr in zip(class_names, presence_thresholds):
        print(f"  {name}: {thr:.4f}")

    # ---- 7. calibrate unknown threshold ---------------------------------------------
    unknown_threshold = float(np.quantile(val_max_np, args.unknown_quantile))
    print(f"Unknown threshold (q={args.unknown_quantile}): {unknown_threshold:.4f}")

    # ---- 8. val sanity-check metrics --------------------------------------------------
    val_pred = (val_sims_np[:, :K] > np.array(presence_thresholds)).astype(np.int64)
    val_f1 = per_sample_f1(val_labels, val_pred)
    val_macro = macro_f1_per_class(val_labels, val_pred, class_names=class_names)
    val_em = exact_match_accuracy(val_labels, val_pred)
    print(f"\nVal sanity (calibration set; test numbers come from run_presence_detection.py):")
    print(f"  per-sample F1: {val_f1:.4f}")
    print(f"  macro F1:      {val_macro['macro']:.4f}")
    print(f"  per-class F1:  " + ", ".join(
        f"{c}={val_macro[c]:.3f}" for c in class_names))
    print(f"  exact match:   {val_em:.4f}")

    # ---- 9. save artifacts -----------------------------------------------------------
    proto_cfg.thresholds = presence_thresholds
    proto_cfg.unknown_threshold = unknown_threshold

    model_path = output_dir / "proto_model.pt"
    torch.save(model.state_dict(), model_path)

    save_json(
        str(output_dir / "config.json"),
        {
            "method": "prototype_matching",
            "splits_path": args.splits_path,
            "backbone": args.backbone,
            "tile_config": asdict(tile_cfg),
            "illumination": args.illumination,
            "illum_sigma": args.illum_sigma,
            "num_prototypes": K,
            "embedding_dim": train_features.shape[1],
            "class_names": class_names,
            "thresholds": presence_thresholds,
            "unknown_threshold": unknown_threshold,
            "calibrate_quantile": args.calibrate_quantile,
            "unknown_quantile": args.unknown_quantile,
        },
    )
    save_json(
        str(output_dir / "train_summary.json"),
        {
            "num_train_images": len(train_paths),
            "num_val_images": len(val_paths),
            "val_per_sample_f1": val_f1,
            "val_macro_f1": val_macro["macro"],
            "val_per_class_f1": {c: val_macro[c] for c in class_names},
            "val_exact_match": val_em,
        },
    )
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
