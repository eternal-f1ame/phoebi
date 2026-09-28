#!/usr/bin/env python3
"""Train Method A — Simplex Unmixing — on the real bacterial dataset.

Pipeline
--------
1. Load train + val splits from `data/splits.json` (built by tools/build_splits.py).
2. Tile every frame at full resolution (Assumption H makes tiles label-preserving) and
   extract DINOv2 features once. Optional illumination normalization is applied to the
   full-resolution image before tiling.
3. Initialize prototypes from pure-culture videos when available (mean of tile features
   per single-species video), else fall back to K-means on the tile pool.
4. Train UnmixerModel on per-tile reconstruction MSE — no entropy regularizer (sparsity
   is enforced by sparsemax itself; the prior +lambda*H(w) term pushed weights toward
   uniform and fought the projection).
5. Aggregate val tile predictions to image level via scatter_mean (justified by
   Assumption H).
6. Calibrate per-class presence thresholds at the val-image level: for each class k,
   theta_k = (calibrate_quantile)-th percentile of weights w_k(x) across positive images.
7. Calibrate the residual unknown threshold at the val-image level:
   theta_unk = (unknown_quantile)-th percentile of residual norms r(x).
8. Save model + tile config + thresholds + class names. Headline numbers (test split)
   come from experiments/run_presence_detection.py.

CLI defaults are set so a fresh run looks like:
    python -m src.simplex_unmixing.train
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    sparsity_score,
)
from src.common.prototypes import init_prototypes_from_pure_cultures
from src.common.sinkhorn import sinkhorn_cluster
from src.common.tiling import TileConfig
from src.simplex_unmixing.model import (
    ModelConfig,
    UnmixerModel,
    greedy_cosine_clustering,
    initialize_prototypes,
)


def calibrate_per_class_thresholds(
    image_weights: np.ndarray,
    image_labels: np.ndarray,
    quantile: float,
    fallback: float = 0.05,
) -> List[float]:
    """For each class k, threshold = q-th percentile of w_k over positive images."""
    K = image_labels.shape[1]
    thresholds: List[float] = []
    for k in range(K):
        positive_mask = image_labels[:, k] == 1
        if positive_mask.sum() == 0:
            thresholds.append(fallback)
            continue
        positives = image_weights[positive_mask, k]
        thresholds.append(float(np.quantile(positives, quantile)))
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Method A: Simplex Unmixing")

    # data
    parser.add_argument("--splits_path", type=str, default="data/splits.json")
    parser.add_argument("--output_dir", type=str, default="outputs/simplex_unmixing/default")
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
                        help="Frames per GPU batch in the GPU illumination path. "
                             "Each frame yields eval_grid_size^2 tiles, so the effective "
                             "tile batch is frame_batch_size * eval_grid_size^2.")
    parser.add_argument("--num_workers", type=int, default=4)

    # model + training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--num_prototypes", type=int, default=0,
                        help="0 = use len(class_names)")
    parser.add_argument("--init", type=str, default="auto",
                        choices=["auto", "pure", "kmeans", "random"])

    # discovery (open-world)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--discovery_every", type=int, default=5)
    parser.add_argument("--residual_threshold", type=float, default=0.15)
    parser.add_argument("--cluster_similarity", type=float, default=0.9)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--residual_buffer", type=int, default=1000)
    parser.add_argument("--cluster_method", type=str, default="sinkhorn",
                        choices=["greedy", "sinkhorn"],
                        help="greedy: variable-K cosine clustering. sinkhorn: balanced k-means with --sinkhorn_k clusters per discovery round.")
    parser.add_argument("--sinkhorn_k", type=int, default=1,
                        help="Number of new prototypes to propose per discovery round when --cluster_method=sinkhorn.")

    # calibration
    parser.add_argument("--calibrate_quantile", type=float, default=0.05,
                        help="Per-class presence threshold = q-th percentile of positive weights")
    parser.add_argument("--unknown_quantile", type=float, default=0.95,
                        help="Residual unknown threshold = q-th percentile of val residual norms")

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

    # GPU illumination path: workers do nothing but JPEG decode, illumination + tiling +
    # forward all happen on GPU. Falls back to CPU preprocess_fn when CUDA is unavailable.
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

    # ---- 4. build + train model ------------------------------------------------------
    model_cfg = ModelConfig(
        embedding_dim=train_features.shape[1],
        num_prototypes=K,
        temperature=args.temperature,
    )
    model = UnmixerModel(model_cfg).to(device)
    model.prototypes.data.copy_(proto_init.to(device))

    dataset = TensorDataset(train_features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    residual_buffer: List[torch.Tensor] = []

    print(f"Training for {args.epochs} epochs on {len(dataset)} tile features...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            z_recon, weights, _ = model(batch)
            loss = F.mse_loss(z_recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)

        # Optional open-world prototype discovery.
        # Bug-fix vs original: the cluster centroids must be in FEATURE space
        # (not residual space), because prototypes are evaluated by cosine
        # similarity against features. We buffer the FEATURES of high-residual
        # tiles and cluster those.
        if args.discover and epoch % args.discovery_every == 0:
            model.eval()
            with torch.no_grad():
                _, _, residuals = model(train_features.to(device))
                residual_norms = residuals.detach().cpu().norm(p=2, dim=1)
                outlier_mask = residual_norms > args.residual_threshold
                if outlier_mask.any():
                    novel_feats = train_features[outlier_mask].cpu()
                    residual_buffer.extend(list(novel_feats))
                    if len(residual_buffer) > args.residual_buffer:
                        residual_buffer = residual_buffer[-args.residual_buffer:]
                if len(residual_buffer) >= args.min_cluster_size:
                    buf = torch.stack(residual_buffer)
                    if args.cluster_method == "sinkhorn":
                        new_protos, _ = sinkhorn_cluster(
                            buf.to(device),
                            num_clusters=args.sinkhorn_k,
                            num_iters=30,
                            sk_iters=3,
                            sk_epsilon=0.05,
                            device=device,
                        )
                    else:
                        new_protos = greedy_cosine_clustering(
                            buf.to(device),
                            threshold=args.cluster_similarity,
                            min_size=args.min_cluster_size,
                        )
                    if new_protos is not None and new_protos.numel() > 0:
                        new_protos = new_protos.detach().cpu()
                        current = model.prototypes.data.cpu()
                        updated = torch.cat([current, new_protos], dim=0)
                        model.prototypes = nn.Parameter(updated.to(device))
                        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                                weight_decay=args.weight_decay)
                        residual_buffer = []

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | recon-MSE {epoch_loss:.6f} | "
                  f"prototypes {model.prototypes.shape[0]}")

    # ---- 5. forward val tiles, aggregate to image level ------------------------------
    print("Computing val image-level predictions...")
    model.eval()
    with torch.no_grad():
        _, val_weights_tile, val_residuals_tile = model(val_features.to(device))
    val_weights_tile = val_weights_tile.cpu()
    val_residual_norms_tile = val_residuals_tile.cpu().norm(p=2, dim=1)

    num_val = len(val_paths)
    val_weights_image = scatter_mean_by_image(val_weights_tile, val_image_index, num_val)
    val_residual_norms_image = scatter_mean_by_image(
        val_residual_norms_tile.unsqueeze(-1), val_image_index, num_val
    ).squeeze(-1)

    val_weights_np = val_weights_image.numpy()
    val_residuals_np = val_residual_norms_image.numpy()

    # ---- 6. calibrate per-class presence thresholds ----------------------------------
    presence_thresholds = calibrate_per_class_thresholds(
        image_weights=val_weights_np[:, :K],
        image_labels=val_labels,
        quantile=args.calibrate_quantile,
    )
    print(f"Per-class presence thresholds (q={args.calibrate_quantile}):")
    for name, thr in zip(class_names, presence_thresholds):
        print(f"  {name}: {thr:.4f}")

    # ---- 7. calibrate residual unknown threshold -------------------------------------
    residual_unknown_threshold = float(np.quantile(val_residuals_np, args.unknown_quantile))
    print(f"Residual unknown threshold (q={args.unknown_quantile}): "
          f"{residual_unknown_threshold:.4f}")

    # ---- 8. val sanity-check metrics --------------------------------------------------
    val_pred = (val_weights_np[:, :K] > np.array(presence_thresholds)).astype(np.int64)
    val_f1 = per_sample_f1(val_labels, val_pred)
    val_macro = macro_f1_per_class(val_labels, val_pred, class_names=class_names)
    val_em = exact_match_accuracy(val_labels, val_pred)
    val_sparsity = sparsity_score(val_weights_np)
    print(f"\nVal sanity (calibration set; test numbers come from run_presence_detection.py):")
    print(f"  per-sample F1: {val_f1:.4f}")
    print(f"  macro F1:      {val_macro['macro']:.4f}")
    print(f"  per-class F1:  " + ", ".join(
        f"{c}={val_macro[c]:.3f}" for c in class_names))
    print(f"  exact match:   {val_em:.4f}")
    print(f"  sparsity:      {val_sparsity:.4f}")

    # ---- 9. save artifacts -----------------------------------------------------------
    model_path = output_dir / "phoebi_model.pt"
    torch.save(model.state_dict(), model_path)

    save_json(
        str(output_dir / "config.json"),
        {
            "method": "simplex_unmixing",
            "splits_path": args.splits_path,
            "backbone": args.backbone,
            "tile_config": asdict(tile_cfg),
            "illumination": args.illumination,
            "illum_sigma": args.illum_sigma,
            "num_prototypes": int(model.prototypes.shape[0]),
            "embedding_dim": train_features.shape[1],
            "temperature": args.temperature,
            "class_names": class_names,
            "thresholds": presence_thresholds,
            "unknown_threshold": residual_unknown_threshold,
            "calibrate_quantile": args.calibrate_quantile,
            "unknown_quantile": args.unknown_quantile,
        },
    )
    save_json(
        str(output_dir / "train_summary.json"),
        {
            "epochs": args.epochs,
            "final_recon_mse": epoch_loss,
            "num_train_images": len(train_paths),
            "num_val_images": len(val_paths),
            "val_per_sample_f1": val_f1,
            "val_macro_f1": val_macro["macro"],
            "val_per_class_f1": {c: val_macro[c] for c in class_names},
            "val_exact_match": val_em,
            "val_sparsity": val_sparsity,
        },
    )
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
