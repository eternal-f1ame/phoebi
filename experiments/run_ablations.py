#!/usr/bin/env python3
"""Ablation harness for PHOEBI — real-data tile pipeline.

Each subcommand produces one paper table:

    tile_count     ->  tab_tile_sweep.tex   (T in {1,4,8,16,32})
    tile_size      ->  tab_ablations.tex    (s in {168, 224, 336, 518})  [section]
    illumination   ->  tab_ablations.tex    (none / subtract / divide)   [section]
    projection     ->  tab_ablations.tex    (sparsemax vs softmax)       [section, A only]
    threshold      ->  tab_ablations.tex    (q in {0.01, 0.05, 0.10, 0.20}) [section]
    proto_init     ->  tab_proto_init.tex   (pure-culture vs k-means)

Design
------
The expensive steps are (1) decoding JPEGs, (2) illumination normalization, and (3)
DINOv2 forward. Both methods' feature caches (train_features_cache.pt,
val_features_cache.pt, and a test cache built here) are keyed on
(paths, tile_config, backbone, illumination, sigma). When an ablation changes only
a downstream knob (threshold quantile, projection function, init method, tile count
via subsampling), we reuse features; when it changes tile size or illumination we
re-extract into a sweep-local cache.

Each subcommand writes:
    outputs/ablations/<sweep>/results.csv
    outputs/ablations/<sweep>/results.json
    outputs/ablations/<sweep>/table_body.tex   # rows ready to splice into the paper

The reference config mirrors the defaults in src.simplex_unmixing.train and
src.prototype_matching.train: tile_size=224, eval_grid_size=4, illum=divide,
sigma=64, temperature=10, calibrate_quantile=0.05.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
    sparsity_score,
)
from src.common.prototypes import init_prototypes_from_pure_cultures
from src.common.tiling import TileConfig
from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig
from src.simplex_unmixing.model import (
    ModelConfig,
    UnmixerModel,
    initialize_prototypes,
    sparsemax,
)


# ---------------------------------------------------------------------------
# Feature extraction (shared)
# ---------------------------------------------------------------------------


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
    cache_path: Path,
    max_images: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], np.ndarray, List[str]]:
    """Return (features [N*T, D], image_index [N*T], class_names, labels [N, K], video_ids)."""
    paths, labels, class_names, video_ids = load_real_split(splits_path, split)
    if max_images is not None:
        paths = paths[:max_images]
        labels = labels[:max_images]
        video_ids = video_ids[:max_images]

    use_gpu = device.type == "cuda" and illum_method != "none"
    if use_gpu:
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
    return features, image_index, class_names, labels, video_ids


# ---------------------------------------------------------------------------
# Training helpers (shared)
# ---------------------------------------------------------------------------


def train_simplex(
    train_features: torch.Tensor,
    proto_init: torch.Tensor,
    K: int,
    temperature: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    activation: str = "sparsemax",
) -> UnmixerModel:
    """Gradient-train a UnmixerModel from a prototype init. activation chooses the
    simplex projection: 'sparsemax' (default) or 'softmax'."""
    model_cfg = ModelConfig(
        embedding_dim=train_features.shape[1],
        num_prototypes=K,
        temperature=temperature,
    )
    model = UnmixerModel(model_cfg).to(device)
    model.prototypes.data.copy_(proto_init.to(device))

    if activation == "softmax":
        # Swap the projection for this ablation only. The model docstring specifies
        # sparsemax as the default (Euclidean projection onto the simplex); softmax is
        # the dense alternative used here as a baseline. `self.activation` must remain
        # an nn.Module, so we wrap softmax in one rather than patching a plain fn.
        import torch.nn as nn
        class _SoftmaxModule(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.softmax(x, dim=1)
        model.activation = _SoftmaxModule()

    dataset = TensorDataset(train_features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            z_recon, _, _ = model(batch)
            loss = F.mse_loss(z_recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    simplex epoch {epoch:03d}: recon-MSE={total / len(dataset):.6f}")
    return model


def forward_simplex_image_level(
    model: UnmixerModel,
    tile_features: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
    device: torch.device,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (weights_image [N, K], residual_norms_image [N])."""
    model.eval()
    with torch.no_grad():
        _, w_tile, r_tile = model(tile_features.to(device))
    w_tile = w_tile.cpu()
    r_norms_tile = r_tile.cpu().norm(p=2, dim=1)
    w_image = scatter_mean_by_image(w_tile, image_index, num_images)
    r_image = scatter_mean_by_image(
        r_norms_tile.unsqueeze(-1), image_index, num_images
    ).squeeze(-1)
    return w_image[:, :K].numpy(), r_image.numpy()


def forward_prototype_image_level(
    prototypes: torch.Tensor,
    tile_features: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (similarities_image [N, K], max_sim_image [N])."""
    proto_cfg = ProtoConfig(
        embedding_dim=prototypes.shape[1],
        num_prototypes=prototypes.shape[0],
    )
    model = PrototypeMatchingModel(proto_cfg).to(device)
    model.prototypes.data.copy_(prototypes.to(device))
    model.eval()
    with torch.no_grad():
        sims_tile, max_sim_tile = model(tile_features.to(device))
    sims_tile = sims_tile.cpu()
    max_sim_tile = max_sim_tile.cpu()
    sims_image = scatter_mean_by_image(sims_tile, image_index, num_images)
    max_sim_image = scatter_mean_by_image(
        max_sim_tile.unsqueeze(-1), image_index, num_images
    ).squeeze(-1)
    return sims_image.numpy(), max_sim_image.numpy()


def calibrate_thresholds(
    scores: np.ndarray, labels: np.ndarray, quantile: float, fallback: float = 0.05
) -> np.ndarray:
    K = labels.shape[1]
    out = np.full(K, fallback, dtype=np.float32)
    for k in range(K):
        mask = labels[:, k] == 1
        if mask.sum() == 0:
            continue
        out[k] = float(np.quantile(scores[mask, k], quantile))
    return out


def summarize(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    pcf1 = macro_f1_per_class(y_true, y_pred, class_names)
    return {
        "per_sample_f1": float(per_sample_f1(y_true, y_pred)),
        "macro_f1": float(pcf1["macro"]),
        "exact_match": float(exact_match_accuracy(y_true, y_pred)),
        "per_class_f1": {c: float(pcf1[c]) for c in class_names},
    }


# ---------------------------------------------------------------------------
# Sweep 1: Tile-count subsampling (cheap — reuses the T=16 cache)
# ---------------------------------------------------------------------------


def subsample_tiles_by_image(
    features: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
    k_tiles: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each image, pick up to k_tiles tiles (without replacement). Returns the
    filtered feature tensor and matching image_index."""
    image_index_np = image_index.numpy()
    keep_rows: List[int] = []
    for i in range(num_images):
        rows_i = np.where(image_index_np == i)[0]
        if len(rows_i) == 0:
            continue
        take = min(k_tiles, len(rows_i))
        picked = rng.choice(rows_i, size=take, replace=False)
        keep_rows.extend(picked.tolist())
    keep = torch.as_tensor(keep_rows, dtype=torch.long)
    return features[keep], image_index[keep]


def sweep_tile_count(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / "tile_count"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = TileConfig(tile_size=224, train_tiles_per_image=16, eval_grid_size=4)
    train_feat, train_idx, class_names, train_labels, train_videos = extract_split(
        "train", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "train_features_cache.pt", args.max_train_images,
    )
    val_feat, val_idx, _, val_labels, _ = extract_split(
        "val", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "val_features_cache.pt", args.max_val_images,
    )
    test_feat, test_idx, _, test_labels, _ = extract_split(
        "test", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "test_features_cache.pt", args.max_test_images,
    )

    K = len(class_names)
    N_val = val_labels.shape[0]
    N_test = test_labels.shape[0]

    proto_init = init_prototypes_from_pure_cultures(
        tile_features=train_feat, image_index=train_idx,
        video_ids=train_videos, class_names=class_names,
    )
    if proto_init is None:
        proto_init = initialize_prototypes(train_feat, K, init="kmeans")

    simplex = train_simplex(
        train_feat, proto_init, K, args.temperature, args.epochs, args.lr,
        args.batch_size, device,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    tile_counts = [1, 4, 8, 16]
    for T in tile_counts:
        # Val subsample for calibration
        val_feat_sub, val_idx_sub = subsample_tiles_by_image(val_feat, val_idx, N_val, T, rng)
        test_feat_sub, test_idx_sub = subsample_tiles_by_image(test_feat, test_idx, N_test, T, rng)

        # --- Method A ---
        w_val_A, _ = forward_simplex_image_level(simplex, val_feat_sub, val_idx_sub, N_val, device, K)
        w_test_A, _ = forward_simplex_image_level(simplex, test_feat_sub, test_idx_sub, N_test, device, K)
        thr_A = calibrate_thresholds(w_val_A, val_labels, args.calibrate_quantile)
        yhat_A = (w_test_A > thr_A).astype(np.int64)
        res_A = summarize(test_labels, yhat_A, class_names)

        # --- Method B ---
        s_val_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), val_feat_sub, val_idx_sub, N_val, device)
        s_test_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), test_feat_sub, test_idx_sub, N_test, device)
        thr_B = calibrate_thresholds(s_val_B[:, :K], val_labels, args.calibrate_quantile)
        yhat_B = (s_test_B[:, :K] > thr_B).astype(np.int64)
        res_B = summarize(test_labels, yhat_B, class_names)

        rows.append({
            "T": T,
            "A_per_sample_f1": res_A["per_sample_f1"],
            "A_macro_f1": res_A["macro_f1"],
            "B_per_sample_f1": res_B["per_sample_f1"],
            "B_macro_f1": res_B["macro_f1"],
        })
        print(f"  T={T:2d}: A F1={res_A['per_sample_f1']:.4f}  B F1={res_B['per_sample_f1']:.4f}")

    _write_results(out_dir, rows)
    _render_tile_count_tex(out_dir / "table_body.tex", rows)


# ---------------------------------------------------------------------------
# Sweep 2: Threshold quantile (cheap — uses the T=16 trained model)
# ---------------------------------------------------------------------------


def sweep_threshold(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / "threshold"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = TileConfig(tile_size=224, train_tiles_per_image=16, eval_grid_size=4)
    train_feat, train_idx, class_names, _, train_videos = extract_split(
        "train", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "train_features_cache.pt", args.max_train_images,
    )
    val_feat, val_idx, _, val_labels, _ = extract_split(
        "val", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "val_features_cache.pt", args.max_val_images,
    )
    test_feat, test_idx, _, test_labels, _ = extract_split(
        "test", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "test_features_cache.pt", args.max_test_images,
    )
    K = len(class_names)
    N_val = val_labels.shape[0]
    N_test = test_labels.shape[0]

    proto_init = init_prototypes_from_pure_cultures(
        tile_features=train_feat, image_index=train_idx,
        video_ids=train_videos, class_names=class_names,
    )
    if proto_init is None:
        proto_init = initialize_prototypes(train_feat, K, init="kmeans")

    simplex = train_simplex(
        train_feat, proto_init, K, args.temperature, args.epochs, args.lr,
        args.batch_size, device,
    )
    w_val_A, _ = forward_simplex_image_level(simplex, val_feat, val_idx, N_val, device, K)
    w_test_A, _ = forward_simplex_image_level(simplex, test_feat, test_idx, N_test, device, K)
    s_val_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), val_feat, val_idx, N_val, device)
    s_test_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), test_feat, test_idx, N_test, device)

    rows = []
    for q in [0.01, 0.05, 0.10, 0.20]:
        thr_A = calibrate_thresholds(w_val_A, val_labels, q)
        thr_B = calibrate_thresholds(s_val_B[:, :K], val_labels, q)
        yhat_A = (w_test_A > thr_A).astype(np.int64)
        yhat_B = (s_test_B[:, :K] > thr_B).astype(np.int64)
        res_A = summarize(test_labels, yhat_A, class_names)
        res_B = summarize(test_labels, yhat_B, class_names)
        rows.append({
            "q": q,
            "A_per_sample_f1": res_A["per_sample_f1"],
            "A_macro_f1": res_A["macro_f1"],
            "B_per_sample_f1": res_B["per_sample_f1"],
            "B_macro_f1": res_B["macro_f1"],
        })
        print(f"  q={q:.2f}: A F1={res_A['per_sample_f1']:.4f}  B F1={res_B['per_sample_f1']:.4f}")
    _write_results(out_dir, rows)


# ---------------------------------------------------------------------------
# Sweep 3: Projection function (simplex only — sparsemax vs softmax)
# ---------------------------------------------------------------------------


def sweep_projection(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / "projection"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = TileConfig(tile_size=224, train_tiles_per_image=16, eval_grid_size=4)
    train_feat, train_idx, class_names, _, train_videos = extract_split(
        "train", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "train_features_cache.pt", args.max_train_images,
    )
    val_feat, val_idx, _, val_labels, _ = extract_split(
        "val", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "val_features_cache.pt", args.max_val_images,
    )
    test_feat, test_idx, _, test_labels, _ = extract_split(
        "test", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "test_features_cache.pt", args.max_test_images,
    )
    K = len(class_names)
    N_val = val_labels.shape[0]
    N_test = test_labels.shape[0]

    proto_init = init_prototypes_from_pure_cultures(
        tile_features=train_feat, image_index=train_idx,
        video_ids=train_videos, class_names=class_names,
    )
    if proto_init is None:
        proto_init = initialize_prototypes(train_feat, K, init="kmeans")

    rows = []
    for activation in ["sparsemax", "softmax"]:
        model = train_simplex(
            train_feat, proto_init.clone(), K, args.temperature, args.epochs, args.lr,
            args.batch_size, device, activation=activation,
        )
        w_val, _ = forward_simplex_image_level(model, val_feat, val_idx, N_val, device, K)
        w_test, _ = forward_simplex_image_level(model, test_feat, test_idx, N_test, device, K)
        thr = calibrate_thresholds(w_val, val_labels, args.calibrate_quantile)
        yhat = (w_test > thr).astype(np.int64)
        res = summarize(test_labels, yhat, class_names)
        sparsity = sparsity_score(w_test)
        rows.append({
            "activation": activation,
            "per_sample_f1": res["per_sample_f1"],
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
            "sparsity": float(sparsity),
        })
        print(f"  activation={activation}: F1={res['per_sample_f1']:.4f} "
              f"macro={res['macro_f1']:.4f} sparsity={sparsity:.3f}")
    _write_results(out_dir, rows)


# ---------------------------------------------------------------------------
# Sweep 4: Prototype init (pure-culture vs K-means)
# ---------------------------------------------------------------------------


def sweep_proto_init(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / "proto_init"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_cfg = TileConfig(tile_size=224, train_tiles_per_image=16, eval_grid_size=4)
    train_feat, train_idx, class_names, _, train_videos = extract_split(
        "train", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "train_features_cache.pt", args.max_train_images,
    )
    val_feat, val_idx, _, val_labels, _ = extract_split(
        "val", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "val_features_cache.pt", args.max_val_images,
    )
    test_feat, test_idx, _, test_labels, _ = extract_split(
        "test", args.splits_path, tile_cfg, args.backbone, device,
        args.illumination, args.illum_sigma,
        args.frame_batch_size, args.batch_size, args.num_workers,
        out_dir / "test_features_cache.pt", args.max_test_images,
    )
    K = len(class_names)
    N_val = val_labels.shape[0]
    N_test = test_labels.shape[0]

    rows = []
    for init_kind in ["pure", "kmeans"]:
        if init_kind == "pure":
            proto_init = init_prototypes_from_pure_cultures(
                tile_features=train_feat, image_index=train_idx,
                video_ids=train_videos, class_names=class_names,
            )
            if proto_init is None:
                print("  WARN: pure-culture init unavailable — skipping")
                continue
        else:
            proto_init = initialize_prototypes(train_feat, K, init="kmeans")

        for method in ["simplex", "prototype"]:
            if method == "simplex":
                model = train_simplex(
                    train_feat, proto_init.clone(), K, args.temperature, args.epochs,
                    args.lr, args.batch_size, device,
                )
                w_val, _ = forward_simplex_image_level(model, val_feat, val_idx, N_val, device, K)
                w_test, _ = forward_simplex_image_level(model, test_feat, test_idx, N_test, device, K)
                thr = calibrate_thresholds(w_val, val_labels, args.calibrate_quantile)
                yhat = (w_test > thr).astype(np.int64)
                score_name = "weights"
            else:
                s_val, _ = forward_prototype_image_level(
                    proto_init.clone(), val_feat, val_idx, N_val, device,
                )
                s_test, _ = forward_prototype_image_level(
                    proto_init.clone(), test_feat, test_idx, N_test, device,
                )
                thr = calibrate_thresholds(s_val[:, :K], val_labels, args.calibrate_quantile)
                yhat = (s_test[:, :K] > thr).astype(np.int64)
                score_name = "similarities"
            res = summarize(test_labels, yhat, class_names)
            rows.append({
                "init": init_kind,
                "method": method,
                "score": score_name,
                "per_sample_f1": res["per_sample_f1"],
                "macro_f1": res["macro_f1"],
                "exact_match": res["exact_match"],
            })
            print(f"  init={init_kind} method={method}: "
                  f"F1={res['per_sample_f1']:.4f} macro={res['macro_f1']:.4f}")
    _write_results(out_dir, rows)


# ---------------------------------------------------------------------------
# Sweep 5: Illumination (re-extracts features for each variant)
# ---------------------------------------------------------------------------


def sweep_illumination(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / "illumination"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in ["none", "subtract", "divide"]:
        sub = out_dir / method
        sub.mkdir(exist_ok=True)
        tile_cfg = TileConfig(tile_size=224, train_tiles_per_image=16, eval_grid_size=4)
        train_feat, train_idx, class_names, _, train_videos = extract_split(
            "train", args.splits_path, tile_cfg, args.backbone, device,
            method, args.illum_sigma,
            args.frame_batch_size, args.batch_size, args.num_workers,
            sub / "train_features_cache.pt", args.max_train_images,
        )
        val_feat, val_idx, _, val_labels, _ = extract_split(
            "val", args.splits_path, tile_cfg, args.backbone, device,
            method, args.illum_sigma,
            args.frame_batch_size, args.batch_size, args.num_workers,
            sub / "val_features_cache.pt", args.max_val_images,
        )
        test_feat, test_idx, _, test_labels, _ = extract_split(
            "test", args.splits_path, tile_cfg, args.backbone, device,
            method, args.illum_sigma,
            args.frame_batch_size, args.batch_size, args.num_workers,
            sub / "test_features_cache.pt", args.max_test_images,
        )
        K = len(class_names)
        N_val = val_labels.shape[0]
        N_test = test_labels.shape[0]

        proto_init = init_prototypes_from_pure_cultures(
            tile_features=train_feat, image_index=train_idx,
            video_ids=train_videos, class_names=class_names,
        )
        if proto_init is None:
            proto_init = initialize_prototypes(train_feat, K, init="kmeans")

        simplex = train_simplex(
            train_feat, proto_init, K, args.temperature, args.epochs, args.lr,
            args.batch_size, device,
        )
        w_val_A, _ = forward_simplex_image_level(simplex, val_feat, val_idx, N_val, device, K)
        w_test_A, _ = forward_simplex_image_level(simplex, test_feat, test_idx, N_test, device, K)
        s_val_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), val_feat, val_idx, N_val, device)
        s_test_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), test_feat, test_idx, N_test, device)

        thr_A = calibrate_thresholds(w_val_A, val_labels, args.calibrate_quantile)
        thr_B = calibrate_thresholds(s_val_B[:, :K], val_labels, args.calibrate_quantile)
        yhat_A = (w_test_A > thr_A).astype(np.int64)
        yhat_B = (s_test_B[:, :K] > thr_B).astype(np.int64)
        res_A = summarize(test_labels, yhat_A, class_names)
        res_B = summarize(test_labels, yhat_B, class_names)
        rows.append({
            "illumination": method,
            "A_per_sample_f1": res_A["per_sample_f1"],
            "A_macro_f1": res_A["macro_f1"],
            "B_per_sample_f1": res_B["per_sample_f1"],
            "B_macro_f1": res_B["macro_f1"],
        })
        print(f"  illum={method}: A F1={res_A['per_sample_f1']:.4f}  B F1={res_B['per_sample_f1']:.4f}")
    _write_results(out_dir, rows)


# ---------------------------------------------------------------------------
# Sweep 6: Tile size (re-extracts features for each size)
# ---------------------------------------------------------------------------


def sweep_tile_size(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / "tile_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in [168, 224, 336, 518]:
        sub = out_dir / f"s{s}"
        sub.mkdir(exist_ok=True)
        tile_cfg = TileConfig(tile_size=s, train_tiles_per_image=16, eval_grid_size=4)
        try:
            train_feat, train_idx, class_names, _, train_videos = extract_split(
                "train", args.splits_path, tile_cfg, args.backbone, device,
                args.illumination, args.illum_sigma,
                args.frame_batch_size, args.batch_size, args.num_workers,
                sub / "train_features_cache.pt", args.max_train_images,
            )
            val_feat, val_idx, _, val_labels, _ = extract_split(
                "val", args.splits_path, tile_cfg, args.backbone, device,
                args.illumination, args.illum_sigma,
                args.frame_batch_size, args.batch_size, args.num_workers,
                sub / "val_features_cache.pt", args.max_val_images,
            )
            test_feat, test_idx, _, test_labels, _ = extract_split(
                "test", args.splits_path, tile_cfg, args.backbone, device,
                args.illumination, args.illum_sigma,
                args.frame_batch_size, args.batch_size, args.num_workers,
                sub / "test_features_cache.pt", args.max_test_images,
            )
        except Exception as exc:
            print(f"  s={s}: extraction failed ({exc}); skipping")
            continue

        K = len(class_names)
        N_val = val_labels.shape[0]
        N_test = test_labels.shape[0]

        proto_init = init_prototypes_from_pure_cultures(
            tile_features=train_feat, image_index=train_idx,
            video_ids=train_videos, class_names=class_names,
        )
        if proto_init is None:
            proto_init = initialize_prototypes(train_feat, K, init="kmeans")

        simplex = train_simplex(
            train_feat, proto_init, K, args.temperature, args.epochs, args.lr,
            args.batch_size, device,
        )
        w_val_A, _ = forward_simplex_image_level(simplex, val_feat, val_idx, N_val, device, K)
        w_test_A, _ = forward_simplex_image_level(simplex, test_feat, test_idx, N_test, device, K)
        s_val_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), val_feat, val_idx, N_val, device)
        s_test_B, _ = forward_prototype_image_level(simplex.prototypes.detach(), test_feat, test_idx, N_test, device)

        thr_A = calibrate_thresholds(w_val_A, val_labels, args.calibrate_quantile)
        thr_B = calibrate_thresholds(s_val_B[:, :K], val_labels, args.calibrate_quantile)
        yhat_A = (w_test_A > thr_A).astype(np.int64)
        yhat_B = (s_test_B[:, :K] > thr_B).astype(np.int64)
        res_A = summarize(test_labels, yhat_A, class_names)
        res_B = summarize(test_labels, yhat_B, class_names)
        rows.append({
            "tile_size": s,
            "A_per_sample_f1": res_A["per_sample_f1"],
            "A_macro_f1": res_A["macro_f1"],
            "B_per_sample_f1": res_B["per_sample_f1"],
            "B_macro_f1": res_B["macro_f1"],
        })
        print(f"  s={s}: A F1={res_A['per_sample_f1']:.4f}  B F1={res_B['per_sample_f1']:.4f}")
    _write_results(out_dir, rows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_results(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"  no rows to write in {out_dir}")
        return
    save_json(str(out_dir / "results.json"), rows)
    keys = list(rows[0].keys())
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {out_dir / 'results.csv'}")


def _render_tile_count_tex(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        lines.append(
            f"${row['T']:>2d}$ & ${row['A_per_sample_f1']:.4f}$ & "
            f"${row['B_per_sample_f1']:.4f}$ \\\\"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--splits_path", type=str, default="data/splits.json")
    parser.add_argument("--output_dir", type=str, default="outputs/ablations")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--illumination", type=str, default="divide", choices=["divide", "subtract", "none"])
    parser.add_argument("--illum_sigma", type=float, default=64.0)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--frame_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--calibrate_quantile", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--max_test_images", type=int, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHOEBI ablation sweeps.")
    sub = parser.add_subparsers(dest="sweep", required=True)

    for name, fn in [
        ("tile_count", sweep_tile_count),
        ("threshold", sweep_threshold),
        ("projection", sweep_projection),
        ("proto_init", sweep_proto_init),
        ("illumination", sweep_illumination),
        ("tile_size", sweep_tile_size),
    ]:
        p = sub.add_parser(name)
        _add_common(p)
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
