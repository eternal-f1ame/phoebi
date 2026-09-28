#!/usr/bin/env python3
"""Presence detection on the real-data test split.

Loads a trained model (either simplex_unmixing or prototype_matching), extracts
tile features on the test split with the exact tile/illumination config used at
train time, aggregates to image level, applies val-calibrated per-class
thresholds, and reports per-sample F1, macro F1, per-class F1, and exact match.

Outputs the headline numbers that populate tab_headline.tex / tab_per_class.tex.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

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
from src.common.io import load_json, load_real_split, save_json
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import TileConfig


def load_simplex_model(model_dir: Path, config: dict, device: torch.device):
    from src.simplex_unmixing.model import ModelConfig, UnmixerModel

    model_cfg = ModelConfig(
        embedding_dim=config["embedding_dim"],
        num_prototypes=config["num_prototypes"],
        temperature=config["temperature"],
    )
    model = UnmixerModel(model_cfg)
    model.load_state_dict(
        torch.load(model_dir / "phoebi_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def load_prototype_model(model_dir: Path, config: dict, device: torch.device):
    from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig

    model_cfg = ProtoConfig(
        embedding_dim=config["embedding_dim"],
        num_prototypes=config["num_prototypes"],
        thresholds=config.get("thresholds"),
        unknown_threshold=config.get("unknown_threshold", 0.5),
    )
    model = PrototypeMatchingModel(model_cfg)
    model.load_state_dict(
        torch.load(model_dir / "proto_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def extract_test_features(
    paths, tile_cfg, backbone, device, illum_method, illum_sigma,
    frame_batch_size, batch_size, num_workers, cache_path,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return features, image_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Presence detection on the real-data test split.")
    parser.add_argument("--method", choices=["simplex", "prototype"], required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--splits_path", type=str, default="data/splits.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = load_json(str(model_dir / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        args.output_dir = str(model_dir / f"presence_{args.split}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load split ----------------------------------------------------------
    paths, labels, class_names, video_ids = load_real_split(args.splits_path, args.split)
    if args.max_images is not None:
        paths = paths[: args.max_images]
        labels = labels[: args.max_images]
    K = labels.shape[1]
    print(f"Loaded {len(paths)} images from split={args.split}, K={K} classes.")

    # Sanity: config's class_names should match splits class_names
    cfg_class_names = config.get("class_names")
    if cfg_class_names and cfg_class_names != class_names:
        print(
            f"WARN: config class_names {cfg_class_names} != splits class_names {class_names}. "
            "Will align by splits order."
        )

    # ---- extract test-split tile features -----------------------------------
    tile_cfg = TileConfig(**config["tile_config"])
    features, image_index = extract_test_features(
        paths=paths,
        tile_cfg=tile_cfg,
        backbone=config["backbone"],
        device=device,
        illum_method=config.get("illumination", "divide"),
        illum_sigma=config.get("illum_sigma", 64.0),
        frame_batch_size=args.frame_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=output_dir / f"{args.split}_features_cache.pt",
    )

    # ---- forward + aggregate --------------------------------------------------
    num_images = len(paths)

    if args.method == "simplex":
        model = load_simplex_model(model_dir, config, device)
        with torch.no_grad():
            _, weights_tile, residuals_tile = model(features.to(device))
        weights_tile = weights_tile.cpu()
        weights_image = scatter_mean_by_image(weights_tile, image_index, num_images)
        residual_norms_tile = residuals_tile.cpu().norm(p=2, dim=1).unsqueeze(-1)
        residual_image = scatter_mean_by_image(residual_norms_tile, image_index, num_images).squeeze(-1)
        scores = weights_image[:, :K].numpy()
        score_name = "weights"
        aux = residual_image.numpy()
    else:
        model = load_prototype_model(model_dir, config, device)
        with torch.no_grad():
            sims_tile, max_sim_tile = model(features.to(device))
        sims_tile = sims_tile.cpu()
        sims_image = scatter_mean_by_image(sims_tile, image_index, num_images)
        max_sim_tile = max_sim_tile.cpu().unsqueeze(-1)
        max_sim_image = scatter_mean_by_image(max_sim_tile, image_index, num_images).squeeze(-1)
        scores = sims_image[:, :K].numpy()
        score_name = "similarities"
        aux = max_sim_image.numpy()

    # ---- thresholds ----------------------------------------------------------
    thresholds = config.get("thresholds")
    if thresholds is None:
        thresholds = [0.05] * K
    thresholds = np.array(thresholds[:K], dtype=np.float32)
    y_pred = (scores > thresholds).astype(np.int64)

    # ---- metrics -------------------------------------------------------------
    f1_sample = per_sample_f1(labels, y_pred)
    per_class = macro_f1_per_class(labels, y_pred, class_names)
    exact = exact_match_accuracy(labels, y_pred)

    print(f"\n=== {args.method} / split={args.split} ===")
    print(f"  per-sample F1 : {f1_sample:.4f}")
    print(f"  macro F1      : {per_class['macro']:.4f}")
    print(f"  exact match   : {exact:.4f}")
    for name in class_names:
        print(f"    F1({name}) = {per_class[name]:.4f}")

    summary = {
        "method": args.method,
        "split": args.split,
        "num_images": int(num_images),
        "class_names": class_names,
        "per_sample_f1": float(f1_sample),
        "macro_f1": float(per_class["macro"]),
        "exact_match": float(exact),
        "per_class_f1": {name: float(per_class[name]) for name in class_names},
        "thresholds": thresholds.tolist(),
        "score_name": score_name,
    }
    save_json(str(output_dir / "summary.json"), summary)
    np.save(str(output_dir / "scores.npy"), scores)
    np.save(str(output_dir / "predictions.npy"), y_pred)
    np.save(str(output_dir / "labels.npy"), labels)
    np.save(str(output_dir / f"{score_name}_aux.npy"), aux)
    print(f"\nSaved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
