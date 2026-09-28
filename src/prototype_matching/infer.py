#!/usr/bin/env python3
"""Run Method B inference at the image level.

For each input image:
  1. Apply illumination normalization (using config saved at train time).
  2. Tile into a deterministic eval_grid_size x eval_grid_size grid.
  3. DINOv2 forward each tile.
  4. PrototypeMatchingModel forward each tile -> per-tile (similarities, max_sim).
  5. Aggregate to image level via scatter_mean (Assumption H).
  6. Apply per-class presence thresholds and the unknown threshold.
  7. Write CSV + JSON with image-level results.
"""

import argparse
import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.features import (
    extract_features_multicrop,
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.illumination import make_illumination_preprocess
from src.common.io import load_json, save_json
from src.common.tiling import TileConfig
from src.prototype_matching.model import ProtoConfig, PrototypeMatchingModel


def find_images(root: str) -> List[str]:
    image_paths: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                image_paths.append(os.path.join(dirpath, fname))
    return sorted(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Method B image-level inference")
    parser.add_argument("--input", type=str, required=True, help="Image file or folder")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="outputs/prototype_matching/inference_results")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    model_path = model_dir / "proto_model.pt"
    if not model_path.exists() or not config_path.exists():
        raise SystemExit("Missing model artifacts. Expected proto_model.pt and config.json.")

    config = load_json(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- inputs ---------------------------------------------------------------------
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = find_images(args.input)
    if not image_paths:
        raise SystemExit("No images found for inference.")

    # ---- preprocessing matches training --------------------------------------------
    tile_cfg_dict = config["tile_config"]
    tile_cfg = TileConfig(**tile_cfg_dict)

    illum_method = config.get("illumination", "divide")
    illum_sigma = config.get("illum_sigma", 64.0)
    use_gpu_illum = device.type == "cuda" and illum_method != "none"

    # ---- extract tile features ------------------------------------------------------
    if use_gpu_illum:
        features, image_index = extract_features_multicrop_gpu(
            image_paths=image_paths,
            tile_config=tile_cfg,
            backbone=config["backbone"],
            frame_batch_size=args.frame_batch_size,
            num_workers=args.num_workers,
            device=device,
            illum_sigma=illum_sigma,
            illum_method=illum_method,
        )
    else:
        preprocess_fn = make_illumination_preprocess(sigma=illum_sigma, method=illum_method)
        features, image_index = extract_features_multicrop(
            image_paths=image_paths,
            tile_config=tile_cfg,
            backbone=config["backbone"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            mode="eval",
            preprocess_fn=preprocess_fn,
        )
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    # ---- model forward -------------------------------------------------------------
    proto_cfg = ProtoConfig(
        embedding_dim=config["embedding_dim"],
        num_prototypes=config["num_prototypes"],
        thresholds=config["thresholds"],
        unknown_threshold=config["unknown_threshold"],
    )
    model = PrototypeMatchingModel(proto_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        sims_tile, max_tile = model(features.to(device))
    sims_tile = sims_tile.cpu()
    max_tile = max_tile.cpu()

    # ---- aggregate tile -> image ----------------------------------------------------
    num_images = len(image_paths)
    sims_image = scatter_mean_by_image(sims_tile, image_index, num_images)
    max_image = scatter_mean_by_image(
        max_tile.unsqueeze(-1), image_index, num_images
    ).squeeze(-1)

    # ---- thresholds ----------------------------------------------------------------
    class_names = config.get("class_names")
    K = len(class_names) if class_names else config["num_prototypes"]
    presence_thresholds = config["thresholds"]
    presence_thresholds_tensor = torch.tensor(presence_thresholds[:K])
    unknown_threshold = float(config["unknown_threshold"])

    presence = (sims_image[:, :K] > presence_thresholds_tensor).numpy()
    unknown = (max_image < unknown_threshold).numpy()

    # ---- write outputs --------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sims_np = sims_image.numpy()
    max_np = max_image.numpy()
    for idx, path in enumerate(image_paths):
        row = {
            "image": path,
            "max_similarity": float(max_np[idx]),
            "unknown": bool(unknown[idx]),
        }
        if class_names:
            for k, name in enumerate(class_names):
                row[f"sim_{name}"] = float(sims_np[idx, k])
                row[f"present_{name}"] = bool(presence[idx, k])
        else:
            for k in range(K):
                row[f"sim_{k}"] = float(sims_np[idx, k])
                row[f"present_{k}"] = bool(presence[idx, k])
        rows.append(row)

    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_json(str(output_dir / "predictions.json"), rows)
    save_json(
        str(output_dir / "summary.json"),
        {
            "num_images": num_images,
            "tile_config": asdict(tile_cfg),
            "presence_thresholds": presence_thresholds[:K],
            "unknown_threshold": unknown_threshold,
            "class_names": class_names,
        },
    )
    print(f"Saved predictions for {num_images} image(s) to {csv_path}")


if __name__ == "__main__":
    main()
