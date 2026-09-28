"""Method C inference: per-image presence prediction with the trained MC head.

Loads `mc_model.pt` + `config.json` from a Method C training run. Tiles each
input image with the same TileConfig + illumination, forwards through the
linear head (no CRA at eval), aggregates per-tile sigmoids to image-level by
mean, applies the val-calibrated per-class thresholds.

Writes outputs/<run>/infer/predictions.{csv,json}.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.common.features import extract_features_multicrop_gpu, scatter_mean_by_image
from src.common.tiling import TileConfig
from src.mc_channel.model import MCChannelHead, MCConfig


def find_images(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    return sorted(str(q) for q in p.rglob("*.jpg"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    config = json.load(open(model_dir / "config.json"))
    out_dir = Path(args.output_dir) if args.output_dir else model_dir / "infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model from config
    cfg = MCConfig(**config["mc_config"])
    model = MCChannelHead(cfg).to(device)
    state = torch.load(model_dir / "mc_model.pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    tile_cfg = TileConfig(**config["tile_config"])
    presence_thresholds = np.array(config["presence_thresholds"])
    class_names = config["class_names"]
    K = len(class_names)

    paths = find_images(args.input)
    print(f"Found {len(paths)} images under {args.input}")

    feats, image_index = extract_features_multicrop_gpu(
        image_paths=paths, tile_config=tile_cfg, backbone=config["backbone"],
        frame_batch_size=args.frame_batch_size, num_workers=args.num_workers,
        device=device, illum_sigma=config["illum_sigma"],
        illum_method=config["illumination"],
        cache_path=str(out_dir / "infer_features_cache.pt"),
    )
    feats = F.normalize(feats, p=2, dim=1)
    with torch.no_grad():
        sig_tile = torch.sigmoid(model(feats.to(device))).cpu()
    sig_image = scatter_mean_by_image(sig_tile, image_index, len(paths)).numpy()
    pred = (sig_image > presence_thresholds).astype(np.int64)

    # Write outputs
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"] + [f"score_{c}" for c in class_names] + [f"present_{c}" for c in class_names])
        for i, p in enumerate(paths):
            writer.writerow([p] + [f"{s:.4f}" for s in sig_image[i]] + [int(b) for b in pred[i]])

    payload = {
        "model_dir": str(model_dir),
        "input": args.input,
        "n_images": len(paths),
        "class_names": class_names,
        "presence_thresholds": presence_thresholds.tolist(),
        "predictions": [
            {"image": p,
             "scores": dict(zip(class_names, sig_image[i].tolist())),
             "present": dict(zip(class_names, [bool(x) for x in pred[i]]))}
            for i, p in enumerate(paths)
        ],
    }
    json_path = out_dir / "predictions.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
