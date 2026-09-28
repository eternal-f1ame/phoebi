"""Method C test-split evaluation. Loads a trained run from --model_dir,
extracts test tile features (cached to model_dir/test_features_cache.pt),
forwards through the head, mean-aggregates to image level, applies the
val-calibrated thresholds, writes test_summary.json next to the model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common.features import extract_features_multicrop_gpu, scatter_mean_by_image
from src.common.io import load_real_split, save_json
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import TileConfig
from src.mc_channel.model import MCChannelHead, MCConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    config = json.load(open(model_dir / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_paths, test_labels, class_names, _ = load_real_split(args.splits_path, "test")
    K = len(class_names)
    tc = config["tile_config"]
    tile_cfg = TileConfig(
        tile_size=tc["tile_size"],
        train_tiles_per_image=tc["train_tiles_per_image"],
        eval_grid_size=tc["eval_grid_size"],
    )

    test_features, test_image_index = extract_features_multicrop_gpu(
        image_paths=test_paths, tile_config=tile_cfg, backbone=config["backbone"],
        frame_batch_size=args.frame_batch_size, num_workers=args.num_workers,
        device=device, illum_sigma=config["illum_sigma"], illum_method=config["illumination"],
        cache_path=str(model_dir / "test_features_cache.pt"),
    )
    test_features = F.normalize(test_features, p=2, dim=1)

    cfg = MCConfig(**config["mc_config"])
    model = MCChannelHead(cfg).to(device)
    state = torch.load(model_dir / "mc_model.pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    thresholds = np.array(config["presence_thresholds"], dtype=np.float64)

    with torch.no_grad():
        logits_tile = model(test_features.to(device))
        sig_tile = torch.sigmoid(logits_tile).cpu()
        sig_image = scatter_mean_by_image(sig_tile, test_image_index, len(test_paths)).numpy()
    pred = (sig_image > thresholds).astype(np.int64)
    f1 = float(per_sample_f1(test_labels, pred))
    macro = macro_f1_per_class(test_labels, pred, class_names)
    em = float(exact_match_accuracy(test_labels, pred))

    out = {
        "test_per_sample_f1": f1,
        "test_macro_f1": macro,
        "test_exact_match": em,
        "n_test_imgs": len(test_paths),
    }
    save_json(str(model_dir / "test_summary.json"), out)
    print(f"  test per-sample F1: {f1:.4f}  macro: {macro['macro']:.4f}  EM: {em:.4f}")
    print(f"  Wrote {model_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
