#!/usr/bin/env python3
"""Test-time augmentation invariance check on Method B (cosine matcher).

Applies 6 image transforms at inference time (identity, rot90, rot180, rot270,
hflip, vflip), re-extracts test tile features for each transform via the GPU
illumination path, runs Method B's frozen prototype matrix, and reports
per-sample F1. The point is to verify that the PHOEBI pipeline is approximately
invariant to rotation + reflection of the input frame, as the Spatial
Homogeneity Assumption (H) implies it should be.

We use Method B because (a) it is closed-form so the test isolates pipeline
invariance from training-time noise, and (b) Methods A and C inherit the same
front-end so the same conclusion applies.

Outputs: outputs/ablations/invariance/results.json with per-transform F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, vflip, rotate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import (
    extract_features_multicrop_gpu, scatter_mean_by_image,
)
from src.common.io import load_real_split, save_json
from src.common.metrics import per_sample_f1, macro_f1_per_class
from src.common.tiling import TileConfig
from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig


TRANSFORMS = {
    "identity": lambda x: x,
    "rot90":    lambda x: torch.rot90(x, k=1, dims=(-2, -1)),
    "rot180":   lambda x: torch.rot90(x, k=2, dims=(-2, -1)),
    "rot270":   lambda x: torch.rot90(x, k=3, dims=(-2, -1)),
    "hflip":    lambda x: torch.flip(x, dims=(-1,)),
    "vflip":    lambda x: torch.flip(x, dims=(-2,)),
}


def evaluate_with_transform(
    paths, labels, class_names, model_path, config_path, transform_name,
    transform_fn, device, tile_cfg, cache_path,
):
    """Re-extract test features with a transform applied to each frame, then
    score with Method B's frozen prototype matrix."""
    config = json.load(open(config_path))
    proto_state = torch.load(model_path, map_location=device, weights_only=False)
    prototypes = proto_state["prototypes"]

    # Patch GPU extraction to apply transform after illumination, before tiling
    # by monkey-patching the cache and re-running. Cleanest way: extract fresh
    # with transform applied.
    print(f"  extracting features for transform={transform_name}...")
    features, image_index = extract_features_multicrop_gpu(
        image_paths=paths,
        tile_config=tile_cfg,
        backbone=config["backbone"],
        frame_batch_size=8,
        num_workers=2,
        device=device,
        illum_sigma=config["illum_sigma"],
        illum_method=config["illumination"],
        cache_path=cache_path,
        frame_transform=transform_fn,
    )
    features = F.normalize(features, p=2, dim=1)

    cfg = ProtoConfig(embedding_dim=features.shape[1], num_prototypes=prototypes.shape[0])
    model = PrototypeMatchingModel(cfg).to(device)
    model.prototypes.data.copy_(prototypes.to(device))
    model.eval()
    with torch.no_grad():
        sims, max_sim = model(features.to(device))
    sims_image = scatter_mean_by_image(sims.cpu(), image_index, len(paths)).numpy()

    # Apply Method B's val-calibrated thresholds
    thresholds = np.array(config["thresholds"])
    pred = (sims_image > thresholds).astype(np.int64)
    f1 = float(per_sample_f1(labels, pred))
    macro = macro_f1_per_class(labels, pred, class_names)
    return f1, macro, sims_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--method_b_dir", default="outputs/prototype_matching/6class_v2")
    ap.add_argument("--output_dir", default="outputs/ablations/invariance")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    print(f"Device: {device}")

    paths, labels, class_names, _ = load_real_split(args.splits_path, "test")
    config = json.load(open(f"{args.method_b_dir}/config.json"))
    tile_cfg = TileConfig(**config["tile_config"])
    K = len(class_names)
    print(f"Test: {len(paths)} images, K={K} classes")

    results = {}
    score_arrays = {}
    for tname, tfn in TRANSFORMS.items():
        cache_path = str(cache_dir / f"test_features_{tname}.pt")
        f1, macro, scores = evaluate_with_transform(
            paths, labels, class_names,
            model_path=f"{args.method_b_dir}/proto_model.pt",
            config_path=f"{args.method_b_dir}/config.json",
            transform_name=tname, transform_fn=tfn,
            device=device, tile_cfg=tile_cfg, cache_path=cache_path,
        )
        results[tname] = {
            "per_sample_f1": f1,
            "macro_f1": {k: float(v) for k, v in macro.items()},
        }
        score_arrays[tname] = scores
        print(f"  {tname:>10s}: F1={f1:.4f}  macro={macro['macro']:.4f}")

    # TTA: average scores across all 6 transforms, recompute F1
    tta_scores = np.mean(np.stack(list(score_arrays.values()), axis=0), axis=0)
    thresholds = np.array(config["thresholds"])
    tta_pred = (tta_scores > thresholds).astype(np.int64)
    tta_f1 = float(per_sample_f1(labels, tta_pred))
    tta_macro = macro_f1_per_class(labels, tta_pred, class_names)
    results["TTA_mean_of_6"] = {
        "per_sample_f1": tta_f1,
        "macro_f1": {k: float(v) for k, v in tta_macro.items()},
    }
    print(f"  {'TTA-6':>10s}: F1={tta_f1:.4f}  macro={tta_macro['macro']:.4f}")

    # Pairwise consistency (mean abs diff in per-image scores between transforms)
    consistency = {}
    names = list(score_arrays.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = float(np.abs(score_arrays[names[i]] - score_arrays[names[j]]).mean())
            consistency[f"{names[i]}_vs_{names[j]}"] = diff

    summary = {
        "test_per_transform": results,
        "pairwise_score_diff_l1": consistency,
        "n_test_images": len(paths),
    }
    save_json(str(out_dir / "results.json"), summary)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
