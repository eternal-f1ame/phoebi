#!/usr/bin/env python3
"""Tile-count sweep for Method C (channel-grouped head).

`experiments/run_ablations.py --ablation tile_count` sweeps T over Methods A
and B only, so `outputs/ablations/tile_count_method_c/results.csv` had no
producer and the figure fell back to values transcribed off an older render.
This driver fills that gap.

It reproduces the A/B sweep's protocol exactly: the same tile configuration,
the same seeded RNG consumed in the same order (val then test, T ascending),
so Method C is scored on the *same* tile subsets as Methods A and B, and the
same per-T recalibration on val -- by argmax-F1, which is Method C's own rule.

Method C is a 390-parameter head over frozen DINOv2 features, so no feature
extraction is needed when a matching cache exists:

    python experiments/run_tile_count_method_c.py \
        --cache_dir outputs/ablations_gpu/projection

Without --cache_dir the features are extracted first, which needs a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_ablations import subsample_tiles_by_image, summarize, _write_results
from src.common.features import scatter_mean_by_image
from src.common.io import load_real_split
from src.mc_channel.model import MCChannelHead, MCConfig
from src.mc_channel.train import calibrate_argmax_f1

TILE_COUNTS = [1, 4, 8, 16]


def _load_cache(path: Path, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Load a feature cache and refuse it unless every key field matches the
    trained model's configuration."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    want = {
        "backbone": cfg["backbone"],
        "illum_method": cfg["illumination"],
        "illum_sigma": float(cfg["illum_sigma"]),
    }
    for k, v in want.items():
        if d[k] != v:
            raise SystemExit(f"{path}: {k} is {d[k]!r}, model was trained with {v!r}")
    for k in ("tile_size", "eval_grid_size"):
        if d["tile_config"][k] != cfg["tile_config"][k]:
            raise SystemExit(
                f"{path}: tile_config[{k}] is {d['tile_config'][k]}, "
                f"model used {cfg['tile_config'][k]}")
    # src/mc_channel/test_eval.py L2-normalises tile features before the head.
    # Without this the stored thresholds score 0.577 instead of 0.674.
    return F.normalize(d["features"], p=2, dim=1), d["image_index"], d["paths"]


@torch.no_grad()
def _image_sigmoids(model: MCChannelHead, feats: torch.Tensor,
                    idx: torch.Tensor, n_images: int,
                    device: torch.device) -> np.ndarray:
    """Per-tile sigmoid, mean-aggregated to image level -- Method C's own
    inference path (src/mc_channel/test_eval.py)."""
    sig = torch.sigmoid(model(feats.to(device))).cpu()
    return scatter_mean_by_image(sig, idx, n_images).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/mc_channel/default")
    ap.add_argument("--cache_dir", default="",
                    help="directory holding {val,test}_features_cache.pt")
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--output_dir", default="outputs/ablations")
    ap.add_argument("--seed", type=int, default=0,
                    help="must match the A/B sweep to reuse its tile subsets")
    args = ap.parse_args()

    model_dir = ROOT / args.model_dir
    cfg = json.loads((model_dir / "config.json").read_text())

    if not args.cache_dir:
        raise SystemExit(
            "no --cache_dir: feature extraction needs a GPU. Point --cache_dir at a "
            "run directory whose cache matches DINOv2-S/14 @ 224 / divide / sigma=64.")
    cache_dir = ROOT / args.cache_dir

    val_feat, val_idx, val_cache_paths = _load_cache(
        cache_dir / "val_features_cache.pt", cfg)
    test_feat, test_idx, test_cache_paths = _load_cache(
        cache_dir / "test_features_cache.pt", cfg)

    val_paths, val_labels, class_names, _ = load_real_split(args.splits_path, "val")
    test_paths, test_labels, _, _ = load_real_split(args.splits_path, "test")
    for name, a, b in (("val", val_cache_paths, val_paths),
                       ("test", test_cache_paths, test_paths)):
        if list(a) != list(b):
            raise SystemExit(f"{name}: cached paths do not match {args.splits_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc_cfg = MCConfig(**cfg["mc_config"])
    model = MCChannelHead(mc_cfg).to(device)
    model.load_state_dict(torch.load(model_dir / "mc_model.pt",
                                     map_location=device)["state_dict"])
    model.eval()

    n_val, n_test = len(val_paths), len(test_paths)
    rng = np.random.default_rng(args.seed)
    rows = []
    for T in TILE_COUNTS:
        # Same call order as sweep_tile_count, so the subsets are identical.
        vf, vi = subsample_tiles_by_image(val_feat, val_idx, n_val, T, rng)
        tf, ti = subsample_tiles_by_image(test_feat, test_idx, n_test, T, rng)

        sig_val = _image_sigmoids(model, vf, vi, n_val, device)
        sig_test = _image_sigmoids(model, tf, ti, n_test, device)
        thr = calibrate_argmax_f1(sig_val, val_labels)
        res = summarize(test_labels, (sig_test > thr).astype(np.int64), class_names)

        rows.append({
            "T": T,
            "C_per_sample_f1": res["per_sample_f1"],
            "C_macro_f1": res["macro_f1"],
        })
        print(f"  T={T:2d}: C per-sample F1={res['per_sample_f1']:.4f}  "
              f"macro F1={res['macro_f1']:.4f}")

    out_dir = ROOT / args.output_dir / "tile_count_method_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_results(out_dir, rows)
    print(f"wrote {out_dir}/results.csv")


if __name__ == "__main__":
    main()
