#!/usr/bin/env python3
"""Is colour a species cue or a uniform offset?

The released images are RGB with a documented warm cast (the paper's appendix on the
Bayer colour camera). This takes our own pure-culture images, converts them to
grayscale and replicates to three channels, and scores them with the same frozen
RGB-trained model. Everything else about them is unchanged: same instrument, same
magnification, same preparation, same processing.

  - If the similarity to each species' own prototype drops by a near-constant amount
    while the margin over the best competing prototype does not fall, colour is a
    uniform shift that the thresholds were calibrated on, not a species cue.
  - If the drop is species-dependent, colour carries discriminative signal.

Everything runs on the CPU feature path, so the numbers are comparable among
themselves and NOT to the published GPU-path figures.

Usage:
    python experiments/run_grayscale_ablation.py --n_images 40 --species bs pf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import extract_features_multicrop, scatter_mean_by_image  # noqa: E402
from src.common.illumination import make_illumination_preprocess  # noqa: E402
from src.common.io import load_real_split  # noqa: E402
from src.common.tiling import TileConfig  # noqa: E402


def sims(paths: List[str], protos, tile_cfg, backbone, preprocess_fn, device, nw) -> np.ndarray:
    feats, idx = extract_features_multicrop(
        image_paths=paths, tile_config=tile_cfg, backbone=backbone, batch_size=64,
        num_workers=nw, device=device, cache_path=None, mode="eval",
        preprocess_fn=preprocess_fn,
    )
    feats = torch.nn.functional.normalize(feats, dim=1)
    return scatter_mean_by_image(feats @ protos.T, idx, len(paths)).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, default=ROOT / "outputs/prototype_matching/default")
    ap.add_argument("--splits_path", type=str, default="data/splits.json")
    ap.add_argument("--species", nargs="+", default=["bs"])
    ap.add_argument("--n_images", type=int, default=40,
                    help="images per species (not --n: conda run reads that as --name)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--output_dir", type=Path, default=ROOT / "outputs/grayscale_ablation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.model_dir / "config.json").read_text())
    protos = torch.load(args.model_dir / "proto_model.pt", map_location="cpu",
                        weights_only=False)["prototypes"].float()
    protos = torch.nn.functional.normalize(protos, dim=1)
    classes = cfg["class_names"]
    thr = np.asarray(cfg["thresholds"], float)
    tile_cfg = TileConfig(**cfg["tile_config"]) if isinstance(cfg.get("tile_config"), dict) else TileConfig()
    pre = make_illumination_preprocess(sigma=cfg["illum_sigma"], method=cfg["illumination"])

    paths, _, _, combos = load_real_split(args.splits_path, "test")
    gdir = args.output_dir / "_gray"
    gdir.mkdir(parents=True, exist_ok=True)
    out = {}

    for sp in args.species:
        i = classes.index(sp)
        rgb = [p for p, c in zip(paths, combos) if c == sp][: args.n_images]
        gray = []
        for p in rgb:
            q = gdir / f"{sp}_{Path(p).stem}.png"
            Image.open(p).convert("L").convert("RGB").save(q)   # grayscale, replicated to 3ch
            gray.append(str(q))

        s_rgb = sims(rgb, protos, tile_cfg, cfg["backbone"], pre, device, args.num_workers)
        s_gry = sims(gray, protos, tile_cfg, cfg["backbone"], pre, device, args.num_workers)
        a, b = s_rgb[:, i], s_gry[:, i]

        # Discriminability, which is the quantity that actually matters. A uniform drop
        # across every species leaves the margin intact and is recoverable by
        # recalibrating thresholds: that is a domain shift, colour carrying no species
        # information. A margin that shrinks means the drop is species-dependent and
        # colour is carrying discriminative signal, which recalibration cannot restore.
        other = [j for j in range(len(classes)) if j != i]
        marg_rgb = a - s_rgb[:, other].max(axis=1)
        marg_gry = b - s_gry[:, other].max(axis=1)

        out[sp] = {
            "threshold": float(thr[i]),
            "rgb":  {"mean": float(a.mean()), "min": float(a.min()), "pass_rate": float((a > thr[i]).mean())},
            "gray": {"mean": float(b.mean()), "min": float(b.min()), "pass_rate": float((b > thr[i]).mean())},
            "mean_drop": float(a.mean() - b.mean()),
            "mean_sim_all_prototypes": {
                "rgb": {c: float(v) for c, v in zip(classes, s_rgb.mean(0))},
                "gray": {c: float(v) for c, v in zip(classes, s_gry.mean(0))},
            },
            "margin_own_minus_best_other": {
                "rgb": float(marg_rgb.mean()), "gray": float(marg_gry.mean()),
                "change": float(marg_gry.mean() - marg_rgb.mean()),
            },
            "own_prototype_is_argmax": {
                "rgb": float((s_rgb.argmax(1) == i).mean()),
                "gray": float((s_gry.argmax(1) == i).mean()),
            },
        }
        print(f"\n{sp}: threshold {thr[i]:.4f}")
        print(f"  RGB       mean {a.mean():.4f}  min {a.min():.4f}  clears threshold {100*(a>thr[i]).mean():5.1f}%")
        print(f"  grayscale mean {b.mean():.4f}  min {b.min():.4f}  clears threshold {100*(b>thr[i]).mean():5.1f}%")
        print(f"  drop from removing colour: {a.mean()-b.mean():+.4f}")
        print(f"  margin (own minus best other): RGB {marg_rgb.mean():+.4f} -> gray {marg_gry.mean():+.4f} "
              f"({marg_gry.mean()-marg_rgb.mean():+.4f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(out, indent=2))

    if len(out) > 1:
        drops = np.array([v["mean_drop"] for v in out.values()])
        dmarg = np.array([v["margin_own_minus_best_other"]["change"] for v in out.values()])
        print("\n" + "=" * 78)
        print(f"{'species':<10}{'drop':>10}{'margin RGB':>13}{'margin gray':>13}{'margin d':>11}")
        print("-" * 78)
        for sp, v in out.items():
            m = v["margin_own_minus_best_other"]
            print(f"{sp:<10}{v['mean_drop']:>+10.4f}{m['rgb']:>+13.4f}{m['gray']:>+13.4f}{m['change']:>+11.4f}")
        print("=" * 78)
        print(f"\n  self-similarity drop: mean {drops.mean():+.4f}, sd {drops.std():.4f}, "
              f"range {drops.min():+.4f} to {drops.max():+.4f}")
        print(f"  margin change:        mean {dmarg.mean():+.4f}, sd {dmarg.std():.4f}")
        print("\n  Reading. A small sd on the drop, with the margin roughly unchanged, means a")
        print("  uniform domain shift: grayscale sits off the manifold DINOv2 and the prototypes")
        print("  occupy, colour carries no species information, and recalibrated thresholds")
        print("  recover it. A large sd, or a margin that collapses, means the drop is")
        print("  species-dependent and colour is carrying discriminative signal, which")
        print("  recalibration cannot restore and which would support the acquisition-cue concern.")
    print(f"\nWrote {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
