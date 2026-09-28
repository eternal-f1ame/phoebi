#!/usr/bin/env python3
"""Mosaic experiment: compositional coverage from real pure-culture sub-crops.

Under Assumption H a 2x2 mosaic of 112 px sub-crops from illumination-corrected
pure-culture frames is an approximation of a mixture tile. A mosaic *image* is sixteen
such tiles whose quadrants are drawn from the species of a target combination, every
species of the combination appearing in at least one quadrant, so its label is the
combination. Mosaics for every combination of the species that have pure cultures in
the training pool are added to training only; dev and test stay real cultures.

This separates two causes of the failure: if training on all combinations (mosaic)
lifts held-out AUROC, the problem was unseen combinations; if it does not, the unit
embedding is not additive and coverage cannot fix it.

Three training pools at the granularity Stage 1 selected, decoder = Stage 1's best
variant (and Stage 2's best subspace when present):

    real          real training units only (Stage 1 / Stage 2 as run)
    real+mosaic   real training units plus mosaic units
    mosaic        mosaic units only (pure cultures never see a real mixture)

Usage
    python experiments/run_comp_mosaic.py --output_dir outputs/compositional/mosaic
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_comp_stage1_patch_simplex import load_training_units, log, save_scores, train_prototypes
from experiments.run_comp_stage2_mixing_subspace import project_units, train_subspace
from src.common.illumination import gpu_normalize_illumination
from src.compositional.evaluate import constant_predictor, evaluate_split, quantile_thresholds
from src.compositional.protocol import SplitData, load_protocol
from src.compositional.scoring import AGGREGATIONS, ScoreStore, score_keys, score_units
from src.compositional.units import GRANULARITIES, UnitExtractor, fixed_unit_subset, iter_image_units

MOSAIC_TILE = 224
QUAD = 112


def load_pure_pool(P, species: List[str], per_species: int, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    """Illumination-corrected pure-culture frames per species, uint8 (n, 3, H, W) on CPU."""
    rng = random.Random(seed)
    pool = {}
    for k in species:
        paths = [p for p, c in zip(P.train.paths, P.train.combos) if c == k]
        rng.shuffle(paths); paths = paths[:per_species]
        frames = []
        for i in range(0, len(paths), 8):
            batch = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)
                                 for p in paths[i:i + 8]]).to(device)
            corrected = gpu_normalize_illumination(batch, sigma=64.0, method="divide", downsample=8)
            frames.append(corrected.round().clamp(0, 255).to(torch.uint8).cpu())
        pool[k] = torch.cat(frames, dim=0)
        log(f"  pure pool {k}: {pool[k].shape[0]} corrected frames")
    return pool


def make_mosaic_images(pool: Dict[str, torch.Tensor], combo: Tuple[str, ...], n_images: int,
                       rng: random.Random) -> torch.Tensor:
    """(n_images, 16, 3, 224, 224) uint8 mosaic tiles for one combination."""
    out = torch.empty((n_images, 16, 3, MOSAIC_TILE, MOSAIC_TILE), dtype=torch.uint8)
    for i in range(n_images):
        species = [rng.choice(combo) for _ in range(64)]
        for k in combo:                          # coverage: every species in at least one quadrant
            if k not in species:
                species[rng.randrange(64)] = k
        for t in range(16):
            for q in range(4):
                k = species[t * 4 + q]
                fr = pool[k][rng.randrange(pool[k].shape[0])]
                H, W = fr.shape[-2:]
                y, x = rng.randrange(0, H - QUAD + 1), rng.randrange(0, W - QUAD + 1)
                oy, ox = (q // 2) * QUAD, (q % 2) * QUAD
                out[i, t, :, oy:oy + QUAD, ox:ox + QUAD] = fr[:, y:y + QUAD, x:x + QUAD]
    return out


def build_mosaic_cache(P, gname: str, cache_dir: Path, backbone: str, device: torch.device,
                       n_per_combo: int, pool_per_species: int, units_per_image: int, seed: int) -> Path:
    g = GRANULARITIES[gname]
    meta_path = cache_dir / f"{gname}_mosaic_meta.json"; units_path = cache_dir / f"{gname}_mosaic_units.npy"
    if meta_path.exists() and units_path.exists():
        log(f"mosaic cache for {gname} present"); return meta_path
    species = sorted(k for k in P.class_names if k in set(P.train.combos))
    combos = [c for r in range(2, len(species) + 1) for c in itertools.combinations(species, r)]
    log(f"mosaic: pure species {species}; {len(combos)} combinations of order >= 2, {n_per_combo} images each")
    pool = load_pure_pool(P, species, pool_per_species, device, seed)
    # mosaic tiles are already corrected: extractor runs with illumination off, grid = 224 / tile
    ext = UnitExtractor(backbone, g.tile_size, MOSAIC_TILE // g.tile_size, [g.kind], device, illum_method="none")
    rng = random.Random(seed)
    N = len(combos) * n_per_combo
    labels = np.zeros((N, len(P.class_names)), dtype=np.int64); names = []
    mm = None; sel = None; pos = 0; t0 = time.time()
    for combo in combos:
        for i0 in range(0, n_per_combo, 4):
            nb = min(4, n_per_combo - i0)
            tiles = make_mosaic_images(pool, combo, nb, rng).to(device)          # (nb, 16, 3, 224, 224)
            u = ext.units_from_frames(tiles.reshape(nb * 16, 3, MOSAIC_TILE, MOSAIC_TILE))   # (nb*16, T'*U, D)
            u = u[g.kind].reshape(nb, -1, ext.dim)
            if mm is None:
                sel = fixed_unit_subset(u.shape[1], units_per_image, seed)
                U = u.shape[1] if sel is None else int(sel.numel())
                mm = np.lib.format.open_memmap(units_path, mode="w+", dtype=np.float16, shape=(N, U, ext.dim))
            uu = u if sel is None else u[:, sel.to(u.device)]
            mm[pos:pos + nb] = uu.half().cpu().numpy()
            for _ in range(nb):
                for k in combo:
                    labels[pos, P.class_names.index(k)] = 1
                names.append("_".join(combo)); pos += 1
        log(f"  mosaic {'_'.join(combo)}: done ({pos}/{N}, {pos / (time.time() - t0):.1f} img/s)")
    mm.flush()
    meta_path.write_text(json.dumps({"granularity": gname, "n": N, "labels": labels.tolist(), "combos": names,
                                     "species": species, "n_per_combo": n_per_combo}))
    return meta_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits_lco.json")
    ap.add_argument("--output_dir", default="outputs/compositional/mosaic")
    ap.add_argument("--stage0_dir", default=None); ap.add_argument("--stage1_dir", default=None); ap.add_argument("--stage2_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--granularity", default=None, help="default: Stage 1 selection")
    ap.add_argument("--n_per_combo", type=int, default=300)
    ap.add_argument("--pool_per_species", type=int, default=200)
    ap.add_argument("--cache_units_per_image", type=int, default=64)
    ap.add_argument("--pools", nargs="+", default=["real+mosaic", "mosaic"])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16384); ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--absent_lambda", type=float, default=1.0)
    ap.add_argument("--frame_batch_size", type=int, default=8); ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337); ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.parent
    stage0 = Path(args.stage0_dir) if args.stage0_dir else root / "stage0"
    stage1 = Path(args.stage1_dir) if args.stage1_dir else root / "stage1"
    stage2 = Path(args.stage2_dir) if args.stage2_dir else root / "stage2"
    cache_dir = Path(args.cache_dir) if args.cache_dir else root / "cache"
    best1 = json.loads((stage1 / "best.json").read_text())
    gname = args.granularity or best1["granularity"]; g = GRANULARITIES[gname]
    variant = best1["variant"]
    best2 = json.loads((stage2 / "best.json").read_text()) if (stage2 / "best.json").exists() else None
    log(f"device={device} granularity={gname} variant={variant} stage2={best2['tag'] if best2 else None}")

    P = load_protocol(args.splits_path, seed=args.seed, max_images=args.max_images)
    log(P.describe())
    K = len(P.class_names); keys = score_keys(AGGREGATIONS)
    if args.max_images:
        args.n_per_combo = min(args.n_per_combo, 4); args.pool_per_species = min(args.pool_per_species, 4)

    meta_path = build_mosaic_cache(P, gname, cache_dir, args.backbone, device, args.n_per_combo,
                                   args.pool_per_species, args.cache_units_per_image, args.seed)
    mmeta = json.loads(meta_path.read_text())
    mos_units = torch.from_numpy(np.ascontiguousarray(np.load(cache_dir / f"{gname}_mosaic_units.npy", mmap_mode="r")))
    mos_labels = torch.from_numpy(np.array(mmeta["labels"], dtype=np.int64))
    real_units, real_labels, _ = load_training_units(cache_dir, gname, P)
    if real_units.shape[1] != mos_units.shape[1]:
        m = min(real_units.shape[1], mos_units.shape[1]); real_units, mos_units = real_units[:, :m], mos_units[:, :m]
    init = torch.load(stage0 / gname / "prototypes.pt")["xdev"]["prototypes"]

    pools = {"real+mosaic": (torch.cat([real_units, mos_units]), torch.cat([real_labels, mos_labels])),
             "mosaic": (mos_units, mos_labels)}
    trained: Dict[str, Dict] = {}
    for pool_name in args.pools:
        u, y = pools[pool_name]
        tag = f"simplex_{variant}/{pool_name}"; rd = out_dir / gname / tag; rd.mkdir(parents=True, exist_ok=True)
        if (rd / "model.pt").exists():
            trained[tag] = torch.load(rd / "model.pt")
        else:
            t = train_prototypes(u, y, init, variant, args.epochs, args.lr, args.batch_size, args.temperature,
                                 args.absent_lambda, device, args.seed)
            torch.save(t, rd / "model.pt"); trained[tag] = t
        if best2 is not None:
            tag2 = f"subspace_{best2['tag']}/{pool_name}"; rd2 = out_dir / gname / tag2; rd2.mkdir(parents=True, exist_ok=True)
            if (rd2 / "model.pt").exists():
                trained[tag2] = torch.load(rd2 / "model.pt")
            else:
                d = int(best2["tag"].split("_")[0][1:]); v2 = best2["tag"].split("_", 1)[1]
                t2 = train_subspace(u, y, init, d, v2, args.epochs, args.lr, args.batch_size, args.temperature,
                                    args.absent_lambda, device, args.seed)
                torch.save(t2, rd2 / "model.pt"); trained[tag2] = t2
    del pools, real_units, mos_units

    ext = UnitExtractor(args.backbone, g.tile_size, g.grid, [g.kind], device)
    for split_name in ("val", "dev", "test"):
        split: SplitData = getattr(P, split_name)
        if all((out_dir / gname / t / f"scores_{split_name}.npz").exists() for t in trained):
            continue
        stores = {t: ScoreStore(len(split), K, keys) for t in trained}
        log(f"scoring {split_name} ({len(split)} images) for {list(trained)}")
        for idxs, u in iter_image_units(ext, split.paths, args.frame_batch_size, args.num_workers):
            for t, m in trained.items():
                uu = project_units(u[g.kind], m["W"]) if "W" in m else u[g.kind]
                stores[t].put(idxs, score_units(uu, m["prototypes"], args.temperature))
        for t in trained:
            assert stores[t].complete()
            save_scores(out_dir / gname / t / f"scores_{split_name}.npz", stores[t], split)
    del ext

    results: Dict = {"config": vars(args), "granularity": gname, "variant": variant, "mosaic": {k: v for k, v in mmeta.items() if k != "labels"},
                     "runs": {}, "constant": {s: constant_predictor(getattr(P, s).labels, P.class_names) for s in ("val", "dev", "test")}}
    # the real-only reference rows come from Stage 1 / Stage 2 results
    ref_rows = []
    for stage, best in (("stage1", best1), ("stage2", best2)):
        if best is None:
            continue
        R = json.loads((root / stage / "results.json").read_text())["runs"][best["run"]]
        for key, e in R["keys"].items():
            ref_rows.append((f"{stage} {best['run']} (real)", key, e["dev"]["auroc"]["mean"], e["test"]["auroc"]["mean"],
                             e["test"]["per_sample_f1"], e["test"]["macro_f1"]["macro"], e["test"]["mean_predicted"]))
    rows = []
    for t in trained:
        rd = out_dir / gname / t; S = {}
        for s in ("val", "dev", "test"):
            z = np.load(rd / f"scores_{s}.npz"); S[s] = {k.replace("__", "/"): z[k] for k in z.files}
        entry = {"keys": {}}
        for key in keys:
            thr_val = quantile_thresholds(S["val"][key], P.val.labels, 0.05)
            entry["keys"][key] = {
                "dev": evaluate_split(S["dev"][key], P.dev.labels, P.class_names, thr_val, P.dev.combos),
                "test": evaluate_split(S["test"][key], P.test.labels, P.class_names, thr_val, P.test.combos),
            }
            e = entry["keys"][key]
            rows.append((t, key, e["dev"]["auroc"]["mean"], e["test"]["auroc"]["mean"], e["test"]["per_sample_f1"],
                         e["test"]["macro_f1"]["macro"], e["test"]["mean_predicted"]))
            log(f"{t} {key:16s} dev AUROC={e['dev']['auroc']['mean']:.3f} test AUROC={e['test']['auroc']['mean']:.3f} test F1={e['test']['per_sample_f1']:.4f}")
        results["runs"][t] = entry
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    c = results["constant"]["test"]
    md = ["# Mosaic experiment: compositional coverage from pure-culture sub-crops (LCO seed 1337)\n",
          f"Granularity `{gname}`, decoder variant `{variant}`; mosaics cover {len(set(mmeta['combos']))} combinations of {mmeta['species']} "
          f"({mmeta['n_per_combo']} images each), training only. Constant predictor test F1 {c['per_sample_f1']:.4f}, AUROC 0.500.\n",
          "| training pool / run | read-out/agg | dev AUROC | test AUROC | test F1 | test macro F1 | pred/img |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.2f} |" for r in sorted(ref_rows + rows, key=lambda r: (r[0], r[1]))]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out_dir / 'results.json'} and summary.md")


if __name__ == "__main__":
    main()
