#!/usr/bin/env python3
"""Stage 0: which unit granularity carries compositional signal?

Closed-form only, no training. For each tile size (224, 112, 56) one pass over the LCO
training split accumulates the paper's hybrid prototypes for every unit kind that size
supports (CLS, patch-mean, per-patch), then one pass each over val, the nested dev
combinations, and the nine held-out combinations scores every unit against the
prototypes with the two read-outs the paper uses (cosine = ProtoMatch, sparsemax =
SimplexUnmix at init) and three image-level aggregations (mean, 90th percentile, max).

Reported per (granularity, read-out, aggregation): per-species AUROC on dev and test
(threshold-free; the constant predictor scores 0.5) and the paper's thresholded metrics
with thresholds from val. The sanity anchor is cls224 / cosine / mean with prototypes
from the paper's full training split, which must reproduce ProtoMatch's seed-1337
held-out row (per-sample F1 0.6604, AUROC 0.606).

Side product: a fixed subsample of training units per granularity, written as an fp16
memmap under <output_dir>/../cache/, so Stage 1 does not repeat the training pass.

Selection for later stages uses dev only (see src/compositional/protocol.py).

Usage
    python experiments/run_comp_stage0_granularity.py --output_dir outputs/compositional/stage0
    python experiments/run_comp_stage0_granularity.py --max_images 2 --frame_batch_size 2 \
        --granularities cls224 patch224 --output_dir /tmp/x   # CPU smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compositional.evaluate import constant_predictor, evaluate_split, quantile_thresholds
from src.compositional.protocol import SplitData, load_protocol
from src.compositional.scoring import AGGREGATIONS, PrototypeAccumulator, ScoreStore, score_keys, score_units
from src.compositional.units import (GRANULARITIES, UnitExtractor, fixed_unit_subset,
                                     group_by_tile_size, iter_image_units)

DEFAULT_GRANULARITIES = ["cls224", "patchmean224", "patch224", "cls112", "patchmean112", "cls56"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_scores(path: Path, store: ScoreStore, split: SplitData) -> None:
    np.savez_compressed(path, residual=store.residual, labels=split.labels,
                        combos=np.array(split.combos), paths=np.array(split.paths),
                        **{k.replace("/", "__"): v for k, v in store.arrays.items()})


def load_scores(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k.replace("__", "/"): z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits_lco.json")
    ap.add_argument("--output_dir", default="outputs/compositional/stage0")
    ap.add_argument("--cache_dir", default=None, help="default: <output_dir>/../cache")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--granularities", nargs="+", default=DEFAULT_GRANULARITIES)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--cache_units_per_image", type=int, default=64,
                    help="training units kept per image for Stage 1 (all if fewer)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_images", type=int, default=None, help="per-combination cap (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"device={device} output={out_dir} cache={cache_dir}")

    P = load_protocol(args.splits_path, seed=args.seed, max_images=args.max_images)
    log(P.describe())
    K = len(P.class_names)
    keys = score_keys(AGGREGATIONS)
    dev_set = set(P.dev_combos)
    dev_mask_full = np.array([c in dev_set for c in P.train_full.combos])

    results: Dict = {"protocol": {"class_names": P.class_names, "heldout": P.heldout_combos,
                                  "dev": P.dev_combos, "n": {"train": len(P.train), "val": len(P.val),
                                  "dev": len(P.dev), "test": len(P.test), "train_full": len(P.train_full)}},
                     "config": vars(args), "granularities": {},
                     "constant": {s: constant_predictor(getattr(P, s).labels, P.class_names)
                                  for s in ("val", "dev", "test")}}

    for (tile_size, grid), grans in group_by_tile_size(args.granularities).items():
        kinds = [g.kind for g in grans]
        gdirs = {g.name: out_dir / g.name for g in grans}
        for d in gdirs.values():
            d.mkdir(exist_ok=True)
        done = all((gdirs[g.name] / "scores_test.npz").exists() for g in grans)
        if done:
            log(f"tile {tile_size}: scores present for {kinds}, skipping extraction")
        else:
            ext = UnitExtractor(args.backbone, tile_size, grid, kinds, device)
            D = ext.dim

            # ---- pass 1: prototypes over the paper's training split (+ unit cache) ----
            acc_full = {g.name: PrototypeAccumulator(P.class_names, D, device) for g in grans}
            acc_xdev = {g.name: PrototypeAccumulator(P.class_names, D, device) for g in grans}
            cache: Dict[str, np.memmap] = {}
            subsets: Dict[str, torch.Tensor | None] = {}
            n_seen = 0; t0 = time.time()
            log(f"tile {tile_size}: pass 1 (prototypes + cache) over {len(P.train_full)} images, kinds={kinds}")
            for idxs, units in iter_image_units(ext, P.train_full.paths, args.frame_batch_size, args.num_workers):
                i = idxs.numpy()
                labels = P.train_full.labels[i]; combos = [P.train_full.combos[j] for j in i]
                keep = ~dev_mask_full[i]
                for g in grans:
                    u = units[g.kind]
                    acc_full[g.name].add(u, labels, combos)
                    if keep.any():
                        km = torch.from_numpy(keep).to(u.device)
                        acc_xdev[g.name].add(u[km], labels[keep], [c for c, k in zip(combos, keep) if k])
                    if g.name not in cache:
                        n_units = u.shape[1]
                        subsets[g.name] = fixed_unit_subset(n_units, args.cache_units_per_image, args.seed)
                        U = n_units if subsets[g.name] is None else int(subsets[g.name].numel())
                        cache[g.name] = np.lib.format.open_memmap(
                            cache_dir / f"{g.name}_train_units.npy", mode="w+", dtype=np.float16,
                            shape=(len(P.train_full), U, D))
                    sel = subsets[g.name]
                    uu = u if sel is None else u[:, sel.to(u.device)]
                    cache[g.name][i] = uu.half().cpu().numpy()
                n_seen += len(i)
                if n_seen % (args.frame_batch_size * 250) == 0:
                    log(f"  pass 1: {n_seen}/{len(P.train_full)} images, {n_seen / (time.time() - t0):.1f} img/s")
            for g in grans:
                cache[g.name].flush()
                meta = {"granularity": g.name, "tile_size": tile_size, "grid": grid, "dim": D,
                        "units_per_image": int(cache[g.name].shape[1]),
                        "subset_positions": None if subsets[g.name] is None else subsets[g.name].tolist(),
                        "paths": P.train_full.paths, "labels": P.train_full.labels.tolist(),
                        "combos": P.train_full.combos, "dev_combos": P.dev_combos}
                (cache_dir / f"{g.name}_train_meta.json").write_text(json.dumps(meta))
            protos = {}
            for g in grans:
                pf, px = acc_full[g.name].finalize(), acc_xdev[g.name].finalize()
                protos[g.name] = {"full": pf["prototypes"], "xdev": px["prototypes"]}
                torch.save({"full": pf, "xdev": px}, gdirs[g.name] / "prototypes.pt")
                log(f"  {g.name}: prototype sources full={pf['source']} xdev={px['source']}")

            # ---- pass 2: score val / dev / test ----
            for split_name in ("val", "dev", "test"):
                split: SplitData = getattr(P, split_name)
                stores = {g.name: {"xdev": ScoreStore(len(split), K, keys)} for g in grans}
                if split_name != "dev":
                    for g in grans:
                        stores[g.name]["full"] = ScoreStore(len(split), K, keys)
                n_seen = 0; t0 = time.time()
                log(f"tile {tile_size}: scoring {split_name} ({len(split)} images)")
                for idxs, units in iter_image_units(ext, split.paths, args.frame_batch_size, args.num_workers):
                    for g in grans:
                        for which, store in stores[g.name].items():
                            store.put(idxs, score_units(units[g.kind], protos[g.name][which], args.temperature))
                    n_seen += len(idxs)
                    if n_seen % (args.frame_batch_size * 250) == 0:
                        log(f"  {split_name}: {n_seen}/{len(split)}, {n_seen / (time.time() - t0):.1f} img/s")
                for g in grans:
                    for which, store in stores[g.name].items():
                        assert store.complete(), f"{g.name}/{split_name}/{which} incomplete"
                        suffix = "" if which == "xdev" else "_fullprotos"
                        save_scores(gdirs[g.name] / f"scores_{split_name}{suffix}.npz", store, split)
            del ext
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- metrics from saved scores ----
        for g in grans:
            gd = gdirs[g.name]
            S = {s: load_scores(gd / f"scores_{s}.npz") for s in ("val", "dev", "test")}
            Sfull = {s: load_scores(gd / f"scores_{s}_fullprotos.npz") for s in ("val", "test")}
            entry: Dict = {"keys": {}, "anchor": {}}
            for key in keys:
                thr_val = quantile_thresholds(S["val"][key], P.val.labels, 0.05)
                thr_dev = quantile_thresholds(S["dev"][key], P.dev.labels, 0.05)
                entry["keys"][key] = {
                    "thresholds_val": thr_val.tolist(),
                    "val": evaluate_split(S["val"][key], P.val.labels, P.class_names, thr_val),
                    "dev": evaluate_split(S["dev"][key], P.dev.labels, P.class_names, thr_val, P.dev.combos),
                    "test": evaluate_split(S["test"][key], P.test.labels, P.class_names, thr_val, P.test.combos),
                    "test_devthr": evaluate_split(S["test"][key], P.test.labels, P.class_names, thr_dev),
                }
            # anchor: prototypes from the paper's full training split, val-calibrated
            for key in ("cosine/mean", "sparsemax/mean"):
                thr = quantile_thresholds(Sfull["val"][key], P.val.labels, 0.05)
                entry["anchor"][key] = evaluate_split(Sfull["test"][key], P.test.labels, P.class_names, thr)
            results["granularities"][g.name] = entry
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))
            a = entry["anchor"]["cosine/mean"]
            log(f"{g.name}: anchor cosine/mean test F1={a['per_sample_f1']:.4f} AUROC={a['auroc']['mean']:.3f}")
            for key in keys:
                e = entry["keys"][key]
                log(f"  {key:16s} dev AUROC={e['dev']['auroc']['mean']:.3f}  test AUROC={e['test']['auroc']['mean']:.3f}"
                    f"  test F1={e['test']['per_sample_f1']:.4f}  macro={e['test']['macro_f1']['macro']:.4f}"
                    f"  pred/img={e['test']['mean_predicted']:.2f}")

    # ---- selection on dev, summary ----
    rows = []
    for gname, entry in results["granularities"].items():
        for key, e in entry["keys"].items():
            rows.append((gname, key, e["dev"]["auroc"]["mean"], e["test"]["auroc"]["mean"],
                         e["test"]["per_sample_f1"], e["test"]["macro_f1"]["macro"], e["test"]["exact_match"],
                         e["test"]["mean_predicted"], e["test_devthr"]["per_sample_f1"]))
    rows.sort(key=lambda r: -r[2])
    best = {"granularity": rows[0][0], "key": rows[0][1], "readout": rows[0][1].split("/")[0],
            "aggregation": rows[0][1].split("/")[1], "dev_auroc": rows[0][2], "selected_on": "dev AUROC mean"}
    best_per_gran = {}
    for r in rows:
        best_per_gran.setdefault(r[0], {"key": r[1], "dev_auroc": r[2]})
    best["per_granularity"] = best_per_gran
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))

    c = results["constant"]
    md = ["# Stage 0: unit granularity diagnostic (closed form, LCO seed 1337)\n",
          f"Nested dev combinations: `{', '.join(P.dev_combos)}`. Held-out test: `{', '.join(P.heldout_combos)}`.\n",
          f"Constant all-present predictor: test F1 {c['test']['per_sample_f1']:.4f} / macro {c['test']['macro_f1']:.4f} / "
          f"exact {c['test']['exact_match']:.4f}; dev F1 {c['dev']['per_sample_f1']:.4f}; AUROC 0.500 everywhere.\n",
          "Thresholds from val (5th percentile of positives) unless the column says dev.\n",
          "| granularity | read-out/agg | dev AUROC | test AUROC | test F1 | test macro F1 | test exact | pred/img | test F1 (dev thr) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.2f} | {r[8]:.4f} |")
    md.append("\n## Anchor (prototypes from the paper's full training split, val thresholds)\n")
    md.append("| granularity | read-out | test F1 | test macro F1 | test AUROC |")
    md.append("|---|---|---|---|---|")
    for gname, entry in results["granularities"].items():
        for key, a in entry["anchor"].items():
            md.append(f"| {gname} | {key} | {a['per_sample_f1']:.4f} | {a['macro_f1']['macro']:.4f} | {a['auroc']['mean']:.3f} |")
    md.append(f"\nSelected on dev AUROC: **{best['granularity']} / {best['key']}** (dev AUROC {best['dev_auroc']:.3f}).\n")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out_dir / 'results.json'}, summary.md, best.json -> {best['granularity']} / {best['key']}")


if __name__ == "__main__":
    main()
