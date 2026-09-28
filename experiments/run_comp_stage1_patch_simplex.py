#!/usr/bin/env python3
"""Stage 1: simplex unmixing at the unit granularity Stage 0 selected.

The paper's Method A, unchanged in its equations, applied to a different unit: the
prototypes are trained with the reconstruction objective on the cached training units
(Stage 0's fixed subsample per image), then every unit of val / dev / test streams
through the trained prototypes and is aggregated to the image by mean, 90th percentile
and max. Two training variants:

    mse      the paper's objective, E ||z - z_hat||^2
    absent   mse + lambda * mean_k-not-in-label w_k, i.e. mass that sparsemax puts on
             species the image does not contain is penalised. Under Assumption H a unit
             cannot contain an absent species, so this is supervision the paper always
             had available and never used; it is per-species, not per-combination.

Reported per (variant, read-out, aggregation): dev and test AUROC, thresholded metrics
with val thresholds (the paper) and with dev thresholds (Stage 3's question). Read-outs
are sparsemax (the trained decoder) and cosine to the trained prototypes.

Usage
    python experiments/run_comp_stage1_patch_simplex.py --output_dir outputs/compositional/stage1
    ... --granularity cls224 patch224        # override the Stage 0 selection
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compositional.evaluate import constant_predictor, evaluate_split, quantile_thresholds
from src.compositional.protocol import SplitData, load_protocol
from src.compositional.scoring import AGGREGATIONS, ScoreStore, score_keys, score_units
from src.compositional.units import GRANULARITIES, UnitExtractor, iter_image_units
from src.simplex_unmixing.model import ModelConfig, UnmixerModel


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_scores(path: Path, store: ScoreStore, split: SplitData) -> None:
    np.savez_compressed(path, residual=store.residual, labels=split.labels,
                        combos=np.array(split.combos), paths=np.array(split.paths),
                        **{k.replace("/", "__"): v for k, v in store.arrays.items()})


def load_training_units(cache_dir: Path, gname: str, P, exclude_dev: bool = True):
    meta = json.loads((cache_dir / f"{gname}_train_meta.json").read_text())
    if meta["paths"] != P.train_full.paths:
        raise SystemExit(f"cache for {gname} was built on a different training split; rerun Stage 0")
    mm = np.load(cache_dir / f"{gname}_train_units.npy", mmap_mode="r")
    combos = meta["combos"]; labels = np.array(meta["labels"], dtype=np.int64)
    keep = np.array([c not in set(meta["dev_combos"]) for c in combos]) if exclude_dev else np.ones(len(combos), bool)
    idx = np.where(keep)[0]
    units = torch.from_numpy(np.ascontiguousarray(mm[idx]))        # (N, U, D) fp16, CPU
    return units, torch.from_numpy(labels[idx]), meta


def train_prototypes(units: torch.Tensor, image_labels: torch.Tensor, init_protos: torch.Tensor,
                     variant: str, epochs: int, lr: float, batch_size: int, temperature: float,
                     absent_lambda: float, device: torch.device, seed: int) -> Dict:
    N, U, D = units.shape; K = init_protos.shape[0]
    flat = units.reshape(N * U, D)                                   # fp16 CPU view
    unit_labels = image_labels.repeat_interleave(U, dim=0)           # (N*U, K) int64 CPU
    model = UnmixerModel(ModelConfig(embedding_dim=D, num_prototypes=K, temperature=temperature)).to(device)
    with torch.no_grad():
        model.prototypes.copy_(init_protos.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = flat.shape[0]; history = []
    log(f"  training variant={variant} on {n} units ({N} images x {U}), D={D}, {epochs} epochs")
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        run_mse = run_abs = 0.0
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            z = flat[bi].to(device, non_blocking=True).float()
            y = unit_labels[bi].to(device, non_blocking=True).float()
            opt.zero_grad()
            _, w, residual = model(z)
            mse = (residual ** 2).sum(dim=1).mean()
            absent = (w * (1.0 - y)).sum(dim=1).mean()
            loss = mse + (absent_lambda * absent if variant == "absent" else 0.0)
            loss.backward(); opt.step()
            run_mse += mse.item() * z.shape[0]; run_abs += absent.item() * z.shape[0]
        history.append({"epoch": ep + 1, "mse": run_mse / n, "absent_mass": run_abs / n})
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"    ep{ep + 1:02d}/{epochs}  recon MSE={run_mse / n:.5f}  absent mass={run_abs / n:.4f}")
    protos = F.normalize(model.prototypes.detach(), p=2, dim=1).cpu()
    return {"prototypes": protos, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits_lco.json")
    ap.add_argument("--output_dir", default="outputs/compositional/stage1")
    ap.add_argument("--stage0_dir", default=None, help="default: <output_dir>/../stage0")
    ap.add_argument("--cache_dir", default=None, help="default: <output_dir>/../cache")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--granularity", nargs="*", default=None, help="default: Stage 0 selection")
    ap.add_argument("--variants", nargs="+", default=["mse", "absent"])
    ap.add_argument("--absent_lambda", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stage0 = Path(args.stage0_dir) if args.stage0_dir else out_dir.parent / "stage0"
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir.parent / "cache"
    best0 = json.loads((stage0 / "best.json").read_text())
    grans = args.granularity or [best0["granularity"]]
    log(f"device={device} granularities={grans} (Stage 0 selected {best0['granularity']} / {best0['key']})")

    P = load_protocol(args.splits_path, seed=args.seed, max_images=args.max_images)
    log(P.describe())
    K = len(P.class_names); keys = score_keys(AGGREGATIONS)
    results: Dict = {"config": vars(args), "stage0_best": best0, "runs": {},
                     "constant": {s: constant_predictor(getattr(P, s).labels, P.class_names) for s in ("val", "dev", "test")}}

    for gname in grans:
        g = GRANULARITIES[gname]
        units, labels, meta = load_training_units(cache_dir, gname, P)
        init = torch.load(stage0 / gname / "prototypes.pt")["xdev"]["prototypes"]
        trained: Dict[str, Dict] = {}
        for variant in args.variants:
            tag = f"{gname}/{variant}"
            rd = out_dir / gname / variant; rd.mkdir(parents=True, exist_ok=True)
            if (rd / "prototypes.pt").exists():
                trained[variant] = torch.load(rd / "prototypes.pt"); log(f"{tag}: prototypes present, skipping training")
                continue
            t = train_prototypes(units, labels, init, variant, args.epochs, args.lr, args.batch_size,
                                 args.temperature, args.absent_lambda, device, args.seed)
            torch.save(t, rd / "prototypes.pt"); trained[variant] = t
        del units

        # one streaming pass per split scores every variant
        ext = UnitExtractor(args.backbone, g.tile_size, g.grid, [g.kind], device)
        for split_name in ("val", "dev", "test"):
            split: SplitData = getattr(P, split_name)
            if all((out_dir / gname / v / f"scores_{split_name}.npz").exists() for v in args.variants):
                log(f"{gname}: {split_name} scores present, skipping"); continue
            stores = {v: ScoreStore(len(split), K, keys) for v in args.variants}
            n_seen = 0; t0 = time.time()
            log(f"{gname}: scoring {split_name} ({len(split)} images) for {args.variants}")
            for idxs, u in iter_image_units(ext, split.paths, args.frame_batch_size, args.num_workers):
                for v in args.variants:
                    stores[v].put(idxs, score_units(u[g.kind], trained[v]["prototypes"], args.temperature))
                n_seen += len(idxs)
                if n_seen % (args.frame_batch_size * 250) == 0:
                    log(f"  {split_name}: {n_seen}/{len(split)}, {n_seen / (time.time() - t0):.1f} img/s")
            for v in args.variants:
                assert stores[v].complete()
                save_scores(out_dir / gname / v / f"scores_{split_name}.npz", stores[v], split)
        del ext
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for v in args.variants:
            rd = out_dir / gname / v
            S = {}
            for s in ("val", "dev", "test"):
                z = np.load(rd / f"scores_{s}.npz"); S[s] = {k.replace("__", "/"): z[k] for k in z.files}
            entry = {"history": trained[v].get("history"), "keys": {}}
            for key in keys:
                thr_val = quantile_thresholds(S["val"][key], P.val.labels, 0.05)
                thr_dev = quantile_thresholds(S["dev"][key], P.dev.labels, 0.05)
                entry["keys"][key] = {
                    "val": evaluate_split(S["val"][key], P.val.labels, P.class_names, thr_val),
                    "dev": evaluate_split(S["dev"][key], P.dev.labels, P.class_names, thr_val, P.dev.combos),
                    "test": evaluate_split(S["test"][key], P.test.labels, P.class_names, thr_val, P.test.combos),
                    "test_devthr": evaluate_split(S["test"][key], P.test.labels, P.class_names, thr_dev),
                }
                e = entry["keys"][key]
                log(f"{gname}/{v} {key:16s} dev AUROC={e['dev']['auroc']['mean']:.3f} test AUROC={e['test']['auroc']['mean']:.3f}"
                    f" test F1={e['test']['per_sample_f1']:.4f} macro={e['test']['macro_f1']['macro']:.4f}"
                    f" pred/img={e['test']['mean_predicted']:.2f} | dev-thr F1={e['test_devthr']['per_sample_f1']:.4f}")
            results["runs"][f"{gname}/{v}"] = entry
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    rows = [(run, key, e["dev"]["auroc"]["mean"], e["test"]["auroc"]["mean"], e["test"]["per_sample_f1"],
             e["test"]["macro_f1"]["macro"], e["test"]["exact_match"], e["test"]["mean_predicted"],
             e["test_devthr"]["per_sample_f1"])
            for run, entry in results["runs"].items() for key, e in entry["keys"].items()]
    rows.sort(key=lambda r: -r[2])
    best = {"run": rows[0][0], "granularity": rows[0][0].split("/")[0], "variant": rows[0][0].split("/")[1],
            "key": rows[0][1], "dev_auroc": rows[0][2], "selected_on": "dev AUROC mean"}
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))
    c = results["constant"]["test"]
    md = ["# Stage 1: simplex unmixing at the selected granularity (LCO seed 1337)\n",
          f"Stage 0 selection: `{best0['granularity']} / {best0['key']}`. Constant predictor test F1 {c['per_sample_f1']:.4f}, "
          f"macro {c['macro_f1']:.4f}, AUROC 0.500.\n",
          "| run | read-out/agg | dev AUROC | test AUROC | test F1 | test macro F1 | test exact | pred/img | test F1 (dev thr) |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.2f} | {r[8]:.4f} |" for r in rows]
    md.append(f"\nSelected on dev AUROC: **{best['run']} / {best['key']}** ({best['dev_auroc']:.3f}).\n")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out_dir / 'results.json'}, summary.md, best.json -> {best['run']} / {best['key']}")


if __name__ == "__main__":
    main()
