#!/usr/bin/env python3
"""Nested five-fold LCO cross-validation on the Stage 0 unit cache (CPU, no extraction).

Applies experiments/LCO_DEV_PROTOCOL.md to the cached training units
(outputs/compositional/cache/<granularity>_train_units.npy: 64 fp16 units per image of
the paper's LCO training split). Two parts:

  closed    every cached granularity, closed-form prototypes per fold (pure-culture
            means where the fold's pool has them, contaminated means otherwise), read-outs
            cosine / sparsemax, aggregations mean / q90 / max
  decoders  the paper's three decoders at their own unit (cls224): SimplexUnmix (init
            + 30 epochs MSE), ProtoMatch (closed form), ChannelGroup (BCE head, 30 epochs),
            one training per fold

Deviations from the protocol forced by the cache, stated in the outputs: dev images are
the dev combinations' train-split images (val-split images are not cached), and
thresholds are calibrated on the fold's training-pool images (same reason). AUROC, the
primary criterion, is unaffected by both.

Outputs <output_dir>/{results.json,summary.md}: pooled metrics over the 31 held-out
combinations with combination-level 95 % intervals.

Usage
    python experiments/run_comp_nested_cv.py --part closed
    python experiments/run_comp_nested_cv.py --part decoders
    python experiments/run_comp_nested_cv.py --part closed --granularities cls224 --max_images_per_combo 40   # smoke
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compositional.evaluate import argmax_f1_thresholds, quantile_thresholds
from src.compositional.intervals import format_report, heldout_report
from src.compositional.scoring import AGGREGATIONS, score_keys, score_units
from src.mc_channel.model import MCChannelHead, MCConfig
from src.simplex_unmixing.model import ModelConfig, UnmixerModel


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cache(cache_dir: Path, gname: str, max_per_combo: int | None):
    meta = json.loads((cache_dir / f"{gname}_train_meta.json").read_text())
    mm = np.load(cache_dir / f"{gname}_train_units.npy", mmap_mode="r")
    combos = np.array(meta["combos"]); labels = np.array(meta["labels"], dtype=np.int64)
    keep = np.ones(len(combos), dtype=bool)
    if max_per_combo:
        seen: Dict[str, int] = {}
        for i, c in enumerate(combos):
            seen[c] = seen.get(c, 0) + 1; keep[i] = seen[c] <= max_per_combo
    return mm, combos, labels, keep


def image_sums(mm, idx: np.ndarray, chunk: int = 2048):
    """Sum of units per image (N, D) float64, streamed from the memmap."""
    out = np.zeros((len(idx), mm.shape[2]), dtype=np.float64)
    for i in range(0, len(idx), chunk):
        ii = idx[i:i + chunk]
        out[i:i + len(ii)] = np.asarray(mm[ii], dtype=np.float32).sum(axis=1)
    return out


def fold_prototypes(sums: np.ndarray, n_units: int, labels: np.ndarray, combos: np.ndarray,
                    class_names: List[str]) -> torch.Tensor:
    K = len(class_names); protos = np.zeros((K, sums.shape[1]))
    for k, name in enumerate(class_names):
        pure = combos == name
        if pure.any():
            protos[k] = sums[pure].sum(0) / (pure.sum() * n_units)
        else:
            has = labels[:, k] == 1
            protos[k] = sums[has].sum(0) / (has.sum() * n_units)
    return F.normalize(torch.from_numpy(protos).float(), p=2, dim=1)


def score_images(mm, idx: np.ndarray, protos: torch.Tensor, keys: List[str], temperature: float,
                 chunk: int = 512) -> Dict[str, np.ndarray]:
    K = protos.shape[0]; out = {k: np.zeros((len(idx), K), dtype=np.float32) for k in keys}
    for i in range(0, len(idx), chunk):
        ii = idx[i:i + chunk]
        u = torch.from_numpy(np.asarray(mm[ii], dtype=np.float32))
        s = score_units(u, protos, temperature)
        for k in keys:
            out[k][i:i + len(ii)] = s[k].numpy()
    return out


def run_closed(args, folds, class_names, cache_dir: Path, out: Dict) -> None:
    keys = score_keys(AGGREGATIONS)
    for gname in args.granularities:
        if not (cache_dir / f"{gname}_train_units.npy").exists():
            log(f"{gname}: no cache, skipping"); continue
        mm, combos, labels, keep = load_cache(cache_dir, gname, args.max_images_per_combo)
        n_units = mm.shape[1]
        log(f"{gname}: cache {mm.shape}, {keep.sum()} images in use")
        pooled = {k: [] for k in keys}; pooled_thr = {k: [] for k in keys}; pooled_lab = []; pooled_combo = []
        for f in folds:
            dev_set = set(f["dev_combos"])
            tr = np.where(keep & ~np.isin(combos, list(dev_set)))[0]; dv = np.where(keep & np.isin(combos, list(dev_set)))[0]
            sums = image_sums(mm, tr)
            protos = fold_prototypes(sums, n_units, labels[tr], combos[tr], class_names)
            s_tr = score_images(mm, tr, protos, keys, args.temperature)   # calibration on the training pool
            s_dv = score_images(mm, dv, protos, keys, args.temperature)
            for k in keys:
                thr = quantile_thresholds(s_tr[k], labels[tr], 0.05)
                pooled[k].append(s_dv[k]); pooled_thr[k].append(np.broadcast_to(thr, s_dv[k].shape))
            pooled_lab.append(labels[dv]); pooled_combo.append(combos[dv])
            log(f"  fold {f['fold']}: train {len(tr)} dev {len(dv)} images")
        Y = np.concatenate(pooled_lab); C = np.concatenate(pooled_combo)
        out["closed"][gname] = {}
        for k in keys:
            S = np.concatenate(pooled[k]); T = np.concatenate(pooled_thr[k])
            # thresholds differ per fold: apply per image via the broadcast copies
            P = S > T
            rep = heldout_report(S, Y, C, thresholds=np.zeros(S.shape[1]) - np.inf, class_names=class_names,
                                 n_boot=args.n_boot, seed=0)      # thresholds handled below
            # recompute thresholded quantities with the per-fold thresholds
            from src.compositional.intervals import macro_f1, per_sample_f1
            rng = np.random.default_rng(0); units = sorted(set(C)); idx = {g: np.where(C == g)[0] for g in units}
            ones = np.ones_like(Y); f1s = []; mars = []; macs = []; mmars = []
            for _ in range(args.n_boot):
                ii = np.concatenate([idx[g] for g in rng.choice(units, size=len(units), replace=True)])
                f1s.append(per_sample_f1(Y[ii], P[ii])); mars.append(per_sample_f1(Y[ii], P[ii]) - per_sample_f1(Y[ii], ones[ii]))
                macs.append(macro_f1(Y[ii], P[ii])); mmars.append(macro_f1(Y[ii], P[ii]) - macro_f1(Y[ii], ones[ii]))
            ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
            rep["f1"] = {"point": per_sample_f1(Y, P), "ci95": ci(f1s)}
            rep["f1_margin"] = {"point": per_sample_f1(Y, P) - per_sample_f1(Y, ones), "ci95": ci(mars)}
            rep["macro"] = {"point": macro_f1(Y, P), "ci95": ci(macs)}
            rep["macro_margin"] = {"point": macro_f1(Y, P) - macro_f1(Y, ones), "ci95": ci(mmars)}
            rep["mean_predicted"] = float(P.sum(1).mean())
            out["closed"][gname][k] = rep
            log(f"  {gname} {k:16s} AUROC {rep['auroc_mean']['point']:.3f} {rep['auroc_mean']['ci95']}  macro margin {rep['macro_margin']['point']:+.3f}")
        del mm


def train_fold_decoders(units_tr: torch.Tensor, labels_tr: torch.Tensor, protos: torch.Tensor, epochs: int,
                        device: torch.device, seed: int) -> Dict[str, object]:
    N, U, D = units_tr.shape; K = protos.shape[0]
    flat = units_tr.reshape(N * U, D); ylab = labels_tr.repeat_interleave(U, dim=0).float()
    g = torch.Generator().manual_seed(seed)
    A = UnmixerModel(ModelConfig(embedding_dim=D, num_prototypes=K, temperature=10.0)).to(device)
    with torch.no_grad():
        A.prototypes.copy_(protos.to(device))
    optA = torch.optim.Adam(A.parameters(), lr=1e-3)
    C = MCChannelHead(MCConfig(embedding_dim=D, num_classes=K, cra_drop_prob=0.5)).to(device)
    optC = torch.optim.AdamW(C.parameters(), lr=1e-2, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
    n = flat.shape[0]; bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            bi = perm[i:i + bs]; z = flat[bi].to(device).float(); y = ylab[bi].to(device)
            optA.zero_grad(); _, _, r = A(z); ((r ** 2).sum(1).mean()).backward(); optA.step()
            C.train(); optC.zero_grad(); bce(C(z), y).backward(); optC.step()
    C.eval()
    return {"A": A, "C": C}


@torch.no_grad()
def decoder_scores(mm, idx: np.ndarray, models: Dict, protos: torch.Tensor, device, chunk: int = 512):
    A, C = models["A"], models["C"]; K = protos.shape[0]
    out = {m: np.zeros((len(idx), K), dtype=np.float32) for m in ("A_simplex", "B_proto", "C_channel")}
    for i in range(0, len(idx), chunk):
        ii = idx[i:i + chunk]
        u = torch.from_numpy(np.asarray(mm[ii], dtype=np.float32)).to(device)   # (b, U, D)
        b, U, D = u.shape; flat = u.reshape(b * U, D)
        _, w, _ = A(flat); out["A_simplex"][i:i + b] = w.reshape(b, U, K).mean(1).cpu().numpy()
        out["B_proto"][i:i + b] = (flat @ F.normalize(protos.to(device), dim=1).t()).reshape(b, U, K).mean(1).cpu().numpy()
        out["C_channel"][i:i + b] = torch.sigmoid(C(flat)).reshape(b, U, K).mean(1).cpu().numpy()
    return out


def run_decoders(args, folds, class_names, cache_dir: Path, out: Dict) -> None:
    gname = "cls224"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm, combos, labels, keep = load_cache(cache_dir, gname, args.max_images_per_combo)
    n_units = mm.shape[1]; log(f"decoders on {gname}: cache {mm.shape}, {keep.sum()} images, device {device}")
    pooled = {m: [] for m in ("A_simplex", "B_proto", "C_channel")}; pooled_P = {m: [] for m in pooled}
    pooled_lab = []; pooled_combo = []
    for f in folds:
        dev_set = set(f["dev_combos"])
        tr = np.where(keep & ~np.isin(combos, list(dev_set)))[0]; dv = np.where(keep & np.isin(combos, list(dev_set)))[0]
        protos = fold_prototypes(image_sums(mm, tr), n_units, labels[tr], combos[tr], class_names)
        units_tr = torch.from_numpy(np.ascontiguousarray(mm[tr]))            # fp16 CPU
        t0 = time.time()
        models = train_fold_decoders(units_tr, torch.from_numpy(labels[tr]), protos, args.epochs, device, args.seed)
        log(f"  fold {f['fold']}: trained A and C on {len(tr)} images in {time.time() - t0:.0f}s")
        s_tr = decoder_scores(mm, tr, models, protos, device); s_dv = decoder_scores(mm, dv, models, protos, device)
        for m in pooled:
            thr = argmax_f1_thresholds(s_tr[m], labels[tr]) if m == "C_channel" else quantile_thresholds(s_tr[m], labels[tr], 0.05)
            pooled[m].append(s_dv[m]); pooled_P[m].append(s_dv[m] > thr)
        pooled_lab.append(labels[dv]); pooled_combo.append(combos[dv])
    Y = np.concatenate(pooled_lab); C = np.concatenate(pooled_combo)
    from src.compositional.intervals import macro_f1, per_sample_f1
    out["decoders"] = {}
    for m in pooled:
        S = np.concatenate(pooled[m]); P = np.concatenate(pooled_P[m])
        rep = heldout_report(S, Y, C, thresholds=np.zeros(S.shape[1]) - np.inf, class_names=class_names, n_boot=args.n_boot, seed=0)
        rng = np.random.default_rng(0); units = sorted(set(C)); idx = {g: np.where(C == g)[0] for g in units}; ones = np.ones_like(Y)
        f1s, mars, macs, mmars = [], [], [], []
        for _ in range(args.n_boot):
            ii = np.concatenate([idx[g] for g in rng.choice(units, size=len(units), replace=True)])
            f1s.append(per_sample_f1(Y[ii], P[ii])); mars.append(per_sample_f1(Y[ii], P[ii]) - per_sample_f1(Y[ii], ones[ii]))
            macs.append(macro_f1(Y[ii], P[ii])); mmars.append(macro_f1(Y[ii], P[ii]) - macro_f1(Y[ii], ones[ii]))
        ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
        rep["f1"] = {"point": per_sample_f1(Y, P), "ci95": ci(f1s)}; rep["f1_margin"] = {"point": per_sample_f1(Y, P) - per_sample_f1(Y, ones), "ci95": ci(mars)}
        rep["macro"] = {"point": macro_f1(Y, P), "ci95": ci(macs)}; rep["macro_margin"] = {"point": macro_f1(Y, P) - macro_f1(Y, ones), "ci95": ci(mmars)}
        rep["mean_predicted"] = float(P.sum(1).mean())
        out["decoders"][m] = rep
        log(f"  {m}: AUROC {rep['auroc_mean']['point']:.3f} {rep['auroc_mean']['ci95']}  F1 {rep['f1']['point']:.3f} margin {rep['f1_margin']['point']:+.3f} {rep['f1_margin']['ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds_path", default="data/lco_dev_folds.json")
    ap.add_argument("--cache_dir", default="outputs/compositional/cache")
    ap.add_argument("--output_dir", default="outputs/compositional/nested_cv")
    ap.add_argument("--part", choices=["closed", "decoders", "both"], default="both")
    ap.add_argument("--granularities", nargs="+", default=["cls224", "patchmean224", "patch224", "cls112", "patchmean112", "cls56"])
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--max_images_per_combo", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    folds_doc = json.loads(Path(args.folds_path).read_text()); folds = folds_doc["folds"]; class_names = folds_doc["class_names"]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True); cache_dir = Path(args.cache_dir)
    res_path = out_dir / "results.json"
    out = json.loads(res_path.read_text()) if res_path.exists() else {"closed": {}, "decoders": {}, "config": vars(args), "folds": folds_doc}
    out["config"] = vars(args)
    if args.part in ("closed", "both"):
        run_closed(args, folds, class_names, cache_dir, out); res_path.write_text(json.dumps(out, indent=1))
    if args.part in ("decoders", "both"):
        run_decoders(args, folds, class_names, cache_dir, out); res_path.write_text(json.dumps(out, indent=1))
    n_units = next((r["n_units"] for d in out["closed"].values() for r in d.values()), None) \
        or next((r["n_units"] for r in out["decoders"].values()), "?")
    md = [f"# Nested {len(folds)}-fold LCO cross-validation (seed-{folds_doc.get('partition_seed', '?')} partition, {n_units} held-out combinations pooled)\n",
          "Dev images are the dev combinations' train-split images; thresholds are calibrated on the fold's training-pool images (cache limitation, AUROC unaffected). Intervals: 95 % bootstrap over combinations.\n"]
    if out["closed"]:
        md += ["## Closed-form prototypes by granularity\n", "| granularity | read-out/agg | mean AUROC [95 %] | macro-F1 margin over constant [95 %] | F1 | pred/img |", "|---|---|---|---|---|---|"]
        rows = [(g, k, r) for g, d in out["closed"].items() for k, r in d.items()]
        rows.sort(key=lambda t: -t[2]["auroc_mean"]["point"])
        for g, k, r in rows:
            md.append(f"| {g} | {k} | {r['auroc_mean']['point']:.3f} [{r['auroc_mean']['ci95'][0]:.3f}, {r['auroc_mean']['ci95'][1]:.3f}] | "
                      f"{r['macro_margin']['point']:+.3f} [{r['macro_margin']['ci95'][0]:+.3f}, {r['macro_margin']['ci95'][1]:+.3f}] | {r['f1']['point']:.3f} | {r['mean_predicted']:.2f} |")
    if out["decoders"]:
        md += ["\n## The paper's decoders at cls224 (one training per fold)\n"]
        for m, r in out["decoders"].items():
            md += [f"### {m}\n", "```", format_report(r, class_names), "```"]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {res_path} and summary.md")


if __name__ == "__main__":
    main()
