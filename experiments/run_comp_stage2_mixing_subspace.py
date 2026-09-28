#!/usr/bin/env python3
"""Stage 2: learn the subspace in which the linear mixing model holds.

The simplex proposition needs the unit embedding to be a convex mixture of species
prototypes. Stage 2 learns one orthonormal projection W (D x d), shared by every species,
under exactly that constraint: a projected unit must be reconstructed by sparsemax
weights over the projected prototypes. W has no per-combination surface to overfit, which
is the property the paper argues for; it is the learned counterpart of the channel
selection ChannelGroup does by hand.

    z' = normalise(W^T z)          Q in R^{K x d} learned prototypes (init: projected hybrid means)
    w  = sparsemax(tau * Q z')     z'_hat = Q^T w      loss = ||z' - z'_hat||^2 (+ absent-mass penalty)

d in {32, 64, 128} and the two loss variants of Stage 1 are swept; selection is on dev
AUROC. W is initialised to the top-d principal directions of the training units and kept
orthonormal by torch's orthogonal parametrisation.

Usage
    python experiments/run_comp_stage2_mixing_subspace.py --output_dir outputs/compositional/stage2
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

from experiments.run_comp_stage1_patch_simplex import load_training_units, log, save_scores
from src.compositional.evaluate import constant_predictor, evaluate_split, quantile_thresholds
from src.compositional.protocol import SplitData, load_protocol
from src.compositional.scoring import AGGREGATIONS, ScoreStore, score_keys, score_units
from src.compositional.units import GRANULARITIES, UnitExtractor, iter_image_units
from src.simplex_unmixing.model import sparsemax


class SubspaceUnmixer(nn.Module):
    """Orthonormal projection W followed by simplex unmixing against learned prototypes Q."""

    def __init__(self, D: int, d: int, K: int, temperature: float, w_init: torch.Tensor, q_init: torch.Tensor) -> None:
        super().__init__()
        from torch.nn.utils.parametrizations import orthogonal
        # Build on the initialiser's device *before* registering the parametrisation:
        # assigning through `orthogonal` runs a right-inverse that multiplies by the
        # module's own base tensor, so a CPU module and a CUDA initialiser cannot meet.
        self.proj = nn.Linear(D, d, bias=False).to(w_init.device)
        orthogonal(self.proj)
        with torch.no_grad():
            self.proj.weight = w_init.contiguous()   # (d, D) orthonormal rows; right-inverse exists
        self.penalised = False
        self.prototypes = nn.Parameter(q_init.clone().to(w_init.device))  # (K, d)
        self.temperature = temperature

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), p=2, dim=-1)

    def forward(self, z: torch.Tensor):
        zp = self.project(z)
        q = F.normalize(self.prototypes, p=2, dim=1)
        w = sparsemax(zp @ q.t() * self.temperature, dim=1)
        recon = w @ q
        return w, zp - recon

    def orth_penalty(self) -> torch.Tensor:
        if not self.penalised:
            return torch.zeros((), device=self.prototypes.device)
        Wt = self.proj.weight
        eye = torch.eye(Wt.shape[0], device=Wt.device)
        return ((Wt @ Wt.t() - eye) ** 2).sum()

    def projection_matrix(self) -> torch.Tensor:
        return self.proj.weight.detach()             # (d, D)


def pca_init(units_flat: torch.Tensor, d: int, n_sample: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(units_flat.shape[0], generator=g)[:n_sample]
    x = units_flat[idx].float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(x, full_matrices=False)
    return vt[:d].contiguous()                        # (d, D) orthonormal rows


def train_subspace(units: torch.Tensor, image_labels: torch.Tensor, init_protos: torch.Tensor, d: int,
                   variant: str, epochs: int, lr: float, batch_size: int, temperature: float,
                   absent_lambda: float, device: torch.device, seed: int) -> Dict:
    N, U, D = units.shape; K = init_protos.shape[0]
    flat = units.reshape(N * U, D); unit_labels = image_labels.repeat_interleave(U, dim=0)
    w0 = pca_init(flat, d, n_sample=min(200_000, flat.shape[0]), seed=seed).to(device)
    q0 = F.normalize(init_protos.to(device).float() @ w0.t(), p=2, dim=1)
    model = SubspaceUnmixer(D, d, K, temperature, w0, q0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = flat.shape[0]; history = []
    log(f"  training d={d} variant={variant} on {n} units, {epochs} epochs")
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g); run_mse = run_abs = 0.0
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            z = flat[bi].to(device, non_blocking=True).float()
            y = unit_labels[bi].to(device, non_blocking=True).float()
            opt.zero_grad()
            w, residual = model(z)
            mse = (residual ** 2).sum(dim=1).mean()
            absent = (w * (1.0 - y)).sum(dim=1).mean()
            loss = mse + (absent_lambda * absent if variant == "absent" else 0.0) + 10.0 * model.orth_penalty()
            loss.backward(); opt.step()
            run_mse += mse.item() * z.shape[0]; run_abs += absent.item() * z.shape[0]
        history.append({"epoch": ep + 1, "mse": run_mse / n, "absent_mass": run_abs / n})
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"    ep{ep + 1:02d}/{epochs}  recon MSE={run_mse / n:.5f}  absent mass={run_abs / n:.4f}")
    return {"W": model.projection_matrix().cpu(), "prototypes": F.normalize(model.prototypes.detach(), p=2, dim=1).cpu(),
            "history": history, "d": d, "variant": variant}


@torch.no_grad()
def project_units(u: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return F.normalize(u @ W.t().to(u.device, u.dtype), p=2, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits_lco.json")
    ap.add_argument("--output_dir", default="outputs/compositional/stage2")
    ap.add_argument("--stage0_dir", default=None); ap.add_argument("--stage1_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--granularity", default=None, help="default: Stage 1 selection")
    ap.add_argument("--dims", nargs="+", type=int, default=[32, 64, 128])
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
    stage1 = Path(args.stage1_dir) if args.stage1_dir else out_dir.parent / "stage1"
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir.parent / "cache"
    best1 = json.loads((stage1 / "best.json").read_text())
    gname = args.granularity or best1["granularity"]
    g = GRANULARITIES[gname]
    log(f"device={device} granularity={gname} (Stage 1 selected {best1['run']} / {best1['key']})")

    P = load_protocol(args.splits_path, seed=args.seed, max_images=args.max_images)
    log(P.describe())
    K = len(P.class_names); keys = score_keys(AGGREGATIONS)
    results: Dict = {"config": vars(args), "stage1_best": best1, "runs": {},
                     "constant": {s: constant_predictor(getattr(P, s).labels, P.class_names) for s in ("val", "dev", "test")}}

    units, labels, meta = load_training_units(cache_dir, gname, P)
    init = torch.load(stage0 / gname / "prototypes.pt")["xdev"]["prototypes"]
    configs = [(d, v) for d in args.dims for v in args.variants]
    trained: Dict[str, Dict] = {}
    for d, v in configs:
        tag = f"d{d}_{v}"; rd = out_dir / gname / tag; rd.mkdir(parents=True, exist_ok=True)
        if (rd / "model.pt").exists():
            trained[tag] = torch.load(rd / "model.pt"); log(f"{tag}: model present, skipping training"); continue
        t = train_subspace(units, labels, init, d, v, args.epochs, args.lr, args.batch_size,
                           args.temperature, args.absent_lambda, device, args.seed)
        torch.save(t, rd / "model.pt"); trained[tag] = t
    del units

    ext = UnitExtractor(args.backbone, g.tile_size, g.grid, [g.kind], device)
    for split_name in ("val", "dev", "test"):
        split: SplitData = getattr(P, split_name)
        if all((out_dir / gname / t / f"scores_{split_name}.npz").exists() for t in trained):
            log(f"{split_name}: scores present, skipping"); continue
        stores = {t: ScoreStore(len(split), K, keys) for t in trained}
        n_seen = 0; t0 = time.time()
        log(f"scoring {split_name} ({len(split)} images) for {list(trained)}")
        for idxs, u in iter_image_units(ext, split.paths, args.frame_batch_size, args.num_workers):
            for t, m in trained.items():
                stores[t].put(idxs, score_units(project_units(u[g.kind], m["W"]), m["prototypes"], args.temperature))
            n_seen += len(idxs)
            if n_seen % (args.frame_batch_size * 250) == 0:
                log(f"  {split_name}: {n_seen}/{len(split)}, {n_seen / (time.time() - t0):.1f} img/s")
        for t in trained:
            assert stores[t].complete()
            save_scores(out_dir / gname / t / f"scores_{split_name}.npz", stores[t], split)
    del ext

    for t in trained:
        rd = out_dir / gname / t; S = {}
        for s in ("val", "dev", "test"):
            z = np.load(rd / f"scores_{s}.npz"); S[s] = {k.replace("__", "/"): z[k] for k in z.files}
        entry = {"history": trained[t].get("history"), "d": trained[t]["d"], "variant": trained[t]["variant"], "keys": {}}
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
            log(f"{gname}/{t} {key:16s} dev AUROC={e['dev']['auroc']['mean']:.3f} test AUROC={e['test']['auroc']['mean']:.3f}"
                f" test F1={e['test']['per_sample_f1']:.4f} macro={e['test']['macro_f1']['macro']:.4f} pred/img={e['test']['mean_predicted']:.2f}")
        results["runs"][f"{gname}/{t}"] = entry
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    rows = [(run, key, e["dev"]["auroc"]["mean"], e["test"]["auroc"]["mean"], e["test"]["per_sample_f1"],
             e["test"]["macro_f1"]["macro"], e["test"]["exact_match"], e["test"]["mean_predicted"], e["test_devthr"]["per_sample_f1"])
            for run, entry in results["runs"].items() for key, e in entry["keys"].items()]
    rows.sort(key=lambda r: -r[2])
    best = {"run": rows[0][0], "granularity": gname, "tag": rows[0][0].split("/")[1], "key": rows[0][1],
            "dev_auroc": rows[0][2], "selected_on": "dev AUROC mean"}
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))
    c = results["constant"]["test"]
    md = ["# Stage 2: learned orthonormal mixing subspace (LCO seed 1337)\n",
          f"Granularity `{gname}` from Stage 1 (`{best1['run']}`). Constant predictor test F1 {c['per_sample_f1']:.4f}, macro {c['macro_f1']:.4f}, AUROC 0.500.\n",
          "| run | read-out/agg | dev AUROC | test AUROC | test F1 | test macro F1 | test exact | pred/img | test F1 (dev thr) |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.2f} | {r[8]:.4f} |" for r in rows]
    md.append(f"\nSelected on dev AUROC: **{best['run']} / {best['key']}** ({best['dev_auroc']:.3f}).\n")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out_dir / 'results.json'}, summary.md, best.json -> {best['run']} / {best['key']}")


if __name__ == "__main__":
    main()
