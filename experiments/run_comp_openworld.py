#!/usr/bin/env python3
"""Open-world half: open-set rejection and novel-class discovery with finer units,
minority-aware aggregation, class-conditional normalisation, and a gated discovery step.

Protocol is the paper's LOOCV over species on the random split (experiments/
run_osr_score_sweep.py, run_discovery.py): for each held-out species k*, prototypes come
from the five remaining pure cultures of the training split with every image containing
k* removed, the kNN reference pool is that filtered training split, and "unknown" means a
test image that contains k*. What changes here:

  unit          cls224 (the paper's 16 CLS tiles per image) or patch224 (patch tokens)
  score         residual norm (SimplexUnmix), negative max cosine (ProtoMatch), energy
                (T = 0.1), kNN k-th distance (k = 10), and class-normalised variants of
                residual and kNN: each unit's score divided by the median score of
                validation units of its nearest known class (unknown evidence is judged
                against how far *that* class normally sits from its own prototypes)
  aggregation   mean over units (the paper), 90th percentile, max
  selection     inner leave-one-species-out on the validation split (never on test):
                for outer k*, each remaining species k' is treated as unknown in turn on
                validation images that do not contain k*, and the variant with the best
                mean inner AUROC is the selected one. Every variant's test AUROC is
                reported anyway.
  discovery     Sinkhorn K = 1 on the features of gated units; the paper's gate is
                residual > 0.15 on CLS tiles; the new gates flag units whose unknown score
                exceeds the 95th percentile of validation known-unit scores. Recall,
                purity, cluster accuracy and drift follow run_discovery.py.

Anchor: cls224 / knn_k10 / mean must reproduce the paper's 0.70 +/- 0.07 AUROC, and
cls224 / residual > 0.15 / SK K=1 its 0.50 +/- 0.11 cluster accuracy.

Memory: unit tensors stay on the CPU (fp16) and stream through the device in image chunks;
only the kNN pool (<= 1.5 M units, fp16) is resident on the device.

Usage
    python experiments/run_comp_openworld.py --output_dir outputs/compositional/openworld
    python experiments/run_comp_openworld.py --max_images 2 --output_dir /tmp/x   # CPU smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import load_real_split
from src.common.metrics import fpr_at_tpr, open_set_aupr, open_set_auroc, per_sample_f1
from src.common.sinkhorn import sinkhorn_cluster
from src.compositional.units import UnitExtractor, fixed_unit_subset, iter_image_units
from src.simplex_unmixing.model import sparsemax

UNITS = ("cls224", "patch224")
SCORES = ("residual", "neg_max", "energy", "knn", "residual_norm", "knn_norm")
AGGS = ("mean", "q90", "max")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------
def cap_split(paths, labels, combos, max_images):
    if not max_images:
        return paths, labels, combos
    keep, seen = [], {}
    for i, c in enumerate(combos):
        seen[c] = seen.get(c, 0) + 1
        if seen[c] <= max_images:
            keep.append(i)
    return [paths[i] for i in keep], labels[keep], [combos[i] for i in keep]


def build_cache(cache_dir: Path, split: str, paths: List[str], per_image_patches: int, backbone: str,
                device: torch.device, frame_batch_size: int, num_workers: int, seed: int) -> None:
    meta_path = cache_dir / f"{split}_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["paths"] == paths:
        log(f"cache {split}: present"); return
    ext = UnitExtractor(backbone, 224, 4, ["cls", "patch"], device)
    D = ext.dim; n = len(paths)
    cls_mm = np.lib.format.open_memmap(cache_dir / f"{split}_cls224.npy", mode="w+", dtype=np.float16, shape=(n, 16, D))
    patch_mm = None; sel = None; t0 = time.time(); seen = 0
    for idxs, u in iter_image_units(ext, paths, frame_batch_size, num_workers):
        i = idxs.numpy()
        cls_mm[i] = u["cls"].half().cpu().numpy()
        p = u["patch"]
        if patch_mm is None:
            sel = fixed_unit_subset(p.shape[1], per_image_patches, seed)
            U = p.shape[1] if sel is None else int(sel.numel())
            patch_mm = np.lib.format.open_memmap(cache_dir / f"{split}_patch224.npy", mode="w+", dtype=np.float16, shape=(n, U, D))
        patch_mm[i] = (p if sel is None else p[:, sel.to(p.device)]).half().cpu().numpy()
        seen += len(i)
        if seen % (frame_batch_size * 250) == 0:
            log(f"  {split}: {seen}/{n} images, {seen / (time.time() - t0):.1f} img/s")
    cls_mm.flush(); patch_mm.flush()
    meta_path.write_text(json.dumps({"paths": paths, "dim": D, "per_image_patches": int(patch_mm.shape[1])}))
    del ext
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_units_cpu(cache_dir: Path, split: str, unit: str) -> torch.Tensor:
    """Whole split as an fp16 CPU tensor (largest: test patches, 12k x 512 x 384 = 4.7 GB)."""
    return torch.from_numpy(np.ascontiguousarray(np.load(cache_dir / f"{split}_{unit}.npy", mmap_mode="r")))


# ---------------------------------------------------------------------------------------
# scoring primitives: CPU-resident units streamed through the device in image chunks
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def knn_kth_distance(queries: torch.Tensor, pool_t: torch.Tensor, k: int, chunk: int = 512) -> torch.Tensor:
    """queries (M, D) L2-normalised on device; pool_t (D, P) on device -> (M,) distance to the k-th neighbour."""
    out = torch.empty(queries.shape[0], device=queries.device)
    for i in range(0, queries.shape[0], chunk):
        sims = queries[i:i + chunk].to(pool_t.dtype) @ pool_t
        out[i:i + chunk] = 1.0 - sims.topk(k, dim=1).values[:, -1].float()
        del sims
    return out


@torch.no_grad()
def score_units(units_cpu: torch.Tensor, protos: torch.Tensor, pool_t: torch.Tensor, k: int, temperature: float,
                energy_T: float, refs: Optional[Dict[str, torch.Tensor]], device: torch.device,
                images_per_chunk: int = 128) -> Dict[str, torch.Tensor]:
    """units_cpu (N, U, D) fp16 -> {score: (N, U) fp32 CPU}; higher = more unknown."""
    N, U, D = units_cpu.shape
    P = F.normalize(protos.to(device).float(), p=2, dim=1)
    names = ["residual", "neg_max", "energy", "knn"] + (["residual_norm", "knn_norm"] if refs is not None else [])
    out = {n: torch.empty(N, U) for n in names}
    for i in range(0, N, images_per_chunk):
        u = units_cpu[i:i + images_per_chunk].to(device, non_blocking=True)
        b = u.shape[0]
        flat = F.normalize(u.reshape(b * U, D).float(), p=2, dim=1)
        sims = flat @ P.t()
        w = sparsemax(sims * temperature, dim=1)
        residual = (flat - w @ P).norm(dim=1)
        neg_max = -sims.max(dim=1).values
        energy = -energy_T * torch.logsumexp(sims / energy_T, dim=1)
        knn = knn_kth_distance(flat, pool_t, k)
        s = {"residual": residual, "neg_max": neg_max, "energy": energy, "knn": knn}
        if refs is not None:
            nearest = sims.argmax(dim=1)
            s["residual_norm"] = residual / refs["residual"][nearest]
            s["knn_norm"] = knn / refs["knn"][nearest]
        for n in names:
            out[n][i:i + b] = s[n].reshape(b, U).cpu()
        del u, flat, sims, w
    return out


def aggregate(per_unit: torch.Tensor) -> Dict[str, np.ndarray]:
    return {"mean": per_unit.mean(dim=1).numpy(),
            "q90": torch.quantile(per_unit, 0.9, dim=1).numpy(),
            "max": per_unit.max(dim=1).values.numpy()}


@torch.no_grad()
def class_references(val_units: torch.Tensor, val_labels: np.ndarray, val_combos: List[str], kept: List[str],
                     protos: torch.Tensor, pool_t: torch.Tensor, k: int, temperature: float,
                     exclude_species_idx: List[int], device: torch.device) -> Dict[str, torch.Tensor]:
    """Per known class, median residual and kNN distance of validation units from that class's
    pure culture (validation images containing any excluded species are ignored)."""
    refs = {"residual": torch.ones(len(kept), device=device), "knn": torch.ones(len(kept), device=device)}
    for j, name in enumerate(kept):
        rows = [i for i, c in enumerate(val_combos) if c == name and not any(val_labels[i, e] for e in exclude_species_idx)]
        if not rows:
            continue
        s = score_units(val_units[rows], protos, pool_t, k, temperature, 0.1, None, device)
        refs["residual"][j] = s["residual"].median().clamp_min(1e-6).to(device)
        refs["knn"][j] = s["knn"].median().clamp_min(1e-6).to(device)
    return refs


def pure_prototypes(units: torch.Tensor, combos: List[str], kept: List[str], row_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    protos = []
    arr = np.array(combos)
    for name in kept:
        rows = torch.from_numpy(np.where((arr == name) & row_mask)[0])
        u = units[rows].float()
        protos.append(F.normalize(u.reshape(-1, u.shape[-1]).mean(dim=0), p=2, dim=0))
    return torch.stack(protos).to(device)


def pool_transposed(units: torch.Tensor, row_mask: np.ndarray, device: torch.device, max_units: int, seed: int) -> torch.Tensor:
    """(D, P) L2-normalised pool on the device: fp16 on CUDA, fp32 on CPU."""
    rows = torch.from_numpy(np.where(row_mask)[0])
    flat = units[rows].reshape(-1, units.shape[-1])
    if flat.shape[0] > max_units:
        g = torch.Generator().manual_seed(seed)
        flat = flat[torch.randperm(flat.shape[0], generator=g)[:max_units]]
    flat = F.normalize(flat.to(device).float(), p=2, dim=1)
    flat = flat.half() if device.type == "cuda" else flat
    return flat.t().contiguous()


def osr_metrics(y_unknown: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    if y_unknown.sum() == 0 or y_unknown.sum() == len(y_unknown):
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    return {"auroc": open_set_auroc(y_unknown, score), "aupr": open_set_aupr(y_unknown, score),
            "fpr95": fpr_at_tpr(y_unknown, score, 0.95)}


# ---------------------------------------------------------------------------------------
# discovery (run_discovery.py semantics), streamed
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def image_mean_sims(units_cpu: torch.Tensor, protos_n: torch.Tensor, device: torch.device, images_per_chunk: int = 128) -> torch.Tensor:
    """(N, P) mean over units of cosine to each prototype (protos_n L2-normalised on device)."""
    N, U, D = units_cpu.shape; out = torch.empty(N, protos_n.shape[0])
    for i in range(0, N, images_per_chunk):
        u = units_cpu[i:i + images_per_chunk].to(device); b = u.shape[0]
        flat = F.normalize(u.reshape(b * U, D).float(), p=2, dim=1)
        out[i:i + b] = (flat @ protos_n.t()).reshape(b, U, -1).mean(dim=1).cpu()
    return out


@torch.no_grad()
def image_mean_weights(units_cpu: torch.Tensor, protos_n: torch.Tensor, temperature: float, device: torch.device,
                       images_per_chunk: int = 128) -> torch.Tensor:
    N, U, D = units_cpu.shape; out = torch.empty(N, protos_n.shape[0])
    for i in range(0, N, images_per_chunk):
        u = units_cpu[i:i + images_per_chunk].to(device); b = u.shape[0]
        flat = F.normalize(u.reshape(b * U, D).float(), p=2, dim=1)
        out[i:i + b] = sparsemax((flat @ protos_n.t()) * temperature, dim=1).reshape(b, U, -1).mean(dim=1).cpu()
    return out


@torch.no_grad()
def gated_features(units_cpu: torch.Tensor, gate_mask: torch.Tensor, max_units: int, seed: int) -> torch.Tensor:
    """Features of gated units, subsampled to max_units (the K=1 centroid is a mean)."""
    idx = torch.nonzero(gate_mask.reshape(-1), as_tuple=False).squeeze(1)
    if idx.numel() > max_units:
        g = torch.Generator().manual_seed(seed)
        idx = idx[torch.randperm(idx.numel(), generator=g)[:max_units]]
    N, U, D = units_cpu.shape
    return F.normalize(units_cpu.reshape(N * U, D)[idx].float(), p=2, dim=1)


@torch.no_grad()
def discovery(test_units: torch.Tensor, gate_mask: torch.Tensor, known_protos: torch.Tensor, temperature: float,
              test_labels_known: np.ndarray, held_out_mask: np.ndarray, train_units_known: torch.Tensor,
              train_labels_known: np.ndarray, quantile: float, device: torch.device, seed: int,
              min_units: int = 5, max_cluster_units: int = 200_000) -> Dict:
    K_known = known_protos.shape[0]
    Pk = F.normalize(known_protos.to(device).float(), p=2, dim=1)
    out = {"n_flagged_units": int(gate_mask.sum()), "frac_flagged_units": float(gate_mask.float().mean())}
    proposed = None
    if int(gate_mask.sum()) >= min_units:
        novel = gated_features(test_units, gate_mask, max_cluster_units, seed).to(device)
        centroids, _ = sinkhorn_cluster(novel, num_clusters=1, num_iters=30, sk_iters=3, sk_epsilon=0.05, device=device)
        proposed = F.normalize(centroids.float(), p=2, dim=1)
        full = torch.cat([Pk, proposed], dim=0)
        arg = image_mean_sims(test_units, full, device).argmax(dim=1).numpy()
        ho = arg[held_out_mask]; new = ho >= K_known
        n_dom = int(np.bincount(ho[new], minlength=K_known + 1)[K_known:].max()) if new.sum() else 0
        out.update({"n_proposed": 1, "discovery_recall": float(new.mean()) if len(ho) else 0.0,
                    "discovery_purity": float(n_dom / new.sum()) if new.sum() else float("nan"),
                    "cluster_accuracy": float(n_dom / held_out_mask.sum()) if held_out_mask.sum() else 0.0})
        del novel
    else:
        out.update({"n_proposed": 0, "discovery_recall": 0.0, "discovery_purity": float("nan"), "cluster_accuracy": 0.0})
    # drift: known-class presence F1 with thresholds from training (5th percentile of positives)
    wtr = image_mean_weights(train_units_known, Pk, temperature, device).numpy()
    thr = np.array([np.quantile(wtr[train_labels_known[:, j] == 1, j], quantile) if (train_labels_known[:, j] == 1).any() else 0.5
                    for j in range(K_known)])
    before = (image_mean_weights(test_units, Pk, temperature, device).numpy()[:, :K_known] > thr).astype(int)
    f1_before = per_sample_f1(test_labels_known, before)
    if proposed is not None:
        after = (image_mean_weights(test_units, torch.cat([Pk, proposed], dim=0), temperature, device).numpy()[:, :K_known] > thr).astype(int)
        f1_after = per_sample_f1(test_labels_known, after)
    else:
        f1_after = f1_before
    out.update({"drift_known_f1_before": float(f1_before), "drift_known_f1_after": float(f1_after),
                "drift_known_f1_delta": float(f1_after - f1_before)})
    return out


# ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--output_dir", default="outputs/compositional/openworld")
    ap.add_argument("--cache_dir", default=None, help="default: <output_dir>/../cache_openworld")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--train_patches", type=int, default=16, help="patch tokens kept per training image (pool, prototypes)")
    ap.add_argument("--val_patches", type=int, default=64)
    ap.add_argument("--test_patches", type=int, default=512)
    ap.add_argument("--pool_max_units", type=int, default=1_500_000)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--energy_T", type=float, default=0.1)
    ap.add_argument("--residual_gate", type=float, default=0.15, help="the paper's absolute residual gate for discovery")
    ap.add_argument("--gate_quantile", type=float, default=0.95)
    ap.add_argument("--calibrate_quantile", type=float, default=0.05)
    ap.add_argument("--drift_train_images", type=int, default=20000)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_images", type=int, default=None, help="per-combination cap (smoke test)")
    ap.add_argument("--skip_inner", action="store_true", help="skip the inner selection loop")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir.parent / "cache_openworld"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"device={device} output={out_dir} cache={cache_dir}")

    splits = {}
    for s in ("train", "val", "test"):
        paths, labels, class_names, combos = load_real_split(args.splits_path, s)
        paths, labels, combos = cap_split(paths, labels, combos, args.max_images)
        splits[s] = (paths, labels.astype(np.int64), combos)
    class_names = list(class_names); K = len(class_names)
    log("split sizes: " + ", ".join(f"{s}={len(v[0])}" for s, v in splits.items()))
    for s, per in (("train", args.train_patches), ("val", args.val_patches), ("test", args.test_patches)):
        build_cache(cache_dir, s, splits[s][0], per, args.backbone, device, args.frame_batch_size, args.num_workers, args.seed)

    tr_paths, tr_labels, tr_combos = splits["train"]; va_paths, va_labels, va_combos = splits["val"]; te_paths, te_labels, te_combos = splits["test"]
    U_cpu = {(s, u): load_units_cpu(cache_dir, s, u) for s in ("train", "val", "test") for u in UNITS}
    log("units on CPU: " + ", ".join(f"{s}/{u} {tuple(t.shape)}" for (s, u), t in U_cpu.items()))

    results: Dict = {"config": vars(args), "class_names": class_names, "folds": [], "variants": {}}
    variant_names = [f"{u}/{s}/{a}" for u in UNITS for s in SCORES for a in AGGS]
    test_auroc = {v: [] for v in variant_names}; test_aupr = {v: [] for v in variant_names}; test_fpr = {v: [] for v in variant_names}
    inner_auroc = {v: [] for v in variant_names}
    disc_rows: List[Dict] = []

    for k_star in range(K):
        held = class_names[k_star]; kept_idx = [i for i in range(K) if i != k_star]; kept = [class_names[i] for i in kept_idx]
        t0 = time.time(); log(f"=== outer fold: unknown = {held} ===")
        tr_mask = tr_labels[:, k_star] == 0
        y_unknown = (te_labels[:, k_star] == 1).astype(int)
        fold = {"held_out": held, "n_unknown_test": int(y_unknown.sum()), "test": {}, "inner": {}}
        for u in UNITS:
            tr_u, va_u, te_u = U_cpu[("train", u)], U_cpu[("val", u)], U_cpu[("test", u)]
            protos = pure_prototypes(tr_u, tr_combos, kept, tr_mask, device)
            pool_t = pool_transposed(tr_u, tr_mask, device, args.pool_max_units, args.seed)
            refs = class_references(va_u, va_labels, va_combos, kept, protos, pool_t, args.knn_k, args.temperature, [k_star], device)
            per_unit = score_units(te_u, protos, pool_t, args.knn_k, args.temperature, args.energy_T, refs, device)
            for s in SCORES:
                for a, vec in aggregate(per_unit[s]).items():
                    m = osr_metrics(y_unknown, vec); name = f"{u}/{s}/{a}"
                    fold["test"][name] = m; test_auroc[name].append(m["auroc"]); test_aupr[name].append(m["aupr"]); test_fpr[name].append(m["fpr95"])
            log(f"  {u}: test scored ({time.time() - t0:.0f}s); knn/mean AUROC {fold['test'][f'{u}/knn/mean']['auroc']:.3f}, "
                f"knn_norm/q90 {fold['test'][f'{u}/knn_norm/q90']['auroc']:.3f}, residual/mean {fold['test'][f'{u}/residual/mean']['auroc']:.3f}")
            # ---- discovery on this unit ----
            held_mask = te_labels[:, k_star] == 1
            te_known = np.delete(te_labels, k_star, axis=1)
            tr_rows = np.where(tr_mask)[0][:args.drift_train_images]
            tr_known_units = tr_u[torch.from_numpy(tr_rows)]
            tr_known_labels = np.delete(tr_labels[tr_rows], k_star, axis=1)
            va_known_rows = torch.from_numpy(np.where(va_labels[:, k_star] == 0)[0])
            val_scores = score_units(va_u[va_known_rows], protos, pool_t, args.knn_k, args.temperature, args.energy_T, refs, device)
            gates = {"residual_abs0.15": per_unit["residual"] > args.residual_gate}
            for s in ("residual", "residual_norm", "knn", "knn_norm"):
                gates[f"{s}_q{int(args.gate_quantile * 100)}"] = per_unit[s] > torch.quantile(val_scores[s].reshape(-1), args.gate_quantile)
            for gname, gm in gates.items():
                d = discovery(te_u, gm, protos, args.temperature, te_known, held_mask, tr_known_units, tr_known_labels,
                              args.calibrate_quantile, device, args.seed)
                d.update({"held_out": held, "unit": u, "gate": gname}); disc_rows.append(d)
            log(f"  {u}: discovery done ({time.time() - t0:.0f}s); paper gate cluster acc {[d['cluster_accuracy'] for d in disc_rows if d['unit'] == u and d['gate'] == 'residual_abs0.15' and d['held_out'] == held][0]:.3f}")
            # ---- inner selection: each remaining species as the unknown, on validation images without k* ----
            if not args.skip_inner:
                va_keep = torch.from_numpy(np.where(va_labels[:, k_star] == 0)[0])
                vu = va_u[va_keep]; va_lab_keep = va_labels[va_keep.numpy()]
                for k_in in kept_idx:
                    inner_kept = [i for i in kept_idx if i != k_in]; inner_names = [class_names[i] for i in inner_kept]
                    in_mask = tr_mask & (tr_labels[:, k_in] == 0)
                    p_in = pure_prototypes(tr_u, tr_combos, inner_names, in_mask, device)
                    pool_in = pool_transposed(tr_u, in_mask, device, args.pool_max_units, args.seed)
                    refs_in = class_references(va_u, va_labels, va_combos, inner_names, p_in, pool_in, args.knn_k, args.temperature, [k_star, k_in], device)
                    y_in = (va_lab_keep[:, k_in] == 1).astype(int)
                    pu = score_units(vu, p_in, pool_in, args.knn_k, args.temperature, args.energy_T, refs_in, device)
                    for s in SCORES:
                        for a, vec in aggregate(pu[s]).items():
                            name = f"{u}/{s}/{a}"; m = osr_metrics(y_in, vec)
                            inner_auroc[name].append(m["auroc"]); fold["inner"].setdefault(name, []).append(m["auroc"])
                    del pool_in, pu
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                log(f"  {u}: inner selection done ({time.time() - t0:.0f}s)")
            del pool_t, per_unit, val_scores
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["folds"].append(fold)
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
        log(f"  fold {held} done in {time.time() - t0:.0f}s")

    ms = lambda v: (float(np.nanmean(v)), float(np.nanstd(v))) if len(v) else (float("nan"), float("nan"))
    for name in variant_names:
        results["variants"][name] = {"test_auroc": ms(test_auroc[name]), "test_aupr": ms(test_aupr[name]), "test_fpr95": ms(test_fpr[name]),
                                     "inner_auroc": ms(inner_auroc[name]) if not args.skip_inner else (float("nan"), float("nan"))}
    ranked = sorted(variant_names, key=lambda n: -(results["variants"][n]["inner_auroc"][0] if not args.skip_inner else results["variants"][n]["test_auroc"][0]))
    results["selected"] = {"variant": ranked[0], "selected_on": "inner LOO on validation" if not args.skip_inner else "test (no inner loop)",
                           "inner_auroc": results["variants"][ranked[0]]["inner_auroc"], "test_auroc": results["variants"][ranked[0]]["test_auroc"]}
    results["discovery"] = disc_rows
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    md = ["# Open-world half: open-set rejection and discovery with finer units and gated aggregation (LOOCV over species, random split)\n",
          f"Anchor: cls224/knn/mean test AUROC {results['variants']['cls224/knn/mean']['test_auroc'][0]:.3f} ± {results['variants']['cls224/knn/mean']['test_auroc'][1]:.3f} "
          f"(paper 0.701 ± 0.066); cls224/residual/mean {results['variants']['cls224/residual/mean']['test_auroc'][0]:.3f} (paper 0.444).\n",
          f"Selected on the inner validation loop: **{results['selected']['variant']}** (inner AUROC {results['selected']['inner_auroc'][0]:.3f}, "
          f"test AUROC {results['selected']['test_auroc'][0]:.3f} ± {results['selected']['test_auroc'][1]:.3f}).\n",
          "| variant (unit / score / aggregation) | inner AUROC | test AUROC (mean ± std over 6 folds) | test AUPR | test FPR@95 |", "|---|---|---|---|---|"]
    for n in ranked:
        v = results["variants"][n]
        md.append(f"| {n} | {v['inner_auroc'][0]:.3f} | {v['test_auroc'][0]:.3f} ± {v['test_auroc'][1]:.3f} | {v['test_aupr'][0]:.3f} | {v['test_fpr95'][0]:.3f} |")
    md += ["\n## Discovery (Sinkhorn K = 1 on gated units; mean over the 6 folds)\n",
           "| unit | gate | flagged units | recall | purity | cluster accuracy | drift (known F1 after − before) |", "|---|---|---|---|---|---|---|"]
    for u, g in sorted({(d["unit"], d["gate"]) for d in disc_rows}):
        rows = [d for d in disc_rows if d["unit"] == u and d["gate"] == g]
        f = lambda key: float(np.nanmean([r[key] for r in rows]))
        md.append(f"| {u} | {g} | {f('frac_flagged_units'):.3f} | {f('discovery_recall'):.3f} | {f('discovery_purity'):.3f} | "
                  f"{f('cluster_accuracy'):.3f} ± {float(np.nanstd([r['cluster_accuracy'] for r in rows])):.3f} | {f('drift_known_f1_delta'):+.3f} |")
    md.append("\nPaper (cls224, residual > 0.15, SK K=1): cluster accuracy 0.502 ± 0.106, purity 1.000, drift −0.031.\n")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out_dir / 'results.json'} and summary.md; selected {results['selected']['variant']}")


if __name__ == "__main__":
    main()
