#!/usr/bin/env python3
"""Run PHOEBI Methods A, B, C in the leave-combinations-out regime.

Same heldout protocol as baselines/supervised_multilabel_heldout.py and
baselines/finetune_dinov2_phoebi_heldout.py:
  - 9 held-out combinations (1 single / 2 pairs / 3 triples / 2 quadruples /
    1 six-species), seed 1337.
  - 31 trained-on combos -> image-level 90/10 split for train/val.
  - Held-out combos form the entire test set.

Methods A and B are closed-form on cached tile features. Method C trains a
390-parameter channel-grouped head for 30 epochs.

Hybrid prototype init: pure-culture mean for trained-on singletons + mean of
tile features from images containing that species for the held-out singleton
(which has no pure-culture combo in the trained set). This is symmetric to
what the supervised baseline sees.

Outputs to outputs/phoebi_heldout/{results.json, summary.md, per_combo.csv}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import (
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import TileConfig
from src.simplex_unmixing.model import (
    ModelConfig,
    UnmixerModel,
    initialize_prototypes,
)
from src.prototype_matching.model import ProtoConfig, PrototypeMatchingModel
from src.mc_channel.model import MCConfig, MCChannelHead

from baselines.supervised_multilabel_heldout import (
    DEFAULT_HELDOUT_COUNTS,
    parse_heldout_counts,
    select_heldout,
)
from tools.build_splits import discover_class_names, parse_label_tokens


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def collect_entries(frames_dir: Path, combos: List[str],
                    class_names: List[str]) -> List[Tuple[Path, np.ndarray, str]]:
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    K = len(class_names)
    out = []
    for combo in combos:
        tokens = parse_label_tokens(combo)
        label = np.zeros(K, dtype=np.int64)
        for t in tokens:
            if t in cls_to_idx:
                label[cls_to_idx[t]] = 1
        for img in sorted((frames_dir / combo).glob("*.jpg")):
            out.append((img, label, combo))
    return out


def image_level_90_10(entries, seed: int):
    rng = random.Random(seed + 1)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n_val = int(0.1 * len(shuffled))
    return shuffled[n_val:], shuffled[:n_val]


def split_lists(entries):
    paths = [str(p) for p, _, _ in entries]
    labels = np.stack([l for _, l, _ in entries], axis=0)
    combos = [c for _, _, c in entries]
    return paths, labels, combos


# ---------------------------------------------------------------------------
# Hybrid prototype init: per-class fallback when a singleton is held out
# ---------------------------------------------------------------------------
def hybrid_prototype_init(
    train_features: torch.Tensor,           # (N*T, D), L2-normalized
    train_image_index: torch.Tensor,        # (N*T,)
    train_labels: np.ndarray,                # (N, K)
    train_combos: List[str],                 # length N
    class_names: List[str],
) -> torch.Tensor:
    """Per-class init.

    For each class k:
      - If a pure-culture combo (folder name == class name) exists in train,
        prototype k = L2-normalized mean of tiles from those images.
      - Otherwise, prototype k = L2-normalized mean of tiles from images whose
        label[k] == 1 (any combo containing k).
    """
    K = len(class_names)
    D = train_features.shape[1]
    protos = torch.empty((K, D), dtype=train_features.dtype)
    n_pure = 0
    n_fallback = 0
    train_combos_arr = np.array(train_combos)
    for k, name in enumerate(class_names):
        pure_image_idxs = np.where(train_combos_arr == name)[0]
        if len(pure_image_idxs) > 0:
            idx = torch.from_numpy(pure_image_idxs).long()
            mask = torch.isin(train_image_index, idx)
            tiles = train_features[mask]
            n_pure += 1
        else:
            present_image_idxs = np.where(train_labels[:, k] == 1)[0]
            if len(present_image_idxs) == 0:
                raise RuntimeError(f"Class {name}: no images with label[k]=1 in train")
            idx = torch.from_numpy(present_image_idxs).long()
            mask = torch.isin(train_image_index, idx)
            tiles = train_features[mask]
            n_fallback += 1
        if tiles.shape[0] == 0:
            raise RuntimeError(f"Class {name}: zero tiles for prototype init")
        proto = tiles.mean(dim=0)
        protos[k] = F.normalize(proto, p=2, dim=0)
    print(f"[init] {n_pure} pure-culture / {n_fallback} fallback "
          f"(present-in-mixed-combo)", flush=True)
    return protos


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def quantile_thresholds(
    image_scores: np.ndarray,   # (N, K)
    labels: np.ndarray,          # (N, K)
    quantile: float = 0.05,
) -> np.ndarray:
    """5th-percentile-of-positives — used by Methods A and B."""
    K = image_scores.shape[1]
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        positive = image_scores[labels[:, k] == 1, k]
        if len(positive) == 0:
            out[k] = float(image_scores[:, k].max() + 1.0)  # never fires
        else:
            out[k] = float(np.quantile(positive, quantile))
    return out


def argmax_f1_thresholds(image_scores: np.ndarray,
                         labels: np.ndarray) -> np.ndarray:
    K = image_scores.shape[1]
    grid = np.linspace(0.01, 0.99, 99)
    out = np.zeros(K)
    for k in range(K):
        best = -1.0
        for t in grid:
            pred = (image_scores[:, k] > t).astype(np.int64)
            tp = int(((pred == 1) & (labels[:, k] == 1)).sum())
            fp = int(((pred == 1) & (labels[:, k] == 0)).sum())
            fn = int(((pred == 0) & (labels[:, k] == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best:
                best = f1
                out[k] = float(t)
    return out


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------
def aggregate_to_image(per_tile: torch.Tensor, image_index: torch.Tensor,
                      n_images: int) -> np.ndarray:
    return scatter_mean_by_image(per_tile, image_index, n_images).cpu().numpy()


def run_method_a(
    train_features, train_image_index, val_features, val_image_index,
    test_features, test_image_index,
    init_protos, val_labels, test_labels, n_train, n_val, n_test,
    K, D, epochs=30, lr=1e-3, batch_size=4096, device="cuda",
    learned_tau=False,
):
    cfg = ModelConfig(embedding_dim=D, num_prototypes=K, temperature=10.0,
                      learned_temperature=learned_tau)
    model = UnmixerModel(cfg).to(device)
    with torch.no_grad():
        model.prototypes.copy_(init_protos.to(device))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_features_dev = train_features.to(device)
    n_tiles = train_features_dev.shape[0]
    print(f"  Method A: training for {epochs} epochs on {n_tiles} tiles", flush=True)
    for ep in range(epochs):
        perm = torch.randperm(n_tiles, device=device)
        running = 0.0
        for i in range(0, n_tiles, batch_size):
            batch = train_features_dev[perm[i:i + batch_size]]
            opt.zero_grad()
            recon, _, residual = model(batch)
            loss = (residual ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
        if (ep + 1) % 5 == 0:
            print(f"    ep{ep+1:02d}/{epochs}  recon MSE={running/n_tiles:.5f}",
                  flush=True)

    if learned_tau and hasattr(model, "log_tau"):
        taus = model.log_tau.exp().detach().cpu().numpy()
        print(f"  Learned τ (per-class): {[round(float(t), 3) for t in taus]}", flush=True)

    model.eval()
    with torch.no_grad():
        _, val_w, _ = model(val_features.to(device))
        _, test_w, _ = model(test_features.to(device))
    val_w = aggregate_to_image(val_w.cpu(), val_image_index, n_val)
    test_w = aggregate_to_image(test_w.cpu(), test_image_index, n_test)

    thr = quantile_thresholds(val_w, val_labels, quantile=0.05)
    val_pred = (val_w > thr).astype(np.int64)
    test_pred = (test_w > thr).astype(np.int64)
    return thr, val_pred, test_pred, val_w, test_w


def run_method_b(
    val_features, val_image_index, test_features, test_image_index,
    init_protos, val_labels, test_labels, n_val, n_test,
    K, D, device="cuda",
):
    cfg = ProtoConfig(embedding_dim=D, num_prototypes=K)
    model = PrototypeMatchingModel(cfg).to(device)
    with torch.no_grad():
        model.prototypes.copy_(init_protos.to(device))

    model.eval()
    with torch.no_grad():
        val_sim, _ = model(val_features.to(device))
        test_sim, _ = model(test_features.to(device))
    val_sim = aggregate_to_image(val_sim.cpu(), val_image_index, n_val)
    test_sim = aggregate_to_image(test_sim.cpu(), test_image_index, n_test)

    thr = quantile_thresholds(val_sim, val_labels, quantile=0.05)
    val_pred = (val_sim > thr).astype(np.int64)
    test_pred = (test_sim > thr).astype(np.int64)
    return thr, val_pred, test_pred, val_sim, test_sim


def run_method_c(
    train_features, train_image_index, val_features, val_image_index,
    test_features, test_image_index,
    train_labels, val_labels, test_labels, n_train, n_val, n_test,
    K, D, epochs=30, lr=1e-2, batch_size=4096, device="cuda",
):
    if D % K != 0:
        raise SystemExit(f"Method C requires D ({D}) divisible by K ({K})")
    cfg = MCConfig(embedding_dim=D, num_classes=K, cra_drop_prob=0.5)
    model = MCChannelHead(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    train_tile_labels = (
        torch.from_numpy(train_labels)[train_image_index].float()
    )
    train_features_dev = train_features.to(device)
    train_tile_labels_dev = train_tile_labels.to(device)
    n_tiles = train_features_dev.shape[0]
    print(f"  Method C: training for {epochs} epochs on {n_tiles} tiles", flush=True)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tiles, device=device)
        running = 0.0
        for i in range(0, n_tiles, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_features_dev[idx]
            yb = train_tile_labels_dev[idx]
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        if (ep + 1) % 5 == 0:
            print(f"    ep{ep+1:02d}/{epochs}  BCE={running/n_tiles:.4f}",
                  flush=True)

    model.eval()
    with torch.no_grad():
        val_sig = torch.sigmoid(model(val_features.to(device))).cpu()
        test_sig = torch.sigmoid(model(test_features.to(device))).cpu()
    val_sig = aggregate_to_image(val_sig, val_image_index, n_val)
    test_sig = aggregate_to_image(test_sig, test_image_index, n_test)

    thr = argmax_f1_thresholds(val_sig, val_labels)
    val_pred = (val_sig > thr).astype(np.int64)
    test_pred = (test_sig > thr).astype(np.int64)
    return thr, val_pred, test_pred, val_sig, test_sig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/images")
    ap.add_argument("--output_dir", default="outputs/phoebi_heldout")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--tile_size", type=int, default=224)
    ap.add_argument("--eval_grid_size", type=int, default=4)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--method_a_epochs", type=int, default=30)
    ap.add_argument("--method_c_epochs", type=int, default=30)
    ap.add_argument("--method_a_learned_tau", action="store_true",
                    help="Use per-class learned temperature in Method A (Tier 2C ablation)")
    ap.add_argument("--heldout_counts", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats_dir = out_dir / "features"
    feats_dir.mkdir(exist_ok=True)
    print(f"Device: {device}  output: {out_dir}", flush=True)

    frames_dir = ROOT / args.frames_dir
    class_names = discover_class_names(frames_dir)
    K = len(class_names)
    heldout_counts = (parse_heldout_counts(args.heldout_counts)
                      if args.heldout_counts else DEFAULT_HELDOUT_COUNTS)

    heldout, trained, _ = select_heldout(frames_dir, seed=args.seed,
                                          class_names=class_names,
                                          heldout_counts=heldout_counts)
    print(f"Species ({K}): {class_names}", flush=True)
    print(f"Held-out combinations ({len(heldout)}): {heldout}", flush=True)
    print(f"Trained-on: {len(trained)}", flush=True)

    trained_entries = collect_entries(frames_dir, trained, class_names)
    heldout_entries = collect_entries(frames_dir, heldout, class_names)
    train_entries, val_entries = image_level_90_10(trained_entries, seed=args.seed)
    train_paths, train_labels, train_combos = split_lists(train_entries)
    val_paths, val_labels, _ = split_lists(val_entries)
    test_paths, test_labels, test_combos = split_lists(heldout_entries)
    print(f"train={len(train_paths)}  val={len(val_paths)}  "
          f"test(heldout)={len(test_paths)}", flush=True)

    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        train_tiles_per_image=args.eval_grid_size ** 2,
        eval_grid_size=args.eval_grid_size,
    )

    def extract(name, paths):
        cache = str(feats_dir / f"{name}.pt")
        feats, idx = extract_features_multicrop_gpu(
            image_paths=paths, tile_config=tile_cfg, backbone=args.backbone,
            frame_batch_size=args.frame_batch_size,
            num_workers=args.num_workers, device=device,
            illum_sigma=64.0, illum_method="divide", cache_path=cache,
        )
        feats = F.normalize(feats, p=2, dim=1)
        return feats, idx

    print("\n[1/4] Extracting features...", flush=True)
    train_feats, train_idx = extract("train", train_paths)
    val_feats, val_idx = extract("val", val_paths)
    test_feats, test_idx = extract("test", test_paths)
    D = train_feats.shape[1]
    print(f"  train tiles={train_feats.shape}  D={D}", flush=True)

    print("\n[2/4] Building hybrid prototypes...", flush=True)
    init_protos = hybrid_prototype_init(
        train_feats, train_idx, train_labels, train_combos, class_names,
    )

    n_train = len(train_paths)
    n_val = len(val_paths)
    n_test = len(test_paths)

    results = {
        "class_names": class_names,
        "heldout_combos": heldout,
        "trained_combos_count": len(trained),
        "n_train_imgs": n_train,
        "n_val_imgs": n_val,
        "n_test_imgs": n_test,
        "config": vars(args),
        "methods": {},
    }

    print("\n[3/4] Method A (simplex unmix)...", flush=True)
    thr_a, va_p, ta_p, va_s, ta_s = run_method_a(
        train_feats, train_idx, val_feats, val_idx, test_feats, test_idx,
        init_protos, val_labels, test_labels, n_train, n_val, n_test,
        K, D, epochs=args.method_a_epochs, device=device,
        learned_tau=args.method_a_learned_tau,
    )
    print("\n[3/4] Method B (cosine match, closed-form)...", flush=True)
    thr_b, vb_p, tb_p, vb_s, tb_s = run_method_b(
        val_feats, val_idx, test_feats, test_idx,
        init_protos, val_labels, test_labels, n_val, n_test,
        K, D, device=device,
    )
    print("\n[3/4] Method C (channel-grouped head)...", flush=True)
    thr_c, vc_p, tc_p, vc_s, tc_s = run_method_c(
        train_feats, train_idx, val_feats, val_idx, test_feats, test_idx,
        train_labels, val_labels, test_labels, n_train, n_val, n_test,
        K, D, epochs=args.method_c_epochs, device=device,
    )

    print("\n[4/4] Aggregating metrics...", flush=True)
    test_combos_arr = np.array(test_combos)

    def per_method(name, val_pred, test_pred, val_scores, test_scores, thresholds):
        v = {
            "per_sample_f1": float(per_sample_f1(val_labels, val_pred)),
            "macro_f1": {k: float(v) for k, v in
                         macro_f1_per_class(val_labels, val_pred, class_names).items()},
            "exact_match": float(exact_match_accuracy(val_labels, val_pred)),
        }
        t = {
            "per_sample_f1": float(per_sample_f1(test_labels, test_pred)),
            "macro_f1": {k: float(v) for k, v in
                         macro_f1_per_class(test_labels, test_pred, class_names).items()},
            "exact_match": float(exact_match_accuracy(test_labels, test_pred)),
        }
        per_combo = {}
        for combo in heldout:
            mask = test_combos_arr == combo
            if mask.sum() == 0:
                continue
            per_combo[combo] = {
                "n_images": int(mask.sum()),
                "per_sample_f1": float(per_sample_f1(test_labels[mask], test_pred[mask])),
                "exact_match": float(exact_match_accuracy(test_labels[mask], test_pred[mask])),
            }
        np.save(out_dir / f"{name}_test_scores.npy", test_scores)
        return {
            "thresholds": thresholds.tolist(),
            "val_in_distribution": v,
            "test_heldout_combos": t,
            "per_heldout_combo": per_combo,
            "delta_f1": v["per_sample_f1"] - t["per_sample_f1"],
        }

    results["methods"]["A_simplex"] = per_method(
        "method_a", va_p, ta_p, va_s, ta_s, thr_a)
    results["methods"]["B_proto"] = per_method(
        "method_b", vb_p, tb_p, vb_s, tb_s, thr_b)
    results["methods"]["C_channel"] = per_method(
        "method_c", vc_p, tc_p, vc_s, tc_s, thr_c)

    np.save(out_dir / "test_labels.npy", test_labels)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    md = []
    md.append("# PHOEBI Methods A/B/C in held-out-combinations regime\n")
    md.append(f"Held-out combinations ({len(heldout)}): "
              f"`{', '.join(heldout)}`\n")
    md.append(f"train={n_train}  val={n_val}  test(heldout)={n_test}\n\n")
    md.append("| Method | Val F1 | Heldout F1 | Δ F1 ↓ | Heldout Macro F1 | Heldout Exact |")
    md.append("|--------|--------|------------|--------|------------------|----------------|")
    for tag in ["A_simplex", "B_proto", "C_channel"]:
        m = results["methods"][tag]
        md.append(f"| {tag} | {m['val_in_distribution']['per_sample_f1']:.4f} | "
                  f"{m['test_heldout_combos']['per_sample_f1']:.4f} | "
                  f"{m['delta_f1']:+.4f} | "
                  f"{m['test_heldout_combos']['macro_f1']['macro']:.4f} | "
                  f"{m['test_heldout_combos']['exact_match']:.4f} |")
    md.append("\n## Per-held-out-combination per-sample F1\n")
    md.append("| Combo | Order | A | B | C |")
    md.append("|-------|-------|---|---|---|")
    for combo in heldout:
        order = len(parse_label_tokens(combo))
        row = [f"`{combo}`", str(order)]
        for tag in ["A_simplex", "B_proto", "C_channel"]:
            pc = results["methods"][tag]["per_heldout_combo"].get(combo)
            row.append(f"{pc['per_sample_f1']:.4f}" if pc else "n/a")
        md.append("| " + " | ".join(row) + " |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print("\n=== PHOEBI Methods A/B/C, leave-combinations-out ===")
    for tag in ["A_simplex", "B_proto", "C_channel"]:
        m = results["methods"][tag]
        print(f"  {tag:12s}  val F1 = {m['val_in_distribution']['per_sample_f1']:.4f}  "
              f"heldout F1 = {m['test_heldout_combos']['per_sample_f1']:.4f}  "
              f"Δ = {m['delta_f1']:+.4f}")
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
