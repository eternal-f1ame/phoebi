#!/usr/bin/env python3
"""Tier 2C ablation: adaptive per-class temperature τ for Method A.

Runs Method A on the leave-9-combinations-out protocol with fixed τ=10
and learned per-class τ (K=6 nn.Parameters, log-space), for seeds
1337/1338/1339. Reports val F1, heldout F1, and Δ F1 for both variants,
and prints the learned τ values per class.

Output: outputs/learned_tau_ablation/results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import extract_features_multicrop_gpu, scatter_mean_by_image
from src.common.metrics import per_sample_f1, macro_f1_per_class
from src.common.tiling import TileConfig
from src.simplex_unmixing.model import ModelConfig, UnmixerModel

from baselines.supervised_multilabel_heldout import (
    DEFAULT_HELDOUT_COUNTS, parse_heldout_counts, select_heldout,
)
from tools.build_splits import discover_class_names, parse_label_tokens

# Reuse data helpers from run_phoebi_heldout
from experiments.run_phoebi_heldout import (
    collect_entries, image_level_90_10, split_lists,
    hybrid_prototype_init, quantile_thresholds,
)


def aggregate(per_tile, image_index, n):
    return scatter_mean_by_image(per_tile, image_index, n).cpu().numpy()


def train_method_a(
    train_feats, train_idx, val_feats, val_idx, test_feats, test_idx,
    init_protos, val_labels, test_labels, n_val, n_test,
    K, D, epochs=30, lr=1e-3, batch_size=4096, device="cuda",
    learned_tau=False, class_names=None,
):
    cfg = ModelConfig(embedding_dim=D, num_prototypes=K, temperature=10.0,
                      learned_temperature=learned_tau)
    model = UnmixerModel(cfg).to(device)
    with torch.no_grad():
        model.prototypes.copy_(init_protos.to(device))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    feats_dev = train_feats.to(device)
    n_tiles = feats_dev.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(n_tiles, device=device)
        running = 0.0
        for i in range(0, n_tiles, batch_size):
            batch = feats_dev[perm[i:i + batch_size]]
            opt.zero_grad()
            recon, _, residual = model(batch)
            loss = (residual ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()
            running += loss.item() * batch.shape[0]
        if (ep + 1) % 10 == 0:
            print(f"    ep{ep+1:02d}/{epochs}  recon MSE={running/n_tiles:.5f}", flush=True)

    if learned_tau and hasattr(model, "log_tau"):
        taus = model.log_tau.exp().detach().cpu().numpy()
        names = class_names or list(range(K))
        print(f"  Learned τ: {dict(zip(names, [round(float(t), 3) for t in taus]))}", flush=True)

    model.eval()
    with torch.no_grad():
        _, val_w, _ = model(val_feats.to(device))
        _, test_w, _ = model(test_feats.to(device))
    val_w = aggregate(val_w.cpu(), val_idx, n_val)
    test_w = aggregate(test_w.cpu(), test_idx, n_test)

    thr = quantile_thresholds(val_w, val_labels, quantile=0.05)
    val_pred = (val_w > thr).astype(np.int64)
    test_pred = (test_w > thr).astype(np.int64)

    val_f1 = float(per_sample_f1(val_labels, val_pred))
    test_f1 = float(per_sample_f1(test_labels, test_pred))
    return val_f1, test_f1, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/images")
    ap.add_argument("--output_dir", default="outputs/learned_tau_ablation")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--tile_size", type=int, default=224)
    ap.add_argument("--eval_grid_size", type=int, default=4)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="1337,1338,1339")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--heldout_counts", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats_dir = out_dir / "features"
    feats_dir.mkdir(exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    frames_dir = ROOT / args.frames_dir
    class_names = discover_class_names(frames_dir)
    K = len(class_names)
    heldout_counts = (parse_heldout_counts(args.heldout_counts)
                      if args.heldout_counts else DEFAULT_HELDOUT_COUNTS)
    heldout, trained, _ = select_heldout(frames_dir, seed=seeds[0],
                                          class_names=class_names,
                                          heldout_counts=heldout_counts)
    print(f"Classes: {class_names}", flush=True)
    print(f"Held-out ({len(heldout)}): {heldout}", flush=True)

    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        train_tiles_per_image=args.eval_grid_size ** 2,
        eval_grid_size=args.eval_grid_size,
    )

    def extract(tag, paths):
        cache = str(feats_dir / f"{tag}.pt")
        feats, idx = extract_features_multicrop_gpu(
            image_paths=paths, tile_config=tile_cfg, backbone=args.backbone,
            frame_batch_size=args.frame_batch_size,
            num_workers=args.num_workers, device=device,
            illum_sigma=64.0, illum_method="divide", cache_path=cache,
        )
        return F.normalize(feats, p=2, dim=1), idx

    seed_results = []
    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"Seed {seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        trained_entries = collect_entries(frames_dir, trained, class_names)
        heldout_entries = collect_entries(frames_dir, heldout, class_names)
        train_entries, val_entries = image_level_90_10(trained_entries, seed=seed)
        train_paths, train_labels, train_combos = split_lists(train_entries)
        val_paths, val_labels, _ = split_lists(val_entries)
        test_paths, test_labels, _ = split_lists(heldout_entries)

        train_feats, train_idx = extract(f"train_s{seed}", train_paths)
        val_feats, val_idx = extract(f"val_s{seed}", val_paths)
        test_feats, test_idx = extract(f"test_s{seed}", test_paths)
        D = train_feats.shape[1]
        n_val, n_test = len(val_paths), len(test_paths)

        init_protos = hybrid_prototype_init(
            train_feats, train_idx, train_labels, train_combos, class_names,
        )

        for variant, learned in [("fixed_tau", False), ("learned_tau", True)]:
            print(f"\n  {variant}:", flush=True)
            val_f1, test_f1, _ = train_method_a(
                train_feats, train_idx, val_feats, val_idx,
                test_feats, test_idx, init_protos,
                val_labels, test_labels, n_val, n_test,
                K, D, epochs=args.epochs, device=device,
                learned_tau=learned, class_names=class_names,
            )
            print(f"  val F1={val_f1:.4f}  heldout F1={test_f1:.4f}  "
                  f"Δ={val_f1 - test_f1:+.4f}", flush=True)
            seed_results.append({
                "seed": seed, "variant": variant,
                "val_f1": val_f1, "test_f1": test_f1,
                "delta_f1": val_f1 - test_f1,
            })

    # Aggregate across seeds
    summary = {}
    for variant in ["fixed_tau", "learned_tau"]:
        rows = [r for r in seed_results if r["variant"] == variant]
        vals = [r["val_f1"] for r in rows]
        tests = [r["test_f1"] for r in rows]
        deltas = [r["delta_f1"] for r in rows]
        summary[variant] = {
            "val_f1_mean": float(np.mean(vals)),
            "val_f1_std": float(np.std(vals)),
            "test_f1_mean": float(np.mean(tests)),
            "test_f1_std": float(np.std(tests)),
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
        }

    output = {
        "heldout_combos": heldout,
        "class_names": class_names,
        "seeds": seeds,
        "per_seed": seed_results,
        "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2))

    print("\n\n=== Adaptive Temperature τ Ablation ===")
    print(f"{'Variant':<15} {'Val F1':>12} {'Heldout F1':>12} {'Δ F1':>10}")
    for variant, s in summary.items():
        print(f"{variant:<15} {s['val_f1_mean']:.3f}±{s['val_f1_std']:.3f} "
              f"{s['test_f1_mean']:.3f}±{s['test_f1_std']:.3f} "
              f"{s['delta_mean']:+.3f}±{s['delta_std']:.3f}")
    print(f"\nResults: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
