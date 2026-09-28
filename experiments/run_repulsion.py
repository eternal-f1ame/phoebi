#!/usr/bin/env python3
"""Tier 2D ablation: hyperspherical repulsion regularizer for Method A.

Adds λ * mean_{i≠j} exp(P_i·P_j / τ_r) to the reconstruction MSE loss
so prototypes are pushed apart on the unit sphere without changing inference.
Motivated by the bt/bs/mx confusion: bt's prototype collapses into the
convex span of bs and mx because MSE loss does not penalize proximity.

Sweeps λ ∈ {0, 0.01, 0.1, 1.0} with τ_r=0.1 (stronger push than model τ=10).
Seeds: 1337/1338/1339. Reports mean±std val/heldout F1 and mean pairwise
prototype similarity at convergence.

Output: outputs/repulsion_ablation/results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import extract_features_multicrop_gpu, scatter_mean_by_image
from src.common.metrics import per_sample_f1
from src.common.tiling import TileConfig
from src.simplex_unmixing.model import ModelConfig, UnmixerModel

from baselines.supervised_multilabel_heldout import (
    DEFAULT_HELDOUT_COUNTS, parse_heldout_counts, select_heldout,
)
from tools.build_splits import discover_class_names
from experiments.run_phoebi_heldout import (
    collect_entries, image_level_90_10, split_lists,
    hybrid_prototype_init, quantile_thresholds,
)


def aggregate(per_tile, image_index, n):
    return scatter_mean_by_image(per_tile, image_index, n).cpu().numpy()


def mean_pairwise_sim(prototypes: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity of prototype matrix."""
    p = F.normalize(prototypes, p=2, dim=1)
    gram = torch.mm(p, p.t())
    K = p.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=p.device)
    return float(gram[mask].mean().item())


def train_with_repulsion(
    train_feats, train_idx, val_feats, val_idx, test_feats, test_idx,
    init_protos, val_labels, test_labels, n_val, n_test,
    K, D, epochs=30, lr=1e-3, batch_size=4096, device="cuda",
    repulsion_lambda=0.0, tau_repul=0.1, class_names=None,
):
    cfg = ModelConfig(embedding_dim=D, num_prototypes=K, temperature=10.0)
    model = UnmixerModel(cfg).to(device)
    with torch.no_grad():
        model.prototypes.copy_(init_protos.to(device))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    feats_dev = train_feats.to(device)
    n_tiles = feats_dev.shape[0]

    off_diag = ~torch.eye(K, dtype=torch.bool, device=device)

    for ep in range(epochs):
        perm = torch.randperm(n_tiles, device=device)
        running_recon = running_repul = 0.0
        for i in range(0, n_tiles, batch_size):
            batch = feats_dev[perm[i:i + batch_size]]
            opt.zero_grad()
            recon, _, residual = model(batch)
            recon_loss = (residual ** 2).sum(dim=1).mean()

            if repulsion_lambda > 0:
                p_norm = F.normalize(model.prototypes, p=2, dim=1)
                gram = torch.mm(p_norm, p_norm.t())
                repulsion = (gram[off_diag] / tau_repul).exp().mean()
                loss = recon_loss + repulsion_lambda * repulsion
                running_repul += repulsion.item() * batch.shape[0]
            else:
                loss = recon_loss

            loss.backward()
            opt.step()
            running_recon += recon_loss.item() * batch.shape[0]

        if (ep + 1) % 10 == 0:
            msg = f"    ep{ep+1:02d}/{epochs}  recon={running_recon/n_tiles:.5f}"
            if repulsion_lambda > 0:
                msg += f"  repul={running_repul/n_tiles:.5f}"
            print(msg, flush=True)

    proto_sim = mean_pairwise_sim(model.prototypes)

    model.eval()
    with torch.no_grad():
        _, val_w, _ = model(val_feats.to(device))
        _, test_w, _ = model(test_feats.to(device))
    val_w = aggregate(val_w.cpu(), val_idx, n_val)
    test_w = aggregate(test_w.cpu(), test_idx, n_test)

    thr = quantile_thresholds(val_w, val_labels, quantile=0.05)
    val_f1 = float(per_sample_f1(val_labels, (val_w > thr).astype(np.int64)))
    test_f1 = float(per_sample_f1(test_labels, (test_w > thr).astype(np.int64)))
    return val_f1, test_f1, proto_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/images")
    ap.add_argument("--output_dir", default="outputs/repulsion_ablation")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--tile_size", type=int, default=224)
    ap.add_argument("--eval_grid_size", type=int, default=4)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="1337,1338,1339")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lambdas", type=str, default="0,0.01,0.1,1.0",
                    help="Comma-separated repulsion λ values to sweep")
    ap.add_argument("--tau_repul", type=float, default=0.1,
                    help="Temperature for repulsion Gram matrix (not the model's τ=10)")
    ap.add_argument("--heldout_counts", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats_dir = out_dir / "features"
    feats_dir.mkdir(exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    lambdas = [float(l) for l in args.lambdas.split(",")]

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
    print(f"λ sweep: {lambdas}  τ_repul={args.tau_repul}", flush=True)

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

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}", flush=True)
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

        for lam in lambdas:
            print(f"\n  λ={lam}:", flush=True)
            val_f1, test_f1, proto_sim = train_with_repulsion(
                train_feats, train_idx, val_feats, val_idx,
                test_feats, test_idx, init_protos,
                val_labels, test_labels, n_val, n_test,
                K, D, epochs=args.epochs, device=device,
                repulsion_lambda=lam, tau_repul=args.tau_repul,
                class_names=class_names,
            )
            print(f"  val F1={val_f1:.4f}  heldout F1={test_f1:.4f}  "
                  f"proto_sim={proto_sim:.4f}", flush=True)
            all_results.append({
                "seed": seed, "lambda": lam,
                "val_f1": val_f1, "test_f1": test_f1,
                "delta_f1": val_f1 - test_f1,
                "mean_proto_sim": proto_sim,
            })

    # Aggregate across seeds per lambda
    summary = {}
    for lam in lambdas:
        rows = [r for r in all_results if r["lambda"] == lam]
        summary[str(lam)] = {
            "lambda": lam,
            "val_f1_mean": float(np.mean([r["val_f1"] for r in rows])),
            "val_f1_std": float(np.std([r["val_f1"] for r in rows])),
            "test_f1_mean": float(np.mean([r["test_f1"] for r in rows])),
            "test_f1_std": float(np.std([r["test_f1"] for r in rows])),
            "delta_mean": float(np.mean([r["delta_f1"] for r in rows])),
            "delta_std": float(np.std([r["delta_f1"] for r in rows])),
            "proto_sim_mean": float(np.mean([r["mean_proto_sim"] for r in rows])),
        }

    output = {
        "heldout_combos": heldout, "class_names": class_names,
        "seeds": seeds, "lambdas": lambdas, "tau_repul": args.tau_repul,
        "per_seed": all_results, "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2))

    print("\n\n=== Repulsion λ Ablation ===")
    print(f"{'λ':<8} {'Val F1':>14} {'Heldout F1':>14} {'Δ F1':>12} {'Proto sim':>10}")
    for lam in lambdas:
        s = summary[str(lam)]
        print(f"{lam:<8} {s['val_f1_mean']:.3f}±{s['val_f1_std']:.3f} "
              f"{s['test_f1_mean']:.3f}±{s['test_f1_std']:.3f} "
              f"{s['delta_mean']:+.3f}±{s['delta_std']:.3f} "
              f"{s['proto_sim_mean']:.4f}")
    print(f"\nResults: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
