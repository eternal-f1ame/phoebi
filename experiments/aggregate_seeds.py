#!/usr/bin/env python3
"""Aggregate val/heldout F1 across multiple seeds for the PHOEBI LCO protocol.

Reads outputs/phoebi_heldout_seed<seed>/results.json (or outputs/phoebi_heldout/results.json
for the canonical seed 1337) and reports mean +/- std per method per metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_seed_results(out_dir: Path):
    p = out_dir / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_dirs", nargs="+", default=[
        "outputs/phoebi_heldout",
        "outputs/phoebi_heldout_seed1338",
        "outputs/phoebi_heldout_seed1339",
    ])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339])
    args = ap.parse_args()

    by_seed = []
    for seed, d in zip(args.seeds, args.seed_dirs):
        res = load_seed_results(Path(d))
        if res is None:
            print(f"SKIP seed {seed}: no results.json at {d}")
            continue
        by_seed.append((seed, res))
    if len(by_seed) < 2:
        print(f"Only {len(by_seed)} seed result(s) found. Need at least 2 for std.")
        sys.exit(1)

    methods = ["A_simplex", "B_proto", "C_channel"]
    metrics = ["val_in_distribution", "test_heldout_combos"]
    sub_metrics = ["per_sample_f1"]

    print(f"\nAggregating across {len(by_seed)} seeds: {[s for s, _ in by_seed]}")
    print()
    print(f"{'Method':<12s} {'Metric':<24s} {'Mean':>8s} {'Std':>8s} {'Per-seed':<30s}")
    print("-" * 80)
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for metric in metrics:
            for sm in sub_metrics:
                vals = []
                for seed, res in by_seed:
                    v = res["methods"].get(method, {}).get(metric, {}).get(sm)
                    if v is not None:
                        vals.append(v)
                if not vals:
                    continue
                vals_np = np.array(vals)
                mean = float(vals_np.mean())
                std = float(vals_np.std())
                agg_key = f"{metric}_{sm}"
                aggregate[method][agg_key] = {
                    "mean": mean, "std": std, "per_seed": vals,
                }
                per_seed_str = ", ".join(f"{v:.4f}" for v in vals)
                print(f"{method:<12s} {metric:<24s} {mean:>8.4f} {std:>8.4f} [{per_seed_str}]")
        # Also delta (val - heldout)
        v_vals, t_vals = [], []
        for seed, res in by_seed:
            v = res["methods"].get(method, {}).get("val_in_distribution", {}).get("per_sample_f1")
            t = res["methods"].get(method, {}).get("test_heldout_combos", {}).get("per_sample_f1")
            if v is not None and t is not None:
                v_vals.append(v)
                t_vals.append(t)
        if v_vals:
            deltas = np.array(v_vals) - np.array(t_vals)
            mean = float(deltas.mean())
            std = float(deltas.std())
            aggregate[method]["delta_f1"] = {
                "mean": mean, "std": std, "per_seed": deltas.tolist(),
            }
            per_seed_str = ", ".join(f"{v:+.4f}" for v in deltas)
            print(f"{method:<12s} {'delta_f1 (val - heldout)':<24s} {mean:>+8.4f} {std:>8.4f} [{per_seed_str}]")
        # Also macro F1 on heldout
        m_vals = []
        for seed, res in by_seed:
            m = res["methods"].get(method, {}).get("test_heldout_combos", {}).get("macro_f1", {}).get("macro")
            if m is not None:
                m_vals.append(m)
        if m_vals:
            m_np = np.array(m_vals)
            aggregate[method]["heldout_macro_f1"] = {
                "mean": float(m_np.mean()), "std": float(m_np.std()),
                "per_seed": m_vals,
            }
            per_seed_str = ", ".join(f"{v:.4f}" for v in m_vals)
            print(f"{method:<12s} {'heldout_macro_f1':<24s} {m_np.mean():>8.4f} {m_np.std():>8.4f} [{per_seed_str}]")
        print("-" * 80)

    out = {
        "seeds": [s for s, _ in by_seed],
        "aggregate": aggregate,
    }
    out_path = Path("outputs/phoebi_heldout_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
