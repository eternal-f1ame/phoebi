#!/usr/bin/env python3
"""Per-combination-order F1 breakdown for the LCO held-out regime.

Aggregates per_heldout_combo F1 across seeds 1337/1338/1339 and prints a
table by combination order (1,2,3,4,6). Writes
outputs/phoebi_heldout_per_order.json for downstream figure rendering.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED_DIRS = [
    "outputs/phoebi_heldout",
    "outputs/phoebi_heldout_seed1338",
    "outputs/phoebi_heldout_seed1339",
]
METHODS = ["A_simplex", "B_proto", "C_channel"]
METHOD_LABELS = {"A_simplex": "A (simplex)", "B_proto": "B (proto)", "C_channel": "C (channel)"}


def combo_order(combo: str) -> int:
    return len(combo.split("_"))


def main() -> None:
    # Collect per-combo F1 per seed, keyed by (method, combo)
    # Each seed has its own 9-combo selection; we average same-order combos
    per_seed: list[dict] = []
    for sd in SEED_DIRS:
        p = Path(sd) / "results.json"
        if not p.exists():
            print(f"  SKIP {sd}: no results.json")
            continue
        d = json.loads(p.read_text())
        per_seed.append(d)

    print(f"Loaded {len(per_seed)} seed results\n")

    # For each seed, group per-combo F1 by order
    # Structure: order -> method -> list of F1 values (one per combo across seeds)
    order_f1: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in per_seed:
        for method in METHODS:
            combos = d["methods"][method].get("per_heldout_combo", {})
            for combo, vals in combos.items():
                order = combo_order(combo)
                order_f1[order][method].append(vals["per_sample_f1"])

    # Print table
    orders = sorted(order_f1.keys())
    print(f"{'Order':<6} {'Method A':<14} {'Method B':<14} {'Method C':<14}")
    print("-" * 50)
    out = {}
    for order in orders:
        row = {"order": order}
        parts = []
        for method in METHODS:
            vals = np.array(order_f1[order][method])
            mean, std = float(vals.mean()), float(vals.std())
            row[method] = {"mean": mean, "std": std, "n": len(vals)}
            parts.append(f"{mean:.3f}±{std:.3f}")
        print(f"{order:<6} {parts[0]:<14} {parts[1]:<14} {parts[2]:<14}")
        out[order] = row

    # Write JSON
    out_path = Path("outputs/phoebi_heldout_per_order.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
