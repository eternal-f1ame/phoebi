#!/usr/bin/env python3
"""Build the standard development folds for the PHOEBI leave-combinations-out protocol.

Labels are constant within a combination, so any held-out per-species metric compares
whole combinations and its effective sample size is the number of held-out combinations.
The nine test combinations are for reporting, once. Model selection, calibration and any
species-level claim made during development therefore need their own held-out
combinations, and enough of them: this tool partitions the trained-on combinations of
order >= 2 of an LCO partition into F folds so that each is held out exactly once with a
spread of orders, while singletons stay in every fold's training pool. The partition's
own held-out singleton (bt at seed 1337) already leaves one species without a pure
culture in every fold, which is the test partition's condition; `--hold_out_singletons`
reproduces the first (harder, non-standard) design for comparison.

Output (one JSON, copied to the release directory unchanged: it names combinations only):

    {"protocol": "lco_dev_folds", "partition_seed": 1337, "n_folds": 5,
     "heldout_test_combos": [...],                 # never used during development
     "folds": [{"fold": 0, "dev_combos": [...], "train_combos": [...]}, ...],
     "rules": {...}}

Usage
    python tools/build_lco_dev_folds.py                      # data/splits_lco.json -> data/lco_dev_folds.json
    python tools/build_lco_dev_folds.py --n_folds 5 --seed 1337 --release_copy data/release/lco_dev_folds.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_splits import parse_label_tokens


def build_folds(trained: List[str], class_names: List[str], n_folds: int, seed: int,
                hold_out_singletons: bool = False) -> List[Dict]:
    """Partition the trained-on combinations into folds.

    Default: singletons stay in every fold's training pool. The partition's own held-out
    singleton (bt at seed 1337) already leaves one species without a pure culture in every
    fold, which is exactly the test partition's condition; holding out a second singleton
    per fold makes the folds harder than the test they stand in for (ChannelGroup 0.41
    dev AUROC against 0.63 on the pooled test partitions). The old behaviour is kept
    behind `hold_out_singletons` for that comparison.
    """
    by_order: Dict[int, List[str]] = defaultdict(list)
    for c in trained:
        by_order[len(parse_label_tokens(c))].append(c)
    rng = random.Random(seed)
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    singletons = sorted(by_order.get(1, []))
    if hold_out_singletons:
        if len(singletons) != n_folds:
            raise SystemExit(f"need exactly one singleton per fold: {len(singletons)} singletons, {n_folds} folds")
        rng.shuffle(singletons)
        for i, c in enumerate(singletons):
            folds[i].append(c)
    # higher orders: shuffle within order, deal round-robin starting from the fold with
    # the fewest combinations so totals stay balanced
    for order in sorted(o for o in by_order if o > 1):
        pool = sorted(by_order[order]); rng.shuffle(pool)
        for c in pool:
            target = min(range(n_folds), key=lambda f: (len(folds[f]), f))
            folds[target].append(c)
    out = []
    for f, dev in enumerate(folds):
        dev_set = set(dev)
        train = [c for c in trained if c not in dev_set]
        covered = {t for c in train for t in parse_label_tokens(c)}
        missing = [k for k in class_names if k not in covered]
        if missing:
            raise SystemExit(f"fold {f}: training pool misses {missing}")
        out.append({"fold": f,
                    "dev_combos": sorted(dev, key=lambda c: (len(parse_label_tokens(c)), c)),
                    "train_combos": train,
                    "dev_orders": sorted(len(parse_label_tokens(c)) for c in dev),
                    "species_without_pure_culture": [c for c in dev if len(parse_label_tokens(c)) == 1]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits_lco.json")
    ap.add_argument("--output", default="data/lco_dev_folds.json")
    ap.add_argument("--release_copy", default="data/release/lco_dev_folds.json")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--hold_out_singletons", action="store_true",
                    help="also hold out one singleton per fold (harder than the test partition; not the standard)")
    args = ap.parse_args()
    d = json.loads(Path(args.splits_path).read_text())
    class_names = list(d["class_names"])
    trained = list(d["trained_combos"]); heldout = list(d["heldout_combos"])
    folds = build_folds(trained, class_names, args.n_folds, args.seed, args.hold_out_singletons)
    payload = {
        "protocol": "lco_dev_folds",
        "partition_seed": int(d.get("seed", args.seed)),
        "fold_seed": args.seed,
        "n_folds": args.n_folds,
        "singletons_held_out": bool(args.hold_out_singletons),
        "species_without_pure_culture_in_every_fold": [k for k in class_names if k not in {c for c in trained if len(parse_label_tokens(c)) == 1}],
        "class_names": class_names,
        "heldout_test_combos": heldout,
        "trained_combos": trained,
        "folds": folds,
        "rules": {
            "train": "the fold's train_combos: their images from the partition's train split",
            "calibrate": "the fold's train_combos: their images from the partition's val split (thresholds, hyper-parameters)",
            "evaluate": "the fold's dev_combos: all their images (train and val splits of the partition)",
            "pool": "concatenate dev predictions over all folds; every trained-on combination of order >= 2 is held out exactly once (singletons stay in training unless singletons_held_out)",
            "intervals": "bootstrap over combinations (2,000 resamples, 95 %) on every pooled metric",
            "select": "primary: pooled mean per-species AUROC; secondary: pooled macro-F1 margin over the constant all-present predictor",
            "report": "the heldout_test_combos once, after selection, with the same intervals; per-species claims only when the interval excludes chance",
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=1))
    if args.release_copy:
        shutil.copyfile(args.output, args.release_copy)
    print(f"singletons held out: {args.hold_out_singletons}")
    print(f"wrote {args.output}" + (f" and {args.release_copy}" if args.release_copy else ""))
    for f in folds:
        print(f"  fold {f['fold']}: {len(f['dev_combos'])} dev combos, orders {f['dev_orders']}, no pure culture for {f['species_without_pure_culture']}: {f['dev_combos']}")


if __name__ == "__main__":
    main()
