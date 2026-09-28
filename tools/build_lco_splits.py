#!/usr/bin/env python3
"""Emit the leave-combinations-out (LCO) partition as a standalone splits file.

The LCO protocol is defined by
``baselines.supervised_multilabel_heldout.select_heldout(seed=1337)``, which
requires running code. This writes it as a machine-readable artifact alongside
``splits.json``.

The partition is not re-derived here. This imports the exact functions the
experiment drivers use -- ``select_heldout``, ``collect_entries``,
``image_level_90_10`` -- so the emitted file is the same partition
``experiments/run_phoebi_heldout.py`` evaluates on, by construction rather than by
coincidence.

Protocol (seed 1337): hold out 9 of the 40 combinations -- 1 single, 2 pairs,
3 triples, 2 quadruples, 1 six-species -- re-rolling until every species still
appears in at least one trained-on combination, so the task is compositional
generalization and not novel-class detection. The 31 trained-on combinations are
split 90/10 at image level into train/val. The held-out combinations form the
entire test set.

Output schema matches ``splits.json`` (``train``/``val``/``test`` under
``splits``, per-entry ``combo`` key, ``combo_to_label`` map) so the released
``load_real_split`` reads it unchanged. ``test`` is the held-out-combination set.

Usage:
    # working tree, paths under data/images/
    python tools/build_lco_splits.py --output data/splits_lco.json

    # release layout, paths rewritten to data/images/ as in data/release/splits.json
    python tools/build_lco_splits.py \
        --output data/release/splits_lco.json --path_prefix data/images
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.supervised_multilabel_heldout import (  # noqa: E402
    DEFAULT_HELDOUT_COUNTS,
    collect_entries,
    discover_class_names,
    image_level_90_10,
    parse_label_tokens,
    select_heldout,
)


def _entry(path: Path, label, combo: str, aug_dir: Path, prefix: str) -> dict:
    rel = path.relative_to(aug_dir)
    return {"path": str(Path(prefix) / rel),
            "label": [int(v) for v in label], "combo": combo}


def _default_prefix(aug_dir: Path) -> str:
    """Repo-relative image root, matching how splits.json spells its paths.

    Falls back to the absolute directory only if aug_dir sits outside the repo.
    """
    try:
        return str(aug_dir.resolve().relative_to(ROOT))
    except ValueError:
        return str(aug_dir)


def build(aug_dir: Path, output: Path, seed: int, prefix: str | None) -> dict:
    prefix = prefix or _default_prefix(aug_dir)
    class_names = discover_class_names(aug_dir)
    heldout, trained, _ = select_heldout(
        aug_dir, seed=seed, class_names=class_names,
        heldout_counts=DEFAULT_HELDOUT_COUNTS,
    )

    trained_entries = collect_entries(aug_dir, trained, class_names)
    heldout_entries = collect_entries(aug_dir, heldout, class_names)
    train_entries, val_entries = image_level_90_10(trained_entries, seed=seed)

    splits = {
        name: [_entry(p, l, c, aug_dir, prefix) for p, l, c in entries]
        for name, entries in (
            ("train", train_entries), ("val", val_entries), ("test", heldout_entries)
        )
    }

    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    combo_to_label = {}
    for combo in sorted(set(trained) | set(heldout)):
        vec = [0] * len(class_names)
        for t in parse_label_tokens(combo):
            if t in cls_to_idx:
                vec[cls_to_idx[t]] = 1
        combo_to_label[combo] = vec

    payload = {
        "class_names": class_names,
        "protocol": "leave-combinations-out (LCO)",
        "protocol_note": (
            "'test' is the held-out-combination set: every image of 9 combinations "
            "that appear in neither train nor val. train/val are an image-level 90/10 "
            "split of the remaining 31 combinations. Every species appears in at least "
            "one trained-on combination, so this measures compositional generalization "
            "rather than novel-class detection."
        ),
        "seed": seed,
        "heldout_counts_by_order": {str(k): v for k, v in DEFAULT_HELDOUT_COUNTS.items()},
        "heldout_combos": heldout,
        "trained_combos": trained,
        "splits": splits,
        "combo_to_label": combo_to_label,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    n = {k: len(v) for k, v in splits.items()}
    print(f"Wrote {output}")
    print(f"  species ({len(class_names)}): {class_names}")
    print(f"  held-out ({len(heldout)}): {heldout}")
    print(f"  trained-on: {len(trained)}")
    print(f"  train={n['train']}  val={n['val']}  test(heldout)={n['test']}"
          f"  total={sum(n.values())}")
    print(f"  image paths under: {prefix}/")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug_dir", type=Path, default=ROOT / "data/images")
    ap.add_argument("--output", type=Path, default=ROOT / "data/splits_lco.json")
    ap.add_argument("--seed", type=int, default=1337,
                    help="canonical protocol seed; 1337 reproduces the paper")
    ap.add_argument("--path_prefix", type=str, default=None,
                    help="rewrite image paths under this prefix "
                         "(use 'data/images' to match data/release/splits.json)")
    args = ap.parse_args()
    build(args.aug_dir, args.output, args.seed, args.path_prefix)


if __name__ == "__main__":
    main()
