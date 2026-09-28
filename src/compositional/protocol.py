"""The LCO partition, plus a nested development split for model selection.

The partition is read from `data/splits_lco.json`, which `tools/build_lco_splits.py`
produced with the same `select_heldout(seed=1337)` / `image_level_90_10` calls as
`experiments/run_phoebi_heldout.py`, so train/val/test here are the paper's exactly:
83,700 / 9,300 / 27,000 images, nine held-out combinations as test.

The nested split addresses a selection problem the paper did not have to face: the
validation split is in-distribution, and the whole point of these experiments is that
in-distribution numbers do not predict held-out ones. So a few *trained-on* combinations
are held out again, deterministically, as a development set. Anything selected in this
package (granularity, subspace dimension, thresholds) is selected on that development set,
never on the nine test combinations.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from tools.build_splits import parse_label_tokens

DEFAULT_DEV_COUNTS = {2: 1, 3: 2, 4: 1}     # nested dev: one pair, two triples, one quadruple


@dataclass
class SplitData:
    paths: List[str]
    labels: np.ndarray                  # (N, K) int64
    combos: List[str]
    class_names: List[str]

    def __len__(self) -> int:
        return len(self.paths)

    def subset(self, mask: np.ndarray) -> "SplitData":
        idx = np.where(mask)[0]
        return SplitData([self.paths[i] for i in idx], self.labels[idx],
                         [self.combos[i] for i in idx], self.class_names)

    def take(self, n: int) -> "SplitData":
        return self.subset(np.arange(len(self)) < n)


@dataclass
class LCOProtocol:
    class_names: List[str]
    heldout_combos: List[str]
    trained_combos: List[str]
    dev_combos: List[str]
    train: SplitData        # trained combos minus dev combos, image-level 90 %
    val: SplitData          # trained combos minus dev combos, image-level 10 %
    dev: SplitData          # dev combos, all their train+val images
    test: SplitData         # the nine held-out combinations
    train_full: SplitData   # the paper's train split, dev combos included (for the anchor)
    extra: Dict = field(default_factory=dict)

    def describe(self) -> str:
        return (f"K={len(self.class_names)} {self.class_names}\n"
                f"held-out ({len(self.heldout_combos)}): {self.heldout_combos}\n"
                f"nested dev ({len(self.dev_combos)}): {self.dev_combos}\n"
                f"train={len(self.train)} val={len(self.val)} dev={len(self.dev)} "
                f"test={len(self.test)} (paper train={len(self.train_full)})")


def _load_split(d: dict, name: str, class_names: List[str]) -> SplitData:
    entries = d["splits"][name]
    paths = [e["path"] for e in entries]
    labels = np.array([e["label"] for e in entries], dtype=np.int64)
    combos = [e.get("combo", e.get("video")) for e in entries]
    return SplitData(paths, labels, combos, class_names)


def select_dev_combos(trained: Sequence[str], class_names: Sequence[str], seed: int,
                      counts: Dict[int, int] = DEFAULT_DEV_COUNTS) -> List[str]:
    """Mirror of `select_heldout`: pick dev combos by order, re-rolling until every
    species still appears in a remaining trained combination that is not a singleton
    of that species only... i.e. coverage is preserved for the reduced training pool."""
    by_order = defaultdict(list)
    for c in trained:
        by_order[len(parse_label_tokens(c))].append(c)
    attempt = 0
    while True:
        rng = random.Random(seed * 7919 + attempt)
        dev: List[str] = []
        for order, count in counts.items():
            pool = sorted(by_order.get(order, []))
            if len(pool) < count:
                raise SystemExit(f"nested dev needs {count} combos of order {order}, have {len(pool)}")
            rng.shuffle(pool)
            dev.extend(pool[:count])
        remaining = [c for c in trained if c not in set(dev)]
        covered = {t for c in remaining for t in parse_label_tokens(c)}
        if all(k in covered for k in class_names):
            return sorted(dev, key=lambda c: (len(parse_label_tokens(c)), c))
        attempt += 1
        if attempt > 100:
            raise SystemExit("could not pick a nested dev split that preserves species coverage")


def load_protocol(splits_path: str | Path, seed: int = 1337,
                  dev_counts: Dict[int, int] = DEFAULT_DEV_COUNTS,
                  max_images: int | None = None) -> LCOProtocol:
    d = json.loads(Path(splits_path).read_text())
    class_names = list(d["class_names"])
    heldout = list(d.get("heldout_combos") or sorted({c for c in _load_split(d, "test", class_names).combos}))
    train_full = _load_split(d, "train", class_names)
    val_full = _load_split(d, "val", class_names)
    test = _load_split(d, "test", class_names)
    trained = list(d.get("trained_combos") or sorted(set(train_full.combos)))
    dev_combos = select_dev_combos(trained, class_names, seed, dev_counts)
    dev_set = set(dev_combos)
    tr_mask = np.array([c not in dev_set for c in train_full.combos])
    va_mask = np.array([c not in dev_set for c in val_full.combos])
    train = train_full.subset(tr_mask)
    val = val_full.subset(va_mask)
    dev_parts = [train_full.subset(~tr_mask), val_full.subset(~va_mask)]
    dev = SplitData(dev_parts[0].paths + dev_parts[1].paths,
                    np.concatenate([dev_parts[0].labels, dev_parts[1].labels], axis=0),
                    dev_parts[0].combos + dev_parts[1].combos, class_names)
    proto = LCOProtocol(class_names, heldout, trained, dev_combos, train, val, dev, test,
                        train_full, extra={"seed": seed, "splits_path": str(splits_path)})
    if max_images is not None:
        # smoke-test mode: a per-combination cap so every combo stays represented
        def cap(s: SplitData) -> SplitData:
            keep = np.zeros(len(s), dtype=bool); seen = defaultdict(int)
            for i, c in enumerate(s.combos):
                if seen[c] < max_images:
                    keep[i] = True; seen[c] += 1
            return s.subset(keep)
        proto.train, proto.val, proto.dev, proto.test, proto.train_full = (
            cap(train), cap(val), cap(dev), cap(test), cap(train_full))
    return proto


def image_order(labels: np.ndarray) -> np.ndarray:
    return labels.sum(axis=1)
