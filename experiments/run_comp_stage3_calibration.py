#!/usr/bin/env python3
"""Stage 3: calibration under shift, on the scores the earlier stages saved.

The paper's thresholds are the 5th percentile of each class's score over positive
*validation* images, and validation is in distribution; under compositional shift those
thresholds slide the decoders toward "everything present". This stage re-thresholds the
saved per-image scores of the selected runs with four rules and reports held-out
behaviour for each:

    val_q05    the paper
    dev_q05    same rule, calibrated on the nested held-out-combination split
    dev_f1     argmax-F1 thresholds on the nested split
    val_f1     argmax-F1 on validation (ChannelGroup's rule)

Calibration cannot create discrimination that AUROC says is not there, which is why it
runs last. No GPU, no extraction: numpy over npz files.

Usage
    python experiments/run_comp_stage3_calibration.py --output_dir outputs/compositional/stage3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compositional.evaluate import (argmax_f1_thresholds, constant_predictor, evaluate_split,
                                        quantile_thresholds)
from src.compositional.protocol import load_protocol


def load(rd: Path) -> Dict[str, Dict[str, np.ndarray]]:
    S = {}
    for s in ("val", "dev", "test"):
        z = np.load(rd / f"scores_{s}.npz"); S[s] = {k.replace("__", "/"): z[k] for k in z.files}
    return S


def candidates(root: Path) -> List[Tuple[str, Path, str]]:
    """(label, run dir, key) for the anchor, Stage 0's best, and every Stage 1/2 best."""
    out = []
    b0 = json.loads((root / "stage0" / "best.json").read_text())
    out.append(("stage0 anchor cls224 cosine/mean", root / "stage0" / "cls224", "cosine/mean"))
    out.append((f"stage0 best {b0['granularity']} {b0['key']}", root / "stage0" / b0["granularity"], b0["key"]))
    for stage in ("stage1", "stage2"):
        bp = root / stage / "best.json"
        if bp.exists():
            b = json.loads(bp.read_text())
            out.append((f"{stage} best {b['run']} {b['key']}", root / stage / b["run"], b["key"]))
            # also every other read-out/agg of the same run
            for key in json.loads((root / stage / "results.json").read_text())["runs"][b["run"]]["keys"]:
                if key != b["key"]:
                    out.append((f"{stage} {b['run']} {key}", root / stage / b["run"], key))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_path", default="data/splits_lco.json")
    ap.add_argument("--output_dir", default="outputs/compositional/stage3")
    ap.add_argument("--root", default=None, help="default: <output_dir>/..")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root) if args.root else out_dir.parent
    P = load_protocol(args.splits_path, seed=args.seed, max_images=args.max_images)
    cn = P.class_names
    rules = {
        "val_q05": lambda S: quantile_thresholds(S["val"], P.val.labels, 0.05),
        "dev_q05": lambda S: quantile_thresholds(S["dev"], P.dev.labels, 0.05),
        "dev_f1": lambda S: argmax_f1_thresholds(S["dev"], P.dev.labels),
        "val_f1": lambda S: argmax_f1_thresholds(S["val"], P.val.labels),
    }
    results: Dict = {"constant": constant_predictor(P.test.labels, cn), "runs": {}}
    rows = []
    for label, rd, key in candidates(root):
        if not (rd / "scores_test.npz").exists():
            continue
        S = load(rd); Sk = {s: S[s][key] for s in S}
        entry = {"key": key, "dir": str(rd), "rules": {}}
        for rule, fn in rules.items():
            thr = fn(Sk)
            entry["rules"][rule] = {
                "thresholds": thr.tolist(),
                "dev": evaluate_split(Sk["dev"], P.dev.labels, cn, thr),
                "test": evaluate_split(Sk["test"], P.test.labels, cn, thr, P.test.combos),
            }
            t = entry["rules"][rule]["test"]
            rows.append((label, rule, t["auroc"]["mean"], t["per_sample_f1"], t["macro_f1"]["macro"],
                         t["exact_match"], t["mean_predicted"], entry["rules"][rule]["dev"]["per_sample_f1"]))
        results["runs"][label] = entry
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    c = results["constant"]
    md = ["# Stage 3: calibration under shift (LCO seed 1337)\n",
          f"Constant all-present predictor on test: F1 {c['per_sample_f1']:.4f}, macro {c['macro_f1']:.4f}, exact {c['exact_match']:.4f}.\n",
          "AUROC is threshold-free and therefore identical across rules for a run; the other columns change with the rule.\n",
          "| run | rule | test AUROC | test F1 | test macro F1 | test exact | pred/img | dev F1 |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.2f} | {r[7]:.4f} |" for r in rows]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
