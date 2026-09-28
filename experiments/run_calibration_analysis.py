#!/usr/bin/env python3
"""Calibration and per-species confusion analysis for held-out test set.

Reads score arrays from outputs/phoebi_heldout/ and computes:
1. Per-species reliability statistics (calibration) for Methods A, B, C
2. Per-species FPR/FNR at the calibrated threshold (confusion breakdown)

Writes outputs/phoebi_heldout_calibration.json and prints a summary table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCORE_DIR = Path("outputs/phoebi_heldout")
RESULTS_JSON = SCORE_DIR / "results.json"


def reliability_stats(scores_k: np.ndarray, labels_k: np.ndarray, n_bins: int = 10):
    """Fraction of positives in each equal-count bin of scores."""
    idx = np.argsort(scores_k)
    bins = np.array_split(idx, n_bins)
    bin_means, bin_fracs = [], []
    for b in bins:
        if len(b) == 0:
            continue
        bin_means.append(float(scores_k[b].mean()))
        bin_fracs.append(float(labels_k[b].mean()))
    return bin_means, bin_fracs


def main() -> None:
    d = json.loads(RESULTS_JSON.read_text())
    class_names = d["class_names"]

    A = np.load(SCORE_DIR / "method_a_test_scores.npy")
    B = np.load(SCORE_DIR / "method_b_test_scores.npy")
    C = np.load(SCORE_DIR / "method_c_test_scores.npy")
    lbl = np.load(SCORE_DIR / "test_labels.npy")

    a_thresh = np.array(d["methods"]["A_simplex"]["thresholds"])
    b_thresh = np.array(d["methods"]["B_proto"]["thresholds"])
    c_thresh = np.array(d["methods"]["C_channel"]["thresholds"])

    results = {}

    # Per-species FPR/FNR at calibrated threshold
    print("\nPer-species FPR / FNR at calibrated threshold (held-out test set)")
    print(f"{'Species':<6} {'Method':<4} {'Thresh':>7} {'FPR':>7} {'FNR':>7} {'Support':>8}")
    print("-" * 50)

    for method_name, scores, thresholds in [
        ("A", A, a_thresh), ("B", B, b_thresh), ("C", C, c_thresh)
    ]:
        results[method_name] = {}
        for k, sp in enumerate(class_names):
            preds = (scores[:, k] > thresholds[k]).astype(int)
            pos_mask = lbl[:, k] == 1
            neg_mask = lbl[:, k] == 0
            fpr = float(preds[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
            fnr = float((1 - preds[pos_mask]).mean()) if pos_mask.sum() > 0 else 0.0
            support_pos = int(pos_mask.sum())
            results[method_name][sp] = {
                "threshold": float(thresholds[k]),
                "fpr": fpr, "fnr": fnr,
                "n_pos": support_pos, "n_neg": int(neg_mask.sum()),
            }
            print(f"{sp:<6} {method_name:<4} {thresholds[k]:>7.3f} {fpr:>7.3f} {fnr:>7.3f} {support_pos:>8d}")
        print()

    # Calibration: rank correlation between predicted score and label
    print("\nMean score for positive vs negative examples (held-out test set)")
    print(f"{'Method':<4} {'Species':<6} {'Mean(pos)':>10} {'Mean(neg)':>10} {'Separation':>12}")
    print("-" * 50)
    for method_name, scores in [("A", A), ("B", B), ("C", C)]:
        for k, sp in enumerate(class_names):
            pos_mask = lbl[:, k] == 1
            neg_mask = lbl[:, k] == 0
            mean_pos = float(scores[pos_mask, k].mean()) if pos_mask.sum() > 0 else 0.0
            mean_neg = float(scores[neg_mask, k].mean()) if neg_mask.sum() > 0 else 0.0
            sep = mean_pos - mean_neg
            print(f"{method_name:<4} {sp:<6} {mean_pos:>10.4f} {mean_neg:>10.4f} {sep:>12.4f}")
        print()

    # Save
    out = {"class_names": class_names, "per_method_per_species": results}
    out_path = Path("outputs/phoebi_heldout_calibration.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
