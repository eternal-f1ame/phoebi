"""Metrics for per-image score matrices: thresholded (the paper's) and threshold-free.

Thresholds follow the paper: the 5th percentile of each class's score over positive
images of the calibration split. Which split calibrates is a parameter, because Stage 3
is precisely about that choice. AUROC is per class over images, the constant predictor
scores 0.5, and it is the number that exposed the problem in the first place.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from src.common.metrics import exact_match_accuracy, macro_f1_per_class, per_sample_f1


def quantile_thresholds(scores: np.ndarray, labels: np.ndarray, q: float = 0.05) -> np.ndarray:
    K = scores.shape[1]
    out = np.zeros(K)
    for k in range(K):
        pos = scores[labels[:, k] == 1, k]
        out[k] = float(np.quantile(pos, q)) if len(pos) else float(scores[:, k].max() + 1.0)
    return out


def argmax_f1_thresholds(scores: np.ndarray, labels: np.ndarray, n_grid: int = 199) -> np.ndarray:
    K = scores.shape[1]
    out = np.zeros(K)
    for k in range(K):
        lo, hi = float(scores[:, k].min()), float(scores[:, k].max())
        grid = np.linspace(lo, hi, n_grid)[:-1]
        best, best_t = -1.0, lo
        y = labels[:, k]
        for t in grid:
            p = scores[:, k] > t
            tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int(((~p) & (y == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom else 0.0
            if f1 > best:
                best, best_t = f1, float(t)
        out[k] = best_t
    return out


def per_class_auroc(scores: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    out = {}
    vals = []
    for k, name in enumerate(class_names):
        y = labels[:, k]
        if y.min() == y.max():
            out[name] = float("nan")
            continue
        a = float(roc_auc_score(y, scores[:, k]))
        out[name] = a
        vals.append(a)
    out["mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def thresholded_metrics(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray,
                        class_names: List[str], combos: Optional[List[str]] = None) -> Dict:
    pred = (scores > thresholds).astype(np.int64)
    out = {
        "per_sample_f1": float(per_sample_f1(labels, pred)),
        "macro_f1": {k: float(v) for k, v in macro_f1_per_class(labels, pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(labels, pred)),
        "mean_predicted": float(pred.sum(axis=1).mean()),
        "mean_true": float(labels.sum(axis=1).mean()),
        "fpr": {}, "tpr": {},
    }
    for k, name in enumerate(class_names):
        neg = labels[:, k] == 0; pos = ~neg
        out["fpr"][name] = float(pred[neg, k].mean()) if neg.any() else float("nan")
        out["tpr"][name] = float(pred[pos, k].mean()) if pos.any() else float("nan")
    if combos is not None:
        arr = np.array(combos); per = {}
        for c in sorted(set(combos)):
            m = arr == c
            per[c] = {"n": int(m.sum()), "per_sample_f1": float(per_sample_f1(labels[m], pred[m]))}
        out["per_combo"] = per
    return out


def constant_predictor(labels: np.ndarray, class_names: List[str]) -> Dict:
    ones = np.ones_like(labels)
    return {
        "per_sample_f1": float(per_sample_f1(labels, ones)),
        "macro_f1": float(macro_f1_per_class(labels, ones, class_names)["macro"]),
        "exact_match": float(exact_match_accuracy(labels, ones)),
        "auroc_mean": 0.5,
    }


def evaluate_split(scores: np.ndarray, labels: np.ndarray, class_names: List[str],
                   thresholds: np.ndarray, combos: Optional[List[str]] = None) -> Dict:
    m = thresholded_metrics(scores, labels, thresholds, class_names, combos)
    m["auroc"] = per_class_auroc(scores, labels, class_names)
    return m
