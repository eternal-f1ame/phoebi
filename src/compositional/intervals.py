"""Combination-level uncertainty for held-out metrics.

Labels are constant within a combination, so the unit of replication for any held-out
per-species metric is the combination, not the image. Every function here resamples
combinations with replacement and recomputes the metric from the pooled per-image
predictions of the resampled combinations.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def rank_auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; nan if one class is absent."""
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # vectorised average ranks (ties share the mean of their rank range)
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    _, inverse, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    last = np.cumsum(counts).astype(np.float64)
    first = last - counts + 1.0
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ((first + last) / 2.0)[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_sample_f1(Y: np.ndarray, P: np.ndarray) -> float:
    Y = Y.astype(bool); P = P.astype(bool)
    tp = (Y & P).sum(1); d = 2 * tp + ((~Y) & P).sum(1) + (Y & (~P)).sum(1)
    return float(np.where(d > 0, 2 * tp / np.maximum(d, 1), 1.0).mean())


def macro_f1(Y: np.ndarray, P: np.ndarray) -> float:
    Y = Y.astype(bool); P = P.astype(bool); vals = []
    for k in range(Y.shape[1]):
        tp = (Y[:, k] & P[:, k]).sum(); fp = ((~Y[:, k]) & P[:, k]).sum(); fn = (Y[:, k] & (~P[:, k])).sum()
        d = 2 * tp + fp + fn
        vals.append(2 * tp / d if d else 0.0)
    return float(np.mean(vals))


def heldout_report(scores: np.ndarray, labels: np.ndarray, combos: Sequence[str], thresholds: np.ndarray,
                   class_names: List[str], n_boot: int = 2000, seed: int = 0,
                   groups: Optional[Sequence[str]] = None) -> Dict:
    """Point estimates plus 95 % combination-level bootstrap intervals.

    `groups` overrides the resampling unit (e.g. "seed:combo" when pooling partitions).
    Returns per-sample F1, macro F1, both margins over the constant all-present predictor,
    mean per-species AUROC and each species' AUROC with a `resolved` flag (interval
    excludes 0.5).
    """
    rng = np.random.default_rng(seed)
    Y = labels.astype(bool); P = scores > thresholds; ones = np.ones_like(Y)
    G = np.array(list(groups if groups is not None else combos))
    units = sorted(set(G)); idx = {g: np.where(G == g)[0] for g in units}
    K = Y.shape[1]

    def stats(ii):
        y, s, p = Y[ii], scores[ii], P[ii]
        a = [rank_auroc(s[:, k], y[:, k]) for k in range(K)]
        return {"f1": per_sample_f1(y, p), "f1_margin": per_sample_f1(y, p) - per_sample_f1(y, ones[ii]),
                "macro": macro_f1(y, p), "macro_margin": macro_f1(y, p) - macro_f1(y, ones[ii]),
                "auroc_mean": float(np.nanmean(a)), "auroc": a}

    point = stats(np.arange(len(Y)))
    boots = {k: [] for k in ("f1", "f1_margin", "macro", "macro_margin", "auroc_mean")}
    per = [[] for _ in range(K)]
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        b = stats(ii)
        for k in boots:
            boots[k].append(b[k])
        for k in range(K):
            if b["auroc"][k] == b["auroc"][k]:
                per[k].append(b["auroc"][k])
    ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [float("nan")] * 2
    out = {k: {"point": point[k], "ci95": ci(boots[k])} for k in boots}
    out["n_units"] = len(units)
    out["per_species"] = {}
    for k, name in enumerate(class_names):
        lo, hi = ci(per[k])
        out["per_species"][name] = {"auroc": point["auroc"][k], "ci95": [lo, hi],
                                    "resolved": bool(lo == lo and (lo > 0.5 or hi < 0.5))}
    return out


def format_report(r: Dict, class_names: List[str]) -> str:
    f = lambda d: f"{d['point']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    lines = [f"units (combinations) = {r['n_units']}",
             f"per-sample F1 {f(r['f1'])}   margin over constant {f(r['f1_margin'])}",
             f"macro F1      {f(r['macro'])}   margin over constant {f(r['macro_margin'])}",
             f"mean AUROC    {f(r['auroc_mean'])}"]
    for name in class_names:
        p = r["per_species"][name]
        lines.append(f"  {name:3s} AUROC {p['auroc']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]{'  resolved' if p['resolved'] else ''}")
    return "\n".join(lines)
