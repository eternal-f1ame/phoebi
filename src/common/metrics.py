"""Multilabel presence/absence and open-set metrics for PHOEBI.

All metrics are strictly presence/absence — no proportion estimation. The headline
classification metric is per-sample F1 (PlantCLEF style):

    For each sample i with K-dim binary label y_i and prediction p_i,
        TP_i = sum_k 1[y_{i,k}=1 and p_{i,k}=1]
        FP_i = sum_k 1[y_{i,k}=0 and p_{i,k}=1]
        FN_i = sum_k 1[y_{i,k}=1 and p_{i,k}=0]
        F1_i = 2 * TP_i / (2*TP_i + FP_i + FN_i)
    per_sample_f1 = mean over samples of F1_i

per_sample_f1 punishes both missed species and hallucinated species, weighted equally
per-sample regardless of K. Macro per-class F1 is reported alongside as a per-class
diagnostic.

Edge case: a sample with y_i = 0 and p_i = 0 (the model correctly says "nothing here")
has 2*0 / (2*0 + 0 + 0) = 0/0. We define this as F1 = 1.0 (perfect agreement on
emptiness). In bacterial cultures every frame has at least one species so this case
should not arise — but it WILL arise during open-set evaluation when we ask the model
to abstain.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def _as_binary(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.int64)


def per_sample_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-sample F1 over multilabel predictions (PlantCLEF metric).

    Both inputs are (N, K) binary arrays. F1 is computed independently per row and
    averaged. Empty-vs-empty rows score 1.0; empty-vs-nonempty (or vice versa) score 0.
    """
    y_true = _as_binary(y_true)
    y_pred = _as_binary(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1).astype(np.float64)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1).astype(np.float64)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn

    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 1.0)
    return float(f1.mean())


def macro_f1_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> Dict[str, float]:
    """Per-class F1 (computed across samples) plus the macro average.

    Returns {class_name: f1, ..., 'macro': mean_over_classes}. If `class_names` is
    None the keys are 'class_0', 'class_1', ....
    """
    y_true = _as_binary(y_true)
    y_pred = _as_binary(y_pred)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0.0)
    macro = float(np.mean(per_class))

    if class_names is None:
        names = [f"class_{k}" for k in range(len(per_class))]
    else:
        if len(class_names) != len(per_class):
            raise ValueError(
                f"class_names has length {len(class_names)} but predictions have "
                f"{len(per_class)} classes"
            )
        names = list(class_names)

    out = {name: float(per_class[k]) for k, name in enumerate(names)}
    out["macro"] = macro
    return out


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples whose full label vector exactly matches the prediction."""
    y_true = _as_binary(y_true)
    y_pred = _as_binary(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.all(y_true == y_pred, axis=1).mean())


def sparsity_score(weights: np.ndarray, eps: float = 1e-8) -> float:
    """Fraction of weights at exactly zero. Useful for sparsemax diagnostics."""
    return float((np.abs(weights) <= eps).mean())


def open_set_auroc(y_true_unknown: np.ndarray, scores: np.ndarray) -> float:
    """AUROC for unknown detection.

    y_true_unknown : binary, 1 if the sample is from an unknown class.
    scores         : higher = more likely unknown
                     (e.g. residual norm for Method A, 1 - max_sim for Method B).
    """
    return float(roc_auc_score(y_true_unknown, scores))


def open_set_aupr(y_true_unknown: np.ndarray, scores: np.ndarray) -> float:
    """AUPR for unknown detection. Same convention as open_set_auroc."""
    return float(average_precision_score(y_true_unknown, scores))


def fpr_at_tpr(y_true_unknown: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
    """False positive rate at the threshold that achieves at least `tpr_target` TPR.

    "Positive" here means "unknown". A lower FPR@95TPR is better. This is the
    AnimalCLEF / FathomNet operating-point metric.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true_unknown, scores)
    idxs = np.where(tpr >= tpr_target)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])


def combined_metric(f1: float, auroc: float, alpha: float = 0.5) -> float:
    """FathomNet-style combined headline: alpha * F1 + (1 - alpha) * AUROC."""
    return float(alpha * f1 + (1.0 - alpha) * auroc)


# ----- presence F1 from scores ---------------------------------------------------------

def presence_f1(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.05) -> float:
    """Per-sample F1 of `y_scores` thresholded at `threshold`.

    Thresholds the scores and forwards to `per_sample_f1`, which takes
    pre-thresholded predictions.
    """
    y_pred = (np.asarray(y_scores) >= threshold).astype(np.int64)
    return per_sample_f1(y_true, y_pred)
