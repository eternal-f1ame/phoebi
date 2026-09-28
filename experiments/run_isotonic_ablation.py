#!/usr/bin/env python3
"""Isotonic recalibration ablation for Method B.

Claim being tested (paper §5.isotonic): fitting a per-class isotonic regressor
on validation-split similarities improves per-sample F1 by ~0.5 pp without
changing AUROC (isotonic is monotone, so rank order is preserved).

Pipeline:
  1. Load cached val + test tile features from outputs/prototype_matching/6class/
  2. Mean-pool tiles to image-level similarities s_k(x)
  3. For each class k, fit IsotonicRegression on (s_k(val), y_k(val))
  4. Apply calibrator to s_k(test) -> calibrated probabilities p_k(test)
  5. Pick per-class threshold on CALIBRATED val probs by argmax F1 over
     [0.01, 0.99]
  6. Score test with both (a) baseline thresholds on raw similarities,
     (b) argmax-F1 thresholds on raw similarities (fair comparison), and
     (c) argmax-F1 thresholds on isotonic-calibrated probs.
  7. Report per-sample F1, macro F1, exact match for all three; also AUROC
     per class (should be identical between raw and calibrated).

Writes outputs/isotonic_ablation/{results.json,summary.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import scatter_mean_by_image
from src.common.io import load_real_split
from src.common.metrics import per_sample_f1, macro_f1_per_class, exact_match_accuracy
from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig


def image_level_similarities(
    tile_feats: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
    model: PrototypeMatchingModel,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        sims_tile, _ = model(tile_feats.to(device))
    sims_img = scatter_mean_by_image(sims_tile.cpu(), image_index, num_images)
    return sims_img.numpy()


def per_class_argmax_f1_threshold(
    scores: np.ndarray, y_true: np.ndarray, grid: np.ndarray,
) -> np.ndarray:
    """scores, y_true: (N, K). Returns per-class threshold that maximizes F1."""
    K = scores.shape[1]
    thr = np.zeros(K)
    for k in range(K):
        best_f1 = -1.0
        for t in grid:
            pred = (scores[:, k] > t).astype(np.int64)
            tp = int(((pred == 1) & (y_true[:, k] == 1)).sum())
            fp = int(((pred == 1) & (y_true[:, k] == 0)).sum())
            fn = int(((pred == 0) & (y_true[:, k] == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                thr[k] = float(t)
    return thr


def score_with_thresholds(
    scores: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray,
    class_names: List[str],
) -> Dict:
    pred = (scores > thresholds).astype(np.int64)
    return {
        "per_sample_f1": float(per_sample_f1(y_true, pred)),
        "macro_f1": {k: float(v) for k, v in
                     macro_f1_per_class(y_true, pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(y_true, pred)),
        "thresholds": thresholds.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto_dir", type=str,
                    default="outputs/prototype_matching/6class")
    ap.add_argument("--splits_path", type=str,
                    default="data/splits.json")
    ap.add_argument("--output_dir", type=str,
                    default="outputs/isotonic_ablation")
    args = ap.parse_args()

    proto_dir = ROOT / args.proto_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- config ----
    cfg = json.loads((proto_dir / "config.json").read_text())
    class_names = cfg["class_names"]
    K = len(class_names)
    baseline_thresh = np.asarray(cfg["thresholds"])

    # ---- load model ----
    mcfg = ProtoConfig(
        embedding_dim=cfg["embedding_dim"],
        num_prototypes=cfg["num_prototypes"],
    )
    model = PrototypeMatchingModel(mcfg).to(device)
    model.load_state_dict(torch.load(proto_dir / "proto_model.pt",
                                     map_location=device, weights_only=True))

    # ---- cached tile features ----
    val_cache = torch.load(proto_dir / "val_features_cache.pt", weights_only=False,
                           map_location="cpu")
    train_val_paths_val, val_labels, _, _ = load_real_split(str(ROOT / args.splits_path),
                                                            "val")
    n_val = len(train_val_paths_val)
    val_sims = image_level_similarities(
        val_cache["features"], val_cache["image_index"], n_val, model, device,
    )

    # For test, we already have presence_test cached; but reload from proto_dir
    # rather than duplicating extraction. The presence-test run saved a cache at
    # proto_dir/presence_test/test_features_cache.pt.
    test_cache_path = proto_dir / "presence_test" / "test_features_cache.pt"
    if not test_cache_path.exists():
        # Fall back to running full test extraction — out of scope here.
        raise SystemExit(
            f"Missing {test_cache_path}. Run `python experiments/run_presence_detection.py "
            f"--method prototype --model_dir {proto_dir}` first."
        )
    test_cache = torch.load(test_cache_path, weights_only=False, map_location="cpu")
    test_paths, test_labels, _, _ = load_real_split(str(ROOT / args.splits_path),
                                                    "test")
    n_test = len(test_paths)
    test_sims = image_level_similarities(
        test_cache["features"], test_cache["image_index"], n_test, model, device,
    )

    # ---- AUROC should be identical for raw vs calibrated (monotone transform) ----
    raw_auroc = {}
    for k in range(K):
        if val_labels[:, k].sum() in (0, n_val):
            raw_auroc[class_names[k]] = float("nan")
            continue
        raw_auroc[class_names[k]] = float(roc_auc_score(
            test_labels[:, k], test_sims[:, k],
        )) if test_labels[:, k].sum() not in (0, n_test) else float("nan")

    # ---- (a) baseline: config["thresholds"] on raw similarities ----
    res_a = score_with_thresholds(test_sims, test_labels, baseline_thresh, class_names)

    # ---- (b) fair: argmax-F1 val thresholds on raw similarities ----
    grid = np.linspace(0.01, 0.99, 99)
    raw_thr = per_class_argmax_f1_threshold(val_sims, val_labels, grid)
    res_b = score_with_thresholds(test_sims, test_labels, raw_thr, class_names)

    # ---- (c) isotonic: fit per-class IsotonicRegression on val ----
    val_calibrated = np.zeros_like(val_sims)
    test_calibrated = np.zeros_like(test_sims)
    calibrators = []
    for k in range(K):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(val_sims[:, k], val_labels[:, k].astype(np.float64))
        val_calibrated[:, k] = ir.transform(val_sims[:, k])
        test_calibrated[:, k] = ir.transform(test_sims[:, k])
        calibrators.append(ir)

    iso_thr = per_class_argmax_f1_threshold(val_calibrated, val_labels, grid)
    res_c = score_with_thresholds(test_calibrated, test_labels, iso_thr, class_names)

    # ---- AUROC on calibrated (must equal raw) ----
    iso_auroc = {}
    for k in range(K):
        if test_labels[:, k].sum() in (0, n_test):
            iso_auroc[class_names[k]] = float("nan")
            continue
        iso_auroc[class_names[k]] = float(roc_auc_score(
            test_labels[:, k], test_calibrated[:, k],
        ))

    # ---- aggregate ----
    results = {
        "class_names": class_names,
        "n_val": int(n_val),
        "n_test": int(n_test),
        "raw_per_class_auroc": raw_auroc,
        "iso_per_class_auroc": iso_auroc,
        "A_config_thresholds_on_raw":         res_a,
        "B_argmax_f1_thresholds_on_raw":      res_b,
        "C_argmax_f1_thresholds_on_isotonic": res_c,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    md = []
    md.append("# Isotonic recalibration ablation (Method B)\n")
    md.append(f"Val images: {n_val}, Test images: {n_test}\n")
    md.append("| Variant | Per-sample F1 | Macro F1 | Exact match |")
    md.append("|---------|---------------|----------|-------------|")
    for name, res in [("A. Config (q=0.05 val) thresholds on raw sims", res_a),
                      ("B. Argmax-F1 thresholds on raw sims", res_b),
                      ("C. Argmax-F1 thresholds on isotonic-calibrated", res_c)]:
        md.append(f"| {name} | {res['per_sample_f1']:.4f} | "
                  f"{res['macro_f1']['macro']:.4f} | {res['exact_match']:.4f} |")
    md.append("")
    md.append("Delta (C − A): per-sample F1 = "
              f"{100 * (res_c['per_sample_f1'] - res_a['per_sample_f1']):+.2f} pp")
    md.append("Delta (C − B): per-sample F1 = "
              f"{100 * (res_c['per_sample_f1'] - res_b['per_sample_f1']):+.2f} pp "
              "(threshold-protocol-matched comparison)")
    md.append("")
    md.append("Per-class AUROC comparison (raw vs isotonic — should be identical):")
    md.append("| Class | Raw AUROC | Isotonic AUROC |")
    md.append("|-------|-----------|----------------|")
    for c in class_names:
        md.append(f"| {c} | {raw_auroc.get(c, float('nan')):.4f} | "
                  f"{iso_auroc.get(c, float('nan')):.4f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print("\n=== Isotonic recalibration ablation ===")
    print(f"  A. baseline (config q=0.05 thresholds on raw):    "
          f"F1 = {res_a['per_sample_f1']:.4f}")
    print(f"  B. argmax-F1 thresholds on raw:                   "
          f"F1 = {res_b['per_sample_f1']:.4f}")
    print(f"  C. argmax-F1 thresholds on isotonic-calibrated:   "
          f"F1 = {res_c['per_sample_f1']:.4f}")
    print(f"  Delta (C - A):  {100*(res_c['per_sample_f1']-res_a['per_sample_f1']):+.2f} pp")
    print(f"  Delta (C - B):  {100*(res_c['per_sample_f1']-res_b['per_sample_f1']):+.2f} pp")
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
