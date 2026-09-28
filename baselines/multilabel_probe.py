"""Multi-label linear-probe baseline across vision encoders.

For each frozen encoder, extracts tile features via the shared PHOEBI pipeline
(illumination-corrected + 4x4 grid tiling), trains a single `nn.Linear(D, K)`
head with BCE, calibrates per-class thresholds on val, and scores test.

One row per encoder written to ``outputs/encoder_probe/results.csv`` and optionally
rendered to ``outputs/encoder_probe/tab_encoder_probe.tex``.

The lineup is breadth across pretraining objectives, not depth within one family
(see plan: frames are pure shape/texture phase-contrast microscopy, so objective
diversity > model scaling).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.common.features import (
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.io import load_real_split
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import TileConfig


# -----------------------------------------------------------------------------
# Encoder lineup — breadth across pretraining objectives
# -----------------------------------------------------------------------------

ENCODERS: List[Tuple[str, str]] = [
    ("resnet50.a1_in1k",                              "ResNet-50"),
    ("convnext_base.fb_in22k_ft_in1k",                "ConvNeXt-B"),
    ("vit_base_patch16_224.augreg_in21k_ft_in1k",     "ViT-B/16 IN21k"),
    ("vit_small_patch14_dinov2.lvd142m",              "DINOv2 ViT-S/14"),
    ("vit_small_patch16_dinov3.lvd1689m",             "DINOv3 ViT-S/16"),
    ("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k","CLIP ViT-B/16"),
    ("vit_base_patch16_siglip_224.webli",             "SigLIP ViT-B/16"),
    ("eva02_base_patch16_clip_224.merged2b",          "EVA-02 CLIP B/16"),
    ("davit_base_fl.msft_florence2",                  "Florence-2 DaViT-B"),
    # Bio/pathology foundation models (gated; require HF auth with approved access)
    ("hf-hub:prov-gigapath/prov-gigapath",            "Prov-GigaPath ViT-G"),
    ("hf-hub:MahmoodLab/UNI",                         "UNI ViT-L"),
]

CSV_COLUMNS = [
    "backbone", "pretty", "params_M", "feat_dim",
    "per_sample_f1", "macro_f1", "exact_match",
    "f1_bs", "f1_bt", "f1_fj", "f1_ka", "f1_mx", "f1_pf",
    "per_sample_f1_q05",
    "chosen_lr", "train_seconds",
]


@dataclass
class ProbeResult:
    backbone: str
    pretty: str
    params_M: float
    feat_dim: int
    per_sample_f1: float
    macro_f1: float
    exact_match: float
    per_class_f1: Dict[str, float]  # keyed by class_name
    per_sample_f1_q05: float
    chosen_lr: float
    train_seconds: float


# -----------------------------------------------------------------------------
# Threshold calibration
# -----------------------------------------------------------------------------

def pick_per_class_thresholds(
    val_scores: np.ndarray,  # (N, K), sigmoid probabilities
    val_labels: np.ndarray,  # (N, K) binary
) -> np.ndarray:
    """For each class, sweep t ∈ [0.01, 0.99] and pick the t maximizing F1."""
    K = val_labels.shape[1]
    thresholds = np.zeros(K, dtype=np.float32)
    grid = np.linspace(0.01, 0.99, 99)
    for k in range(K):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred_k = (val_scores[:, k] > t).astype(np.int64)
            if pred_k.sum() == 0 and val_labels[:, k].sum() > 0:
                continue
            f1_k = f1_score(val_labels[:, k], pred_k, zero_division=0.0)
            if f1_k > best_f1:
                best_f1 = f1_k
                best_t = float(t)
        thresholds[k] = best_t
    return thresholds


def quantile_thresholds(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    q: float = 0.05,
) -> np.ndarray:
    """Method-B style: per-class q-th percentile over positives."""
    K = val_labels.shape[1]
    thr = np.zeros(K, dtype=np.float32)
    for k in range(K):
        pos = val_scores[val_labels[:, k] == 1, k]
        if len(pos) == 0:
            thr[k] = 0.5
        else:
            thr[k] = float(np.quantile(pos, q))
    return thr


# -----------------------------------------------------------------------------
# Probe one encoder
# -----------------------------------------------------------------------------

def _extract_all_splits(
    backbone: str,
    splits_path: str,
    tile_cfg: TileConfig,
    cache_dir: str,
    cli: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str]]]:
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for split in ("train", "val", "test"):
        paths, labels, class_names, _ = load_real_split(splits_path, split)
        feats, image_index = extract_features_multicrop_gpu(
            image_paths=paths,
            tile_config=tile_cfg,
            backbone=backbone,
            frame_batch_size=cli.frame_batch_size,
            num_workers=cli.num_workers,
            device=device,
            illum_sigma=cli.illum_sigma,
            illum_method=cli.illum_method,
            cache_path=os.path.join(cache_dir, f"{split}_features_cache.pt"),
        )
        feats = F.normalize(feats, dim=1)
        out[split] = (feats, image_index, labels, class_names)
    return out


def _train_linear_head(
    train_feats: torch.Tensor,
    train_tile_y: torch.Tensor,
    val_feats: torch.Tensor,
    val_image_index: torch.Tensor,
    val_labels: np.ndarray,
    cli: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Linear, float, np.ndarray, float]:
    """Train a BCE linear head with LR grid search; pick best by val per-sample F1.

    Returns (best_head, best_lr, best_thresholds, best_val_f1).
    """
    D = train_feats.shape[1]
    K = train_tile_y.shape[1]
    n_val = val_labels.shape[0]

    train_feats_gpu = train_feats.to(device)
    train_tile_y_gpu = train_tile_y.to(device)
    val_feats_gpu = val_feats.to(device)

    best: Dict = {"val_f1": -1.0}

    for lr in cli.lr_grid:
        torch.manual_seed(cli.seed)
        head = nn.Linear(D, K).to(device)
        opt = torch.optim.SGD(
            head.parameters(),
            lr=lr, momentum=0.9, nesterov=True,
            weight_decay=cli.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        N_tiles = train_feats_gpu.shape[0]
        g = torch.Generator(device="cpu").manual_seed(cli.seed)
        for epoch in range(cli.epochs):
            perm = torch.randperm(N_tiles, generator=g)
            for start in range(0, N_tiles, cli.batch_size):
                idx = perm[start : start + cli.batch_size]
                xb = train_feats_gpu[idx]
                yb = train_tile_y_gpu[idx]
                logits = head(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            tile_sig = torch.sigmoid(head(val_feats_gpu)).cpu()
        val_scores = scatter_mean_by_image(tile_sig, val_image_index, n_val).numpy()
        thr = pick_per_class_thresholds(val_scores, val_labels)
        val_pred = (val_scores > thr).astype(np.int64)
        val_f1 = per_sample_f1(val_labels, val_pred)

        if val_f1 > best["val_f1"]:
            best = {
                "val_f1": val_f1,
                "lr": float(lr),
                "state": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "thresholds": thr,
                "val_scores": val_scores,
            }
        print(f"    lr={lr:.2e}  val per-sample F1 = {val_f1:.4f}")

    head = nn.Linear(D, K).to(device)
    head.load_state_dict(best["state"])
    return head, best["lr"], best["thresholds"], best["val_f1"]


def probe_one_encoder(
    backbone: str,
    pretty: str,
    splits_path: str,
    tile_cfg: TileConfig,
    cache_dir: str,
    cli: argparse.Namespace,
    device: torch.device,
) -> ProbeResult:
    print(f"\n=== {pretty}  ({backbone}) ===")
    t0 = time.time()
    stash = _extract_all_splits(backbone, splits_path, tile_cfg, cache_dir, cli, device)
    t_extract = time.time() - t0

    train_feats, train_image_index, train_labels, class_names = stash["train"]
    val_feats, val_image_index, val_labels, _ = stash["val"]
    test_feats, test_image_index, test_labels, _ = stash["test"]
    D = train_feats.shape[1]

    train_tile_y = torch.from_numpy(train_labels.astype(np.float32))[train_image_index]

    t0 = time.time()
    head, chosen_lr, thresholds, _ = _train_linear_head(
        train_feats, train_tile_y,
        val_feats, val_image_index, val_labels,
        cli, device,
    )
    t_train = time.time() - t0

    # Test scoring (argmax-F1 thresholds)
    with torch.no_grad():
        tile_sig = torch.sigmoid(head(test_feats.to(device))).cpu()
    test_scores = scatter_mean_by_image(tile_sig, test_image_index, test_labels.shape[0]).numpy()
    test_pred = (test_scores > thresholds).astype(np.int64)

    per_sample = per_sample_f1(test_labels, test_pred)
    macro = macro_f1_per_class(test_labels, test_pred, class_names)
    em = exact_match_accuracy(test_labels, test_pred)

    # Supplementary: quantile thresholds on val, applied to test
    with torch.no_grad():
        val_tile_sig = torch.sigmoid(head(val_feats.to(device))).cpu()
    val_scores = scatter_mean_by_image(val_tile_sig, val_image_index, val_labels.shape[0]).numpy()
    thr_q05 = quantile_thresholds(val_scores, val_labels, q=0.05)
    test_pred_q05 = (test_scores > thr_q05).astype(np.int64)
    per_sample_q05 = per_sample_f1(test_labels, test_pred_q05)

    # Count backbone params (frozen, but report anyway)
    n_params = sum(
        p.numel() for p in timm.create_model(backbone, pretrained=False).parameters()
    )
    params_M = n_params / 1e6

    print(f"  extract={t_extract:.1f}s  train={t_train:.1f}s  chosen_lr={chosen_lr:.2e}")
    print(f"  test per-sample F1 = {per_sample:.4f}  macro F1 = {macro['macro']:.4f}  exact = {em:.4f}")
    print(f"  (quantile q=0.05 thresholds: per-sample F1 = {per_sample_q05:.4f})")

    return ProbeResult(
        backbone=backbone,
        pretty=pretty,
        params_M=params_M,
        feat_dim=D,
        per_sample_f1=per_sample,
        macro_f1=macro["macro"],
        exact_match=em,
        per_class_f1={c: macro[c] for c in class_names},
        per_sample_f1_q05=per_sample_q05,
        chosen_lr=chosen_lr,
        train_seconds=t_train,
    )


# -----------------------------------------------------------------------------
# CSV / LaTeX IO
# -----------------------------------------------------------------------------

def _row_from_result(r: ProbeResult, class_names: Sequence[str]) -> Dict[str, str]:
    row = {
        "backbone": r.backbone,
        "pretty": r.pretty,
        "params_M": f"{r.params_M:.1f}",
        "feat_dim": str(r.feat_dim),
        "per_sample_f1": f"{r.per_sample_f1:.4f}",
        "macro_f1": f"{r.macro_f1:.4f}",
        "exact_match": f"{r.exact_match:.4f}",
        "per_sample_f1_q05": f"{r.per_sample_f1_q05:.4f}",
        "chosen_lr": f"{r.chosen_lr:.2e}",
        "train_seconds": f"{r.train_seconds:.1f}",
    }
    for c in class_names:
        row[f"f1_{c}"] = f"{r.per_class_f1.get(c, 0.0):.4f}"
    return row


def _append_csv(csv_path: str, row: Dict[str, str]) -> None:
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _read_existing_backbones(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [row["backbone"] for row in csv.DictReader(f)]


def render_latex_table(csv_path: str, tex_path: str) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print(f"No rows in {csv_path}; skipping tex render.")
        return

    numeric_cols = [
        "per_sample_f1", "macro_f1", "exact_match",
        "f1_bs", "f1_bt", "f1_fj", "f1_ka", "f1_mx", "f1_pf",
    ]
    best_per_col = {}
    for c in numeric_cols:
        vals = [float(r[c]) for r in rows]
        best_per_col[c] = max(vals)

    def fmt(r, col):
        v = float(r[col])
        s = f"{v:.3f}"
        if abs(v - best_per_col[col]) < 1e-6:
            return r"$\mathbf{" + s + r"}$"
        return f"${s}$"

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Multi-label linear-probe across frozen vision encoders (test split, "
        "$6$-class). Each row: tile embeddings from the named backbone under the shared "
        "PHOEBI tile+illumination pipeline, mean-pooled to image level, classified by a "
        "single \\texttt{nn.Linear(D,K)} BCE head trained on the train split with per-"
        "class thresholds tuned on val. Bold = best per column.}\n"
        "\\label{tab:encoder-probe}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lcc ccc cccccc}\n"
        "\\toprule\n"
        "\\textbf{Encoder} & \\textbf{Params (M)} & \\textbf{$D$} & "
        "\\textbf{Per-sample F1} $\\uparrow$ & \\textbf{Macro F1} $\\uparrow$ & "
        "\\textbf{Exact} $\\uparrow$ & "
        "\\textbf{bs} & \\textbf{bt} & \\textbf{fj} & \\textbf{ka} & "
        "\\textbf{mx} & \\textbf{pf} \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for r in rows:
        pretty = r["pretty"].replace("&", "\\&")
        lines.append(
            " & ".join([
                pretty,
                r["params_M"],
                r["feat_dim"],
                fmt(r, "per_sample_f1"),
                fmt(r, "macro_f1"),
                fmt(r, "exact_match"),
                fmt(r, "f1_bs"), fmt(r, "f1_bt"), fmt(r, "f1_fj"),
                fmt(r, "f1_ka"), fmt(r, "f1_mx"), fmt(r, "f1_pf"),
            ]) + " \\\\"
        )
    body = "\n".join(lines)
    footer = "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    os.makedirs(os.path.dirname(tex_path) or ".", exist_ok=True)
    with open(tex_path, "w") as f:
        f.write(header + body + footer)
    print(f"Wrote {tex_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits_path", default="data/splits.json")
    p.add_argument("--output_dir", default="outputs/encoder_probe")
    p.add_argument("--tex_path", default="outputs/encoder_probe/tab_encoder_probe.tex")

    p.add_argument("--tile_size", type=int, default=224)
    p.add_argument("--eval_grid_size", type=int, default=4)
    p.add_argument("--illum_sigma", type=float, default=64.0)
    p.add_argument("--illum_method", default="divide", choices=["divide", "subtract", "none"])

    p.add_argument("--frame_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr_grid", type=float, nargs="+", default=[1e-2, 3e-2, 1e-1])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--encoders", nargs="*", default=None,
        help="Subset of backbones (timm tags) to run. Default: all 9.",
    )
    p.add_argument("--skip_if_cached", action="store_true",
                   help="Skip encoders whose row already exists in results.csv")
    p.add_argument("--render_tex", action="store_true",
                   help="After the sweep (or with --render_only), regenerate the LaTeX table.")
    p.add_argument("--render_only", action="store_true",
                   help="Skip training entirely; just render the LaTeX table from existing CSV.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")

    if args.render_only:
        render_latex_table(csv_path, args.tex_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_cfg = TileConfig(tile_size=args.tile_size, eval_grid_size=args.eval_grid_size)

    lineup = ENCODERS
    if args.encoders:
        wanted = set(args.encoders)
        lineup = [(bb, pretty) for bb, pretty in ENCODERS if bb in wanted]
        if not lineup:
            raise SystemExit(f"No matching encoders in {args.encoders}; available: {[e[0] for e in ENCODERS]}")

    already_done = set(_read_existing_backbones(csv_path)) if args.skip_if_cached else set()

    _, _, class_names, _ = load_real_split(args.splits_path, "val")

    for backbone, pretty in lineup:
        if backbone in already_done:
            print(f"[skip] {pretty} already in {csv_path}")
            continue
        cache_dir = os.path.join(args.output_dir, backbone.replace("/", "_"), "features")
        try:
            result = probe_one_encoder(
                backbone, pretty, args.splits_path, tile_cfg, cache_dir, args, device,
            )
        except Exception as e:
            print(f"[fail] {pretty} ({backbone}): {e}")
            continue
        row = _row_from_result(result, class_names)
        _append_csv(csv_path, row)
        print(f"  wrote row -> {csv_path}")

    if args.render_tex:
        render_latex_table(csv_path, args.tex_path)


if __name__ == "__main__":
    main()
