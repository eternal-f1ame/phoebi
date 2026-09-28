"""Method C training: BCE on per-class channel-grouped DINOv2 tile embeddings.

Pipeline mirrors Methods A/B (src/simplex_unmixing/train.py and
src/prototype_matching/train.py):
  1. load_real_split for train + val (image-level random 80/10/10).
  2. Extract DINOv2 tile features via the GPU illumination path (cached).
  3. Replicate per-image labels to per-tile labels (all tiles inherit the
     image label by Spatial Homogeneity Assumption H).
  4. Train MCChannelHead with BCE on (tile_logit, tile_label) for `--epochs`.
     CRA dropout is active during training.
  5. Forward val tiles, mean-aggregate sigmoids per image, calibrate per-class
     thresholds at the val image level by argmax-F1 on val (matches the
     supervised baseline calibration; Method C is discriminative so val-F1
     calibration is the right analog).
  6. Save model.pt + config.json + train_summary.json.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common.features import extract_features_multicrop_gpu, scatter_mean_by_image
from src.common.io import load_real_split, save_json
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import TileConfig
from src.mc_channel.model import MCChannelHead, MCConfig


def calibrate_argmax_f1(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-class threshold = argmax F1 on a 99-point grid (matches
    baselines/multilabel_probe.py and supervised_multilabel.py)."""
    K = scores.shape[1]
    grid = np.linspace(0.01, 0.99, 99)
    out = np.zeros(K)
    for k in range(K):
        best = -1.0
        for t in grid:
            pred = (scores[:, k] > t).astype(np.int64)
            tp = int(((pred == 1) & (labels[:, k] == 1)).sum())
            fp = int(((pred == 1) & (labels[:, k] == 0)).sum())
            fn = int(((pred == 0) & (labels[:, k] == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best:
                best = f1
                out[k] = float(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", type=str, default="data/splits.json")
    ap.add_argument("--output_dir", type=str, default="outputs/mc_channel/default")
    ap.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--tile_size", type=int, default=224)
    ap.add_argument("--eval_grid_size", type=int, default=4)
    ap.add_argument("--train_tiles_per_image", type=int, default=16)
    ap.add_argument("--illumination", type=str, default="divide",
                    choices=["divide", "subtract", "none"])
    ap.add_argument("--illum_sigma", type=float, default=64.0)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096,
                    help="Tile-batch size; large because the head is tiny.")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cra_drop_prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--loss", type=str, default="bce", choices=["bce", "asl"],
                    help="Loss function. asl=Asymmetric Loss (Ridnik 2021).")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip_neg", type=float, default=0.05,
                    help="Margin shift on negative probabilities (Ridnik 2021).")

    ap.add_argument("--mixup_alpha", type=float, default=0.0,
                    help="Tile-level mixup: if > 0, sample lambda from Beta(alpha, alpha) "
                         "and mix tile pairs with union of labels (max). Acts as compositional augmentation.")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | output: {output_dir}")

    # ---- 1. extract features ----
    tile_cfg = TileConfig(
        tile_size=args.tile_size,
        train_tiles_per_image=args.train_tiles_per_image,
        eval_grid_size=args.eval_grid_size,
    )
    print("\n[1/4] Extracting train + val tile features...")
    train_paths, train_labels, class_names, _ = load_real_split(args.splits_path, "train")
    val_paths, val_labels, _, _ = load_real_split(args.splits_path, "val")
    K = len(class_names)

    train_features, train_image_index = extract_features_multicrop_gpu(
        image_paths=train_paths, tile_config=tile_cfg, backbone=args.backbone,
        frame_batch_size=args.frame_batch_size, num_workers=args.num_workers,
        device=device, illum_sigma=args.illum_sigma, illum_method=args.illumination,
        cache_path=str(output_dir / "train_features_cache.pt"),
    )
    val_features, val_image_index = extract_features_multicrop_gpu(
        image_paths=val_paths, tile_config=tile_cfg, backbone=args.backbone,
        frame_batch_size=args.frame_batch_size, num_workers=args.num_workers,
        device=device, illum_sigma=args.illum_sigma, illum_method=args.illumination,
        cache_path=str(output_dir / "val_features_cache.pt"),
    )
    train_features = F.normalize(train_features, p=2, dim=1)
    val_features = F.normalize(val_features, p=2, dim=1)
    D = train_features.shape[1]
    print(f"  train tile features: {tuple(train_features.shape)}; classes: {class_names}")

    # ---- 2. per-tile labels ----
    train_tile_labels = torch.from_numpy(train_labels)[train_image_index].float()  # (T, K)

    # ---- 3. build model ----
    cfg = MCConfig(embedding_dim=D, num_classes=K, cra_drop_prob=args.cra_drop_prob)
    model = MCChannelHead(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def asymmetric_loss(logits, target, gamma_pos, gamma_neg, clip_neg):
        """Asymmetric Loss for multi-label (Ridnik 2021).
        Down-weights easy negatives by raising (1-p_shifted)^gamma_neg, where
        p_shifted = max(p - clip_neg, 0). gamma_pos=0 keeps the positive
        cross-entropy unchanged."""
        p = torch.sigmoid(logits)
        # Positive part
        pos_loss = -(1 - p).clamp(min=1e-6).pow(gamma_pos) * torch.log(p.clamp(min=1e-6))
        # Negative part with probability shift
        p_shift = (p - clip_neg).clamp(min=0.0)
        neg_loss = -p_shift.clamp(min=1e-6).pow(gamma_neg) * torch.log((1 - p_shift).clamp(min=1e-6))
        loss = target * pos_loss + (1 - target) * neg_loss
        return loss.mean()

    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        print(f"  loss: BCE")
    else:
        def loss_fn(logits, target):
            return asymmetric_loss(logits, target, args.asl_gamma_pos,
                                   args.asl_gamma_neg, args.asl_clip_neg)
        print(f"  loss: ASL (gamma_pos={args.asl_gamma_pos}, gamma_neg={args.asl_gamma_neg}, "
              f"clip_neg={args.asl_clip_neg})")
    print(f"  MCConfig: {asdict(cfg)} (channels_per_class={cfg.channels_per_class})")

    # ---- 4. train ----
    print(f"\n[2/4] Training for {args.epochs} epochs on {train_features.shape[0]} tiles...")
    loader = DataLoader(
        TensorDataset(train_features, train_tile_labels),
        batch_size=args.batch_size, shuffle=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            # Tile-level Mixup with multi-label union (max of labels). Principled
            # under Assumption H: any tile features representing label sets A,B
            # can be mixed and the mixed tile genuinely represents A union B.
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                yb = torch.maximum(yb, yb[perm])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        if epoch == 1 or epoch % 5 == 0:
            # Quick val check
            model.eval()
            with torch.no_grad():
                vlogits_tile = model(val_features.to(device))
                vsig_tile = torch.sigmoid(vlogits_tile).cpu()
                vsig_image = scatter_mean_by_image(vsig_tile, val_image_index, len(val_paths)).numpy()
            thr = calibrate_argmax_f1(vsig_image, val_labels)
            vpred = (vsig_image > thr).astype(np.int64)
            vf1 = float(per_sample_f1(val_labels, vpred))
            print(f"  ep{epoch:03d}  train_loss={total/n_seen:.4f}  val per-sample F1={vf1:.4f}")

    # ---- 5. final calibration on val ----
    print("\n[3/4] Final val calibration...")
    model.eval()
    with torch.no_grad():
        vlogits_tile = model(val_features.to(device))
        vsig_tile = torch.sigmoid(vlogits_tile).cpu()
        vsig_image = scatter_mean_by_image(vsig_tile, val_image_index, len(val_paths)).numpy()
    presence_thresholds = calibrate_argmax_f1(vsig_image, val_labels)
    vpred = (vsig_image > presence_thresholds).astype(np.int64)
    val_f1 = float(per_sample_f1(val_labels, vpred))
    val_macro = macro_f1_per_class(val_labels, vpred, class_names)
    val_em = float(exact_match_accuracy(val_labels, vpred))
    print(f"  val per-sample F1: {val_f1:.4f}  macro: {val_macro['macro']:.4f}  EM: {val_em:.4f}")
    print(f"  per-class thresholds: {dict(zip(class_names, presence_thresholds.tolist()))}")

    # ---- 6. save ----
    print("\n[4/4] Saving model + config...")
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, output_dir / "mc_model.pt")
    config_payload = {
        "backbone": args.backbone,
        "class_names": class_names,
        "tile_config": asdict(tile_cfg),
        "illumination": args.illumination,
        "illum_sigma": args.illum_sigma,
        "mc_config": asdict(cfg),
        "presence_thresholds": presence_thresholds.tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "cra_drop_prob": args.cra_drop_prob,
    }
    save_json(str(output_dir / "config.json"), config_payload)
    save_json(str(output_dir / "train_summary.json"), {
        "val_per_sample_f1": val_f1,
        "val_macro_f1": val_macro,
        "val_exact_match": val_em,
        "n_train_tiles": int(train_features.shape[0]),
        "n_val_tiles": int(val_features.shape[0]),
    })
    print(f"\n=== Method C ({args.backbone}) trained ===")
    print(f"  val per-sample F1: {val_f1:.4f}")
    print(f"  Wrote {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
