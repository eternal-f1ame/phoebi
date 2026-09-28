#!/usr/bin/env python3
"""End-to-end fine-tuning of DINOv2-S/14 through the PHOEBI GPU pipeline.

Workers only JPEG-decode raw uint8 frames; illumination correction, tile
slicing, ImageNet normalization, and DINOv2 forward (with gradient) all happen
on the GPU. This is the same throughput pattern as
extract_features_multicrop_gpu but with backbone gradient flow.

Training tile sampling: deterministic 4x4 grid + random per-tile h/v flips.
The grid is fixed; the diversity comes from flips and from epochs * dataset
shuffling. This sacrifices random-crop diversity for an order-of-magnitude
throughput win, which is the right trade for end-to-end fine-tuning where
wall-clock matters.

Outputs: outputs/finetune_dinov2_phoebi/{model.pt,results.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import scatter_mean_by_image
from src.common.illumination import gpu_normalize_illumination
from src.common.io import load_real_split
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import FullFrameDataset, _grid_offsets


def calibrate_argmax_f1(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
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


def gpu_grid_tiles(frames: torch.Tensor, tile_size: int, grid_size: int) -> torch.Tensor:
    """frames: (B, 3, H, W) float. Returns (B*T, 3, tile_size, tile_size) where
    T = grid_size**2 with the same span-spaced offsets as the eval pipeline."""
    B, C, H, W = frames.shape
    y_offs = _grid_offsets(H, tile_size, grid_size)
    x_offs = _grid_offsets(W, tile_size, grid_size)
    tiles = []
    for y in y_offs:
        for x in x_offs:
            tiles.append(frames[:, :, y:y + tile_size, x:x + tile_size])
    return torch.stack(tiles, dim=1).reshape(B * grid_size * grid_size, C, tile_size, tile_size)


def gpu_random_tile_flips(tiles: torch.Tensor, training: bool) -> torch.Tensor:
    """In-place random h/v flips per tile during training; identity at eval."""
    if not training:
        return tiles
    B = tiles.shape[0]
    h = torch.rand(B, device=tiles.device) > 0.5
    v = torch.rand(B, device=tiles.device) > 0.5
    if h.any():
        tiles[h] = torch.flip(tiles[h], dims=(-1,))
    if v.any():
        tiles[v] = torch.flip(tiles[v], dims=(-2,))
    return tiles


class Net(nn.Module):
    def __init__(self, backbone: nn.Module, D: int, K: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(D, K)

    def forward(self, x):
        return self.head(self.backbone(x))


def evaluate(model, loader, labels, num_images, tile_size, grid_size,
             illum_sigma, illum_method, mean, std, device):
    model.eval()
    sig_all, idx_all = [], []
    with torch.no_grad():
        for frames, idxs in loader:
            frames = frames.to(device, non_blocking=True)
            frames_f = gpu_normalize_illumination(
                frames, sigma=illum_sigma, method=illum_method, downsample=8,
            )
            tiles = gpu_grid_tiles(frames_f, tile_size, grid_size)
            tiles = tiles / 255.0
            tiles = (tiles - mean) / std
            sig = torch.sigmoid(model(tiles)).cpu()
            sig_all.append(sig)
            T = grid_size * grid_size
            tile_idx = idxs.unsqueeze(1).expand(-1, T).reshape(-1)
            idx_all.append(tile_idx)
    sig_all = torch.cat(sig_all, dim=0)
    idx_all = torch.cat(idx_all, dim=0)
    image_sig = scatter_mean_by_image(sig_all, idx_all, num_images).numpy()
    return image_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--output_dir", default="outputs/finetune_dinov2_phoebi")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--grid_size", type=int, default=4)
    ap.add_argument("--illum_method", default="divide", choices=["divide", "subtract", "none"])
    ap.add_argument("--illum_sigma", type=float, default=64.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--frame_batch_size", type=int, default=8,
                    help="Frames per batch; tile batch is frame_batch_size * grid_size**2.")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--loss", type=str, default="bce", choices=["bce", "asl"])
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip_neg", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.0,
                    help="Tile-level mixup. Sample lambda from Beta(alpha,alpha) "
                         "and mix tiles + take max-of-labels. >0 enables.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from model.pt + best.json in output_dir.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  output: {out_dir}", flush=True)

    train_paths, train_labels, class_names, _ = load_real_split(args.splits_path, "train")
    val_paths, val_labels, _, _ = load_real_split(args.splits_path, "val")
    test_paths, test_labels, _, _ = load_real_split(args.splits_path, "test")
    K = len(class_names)
    T = args.grid_size * args.grid_size
    print(f"Splits: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}, K={K}, tiles/img={T}", flush=True)

    # Loaders
    def make_loader(paths, shuffle):
        ds = FullFrameDataset(paths)
        return DataLoader(
            ds, batch_size=args.frame_batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
    train_loader = make_loader(train_paths, shuffle=True)
    val_loader = make_loader(val_paths, shuffle=False)
    test_loader = make_loader(test_paths, shuffle=False)

    # Model
    backbone = timm.create_model(args.backbone, pretrained=True, num_classes=0,
                                 img_size=args.input_size)
    with torch.no_grad():
        D = backbone(torch.zeros(1, 3, args.input_size, args.input_size)).shape[-1]
    model = Net(backbone, D, K).to(device)
    print(f"Backbone: {args.backbone}  D={D}  K={K}", flush=True)

    cfg = backbone.default_cfg
    mean_tup = cfg.get("mean", (0.485, 0.456, 0.406))
    std_tup = cfg.get("std", (0.229, 0.224, 0.225))
    mean = torch.tensor(mean_tup, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_tup, device=device).view(1, 3, 1, 1)

    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    def asymmetric_loss(logits, target, gamma_neg=4.0, clip_neg=0.05):
        p = torch.sigmoid(logits)
        pos_loss = -torch.log(p.clamp(min=1e-6))
        p_shift = (p - clip_neg).clamp(min=0.0)
        neg_loss = -p_shift.clamp(min=1e-6).pow(gamma_neg) * torch.log((1 - p_shift).clamp(min=1e-6))
        return (target * pos_loss + (1 - target) * neg_loss).mean()

    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        print(f"  loss: BCE", flush=True)
    else:
        def loss_fn(logits, target):
            return asymmetric_loss(logits, target, args.asl_gamma_neg, args.asl_clip_neg)
        print(f"  loss: ASL (gamma_neg={args.asl_gamma_neg}, clip_neg={args.asl_clip_neg})", flush=True)
    bce = loss_fn  # alias for the rest of the loop
    train_labels_t = torch.from_numpy(train_labels).to(device).float()

    best_val_f1 = -1.0
    best_state = None
    best_thr = None
    start_epoch = 1

    if args.resume and (out_dir / "model.pt").exists() and (out_dir / "best.json").exists():
        ckpt = json.loads((out_dir / "best.json").read_text())
        best_val_f1 = ckpt["best_val_per_sample_f1"]
        best_thr = np.array(ckpt["thresholds"])
        last_f = out_dir / "last.json"
        last_epoch = (json.loads(last_f.read_text())["last_epoch"]
                      if last_f.exists() else ckpt["epoch"])
        start_epoch = last_epoch + 1
        model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Resumed from epoch {last_epoch} (best val F1={best_val_f1:.4f}), "
              f"starting epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for step, (frames, idxs) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            B = frames.shape[0]
            frames_f = gpu_normalize_illumination(
                frames, sigma=args.illum_sigma, method=args.illum_method, downsample=8,
            )
            tiles = gpu_grid_tiles(frames_f, args.input_size, args.grid_size)
            tiles = gpu_random_tile_flips(tiles, training=True)
            tiles = tiles / 255.0
            tiles = (tiles - mean) / std
            tile_labels = train_labels_t[idxs.to(device)].unsqueeze(1).expand(B, T, K).reshape(B * T, K)

            # Tile-level Mixup: mix per-tile features with their permutation,
            # take element-wise max of labels (multi-label union). Principled
            # under Assumption H: any combination of tiles still represents the
            # union of label sets present in the source images.
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(tiles.shape[0], device=device)
                tiles = lam * tiles + (1.0 - lam) * tiles[perm]
                tile_labels = torch.maximum(tile_labels, tile_labels[perm])

            opt.zero_grad()
            loss = bce(model(tiles), tile_labels)
            loss.backward()
            opt.step()

            running += loss.item() * tiles.shape[0]
            n_seen += tiles.shape[0]
            if step % 50 == 0:
                print(f"  ep{epoch} step{step}/{len(train_loader)}  running loss={running/max(n_seen,1):.4f}", flush=True)

        # Val
        v_sig = evaluate(model, val_loader, val_labels, len(val_paths),
                         args.input_size, args.grid_size,
                         args.illum_sigma, args.illum_method, mean, std, device)
        thr = calibrate_argmax_f1(v_sig, val_labels)
        v_pred = (v_sig > thr).astype(np.int64)
        v_f1 = float(per_sample_f1(val_labels, v_pred))
        print(f"  ep{epoch}  train_loss={running/n_seen:.4f}  val per-sample F1={v_f1:.4f}", flush=True)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr.copy()
            torch.save(best_state, out_dir / "model.pt")
            (out_dir / "best.json").write_text(json.dumps(
                {"epoch": epoch, "best_val_per_sample_f1": best_val_f1,
                 "thresholds": best_thr.tolist()},
                indent=2,
            ))
        (out_dir / "last.json").write_text(json.dumps({"last_epoch": epoch}))

    # Test
    print(f"\nLoading best val state (val F1 = {best_val_f1:.4f}) for test eval", flush=True)
    model.load_state_dict(best_state)
    t_sig = evaluate(model, test_loader, test_labels, len(test_paths),
                     args.input_size, args.grid_size,
                     args.illum_sigma, args.illum_method, mean, std, device)
    t_pred = (t_sig > best_thr).astype(np.int64)
    t_f1 = float(per_sample_f1(test_labels, t_pred))
    t_macro = macro_f1_per_class(test_labels, t_pred, class_names)
    t_em = float(exact_match_accuracy(test_labels, t_pred))

    results = {
        "backbone": args.backbone,
        "class_names": class_names,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_test": len(test_paths),
        "thresholds": best_thr.tolist(),
        "best_val_per_sample_f1": best_val_f1,
        "test_per_sample_f1": t_f1,
        "test_macro_f1": {k: float(v) for k, v in t_macro.items()},
        "test_exact_match": t_em,
        "config": vars(args),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(best_state, out_dir / "model.pt")
    np.save(out_dir / "test_scores.npy", t_sig)
    np.save(out_dir / "test_labels.npy", test_labels)

    print(f"\n=== End-to-end DINOv2 fine-tune ===")
    print(f"  test per-sample F1: {t_f1:.4f}")
    print(f"  test macro F1:      {t_macro['macro']:.4f}")
    print(f"  test exact match:   {t_em:.4f}")


if __name__ == "__main__":
    main()
