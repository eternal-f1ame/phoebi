#!/usr/bin/env python3
"""End-to-end fine-tuning of DINOv2-S/14 through the PHOEBI GPU pipeline,
under the leave-combinations-out protocol used by
baselines/supervised_multilabel_heldout.py.

Same training pipeline as baselines/finetune_dinov2_phoebi.py:
  workers JPEG-decode raw uint8 frames at 2592x1944; illumination correction,
  4x4 grid tile slicing, ImageNet normalization, and DINOv2 forward (with
  gradient) all run on the GPU.

Same heldout split as baselines/supervised_multilabel_heldout.py:
  9 held-out combinations (1 single / 2 pairs / 3 triples / 2 quadruples /
  1 six-species), trained-on combos split image-level 90/10 train/val,
  the 9 held-out combos form the entire test set.

Outputs to outputs/finetune_dinov2_phoebi_heldout/<run>/{model.pt,results.json,
summary.md,test_scores.npy,test_labels.npy}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import scatter_mean_by_image
from src.common.illumination import gpu_normalize_illumination
from src.common.metrics import (
    exact_match_accuracy,
    macro_f1_per_class,
    per_sample_f1,
)
from src.common.tiling import FullFrameDataset, _grid_offsets

from baselines.finetune_dinov2_phoebi import (
    Net,
    calibrate_argmax_f1,
    evaluate,
    gpu_grid_tiles,
    gpu_random_tile_flips,
)
from baselines.supervised_multilabel_heldout import (
    DEFAULT_HELDOUT_COUNTS,
    parse_heldout_counts,
    select_heldout,
)
from tools.build_splits import discover_class_names, parse_label_tokens


def collect_entries(frames_dir: Path, combos: List[str],
                    class_names: List[str]) -> List[Tuple[Path, np.ndarray, str]]:
    """For each combo folder, enumerate frame_*.jpg and assign the combo's
    multi-hot label vector. Returns (path, label, combo) triples."""
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    K = len(class_names)
    out = []
    for combo in combos:
        tokens = parse_label_tokens(combo)
        label = np.zeros(K, dtype=np.int64)
        for t in tokens:
            if t in cls_to_idx:
                label[cls_to_idx[t]] = 1
        for img in sorted((frames_dir / combo).glob("*.jpg")):
            out.append((img, label, combo))
    return out


def image_level_90_10(entries, seed: int):
    rng = random.Random(seed + 1)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n_val = int(0.1 * len(shuffled))
    return shuffled[n_val:], shuffled[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/images",
                    help="Native-resolution frame folders, one per combination.")
    ap.add_argument("--output_dir", default="outputs/finetune_dinov2_phoebi_heldout/dinov2_s14")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--grid_size", type=int, default=4)
    ap.add_argument("--illum_method", default="divide", choices=["divide", "subtract", "none"])
    ap.add_argument("--illum_sigma", type=float, default=64.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--heldout_counts", type=str, default="",
                    help="Override heldout per order, e.g. '1:1,2:2,3:3,4:2,6:1'.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from model.pt + best.json in output_dir.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  output: {out_dir}", flush=True)

    frames_dir = ROOT / args.frames_dir
    class_names = discover_class_names(frames_dir)
    K = len(class_names)
    T = args.grid_size * args.grid_size
    heldout_counts = (parse_heldout_counts(args.heldout_counts)
                      if args.heldout_counts else DEFAULT_HELDOUT_COUNTS)

    heldout, trained, _ = select_heldout(frames_dir, seed=args.seed,
                                          class_names=class_names,
                                          heldout_counts=heldout_counts)
    print(f"Species ({K}): {class_names}", flush=True)
    print(f"Held-out combinations ({len(heldout)}): {heldout}", flush=True)
    print(f"Trained-on combinations: {len(trained)}", flush=True)

    trained_entries = collect_entries(frames_dir, trained, class_names)
    heldout_entries = collect_entries(frames_dir, heldout, class_names)
    train_entries, val_entries = image_level_90_10(trained_entries, seed=args.seed)
    print(f"train={len(train_entries)}  val={len(val_entries)}  "
          f"heldout_test={len(heldout_entries)}  K={K}  tiles/img={T}", flush=True)

    def split_lists(entries):
        paths = [str(p) for p, _, _ in entries]
        labels = np.stack([l for _, l, _ in entries], axis=0)
        combos = [c for _, _, c in entries]
        return paths, labels, combos

    train_paths, train_labels, _ = split_lists(train_entries)
    val_paths, val_labels, _ = split_lists(val_entries)
    test_paths, test_labels, test_combos = split_lists(heldout_entries)

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

    backbone = timm.create_model(args.backbone, pretrained=True, num_classes=0,
                                 img_size=args.input_size)
    with torch.no_grad():
        D = backbone(torch.zeros(1, 3, args.input_size, args.input_size)).shape[-1]
    model = Net(backbone, D, K).to(device)
    print(f"Backbone: {args.backbone}  D={D}  K={K}", flush=True)

    cfg = backbone.default_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)),
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)),
                       device=device).view(1, 3, 1, 1)

    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    bce = nn.BCEWithLogitsLoss()

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
            tile_labels = (
                train_labels_t[idxs.to(device)]
                .unsqueeze(1).expand(B, T, K).reshape(B * T, K)
            )

            opt.zero_grad()
            loss = bce(model(tiles), tile_labels)
            loss.backward()
            opt.step()

            running += loss.item() * tiles.shape[0]
            n_seen += tiles.shape[0]
            if step % 50 == 0:
                print(f"  ep{epoch} step{step}/{len(train_loader)}  "
                      f"running loss={running / max(n_seen, 1):.4f}", flush=True)

        v_sig = evaluate(model, val_loader, val_labels, len(val_paths),
                         args.input_size, args.grid_size,
                         args.illum_sigma, args.illum_method, mean, std, device)
        thr = calibrate_argmax_f1(v_sig, val_labels)
        v_pred = (v_sig > thr).astype(np.int64)
        v_f1 = float(per_sample_f1(val_labels, v_pred))
        print(f"  ep{epoch}  train_loss={running / n_seen:.4f}  "
              f"val per-sample F1={v_f1:.4f}", flush=True)
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

    print(f"\nLoading best val state (val F1 = {best_val_f1:.4f}) for test eval", flush=True)
    model.load_state_dict(best_state)

    v_sig = evaluate(model, val_loader, val_labels, len(val_paths),
                     args.input_size, args.grid_size,
                     args.illum_sigma, args.illum_method, mean, std, device)
    v_pred = (v_sig > best_thr).astype(np.int64)
    val_metrics = {
        "per_sample_f1": float(per_sample_f1(val_labels, v_pred)),
        "macro_f1": {k: float(v) for k, v in
                     macro_f1_per_class(val_labels, v_pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(val_labels, v_pred)),
    }

    t_sig = evaluate(model, test_loader, test_labels, len(test_paths),
                     args.input_size, args.grid_size,
                     args.illum_sigma, args.illum_method, mean, std, device)
    t_pred = (t_sig > best_thr).astype(np.int64)
    test_metrics = {
        "per_sample_f1": float(per_sample_f1(test_labels, t_pred)),
        "macro_f1": {k: float(v) for k, v in
                     macro_f1_per_class(test_labels, t_pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(test_labels, t_pred)),
    }

    combos_arr = np.array(test_combos)
    per_combo = {}
    for combo in heldout:
        mask = combos_arr == combo
        if mask.sum() == 0:
            continue
        per_combo[combo] = {
            "n_images": int(mask.sum()),
            "per_sample_f1": float(per_sample_f1(test_labels[mask], t_pred[mask])),
            "exact_match": float(exact_match_accuracy(test_labels[mask], t_pred[mask])),
        }

    results = {
        "backbone": args.backbone,
        "class_names": class_names,
        "heldout_combos": heldout,
        "trained_combos_count": len(trained),
        "n_train_imgs": len(train_paths),
        "n_val_imgs": len(val_paths),
        "n_test_imgs": len(test_paths),
        "thresholds": best_thr.tolist(),
        "best_val_per_sample_f1": best_val_f1,
        "val_in_distribution": val_metrics,
        "test_heldout_combos": test_metrics,
        "per_heldout_combo": per_combo,
        "config": vars(args),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.save(out_dir / "test_scores.npy", t_sig)
    np.save(out_dir / "test_labels.npy", test_labels)
    torch.save(best_state, out_dir / "model.pt")

    md = []
    md.append(f"# PHOEBI-pipeline DINOv2 fine-tune, leave-combinations-out\n")
    md.append(f"Held-out combinations ({len(heldout)}): "
              f"`{', '.join(heldout)}`\n")
    md.append(f"train={len(train_paths)}  val={len(val_paths)}  "
              f"test(heldout)={len(test_paths)}\n")
    md.append(f"\n### Val (in-distribution)")
    md.append(f"Per-sample F1 = {val_metrics['per_sample_f1']:.4f}  |  "
              f"Macro F1 = {val_metrics['macro_f1']['macro']:.4f}  |  "
              f"Exact match = {val_metrics['exact_match']:.4f}\n")
    md.append(f"### Test (held-out combinations)")
    md.append(f"Per-sample F1 = {test_metrics['per_sample_f1']:.4f}  |  "
              f"Macro F1 = {test_metrics['macro_f1']['macro']:.4f}  |  "
              f"Exact match = {test_metrics['exact_match']:.4f}\n")
    md.append("\n### Per-held-out-combination\n")
    md.append("| Combo | Order | Images | Per-sample F1 | Exact match |")
    md.append("|-------|-------|--------|---------------|-------------|")
    for combo in heldout:
        if combo not in per_combo:
            continue
        order = len(parse_label_tokens(combo))
        m = per_combo[combo]
        md.append(f"| `{combo}` | {order} | {m['n_images']} | "
                  f"{m['per_sample_f1']:.4f} | {m['exact_match']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\n=== PHOEBI-pipeline DINOv2 fine-tune (held-out combos) ===")
    print(f"  Val (in-distribution):  F1 = {val_metrics['per_sample_f1']:.4f}")
    print(f"  Test (held-out combos): F1 = {test_metrics['per_sample_f1']:.4f}")
    print(f"  delta F1: {val_metrics['per_sample_f1'] - test_metrics['per_sample_f1']:.4f}")
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
