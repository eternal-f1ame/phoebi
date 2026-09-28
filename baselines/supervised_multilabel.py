#!/usr/bin/env python3
"""End-to-end multilabel classification baseline on images.

The honest "simple multilabel classification setup" for comparison against
Borowa et al. (2022). Uses the dataset in data/images/
(produced by tools/augment.py) and assigns labels by parsing each folder
name (tokens separated by underscores).

Pipeline:
  1. Enumerate data/images/<combo>/*.jpg
  2. Random image-level 80/10/10 split (deterministic seed)
  3. Pretrained ResNet-50 backbone, replace the FC with nn.Linear(2048, K)
     and fine-tune end-to-end under BCEWithLogitsLoss
  4. Train-time aug: random horizontal + vertical flip at 224x224
  5. Eval-time: center crop to 224x224, sigmoid logits -> per-class scores
  6. Report per-class AUROC / AUPR on test + per-sample F1 + macro F1
     using per-class argmax-F1 val thresholds.

Writes:
  outputs/supervised_multilabel/<run>/{model.pt, results.json, summary.md,
                                       test_scores.npy, test_labels.npy}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.metrics import per_sample_f1, macro_f1_per_class, exact_match_accuracy
from tools.build_splits import parse_label_tokens, discover_class_names


class ImagesDataset(Dataset):
    def __init__(self, paths: List[Path], labels: np.ndarray,
                 transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        y = torch.from_numpy(self.labels[i]).float()
        return x, y


def image_level_split(frames_dir: Path, class_names: List[str],
                      seed: int, val_frac: float = 0.1, test_frac: float = 0.1,
                      ) -> Dict[str, List[Tuple[Path, np.ndarray]]]:
    """Random 80/10/10 image-level split across every image in the dataset."""
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    K = len(class_names)

    entries: List[Tuple[Path, np.ndarray]] = []
    for folder in sorted(frames_dir.iterdir()):
        if not folder.is_dir():
            continue
        tokens = parse_label_tokens(folder.name)
        label = np.zeros(K, dtype=np.int64)
        for t in tokens:
            if t in cls_to_idx:
                label[cls_to_idx[t]] = 1
        for img in sorted(folder.glob("*.jpg")):
            entries.append((img, label))

    rng = random.Random(seed)
    rng.shuffle(entries)
    n = len(entries)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    return {
        "test":  entries[:n_test],
        "val":   entries[n_test:n_test + n_val],
        "train": entries[n_test + n_val:],
    }


def _make_loader(entries: List[Tuple[Path, np.ndarray]], transform,
                 batch_size: int, shuffle: bool, num_workers: int,
                 ) -> DataLoader:
    paths = [p for p, _ in entries]
    labels = np.stack([l for _, l in entries], axis=0)
    ds = ImagesDataset(paths, labels, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)


def per_class_argmax_f1_threshold(scores: np.ndarray, y_true: np.ndarray,
                                  ) -> np.ndarray:
    K = scores.shape[1]
    grid = np.linspace(0.01, 0.99, 99)
    thr = np.zeros(K)
    for k in range(K):
        best = -1.0
        for t in grid:
            pred = (scores[:, k] > t).astype(np.int64)
            tp = int(((pred == 1) & (y_true[:, k] == 1)).sum())
            fp = int(((pred == 1) & (y_true[:, k] == 0)).sum())
            fn = int(((pred == 0) & (y_true[:, k] == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best:
                best = f1
                thr[k] = float(t)
    return thr


def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_scores, axis=0), np.concatenate(all_labels, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str,
                    default="data/images")
    ap.add_argument("--output_dir", type=str,
                    default="outputs/supervised_multilabel/resnet50")
    ap.add_argument("--backbone", type=str, default="resnet50.a1_in1k")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    frames_dir = ROOT / args.frames_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = discover_class_names(frames_dir)
    K = len(class_names)
    print(f"Class names ({K}): {class_names}")

    splits = image_level_split(frames_dir, class_names, seed=args.seed)
    print(f"Splits: train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}")

    # Load backbone (pass img_size where the model accepts it — ViT-style
    # backbones interpolate positional embeddings; CNNs ignore it).
    try:
        model_b = timm.create_model(args.backbone, pretrained=True,
                                    num_classes=0, img_size=args.input_size)
    except TypeError:
        model_b = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    cfg = model_b.default_cfg
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    train_tf = transforms.Compose([
        transforms.Resize((args.input_size + 32, args.input_size + 32)),
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = _make_loader(splits["train"], train_tf, args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
    val_loader = _make_loader(splits["val"], eval_tf, args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = _make_loader(splits["test"], eval_tf, args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    # Wrap backbone with an nn.Linear(D, K) head
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.input_size, args.input_size)
        D = model_b(dummy).shape[-1]

    class Net(nn.Module):
        def __init__(self, backbone, D, K):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(D, K)
        def forward(self, x):
            return self.head(self.backbone(x))

    model = Net(model_b, D, K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
            if step % 50 == 0:
                print(f"  ep{epoch} step{step}/{len(train_loader)}  "
                      f"running BCE={total/max(n,1):.4f}")
        val_scores, val_labels = evaluate(model, val_loader, device)
        thr = per_class_argmax_f1_threshold(val_scores, val_labels)
        val_pred = (val_scores > thr).astype(np.int64)
        val_f1 = float(per_sample_f1(val_labels, val_pred))
        val_auroc = float(np.mean([
            roc_auc_score(val_labels[:, k], val_scores[:, k])
            for k in range(K) if 0 < val_labels[:, k].sum() < len(val_labels)
        ]))
        print(f"  ep{epoch}  train_BCE={total/n:.4f}  val per-sample F1={val_f1:.4f}  "
              f"val mean AUROC={val_auroc:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr

    print(f"\nLoading best val state (val F1 = {best_val_f1:.4f}) for test eval")
    model.load_state_dict(best_state)
    test_scores, test_labels = evaluate(model, test_loader, device)
    test_pred = (test_scores > best_thr).astype(np.int64)

    per_class_auroc = {}
    per_class_aupr = {}
    for k, c in enumerate(class_names):
        if 0 < test_labels[:, k].sum() < len(test_labels):
            per_class_auroc[c] = float(roc_auc_score(test_labels[:, k], test_scores[:, k]))
            per_class_aupr[c] = float(average_precision_score(test_labels[:, k], test_scores[:, k]))
        else:
            per_class_auroc[c] = float("nan")
            per_class_aupr[c] = float("nan")

    results = {
        "backbone": args.backbone,
        "class_names": class_names,
        "n_train_imgs": len(splits["train"]),
        "n_val_imgs": len(splits["val"]),
        "n_test_imgs": len(splits["test"]),
        "thresholds": best_thr.tolist(),
        "per_sample_f1": float(per_sample_f1(test_labels, test_pred)),
        "macro_f1": {k: float(v) for k, v in
                     macro_f1_per_class(test_labels, test_pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(test_labels, test_pred)),
        "per_class_auroc": per_class_auroc,
        "per_class_aupr": per_class_aupr,
        "mean_auroc": float(np.mean(list(per_class_auroc.values()))),
        "mean_aupr": float(np.mean(list(per_class_aupr.values()))),
        "best_val_per_sample_f1": best_val_f1,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "test_labels.npy", test_labels)
    torch.save(best_state, out_dir / "model.pt")

    md = []
    md.append(f"# Supervised multilabel baseline — {args.backbone}\n")
    md.append(f"Image-level random split (seed {args.seed}). "
              f"Train folders={results['n_train_imgs']}, val={results['n_val_imgs']}, "
              f"test={results['n_test_imgs']}.\n")
    md.append(f"**Per-sample F1**: {results['per_sample_f1']:.4f}  |  "
              f"**Macro F1**: {results['macro_f1']['macro']:.4f}  |  "
              f"**Exact match**: {results['exact_match']:.4f}\n")
    md.append(f"**Mean AUROC** (across {K} classes): {results['mean_auroc']:.4f}  |  "
              f"**Mean AUPR**: {results['mean_aupr']:.4f}\n")
    md.append("| Class | AUROC | AUPR | F1 |")
    md.append("|-------|-------|------|----|")
    for c in class_names:
        md.append(f"| {c} | {per_class_auroc[c]:.3f} | {per_class_aupr[c]:.3f} | "
                  f"{results['macro_f1'].get(c, float('nan')):.3f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\n=== Supervised multilabel baseline ({args.backbone}) ===")
    print(f"  Per-sample F1: {results['per_sample_f1']:.4f}")
    print(f"  Mean AUROC:    {results['mean_auroc']:.4f}")
    print(f"  Mean AUPR:     {results['mean_aupr']:.4f}")
    print(f"Wrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
