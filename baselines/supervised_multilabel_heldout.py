#!/usr/bin/env python3
"""Leave-combinations-out generalization experiment for the supervised
multi-label baseline.

Holds out 9 entire combinations from training (1 single / 2 pairs / 3 triples
/ 2 quadruples / 1 six-species) and asks whether an end-to-end ResNet-50
trained on the remaining 31 combinations can still identify the species
present in held-out combinations it has never seen plated together.

Pipeline:
  1. Deterministically pick held-out combinations per order (seed 1337).
  2. Split remaining combinations 90/10 image-level -> train / val.
  3. The held-out combinations form the test set (all 9,000 images).
  4. Fine-tune ResNet-50 end-to-end with BCE for 15 epochs.
  5. Pick per-class thresholds on val by argmax F1.
  6. Report:
     - In-distribution val metrics (images from combos seen at train time)
     - Held-out test metrics aggregated
     - Per-held-out-combo F1 / AUROC (which combos the model fails on)

Writes outputs/supervised_multilabel_heldout/<run>/{results.json, summary.md,
test_scores.npy, test_labels.npy, model.pt}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
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


# ----------------------------------------------------------------------------
# Deterministic held-out combination selection
# ----------------------------------------------------------------------------
DEFAULT_HELDOUT_COUNTS = {
    1: 1,   # 1 single-species combination
    2: 2,   # 2 pairwise
    3: 3,   # 3 triples
    4: 2,   # 2 quadruples
    6: 1,   # 1 six-species (we have no 5-species in the dataset)
}


def parse_heldout_counts(spec: str) -> dict:
    """Parse a heldout-counts spec like "1:1,2:2,3:1,4:1" into {order: count}."""
    out = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        order_str, count_str = piece.split(":")
        out[int(order_str)] = int(count_str)
    return out


def _tokens_in(combos):
    s = set()
    for c in combos:
        s.update(parse_label_tokens(c))
    return s


def select_heldout(frames_dir: Path, seed: int,
                   class_names: List[str], heldout_counts: dict,
                   ) -> Tuple[List[str], List[str], dict]:
    """Pick combinations to hold out, ensuring that every one of the K species
    still appears in at least one TRAINED combination. Re-rolls seeds until
    coverage holds (usually succeeds on seed 0)."""
    folders = sorted([f.name for f in frames_dir.iterdir() if f.is_dir()])
    by_order = defaultdict(list)
    for name in folders:
        tokens = parse_label_tokens(name)
        by_order[len(tokens)].append(name)

    attempt = 0
    while True:
        rng = random.Random(seed + attempt)
        heldout: List[str] = []
        for order, count in heldout_counts.items():
            if order not in by_order:
                raise SystemExit(f"No combinations of order {order} in {frames_dir}")
            pool = sorted(by_order[order])
            if len(pool) < count:
                raise SystemExit(
                    f"Need to hold out {count} combinations of order {order} but "
                    f"only {len(pool)} exist in {frames_dir}")
            rng.shuffle(pool)
            heldout.extend(pool[:count])
        # Drop from training any folder whose label set duplicates a held-out
        # folder's label set (e.g. the four-species subset has both `b_f` and `f_b` for
        # the {b,f} combo; otherwise the "held-out" combo leaks into training).
        heldout_label_sets = {frozenset(parse_label_tokens(c)) for c in heldout}
        trained = [
            f for f in folders
            if f not in set(heldout)
            and frozenset(parse_label_tokens(f)) not in heldout_label_sets
        ]
        trained_species = _tokens_in(trained)
        missing = [c for c in class_names if c not in trained_species]
        if not missing:
            if attempt > 0:
                print(f"  (used seed offset {attempt} to preserve species coverage)")
            return heldout, trained, dict(by_order)
        print(f"  seed offset {attempt}: trained set misses {missing}; retry")
        attempt += 1
        if attempt > 50:
            raise SystemExit("Could not find a held-out selection that "
                             "preserves species coverage in training")


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class ImagesDataset(Dataset):
    def __init__(self, paths: List[Path], labels: np.ndarray,
                 combos: List[str], transform) -> None:
        self.paths = paths
        self.labels = labels
        self.combos = combos
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        y = torch.from_numpy(self.labels[i]).float()
        return x, y


def collect_entries(frames_dir: Path, combos: List[str],
                    class_names: List[str]) -> List[Tuple[Path, np.ndarray, str]]:
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
    n = len(shuffled)
    n_val = int(0.1 * n)
    return shuffled[n_val:], shuffled[:n_val]


def make_loader(entries, transform, batch_size, shuffle, num_workers):
    paths = [p for p, _, _ in entries]
    labels = np.stack([l for _, l, _ in entries], axis=0)
    combos = [c for _, _, c in entries]
    ds = ImagesDataset(paths, labels, combos, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0), combos


def per_class_argmax_f1_threshold(scores, y_true):
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


def evaluate(model, loader, device):
    model.eval()
    S, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            S.append(torch.sigmoid(model(x.to(device, non_blocking=True))).cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(S), np.concatenate(Y)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str,
                    default="data/images_256")
    ap.add_argument("--output_dir", type=str,
                    default="outputs/supervised_multilabel_heldout/resnet50")
    ap.add_argument("--backbone", type=str, default="resnet50.a1_in1k")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--heldout_counts", type=str, default="",
                    help="Override heldout per order, e.g. '1:1,2:2,3:1,4:1'. "
                         "Default matches the 6-class sweep: 1:1,2:2,3:3,4:2,6:1.")
    args = ap.parse_args()

    heldout_counts = (parse_heldout_counts(args.heldout_counts)
                      if args.heldout_counts else DEFAULT_HELDOUT_COUNTS)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    frames_dir = ROOT / args.frames_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = discover_class_names(aug_dir)
    K = len(class_names)
    heldout, trained, by_order = select_heldout(frames_dir, seed=args.seed,
                                                 class_names=class_names,
                                                 heldout_counts=heldout_counts)

    print(f"Species ({K}): {class_names}")
    print(f"Held-out combinations ({len(heldout)}): {heldout}")
    print(f"Trained-on combinations ({len(trained)}): "
          f"{[c for c in trained[:6]]}... ({len(trained) - 6} more)")

    # Build entries
    trained_entries = collect_entries(frames_dir, trained, class_names)
    heldout_entries = collect_entries(frames_dir, heldout, class_names)
    train_entries, val_entries = image_level_90_10(trained_entries, seed=args.seed)

    print(f"train={len(train_entries)}  val={len(val_entries)}  "
          f"heldout_test={len(heldout_entries)}")

    try:
        model_b = timm.create_model(args.backbone, pretrained=True,
                                    num_classes=0, img_size=args.input_size)
    except TypeError:
        model_b = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    cfg = model_b.default_cfg
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.0)),
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

    train_loader, _ = make_loader(train_entries, train_tf, args.batch_size,
                                   True, args.num_workers)
    val_loader, _ = make_loader(val_entries, eval_tf, args.batch_size,
                                 False, args.num_workers)
    test_loader, test_combos = make_loader(heldout_entries, eval_tf,
                                            args.batch_size, False, args.num_workers)

    with torch.no_grad():
        D = model_b(torch.zeros(1, 3, args.input_size, args.input_size)).shape[-1]

    class Net(nn.Module):
        def __init__(self, backbone, D, K):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(D, K)
        def forward(self, x): return self.head(self.backbone(x))

    model = Net(model_b, D, K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = 0.0; n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y); loss.backward(); opt.step()
            total += loss.item() * x.shape[0]; n += x.shape[0]
        vs, vy = evaluate(model, val_loader, device)
        thr = per_class_argmax_f1_threshold(vs, vy)
        vpred = (vs > thr).astype(np.int64)
        val_f1 = float(per_sample_f1(vy, vpred))
        val_auroc = float(np.mean([
            roc_auc_score(vy[:, k], vs[:, k])
            for k in range(K) if 0 < vy[:, k].sum() < len(vy)
        ]))
        print(f"  ep{epoch:02d}  train_BCE={total/n:.4f}  val F1={val_f1:.4f}  "
              f"val AUROC={val_auroc:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr

    print(f"\nLoading best val state (val F1 = {best_val_f1:.4f})")
    model.load_state_dict(best_state)

    # Val scores -> use best_thr for both val and test
    vs, vy = evaluate(model, val_loader, device)
    val_pred = (vs > best_thr).astype(np.int64)
    val_metrics = {
        "per_sample_f1": float(per_sample_f1(vy, val_pred)),
        "macro_f1": {k: float(v) for k, v in
                     macro_f1_per_class(vy, val_pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(vy, val_pred)),
        "mean_auroc": float(np.mean([roc_auc_score(vy[:, k], vs[:, k])
                                     for k in range(K)
                                     if 0 < vy[:, k].sum() < len(vy)])),
    }

    # Test (held-out combos)
    ts, ty = evaluate(model, test_loader, device)
    test_pred = (ts > best_thr).astype(np.int64)

    # Per-class AUROC on test (some classes may be all-1 or all-0 because the
    # held-out set is 9 combos; guard against that)
    per_class_auroc = {}
    for k, c in enumerate(class_names):
        if 0 < ty[:, k].sum() < len(ty):
            per_class_auroc[c] = float(roc_auc_score(ty[:, k], ts[:, k]))
        else:
            per_class_auroc[c] = float("nan")

    test_metrics = {
        "per_sample_f1": float(per_sample_f1(ty, test_pred)),
        "macro_f1": {k: float(v) for k, v in
                     macro_f1_per_class(ty, test_pred, class_names).items()},
        "exact_match": float(exact_match_accuracy(ty, test_pred)),
        "per_class_auroc": per_class_auroc,
        "mean_auroc": float(np.nanmean(list(per_class_auroc.values()))),
    }

    # Per-held-out-combo metrics
    per_combo = {}
    combos_arr = np.array(test_combos)
    for combo in heldout:
        mask = combos_arr == combo
        if mask.sum() == 0: continue
        per_combo[combo] = {
            "n_images": int(mask.sum()),
            "per_sample_f1": float(per_sample_f1(ty[mask], test_pred[mask])),
            "exact_match": float(exact_match_accuracy(ty[mask], test_pred[mask])),
        }

    results = {
        "backbone": args.backbone,
        "class_names": class_names,
        "heldout_combos": heldout,
        "trained_combos_count": len(trained),
        "n_train_imgs": len(train_entries),
        "n_val_imgs": len(val_entries),
        "n_test_imgs": len(heldout_entries),
        "thresholds": best_thr.tolist(),
        "val_in_distribution":    val_metrics,
        "test_heldout_combos":    test_metrics,
        "per_heldout_combo":      per_combo,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.save(out_dir / "test_scores.npy", ts)
    np.save(out_dir / "test_labels.npy", ty)
    torch.save(best_state, out_dir / "model.pt")

    md = []
    md.append(f"# Held-out combinations generalization — {args.backbone}\n")
    md.append(f"Held-out combinations ({len(heldout)}): "
              f"`{', '.join(heldout)}`\n")
    md.append(f"train={len(train_entries)}  val={len(val_entries)}  "
              f"test(heldout)={len(heldout_entries)}\n")
    md.append(f"\n### Val (in-distribution, images from trained-on combos)")
    md.append(f"Per-sample F1 = {val_metrics['per_sample_f1']:.4f}  |  "
              f"Macro F1 = {val_metrics['macro_f1']['macro']:.4f}  |  "
              f"Exact match = {val_metrics['exact_match']:.4f}  |  "
              f"Mean AUROC = {val_metrics['mean_auroc']:.4f}\n")
    md.append(f"### Test (held-out combinations — never seen at train time)")
    md.append(f"Per-sample F1 = {test_metrics['per_sample_f1']:.4f}  |  "
              f"Macro F1 = {test_metrics['macro_f1']['macro']:.4f}  |  "
              f"Exact match = {test_metrics['exact_match']:.4f}  |  "
              f"Mean AUROC = {test_metrics['mean_auroc']:.4f}\n")
    md.append("\n### Per-held-out-combination\n")
    md.append("| Combo | Order | Images | Per-sample F1 | Exact match |")
    md.append("|-------|-------|--------|---------------|-------------|")
    for combo in heldout:
        if combo not in per_combo: continue
        order = len(parse_label_tokens(combo))
        m = per_combo[combo]
        md.append(f"| `{combo}` | {order} | {m['n_images']} | "
                  f"{m['per_sample_f1']:.4f} | {m['exact_match']:.4f} |")
    md.append("\n### Per-class AUROC on held-out test")
    md.append("| Class | AUROC |")
    md.append("|-------|-------|")
    for c in class_names:
        v = per_class_auroc[c]
        md.append(f"| {c} | {v:.4f} |" if not np.isnan(v) else f"| {c} | N/A |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\n=== Held-out combinations generalization ({args.backbone}) ===")
    print(f"  Val (in-distribution):  F1 = {val_metrics['per_sample_f1']:.4f}  "
          f"AUROC = {val_metrics['mean_auroc']:.4f}")
    print(f"  Test (held-out combos): F1 = {test_metrics['per_sample_f1']:.4f}  "
          f"AUROC = {test_metrics['mean_auroc']:.4f}")
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
