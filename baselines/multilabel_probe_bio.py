"""Bio-foundation-model encoder probe for PHOEBI.

Probes pathology/biomedical foundation models not available via timm:
  - Phikon (owkin/phikon): ViT-B/16 trained on 6M H&E tiles via DINO.
  - (UNI and others gated/unavailable on this machine.)

Reuses the PHOEBI GPU illumination + 4×4 grid tiling pipeline and the same
linear-probe training as multilabel_probe.py. Appends rows to
outputs/encoder_probe/results.csv and re-renders the LaTeX table.

Usage:
  python baselines/multilabel_probe_bio.py
  python baselines/multilabel_probe_bio.py --models phikon
  python baselines/multilabel_probe_bio.py --render_only
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Bio model wrappers — expose the same interface as timm models:
#   model(tile_batch) -> (B, D) features
#   model.default_cfg  -> {'mean': ..., 'std': ...}
#   model.parameters() -> iterable (for param count)
# ---------------------------------------------------------------------------

class PhikonWrapper(nn.Module):
    """Wraps owkin/phikon (HuggingFace ViT-B/16) to look like a timm model."""

    MODEL_ID = "owkin/phikon"
    default_cfg = {
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    }

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(self.MODEL_ID)
        self._model.eval()
        self.embed_dim = self._model.config.hidden_size  # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(pixel_values=x)
        return out.last_hidden_state[:, 0]  # CLS token: (B, 768)

    def reset_classifier(self, num_classes: int) -> None:
        pass

    @property
    def num_features(self) -> int:
        return self.embed_dim


class BiomedCLIPWrapper(nn.Module):
    """Wraps microsoft/BiomedCLIP visual encoder (open_clip, ViT-B/16, 512-dim)."""

    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # BiomedCLIP uses CLIP normalization (OpenAI stats)
    default_cfg = {
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std":  (0.26862954, 0.26130258, 0.27577711),
    }

    def __init__(self) -> None:
        super().__init__()
        from open_clip import create_model_from_pretrained
        model, _ = create_model_from_pretrained(self.MODEL_ID)
        self._model = model
        self._model.eval()
        self.embed_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._model.encode_image(x)
        return feats  # (B, 512) L2-normalized by open_clip

    def reset_classifier(self, num_classes: int) -> None:
        pass

    @property
    def num_features(self) -> int:
        return self.embed_dim


BIO_MODELS: List[Tuple[str, str, type]] = [
    ("owkin/phikon",    "Phikon ViT-B/16",    PhikonWrapper),
    ("BiomedCLIP-B/16", "BiomedCLIP ViT-B/16", BiomedCLIPWrapper),
]


# ---------------------------------------------------------------------------
# Feature extraction (mirrors extract_features_multicrop_gpu but accepts
# a pre-loaded model object rather than a backbone string)
# ---------------------------------------------------------------------------

def extract_bio_features(
    model: nn.Module,
    image_paths: List[str],
    tile_size: int,
    eval_grid_size: int,
    frame_batch_size: int,
    num_workers: int,
    device: torch.device,
    illum_sigma: float,
    illum_method: str,
    cache_path: Optional[str],
    backbone_tag: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU extraction with PHOEBI illumination + deterministic grid tiling."""
    import warnings
    from src.common.tiling import TileConfig, FullFrameDataset, _grid_offsets
    from src.common.illumination import gpu_normalize_illumination

    tile_cfg = TileConfig(tile_size=tile_size, eval_grid_size=eval_grid_size)
    cfg_dict = asdict(tile_cfg)

    if cache_path and os.path.exists(cache_path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*weights_only=False.*")
            payload = torch.load(cache_path)
        if (
            payload.get("paths") == image_paths
            and payload.get("tile_config") == cfg_dict
            and payload.get("backbone") == backbone_tag
            and payload.get("illum_method") == illum_method
            and payload.get("illum_sigma") == illum_sigma
        ):
            print(f"  [cache hit] {cache_path}")
            return payload["features"], payload["image_index"]

    mean_t = torch.tensor(model.default_cfg["mean"]).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(model.default_cfg["std"]).view(1, 3, 1, 1).to(device)

    dataset = FullFrameDataset(image_paths)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=frame_batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )

    N = len(image_paths)
    T = eval_grid_size ** 2
    D = model.embed_dim
    features    = torch.empty((N * T, D), dtype=torch.float32)
    image_index = torch.empty((N * T,),   dtype=torch.long)
    idx_ptr = 0

    model = model.to(device)
    _offsets_cache: Dict = {}

    with torch.no_grad():
        for frames_uint8, idxs in loader:
            # frames_uint8: (B, 3, H, W) uint8 on CPU
            frames = frames_uint8.to(device)
            B, C, H, W = frames.shape

            # Illumination correction
            frames_f = frames.float() / 255.0
            if illum_method != "none":
                frames_f = gpu_normalize_illumination(
                    frames.to(torch.uint8), illum_sigma, illum_method,
                )
                frames_f = frames_f.float() / 255.0

            # Grid tiling
            hw_key = (H, W)
            if hw_key not in _offsets_cache:
                xs = _grid_offsets(W, tile_size, eval_grid_size)
                ys = _grid_offsets(H, tile_size, eval_grid_size)
                _offsets_cache[hw_key] = (xs, ys)
            xs, ys = _offsets_cache[hw_key]

            tiles_list = []
            for y in ys:
                for x in xs:
                    tiles_list.append(frames_f[:, :, y:y+tile_size, x:x+tile_size])
            tiles = torch.stack(tiles_list, dim=1)  # (B, T, 3, ts, ts)
            tiles = tiles.reshape(B * T, C, tile_size, tile_size)

            # ImageNet normalization
            tiles = (tiles - mean_t) / std_t

            with torch.amp.autocast(device_type=device.type):
                out = model(tiles).cpu().float()

            tile_idxs = idxs.repeat_interleave(T)
            end_ptr = idx_ptr + out.shape[0]
            features[idx_ptr:end_ptr]    = out
            image_index[idx_ptr:end_ptr] = tile_idxs
            idx_ptr = end_ptr

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({
            "features": features, "image_index": image_index,
            "paths": image_paths, "tile_config": cfg_dict,
            "backbone": backbone_tag, "illum_method": illum_method,
            "illum_sigma": illum_sigma, "mode": "eval",
        }, cache_path)
    return features, image_index


# ---------------------------------------------------------------------------
# Probe one bio model (mirrors multilabel_probe.probe_one_encoder)
# ---------------------------------------------------------------------------

def probe_one_bio_model(
    backbone_tag: str,
    pretty: str,
    model_cls: type,
    splits_path: str,
    tile_size: int,
    eval_grid_size: int,
    frame_batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_root: str,
    lr_grid: Sequence[float],
    epochs: int,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> None:
    from src.common.io import load_real_split
    from src.common.metrics import per_sample_f1, macro_f1_per_class, exact_match_accuracy
    from baselines.multilabel_probe import pick_per_class_thresholds, quantile_thresholds

    print(f"\n=== {pretty}  ({backbone_tag}) ===")
    model = model_cls()
    model.eval()
    D = model.embed_dim

    stash = {}
    for split in ("train", "val", "test"):
        paths, labels, class_names, _ = load_real_split(splits_path, split)
        cache_dir = os.path.join(cache_root, backbone_tag.replace("/", "_"), "features")
        cache_path = os.path.join(cache_dir, f"{split}_features_cache.pt")
        feats, img_idx = extract_bio_features(
            model, paths, tile_size, eval_grid_size,
            frame_batch_size, num_workers, device,
            illum_sigma=64.0, illum_method="divide",
            cache_path=cache_path, backbone_tag=backbone_tag,
        )
        feats = F.normalize(feats, dim=1)
        stash[split] = (feats, img_idx, labels, class_names)

    train_feats, train_idx, train_labels, class_names = stash["train"]
    val_feats,   val_idx,   val_labels,   _           = stash["val"]
    test_feats,  test_idx,  test_labels,  _           = stash["test"]

    train_tile_y = torch.from_numpy(train_labels.astype(np.float32))[train_idx]

    # Train linear probe (same as multilabel_probe)
    from src.common.features import scatter_mean_by_image
    K = train_tile_y.shape[1]
    best: Dict = {"val_f1": -1.0}
    t0 = time.time()

    train_feats_gpu = train_feats.to(device)
    train_tile_y_gpu = train_tile_y.to(device)
    val_feats_gpu   = val_feats.to(device)

    for lr in lr_grid:
        torch.manual_seed(seed)
        head = nn.Linear(D, K).to(device)
        opt  = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9,
                               nesterov=True, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        N_tiles = train_feats_gpu.shape[0]
        g = torch.Generator(device="cpu").manual_seed(seed)
        for _epoch in range(epochs):
            perm = torch.randperm(N_tiles, generator=g)
            for start in range(0, N_tiles, batch_size):
                idx = perm[start:start+batch_size]
                logits = head(train_feats_gpu[idx])
                loss = loss_fn(logits, train_tile_y_gpu[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val_sig = torch.sigmoid(head(val_feats_gpu)).cpu()
        val_scores = scatter_mean_by_image(val_sig, val_idx,
                                           val_labels.shape[0]).numpy()
        thr = pick_per_class_thresholds(val_scores, val_labels)
        val_pred = (val_scores > thr).astype(np.int64)
        vf1 = per_sample_f1(val_labels, val_pred)
        print(f"  lr={lr:.2e}  val F1={vf1:.4f}")
        if vf1 > best["val_f1"]:
            best = {"val_f1": vf1, "lr": lr,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()},
                    "thresholds": thr}

    t_train = time.time() - t0
    head.load_state_dict(best["state"])

    # Test scoring
    with torch.no_grad():
        test_sig = torch.sigmoid(head(test_feats.to(device))).cpu()
    test_scores = scatter_mean_by_image(test_sig, test_idx, test_labels.shape[0]).numpy()
    test_pred   = (test_scores > best["thresholds"]).astype(np.int64)
    per_sample  = per_sample_f1(test_labels, test_pred)
    macro       = macro_f1_per_class(test_labels, test_pred, class_names)
    em          = exact_match_accuracy(test_labels, test_pred)

    # q05 thresholds
    with torch.no_grad():
        val_sig2 = torch.sigmoid(head(val_feats.to(device))).cpu()
    val_scores2 = scatter_mean_by_image(val_sig2, val_idx, val_labels.shape[0]).numpy()
    thr_q05 = quantile_thresholds(val_scores2, val_labels, q=0.05)
    psf_q05 = per_sample_f1(test_labels, (test_scores > thr_q05).astype(np.int64))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"  test per-sample F1 = {per_sample:.4f}  macro = {macro['macro']:.4f}  exact = {em:.4f}")
    print(f"  (q05 thresholds: per-sample F1 = {psf_q05:.4f})")
    print(f"  train time = {t_train:.0f}s  params = {n_params:.0f}M  feat_dim = {D}")

    # Append to shared CSV (same path as multilabel_probe.py: outputs/encoder_probe/results.csv)
    csv_path = os.path.join(cache_root, "results.csv")
    row = {
        "backbone": backbone_tag, "pretty": pretty,
        "params_M": f"{n_params:.1f}", "feat_dim": str(D),
        "per_sample_f1": f"{per_sample:.4f}", "macro_f1": f"{macro['macro']:.4f}",
        "exact_match": f"{em:.4f}",
        "per_sample_f1_q05": f"{psf_q05:.4f}",
        "chosen_lr": f"{best['lr']:.2e}", "train_seconds": f"{t_train:.1f}",
    }
    for c in class_names:
        row[f"f1_{c}"] = f"{macro[c]:.4f}"

    fieldnames = [
        "backbone", "pretty", "params_M", "feat_dim",
        "per_sample_f1", "macro_f1", "exact_match",
        "f1_bs", "f1_bt", "f1_fj", "f1_ka", "f1_mx", "f1_pf",
        "per_sample_f1_q05", "chosen_lr", "train_seconds",
    ]
    csv_path = os.path.abspath(csv_path)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Appended to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=[tag for tag, _, _ in BIO_MODELS])
    ap.add_argument("--splits_path", default="data/splits.json")
    ap.add_argument("--tile_size",   type=int, default=224)
    ap.add_argument("--eval_grid_size", type=int, default=4)
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--batch_size",  type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed",        type=int, default=1337)
    ap.add_argument("--lr_grid",     nargs="+", type=float,
                    default=[1e-2, 3e-2, 1e-1])
    ap.add_argument("--cache_root",  default="outputs/encoder_probe")
    ap.add_argument("--render_only", action="store_true")
    args = ap.parse_args()

    if args.render_only:
        from baselines.multilabel_probe import render_latex_table
        render_latex_table(
            os.path.join(args.cache_root, "results.csv"),
            "outputs/encoder_probe/tab_encoder_probe.tex",
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wanted = set(args.models)
    for tag, pretty, cls in BIO_MODELS:
        if tag not in wanted and pretty not in wanted and "phikon" not in tag.lower():
            continue
        if tag not in wanted and pretty not in wanted:
            if not any(w.lower() in tag.lower() or w.lower() in pretty.lower()
                       for w in wanted):
                continue
        probe_one_bio_model(
            backbone_tag=tag, pretty=pretty, model_cls=cls,
            splits_path=args.splits_path,
            tile_size=args.tile_size, eval_grid_size=args.eval_grid_size,
            frame_batch_size=args.frame_batch_size, num_workers=args.num_workers,
            device=device, cache_root=args.cache_root,
            lr_grid=args.lr_grid, epochs=args.epochs,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
            seed=args.seed,
        )

    # Re-render table
    from baselines.multilabel_probe import render_latex_table
    render_latex_table(
        os.path.join(args.cache_root, "results.csv"),
        "outputs/encoder_probe/tab_encoder_probe.tex",
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
