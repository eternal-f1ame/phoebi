import os
import warnings
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from src.common.tiling import FullFrameDataset, MultiCropDataset, TileConfig, PreprocessFn, _grid_offsets
from src.common.illumination import gpu_normalize_illumination


def _load_dinov2(
    backbone: str,
    device: torch.device,
    img_size: Optional[int] = None,
) -> torch.nn.Module:
    """Load any timm backbone at a specific input resolution when supported.

    ViTs (DINOv2/v3, CLIP, SigLIP, EVA-02) accept `img_size=` and interpolate
    positional embeddings. CNNs (ResNet, ConvNeXt) and hybrid backbones (DaViT)
    do not; we fall back to the default input size.
    """
    kwargs = {"pretrained": True}
    if img_size is not None:
        kwargs["img_size"] = img_size
    # UNI (MahmoodLab/UNI) uses LayerScale; requires init_values to load cleanly.
    if "MahmoodLab/UNI" in backbone or "MahmoodLab/uni" in backbone:
        kwargs["init_values"] = 1e-5
        kwargs["dynamic_img_size"] = True
        kwargs.pop("img_size", None)  # UNI uses dynamic sizing
    try:
        model = timm.create_model(backbone, **kwargs)
    except TypeError:
        model = timm.create_model(backbone, pretrained=True)
    model.reset_classifier(0)
    model.to(device)
    model.eval()
    return model


def extract_features_multicrop(
    image_paths: List[str],
    tile_config: TileConfig,
    backbone: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_path: Optional[str] = None,
    mode: str = "eval",
    preprocess_fn: Optional[PreprocessFn] = None,
    seed: int = 1337,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 features for every tile of every image.

    Returns
    -------
    features    : Tensor of shape (N * T, D)
    image_index : LongTensor of shape (N * T,) — for each row, the index into
                  `image_paths` of the parent image. Use scatter-mean to aggregate
                  per-tile predictions to per-image predictions.

    The cache key includes the tile_config and mode, so changing T or tile_size
    invalidates the cache automatically.
    """
    cfg_dict = asdict(tile_config)
    cache_key = {"paths": image_paths, "tile_config": cfg_dict, "mode": mode, "backbone": backbone}

    if cache_path and os.path.exists(cache_path):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch.load` with `weights_only=False`",
            )
            payload = torch.load(cache_path)
        if (
            payload.get("paths") == image_paths
            and payload.get("tile_config") == cfg_dict
            and payload.get("mode") == mode
            and payload.get("backbone") == backbone
        ):
            return payload["features"], payload["image_index"]

    model = _load_dinov2(backbone, device, img_size=tile_config.tile_size)

    dataset = MultiCropDataset(
        image_paths=image_paths,
        tile_config=tile_config,
        mode=mode,
        preprocess_fn=preprocess_fn,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    total_tiles = len(dataset)
    features = torch.empty((total_tiles, model.num_features), dtype=torch.float32)
    image_index = torch.empty((total_tiles,), dtype=torch.long)
    idx_ptr = 0

    with torch.no_grad():
        for tiles, idxs in tqdm(loader, desc=f"Tiling+extracting ({mode})", unit="batch"):
            tiles = tiles.to(device)
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                out = model(tiles)
            out = out.detach().cpu()
            end_ptr = idx_ptr + out.shape[0]
            features[idx_ptr:end_ptr] = out
            image_index[idx_ptr:end_ptr] = idxs
            idx_ptr = end_ptr

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(
            {
                "features": features,
                "image_index": image_index,
                "paths": image_paths,
                "tile_config": cfg_dict,
                "mode": mode,
                "backbone": backbone,
            },
            cache_path,
        )

    return features, image_index


_IMAGENET_MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD_TENSOR = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def extract_features_multicrop_gpu(
    image_paths: List[str],
    tile_config: TileConfig,
    backbone: str,
    frame_batch_size: int,
    num_workers: int,
    device: torch.device,
    illum_sigma: float = 64.0,
    illum_method: str = "divide",
    illum_downsample: int = 8,
    cache_path: Optional[str] = None,
    frame_transform=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-resident multi-crop feature extraction with illumination on the GPU.

    Workers do nothing but JPEG decode (the floor cost on a CPU-starved system); the
    main process moves each frame batch to GPU where illumination, deterministic
    grid tiling, ImageNet normalization, and DINOv2 forward all happen in one shot.
    Eval mode only (deterministic 4x4 grid).

    Returns features (N*T, D) and image_index (N*T,) — same layout as the CPU path.
    """
    cfg_dict = asdict(tile_config)
    if cache_path and os.path.exists(cache_path):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch.load` with `weights_only=False`",
            )
            payload = torch.load(cache_path)
        if (
            payload.get("paths") == image_paths
            and payload.get("tile_config") == cfg_dict
            and payload.get("mode") == "eval"
            and payload.get("backbone") == backbone
            and payload.get("illum_method") == illum_method
            and payload.get("illum_sigma") == illum_sigma
        ):
            return payload["features"], payload["image_index"]

    model = _load_dinov2(backbone, device, img_size=tile_config.tile_size)

    # Per-encoder normalization (CLIP/SigLIP/EVA-02/ViT-IN21k differ from ImageNet).
    mean_tup = model.default_cfg.get("mean", (0.485, 0.456, 0.406))
    std_tup = model.default_cfg.get("std", (0.229, 0.224, 0.225))
    norm_mean = torch.tensor(mean_tup, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(std_tup, device=device).view(1, 3, 1, 1)

    dataset = FullFrameDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=frame_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    grid_size = tile_config.eval_grid_size
    tile_size = tile_config.tile_size
    tiles_per_image = grid_size * grid_size

    total_tiles = len(image_paths) * tiles_per_image
    features = torch.empty((total_tiles, model.num_features), dtype=torch.float32)
    image_index = torch.empty((total_tiles,), dtype=torch.long)
    idx_ptr = 0

    # Pre-compute grid offsets — they only depend on the (fixed) frame size, but we
    # do not assume that until we see the first batch.
    grid_offsets_cache = {}

    with torch.no_grad():
        for frames_uint8, idxs in tqdm(loader, desc="Tiling+extracting (gpu)", unit="batch"):
            frames_uint8 = frames_uint8.to(device, non_blocking=True)
            if frame_transform is not None:
                # Frame-level test-time augmentation (rotation, flip, etc.)
                # applied before illumination correction.
                frames_uint8 = frame_transform(frames_uint8)
            B, C, H, W = frames_uint8.shape

            # Illumination on GPU
            frames_f = gpu_normalize_illumination(
                frames_uint8,
                sigma=illum_sigma,
                method=illum_method,
                downsample=illum_downsample,
            )

            # Deterministic grid offsets (cache by (H, W))
            key = (H, W)
            if key not in grid_offsets_cache:
                grid_offsets_cache[key] = (
                    _grid_offsets(H, tile_size, grid_size),
                    _grid_offsets(W, tile_size, grid_size),
                )
            ys, xs = grid_offsets_cache[key]

            # Extract tiles via slicing — order is row-major (matches grid_tiles())
            tile_list = []
            for y in ys:
                for x in xs:
                    tile_list.append(frames_f[:, :, y : y + tile_size, x : x + tile_size])
            # (T, B, C, h, w) -> (B, T, C, h, w) -> (B*T, C, h, w)
            tiles = torch.stack(tile_list, dim=0).transpose(0, 1).reshape(
                B * tiles_per_image, C, tile_size, tile_size
            )

            # Per-encoder normalized [0, 1] inputs
            tiles = tiles / 255.0
            tiles = (tiles - norm_mean) / norm_std

            # DINOv2 forward (chunked so a giant batch does not OOM)
            chunk = 256
            forward_chunks = []
            for i in range(0, tiles.shape[0], chunk):
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    feat = model(tiles[i : i + chunk])
                forward_chunks.append(feat.float())
            feats = torch.cat(forward_chunks, dim=0).cpu()

            # image_index: each idx in `idxs` repeated tiles_per_image times (row-major matches above)
            tile_image_idx = idxs.repeat_interleave(tiles_per_image)

            end_ptr = idx_ptr + feats.shape[0]
            features[idx_ptr:end_ptr] = feats
            image_index[idx_ptr:end_ptr] = tile_image_idx
            idx_ptr = end_ptr

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(
            {
                "features": features,
                "image_index": image_index,
                "paths": image_paths,
                "tile_config": cfg_dict,
                "mode": "eval",
                "backbone": backbone,
                "illum_method": illum_method,
                "illum_sigma": illum_sigma,
            },
            cache_path,
        )

    return features, image_index


def scatter_mean_by_image(
    per_tile_values: torch.Tensor,
    image_index: torch.Tensor,
    num_images: int,
) -> torch.Tensor:
    """Mean-aggregate per-tile values over their parent images.

    per_tile_values : Tensor of shape (N*T, ...) — per-tile predictions/scores.
    image_index     : LongTensor of shape (N*T,) — parent image index per row.
    num_images      : N, the number of distinct images.

    Returns Tensor of shape (N, ...) where row i is the mean over rows whose
    image_index == i. Justified by Assumption H: each tile is an unbiased estimator of
    the same per-image quantity, so the mean has variance O(1/T).
    """
    if per_tile_values.shape[0] != image_index.shape[0]:
        raise ValueError(
            f"per_tile_values has {per_tile_values.shape[0]} rows but image_index has "
            f"{image_index.shape[0]}"
        )
    feat_shape = per_tile_values.shape[1:]
    out = torch.zeros((num_images, *feat_shape), dtype=per_tile_values.dtype, device=per_tile_values.device)
    counts = torch.zeros((num_images,), dtype=torch.long, device=per_tile_values.device)

    expand_index = image_index
    for _ in feat_shape:
        expand_index = expand_index.unsqueeze(-1)
    expand_index = expand_index.expand_as(per_tile_values)

    out.scatter_add_(0, expand_index, per_tile_values)
    counts.scatter_add_(0, image_index, torch.ones_like(image_index))

    counts_view = counts.clamp_min(1).view(num_images, *([1] * len(feat_shape))).to(out.dtype)
    return out / counts_view

