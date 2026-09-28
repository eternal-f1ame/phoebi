"""Multi-crop tiling for full-resolution microscopy frames.

This module is the operational realization of Assumption H (spatial homogeneity): every
sufficiently large crop of a frame inherits the frame's label, so we can sample many
i.i.d. tiles per frame and aggregate their predictions by averaging.

Design choices and rationale
----------------------------
- Tiles are taken from the *full-resolution* PIL image (no pre-resize). The point of
  tiling is to preserve the per-cell pixel budget that distinguishes morphologies.
- Default tile_size = 224, which is the standard DINOv2 input (16 patches x 14 px) and
  requires no further resizing.
- Training uses random crops (+ optional horizontal/vertical flips). Each frame
  contributes `train_tiles_per_image` tiles per epoch, drawn independently.
- Inference uses a deterministic n x n grid evenly spanning the frame:
      x_i = round( i * (W - tile_size) / (n - 1) )    for i in 0..n-1
  Corners + interior are covered with no overlap. With n = 4 on a 2592 x 1944 frame,
  the four column positions are roughly [0, 789, 1579, 2368] and the four row positions
  [0, 573, 1147, 1720].
- A `preprocess_fn` hook lets Phase 2 (illumination normalization) plug in without
  changing the dataset signature.
"""

import os
import random
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


PreprocessFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class TileConfig:
    """Hyperparameters for the multi-crop pipeline."""

    tile_size: int = 224                # DINOv2 standard input (16 patches x 14 px)
    train_tiles_per_image: int = 16     # random crops per frame per epoch
    eval_grid_size: int = 4             # 4x4 = 16 deterministic tiles per frame
    train_random_hflip: bool = True
    train_random_vflip: bool = True

    @property
    def eval_tiles_per_image(self) -> int:
        return self.eval_grid_size * self.eval_grid_size


def _grid_offsets(image_dim: int, tile_size: int, n: int) -> List[int]:
    """Top-left coordinates for an n-cell deterministic grid spanning [0, image_dim).

    Returns n integers in [0, image_dim - tile_size]. With n = 1 returns a centered tile.
    """
    if image_dim < tile_size:
        raise ValueError(f"image dimension {image_dim} smaller than tile_size {tile_size}")
    if n <= 1:
        return [(image_dim - tile_size) // 2]
    span = image_dim - tile_size
    return [round(i * span / (n - 1)) for i in range(n)]


def grid_tiles(image: Image.Image, tile_size: int, grid_size: int) -> List[Image.Image]:
    """Deterministic grid_size x grid_size tiles spanning the full image."""
    w, h = image.size
    xs = _grid_offsets(w, tile_size, grid_size)
    ys = _grid_offsets(h, tile_size, grid_size)
    tiles: List[Image.Image] = []
    for y in ys:
        for x in xs:
            tiles.append(image.crop((x, y, x + tile_size, y + tile_size)))
    return tiles


def random_tile(image: Image.Image, tile_size: int, rng: random.Random) -> Image.Image:
    """One uniformly random tile_size x tile_size crop from the full image."""
    w, h = image.size
    if w < tile_size or h < tile_size:
        raise ValueError(f"image {w}x{h} smaller than tile_size {tile_size}")
    x = rng.randint(0, w - tile_size)
    y = rng.randint(0, h - tile_size)
    return image.crop((x, y, x + tile_size, y + tile_size))


class FullFrameDataset(Dataset):
    """Returns (full_uint8_chw_tensor, image_idx) — no preprocessing, just JPEG decode.

    Used by the GPU illumination path: workers do nothing but decode JPEGs (the floor
    cost on a CPU-starved system), then the main process moves the full frame to GPU
    where illumination, tiling, and DINOv2 forward all happen in one shot.
    """

    def __init__(self, image_paths: List[str]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        arr = np.array(img, dtype=np.uint8, copy=True)    # (H, W, 3) — writable copy
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W) uint8
        return tensor, idx


# DINOv2 was trained with ImageNet normalization stats.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


def _tile_to_tensor(tile: Image.Image) -> torch.Tensor:
    return _NORMALIZE(_TO_TENSOR(tile))


class MultiCropDataset(Dataset):
    """Yields (tile_tensor, image_idx) pairs.

    Length is `len(image_paths) * tiles_per_image`. The image is loaded fresh each time
    (no caching) because PIL images are too large to keep in memory and the I/O cost is
    dwarfed by the DINOv2 forward pass.

    mode="eval"  : deterministic grid (eval_grid_size^2 tiles per image, in row-major order).
    mode="train" : random crops (train_tiles_per_image tiles per image), optional flips.
    """

    def __init__(
        self,
        image_paths: List[str],
        tile_config: TileConfig,
        mode: str = "eval",
        preprocess_fn: Optional[PreprocessFn] = None,
        seed: int = 1337,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.image_paths = image_paths
        self.config = tile_config
        self.mode = mode
        self.preprocess_fn = preprocess_fn
        self.seed = seed
        self.tiles_per_image = (
            tile_config.train_tiles_per_image if mode == "train" else tile_config.eval_tiles_per_image
        )
        # Per-worker last-image cache. With shuffle=False, consecutive __getitem__ calls
        # within a worker hit the same image_idx for `tiles_per_image` calls in a row.
        # Caching the last preprocessed image makes preprocess_fn (the slow Gaussian blur
        # in illumination normalization) run once per image instead of once per tile —
        # a tiles_per_image speedup on the data path.
        self._cache_image_idx: int = -1
        self._cache_image: Optional[Image.Image] = None

    def __len__(self) -> int:
        return len(self.image_paths) * self.tiles_per_image

    def _load_image(self, idx: int) -> Image.Image:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.preprocess_fn is not None:
            arr = np.asarray(img)
            arr = self.preprocess_fn(arr)
            img = Image.fromarray(arr.astype(np.uint8))
        return img

    def __getitem__(self, flat_idx: int) -> Tuple[torch.Tensor, int]:
        image_idx, tile_idx = divmod(flat_idx, self.tiles_per_image)
        if self._cache_image is None or self._cache_image_idx != image_idx:
            self._cache_image = self._load_image(image_idx)
            self._cache_image_idx = image_idx
        img = self._cache_image

        if self.mode == "eval":
            # Lazy: regenerate the full grid each call. With ~16 tiles this is cheap
            # relative to the DINOv2 forward pass and avoids caching huge PIL images.
            tiles = grid_tiles(img, self.config.tile_size, self.config.eval_grid_size)
            tile = tiles[tile_idx]
        else:
            rng = random.Random((self.seed * 1_000_003) ^ flat_idx)
            tile = random_tile(img, self.config.tile_size, rng)
            if self.config.train_random_hflip and rng.random() < 0.5:
                tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
            if self.config.train_random_vflip and rng.random() < 0.5:
                tile = tile.transpose(Image.FLIP_TOP_BOTTOM)

        return _tile_to_tensor(tile), image_idx
