"""Streaming extraction of per-image *units* at a chosen granularity.

A unit is the thing the decoder scores and the simplex unmixes. The paper's unit is the
CLS token of a 224 px tile (16 units per image). This module generalises that:

    ("cls",       224, grid 4)     16 units     the paper
    ("patchmean", 224, grid 4)     16 units     mean of the 256 patch tokens
    ("patch",     224, grid 4)     4096 units   every patch token
    ("cls",       112, grid 8)     64 units
    ("patchmean", 112, grid 8)     64 units
    ("cls",        56, grid 16)    256 units

Every grid covers the same 896 px span of the 1024 px image as the paper's 4x4 grid of
224. Illumination correction, tiling and the DINOv2 forward all run on the device, as in
`extract_features_multicrop_gpu`; only the read-out of the transformer differs, and
"cls" reproduces that function's output (timm pools by the normed CLS token when
`num_classes=0`).

One forward at a tile size yields every unit kind for that size, so callers group
granularities by tile size. Units are yielded per frame batch and never stored here,
because per-patch units for the LCO training split would be ~260 GB.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.features import _load_dinov2
from src.common.illumination import gpu_normalize_illumination
from src.common.tiling import FullFrameDataset, _grid_offsets

KINDS = ("cls", "patchmean", "patch")


@dataclass(frozen=True)
class Granularity:
    kind: str
    tile_size: int
    grid: int

    @property
    def name(self) -> str:
        return f"{self.kind}{self.tile_size}"

    @property
    def tiles_per_image(self) -> int:
        return self.grid * self.grid


GRANULARITIES: Dict[str, Granularity] = {
    "cls224": Granularity("cls", 224, 4),
    "patchmean224": Granularity("patchmean", 224, 4),
    "patch224": Granularity("patch", 224, 4),
    "cls112": Granularity("cls", 112, 8),
    "patchmean112": Granularity("patchmean", 112, 8),
    "cls56": Granularity("cls", 56, 16),
}


def group_by_tile_size(names: Sequence[str]) -> Dict[Tuple[int, int], List[Granularity]]:
    groups: Dict[Tuple[int, int], List[Granularity]] = {}
    for n in names:
        g = GRANULARITIES[n]
        groups.setdefault((g.tile_size, g.grid), []).append(g)
    return groups


class UnitExtractor:
    """One backbone at one input size; turns frame batches into units of several kinds."""

    def __init__(self, backbone: str, tile_size: int, grid: int, kinds: Sequence[str],
                 device: torch.device, illum_method: str = "divide",
                 illum_sigma: float = 64.0, forward_chunk: int = 256) -> None:
        for k in kinds:
            if k not in KINDS:
                raise ValueError(k)
        self.tile_size, self.grid, self.kinds = tile_size, grid, list(kinds)
        self.device = device
        self.illum_method, self.illum_sigma = illum_method, illum_sigma
        self.forward_chunk = forward_chunk
        self.model = _load_dinov2(backbone, device, img_size=tile_size)
        self.num_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        self.dim = int(self.model.num_features)
        mean_tup = self.model.default_cfg.get("mean", (0.485, 0.456, 0.406))
        std_tup = self.model.default_cfg.get("std", (0.229, 0.224, 0.225))
        self.norm_mean = torch.tensor(mean_tup, device=device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(std_tup, device=device).view(1, 3, 1, 1)
        self._offsets: Dict[Tuple[int, int], Tuple[List[int], List[int]]] = {}

    @property
    def tiles_per_image(self) -> int:
        return self.grid * self.grid

    @torch.no_grad()
    def tiles_from_frames(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) uint8 on device -> (B*T, 3, s, s) normalised tiles, row-major grid."""
        frames = gpu_normalize_illumination(frames_uint8, sigma=self.illum_sigma,
                                            method=self.illum_method, downsample=8)
        B, C, H, W = frames.shape
        s, n = self.tile_size, self.grid
        key = (H, W)
        if key not in self._offsets:
            self._offsets[key] = (_grid_offsets(H, s, n), _grid_offsets(W, s, n))
        ys, xs = self._offsets[key]
        tiles = torch.stack([frames[:, :, y:y + s, x:x + s] for y in ys for x in xs], dim=0)
        tiles = tiles.transpose(0, 1).reshape(B * n * n, C, s, s) / 255.0
        return (tiles - self.norm_mean) / self.norm_std

    @torch.no_grad()
    def units_from_tiles(self, tiles: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(M, 3, s, s) -> {kind: (M, U_kind, D)} L2-normalised units."""
        parts: Dict[str, List[torch.Tensor]] = {k: [] for k in self.kinds}
        for i in range(0, tiles.shape[0], self.forward_chunk):
            with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                tok = self.model.forward_features(tiles[i:i + self.forward_chunk])
            tok = tok.float()
            if "cls" in parts:
                parts["cls"].append(F.normalize(tok[:, :1], p=2, dim=-1))
            if "patchmean" in parts:
                parts["patchmean"].append(F.normalize(tok[:, self.num_prefix:].mean(dim=1, keepdim=True), p=2, dim=-1))
            if "patch" in parts:
                parts["patch"].append(F.normalize(tok[:, self.num_prefix:], p=2, dim=-1))
        return {k: torch.cat(v, dim=0) for k, v in parts.items()}

    @torch.no_grad()
    def units_from_frames(self, frames_uint8: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(B, 3, H, W) uint8 -> {kind: (B, T*U_kind, D)}, tiles in row-major grid order."""
        B = frames_uint8.shape[0]
        out = self.units_from_tiles(self.tiles_from_frames(frames_uint8))
        return {k: v.reshape(B, -1, v.shape[-1]) for k, v in out.items()}


def iter_frame_batches(image_paths: List[str], frame_batch_size: int, num_workers: int,
                       device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    loader = DataLoader(FullFrameDataset(image_paths), batch_size=frame_batch_size,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"),
                        shuffle=False)
    for frames, idxs in loader:
        yield frames.to(device, non_blocking=True), idxs


def iter_image_units(extractor: UnitExtractor, image_paths: List[str], frame_batch_size: int,
                     num_workers: int) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Yields (image indices (B,), {kind: units (B, N_units, D)} on device)."""
    for frames, idxs in iter_frame_batches(image_paths, frame_batch_size, num_workers,
                                           extractor.device):
        yield idxs, extractor.units_from_frames(frames)


def fixed_unit_subset(n_units: int, per_image: Optional[int], seed: int) -> Optional[torch.Tensor]:
    """A fixed random subset of unit positions, shared by every image of a split."""
    if per_image is None or n_units <= per_image:
        return None
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_units, generator=g)[:per_image].sort().values
