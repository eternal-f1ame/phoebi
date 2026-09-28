"""Prototype initialization helpers shared by Method A and Method B.

Two strategies:

1. **Pure-culture mean** — when a single-species video exists for every class
   (e.g., folders `b/`, `f/`, `k/`, `p/`), each prototype is the L2-normalized
   mean of all tile embeddings drawn from that video. This is the cleanest
   anchor we can build without supervised training: it places each prototype at
   the centroid of its class manifold under the frozen DINOv2 metric. This is
   the procedure the AnimalCLEF winners used for known-class prototypes.

2. **K-means** (fallback) — if any class lacks a pure-culture video, fall back
   to K-means on the full pool of tile embeddings. The existing
   `initialize_prototypes` in `src.simplex_unmixing.model` already implements
   this; we expose it here for symmetry.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def init_prototypes_from_pure_cultures(
    tile_features: torch.Tensor,
    image_index: torch.Tensor,
    video_ids: List[str],
    class_names: List[str],
) -> Optional[torch.Tensor]:
    """Compute one prototype per class by averaging tile features from pure-culture videos.

    Parameters
    ----------
    tile_features : (N*T, D) — L2-normalized DINOv2 tile embeddings.
    image_index   : (N*T,)   — parent image index for each tile row.
    video_ids     : list of length N — video name for each parent image.
    class_names   : list of length K — canonical class order.

    Returns
    -------
    (K, D) tensor of L2-normalized prototypes if a pure-culture video exists for
    EVERY class (a video named exactly `class_name`), else None — caller falls
    back to K-means.
    """
    if tile_features.shape[0] != image_index.shape[0]:
        raise ValueError(
            f"tile_features has {tile_features.shape[0]} rows but image_index has "
            f"{image_index.shape[0]}"
        )

    K = len(class_names)
    D = tile_features.shape[1]
    protos = torch.empty((K, D), dtype=tile_features.dtype)

    for k, class_name in enumerate(class_names):
        # Image indices whose video name == class_name (pure-culture videos for class k)
        pure_image_idxs = [i for i, v in enumerate(video_ids) if v == class_name]
        if not pure_image_idxs:
            return None
        idx_tensor = torch.tensor(pure_image_idxs, dtype=torch.long)
        mask = torch.isin(image_index, idx_tensor)
        class_tiles = tile_features[mask]
        if class_tiles.shape[0] == 0:
            return None
        proto = class_tiles.mean(dim=0)
        protos[k] = F.normalize(proto, p=2, dim=0)

    return protos
