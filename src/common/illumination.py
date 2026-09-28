"""Illumination normalization for microscopy frames.

Phase-contrast microscope images have a slow-varying multiplicative gain field caused
by the condenser/lamp geometry. In the PHOEBI frames this manifests as a pink/white
hotspot in the lower-right corner. Without correction, DINOv2 learns to lean on the
hotspot as a feature.

Model
-----
We assume the observed pixel intensity factorizes as

    I_obs(c, x, y) = G(c, x, y) * I_true(c, x, y)

where G is the slow gain field and I_true is the underlying scene. A heavy Gaussian
blur of I_obs estimates G * <I_true>_local (a smoothed version of the observation),
which to first order is proportional to G. Dividing by this estimate produces

    I_corr(c, x, y) ≈ I_true(c, x, y) / <I_true(c, x, y)>_local

i.e. a contrast-normalized image where the gain has been removed. We then rescale by
the global background mean to keep the result in roughly the original intensity range.

Why per-channel
---------------
The hotspot has a color cast, not just a brightness gradient. Per-channel division
removes both. The cells themselves are roughly grayscale at the per-tile scale, so
treating channels independently does not distort their color.

Why sigma = 64 (default)
------------------------
The hotspot has a length scale of several hundred pixels at native 2592x1944
resolution; cells are 5-20 pixels. A 64-pixel Gaussian removes the hotspot (its energy
lives at frequencies far below 1/64 cycles/pixel) without flattening individual cells.
This is the bandlimit Bayesian-deconvolution baseline used in PlantCLEF/AnimalCLEF
preprocessing pipelines.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter


def _gaussian_2d_fast(channel: np.ndarray, sigma: float, downsample: int) -> np.ndarray:
    """Equivalent to gaussian_filter(channel, sigma) but ~50-100x faster for large sigma.

    Trick: downsample by `downsample`, blur on the small image with sigma/downsample,
    upsample back. Uses PIL.Image.resize (bilinear) for the resamples — much faster
    than scipy.ndimage.zoom on our 2592x1944 frames. The result is visually
    indistinguishable for our use case (estimating a slow-varying illumination field)
    and the FLOPs scale as 1/downsample^2.
    """
    if downsample <= 1:
        return gaussian_filter(channel, sigma=sigma, mode="reflect")

    h, w = channel.shape
    target_sigma = max(sigma / downsample, 0.5)
    small_w = max(1, w // downsample)
    small_h = max(1, h // downsample)

    # Downsample with PIL bilinear (much faster than scipy.ndimage.zoom)
    pil_full = Image.fromarray(channel.astype(np.float32), mode="F")
    pil_small = pil_full.resize((small_w, small_h), resample=Image.BILINEAR)
    small = np.asarray(pil_small, dtype=np.float32)

    # Blur on the small image
    blurred_small = gaussian_filter(small, sigma=target_sigma, mode="reflect")

    # Upsample back to original size
    pil_blur = Image.fromarray(blurred_small, mode="F")
    pil_upsampled = pil_blur.resize((w, h), resample=Image.BILINEAR)
    upsampled = np.asarray(pil_upsampled, dtype=np.float32)

    return upsampled.astype(channel.dtype)


def estimate_background(
    image: np.ndarray,
    sigma: float = 64.0,
    downsample: int = 8,
) -> np.ndarray:
    """Slow-varying background estimate via per-channel large-sigma Gaussian blur.

    Parameters
    ----------
    image      : (H, W) or (H, W, C) array, any numeric dtype.
    sigma      : Gaussian standard deviation in pixels (at the original resolution).
    downsample : downsample factor for the fast Gaussian path. Default 8 is safe for
                 sigma >= 32; pass 1 to disable the optimization.

    Returns
    -------
    Background array of the same shape as `image`, in float32.
    """
    img_f = image.astype(np.float32)
    if img_f.ndim == 2:
        return _gaussian_2d_fast(img_f, sigma=sigma, downsample=downsample)
    if img_f.ndim == 3:
        bg = np.empty_like(img_f)
        for c in range(img_f.shape[2]):
            bg[..., c] = _gaussian_2d_fast(img_f[..., c], sigma=sigma, downsample=downsample)
        return bg
    raise ValueError(f"image must be 2-D or 3-D, got shape {image.shape}")


def normalize_illumination(
    image: np.ndarray,
    sigma: float = 64.0,
    method: str = "divide",
    eps: float = 1.0,
) -> np.ndarray:
    """Flatten the illumination field and return a uint8 image of the same shape.

    method = "divide"   : I_corr = (I / (bg + eps)) * mean(bg)
    method = "subtract" : I_corr = I - bg + mean(bg)
    method = "none"     : passthrough (returns the input as uint8)

    Both correction methods preserve the original global brightness so downstream
    DINOv2 (which expects ImageNet-normalized RGB) sees a familiar intensity range.
    """
    if method == "none":
        return image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

    img_f = image.astype(np.float32)
    bg = estimate_background(img_f, sigma=sigma)
    bg_mean = float(bg.mean())

    if method == "divide":
        result = (img_f / (bg + eps)) * bg_mean
    elif method == "subtract":
        result = img_f - bg + bg_mean
    else:
        raise ValueError(f"method must be 'divide', 'subtract', or 'none'; got {method!r}")

    return np.clip(result, 0, 255).astype(np.uint8)


def gpu_normalize_illumination(
    frames_uint8: torch.Tensor,
    sigma: float = 64.0,
    method: str = "divide",
    downsample: int = 8,
    eps: float = 1.0,
) -> torch.Tensor:
    """GPU-native illumination normalization for a batch of full-resolution frames.

    Parameters
    ----------
    frames_uint8 : (B, 3, H, W) uint8 tensor on the target device. Workers decode
                   JPEGs into this format; everything below this point runs on GPU.
    sigma        : Gaussian standard deviation in pixels at the *original* resolution.
    method       : "divide" | "subtract" | "none".
    downsample   : downsample factor for the cheap separable Gaussian (default 8).
    eps          : numerical stabilizer for the divide path.

    Returns
    -------
    (B, 3, H, W) float32 tensor in [0, 255]. Same correction model as the CPU path.
    """
    if method == "none":
        return frames_uint8.float()

    frames = frames_uint8.float()
    B, C, H, W = frames.shape

    # ---- background = downsampled separable Gaussian, then upsampled ----------------
    small_h = max(1, H // downsample)
    small_w = max(1, W // downsample)
    small = F.interpolate(frames, size=(small_h, small_w), mode="bilinear", align_corners=False)

    target_sigma = max(sigma / downsample, 0.5)
    radius = max(1, int(round(3.0 * target_sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=frames.device)
    g = torch.exp(-(coords ** 2) / (2.0 * target_sigma ** 2))
    g = g / g.sum()
    ksize = 2 * radius + 1

    # Depthwise separable Gaussian (horizontal then vertical) — `groups=C` keeps
    # channels independent so the per-channel color cast is preserved.
    kh = g.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).contiguous()
    kv = g.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).contiguous()
    blurred = F.conv2d(small, kh, padding=(0, radius), groups=C)
    blurred = F.conv2d(blurred, kv, padding=(radius, 0), groups=C)

    bg = F.interpolate(blurred, size=(H, W), mode="bilinear", align_corners=False)

    # Per-image global brightness anchor (one scalar per image, broadcast over C/H/W)
    bg_mean = bg.mean(dim=(1, 2, 3), keepdim=True)

    if method == "divide":
        result = (frames / (bg + eps)) * bg_mean
    elif method == "subtract":
        result = frames - bg + bg_mean
    else:
        raise ValueError(f"method must be 'divide', 'subtract', or 'none'; got {method!r}")

    return result.clamp(0.0, 255.0)


def make_illumination_preprocess(
    sigma: float = 64.0,
    method: str = "divide",
    eps: float = 1.0,
):
    """Build a preprocess_fn closure compatible with `MultiCropDataset.preprocess_fn`.

    Returned function: np.ndarray -> np.ndarray (uint8).
    """
    if method == "none":
        return None

    def _fn(arr: np.ndarray) -> np.ndarray:
        return normalize_illumination(arr, sigma=sigma, method=method, eps=eps)

    return _fn
