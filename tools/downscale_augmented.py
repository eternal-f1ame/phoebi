#!/usr/bin/env python3
"""Pre-downscale augmented images to 256x256 so the training DataLoader can
actually keep the GPU fed.

Reads:   data/images/<combo>/*.jpg  (1222 x 1222, ~200 KB each)
Writes:  data/images_256/<combo>/*.jpg (256 x 256, ~15 KB each)

Runs multi-process over class folders. Idempotent: skips any output file that
already exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def resize_one(src: Path, dst: Path, size: int, quality: int) -> bool:
    if dst.exists():
        return False
    img = Image.open(src).convert("RGB")
    img = img.resize((size, size), resample=Image.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=quality, optimize=True)
    return True


def resize_folder(folder: Path, out_folder: Path, size: int, quality: int) -> int:
    n_new = 0
    for src in sorted(folder.glob("*.jpg")):
        dst = out_folder / src.name
        if resize_one(src, dst, size, quality):
            n_new += 1
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str,
                    default="data/images")
    ap.add_argument("--output_dir", type=str,
                    default="data/images_256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_root = root / args.input_dir
    dst_root = root / args.output_dir
    dst_root.mkdir(parents=True, exist_ok=True)

    folders = sorted([f for f in src_root.iterdir() if f.is_dir()])
    print(f"Folders: {len(folders)}  size={args.size}  quality={args.quality}  "
          f"workers={args.workers}")
    print(f"Reading:  {src_root}")
    print(f"Writing:  {dst_root}")

    total_new = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for folder in folders:
            out = dst_root / folder.name
            out.mkdir(parents=True, exist_ok=True)
            futs[ex.submit(resize_folder, folder, out, args.size, args.quality)] = folder.name

        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n = fut.result()
                total_new += n
                print(f"  {name}: +{n}")
            except Exception as e:
                print(f"  {name}: ERROR {e}")

    print(f"\nDone: {total_new} new files written to {dst_root}")


if __name__ == "__main__":
    main()
