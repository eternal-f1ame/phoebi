#!/usr/bin/env python3
"""Build splits.json from the augmented/ directory.

Wraps build_splits.build_splits() with augmented-data defaults so the main
splits.json points to augmented crops (00001.jpg … NNNNN.jpg) rather than
raw video frames.

Usage:
    python tools/build_augmented_splits.py
    python tools/build_augmented_splits.py --aug_dir data/images --output data/splits.json
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.build_splits import build_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build splits.json from the augmented image directory."
    )
    parser.add_argument("--aug_dir", type=Path, default=Path("data/images"),
                        help="Root directory of augmented images (one sub-folder per combo)")
    parser.add_argument("--output", type=Path, default=Path("data/splits.json"),
                        help="Output path for splits.json (overwrites existing file)")
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    args = parser.parse_args()

    build_splits(args.aug_dir, args.output, args.train_frac, args.val_frac)


if __name__ == "__main__":
    main()
