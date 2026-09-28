#!/usr/bin/env python3
"""One-shot real-data preparation: extract frames + build splits.

Run this whenever new videos are dropped into `data/videos/`. It is fully
idempotent — already-extracted videos are skipped, and the splits.json is
rebuilt from the current state of `data/frames/`.

    python tools/prepare_real_data.py

Optional flags forward to the underlying scripts:

    python tools/prepare_real_data.py \
        --video_dir data/videos \
        --frames_dir data/frames \
        --splits_path data/splits.json \
        --n_frames 1000 \
        --train_frac 0.8 --val_frac 0.1
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.build_splits import build_splits  # noqa: E402
from tools.extract_frames import extract_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames + build splits in one shot.")
    parser.add_argument("--video_dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--frames_dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--splits_path", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--n_frames", type=int, default=1000)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--force_extract", action="store_true",
                        help="Re-extract every video even if frames already exist.")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1/2: extract frames")
    print("=" * 60)
    extract_all(args.video_dir, args.frames_dir, args.n_frames, force=args.force_extract)

    print()
    print("=" * 60)
    print("Step 2/2: build splits")
    print("=" * 60)
    build_splits(args.frames_dir, args.splits_path, args.train_frac, args.val_frac)


if __name__ == "__main__":
    main()
