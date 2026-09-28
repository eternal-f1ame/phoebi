#!/usr/bin/env python3
"""Extract frames from bacterial culture videos.

Idempotent: skips any video whose output folder already contains at least
`--n_frames` images. New videos dropped into `--video_dir` are picked up on the
next run; existing extractions are not touched.

Default I/O matches the PHOEBI layout:
    inputs : data/videos/*.mp4
    outputs: data/frames/<video_name>/frame_NNNN.jpg

Folder names encode multilabel via underscore separation: a video named
`b_f_k.mp4` produces a folder `b_f_k/` whose label vector is `{b, f, k}`.
The split builder (`tools/build_splits.py`) parses these later.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def _import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError as exc:
        raise SystemExit(
            "extract_frames requires opencv-python. Install with:\n"
            "    pip install opencv-python"
        ) from exc


def list_videos(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file())


def existing_frame_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def sample_frames(video_path: Path, output_folder: Path, n_frames: int, jpeg_quality: int = 90) -> int:
    """Sample n_frames evenly across the video using sequential decode (no seeking).

    cv2.VideoCapture.set(POS_FRAMES, N) is extremely slow on compressed video because
    the decoder has to rewind to the nearest keyframe each time. Reading sequentially
    and discarding unwanted frames is ~15-20x faster in practice.
    """
    cv2 = _import_cv2()
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: could not open {video_path}", file=sys.stderr)
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  {video_path.name}: total frames={total}, fps={fps:.2f}")

    if total == 0:
        cap.release()
        return 0

    if n_frames >= total:
        target_indices = set(range(total))
    else:
        target_indices = {int(i * total / n_frames) for i in range(n_frames)}

    saved = 0
    frame_idx = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            out_path = output_folder / f"frame_{saved + 1:04d}.jpg"
            cv2.imwrite(str(out_path), frame, encode_params)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def _extract_one(args_tuple):
    video_path, out_folder, n_frames, force, jpeg_quality = args_tuple
    existing = existing_frame_count(out_folder)
    if existing >= n_frames and not force:
        return (video_path.name, 0, "skip")
    if force and out_folder.exists():
        for old in out_folder.iterdir():
            if old.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                old.unlink()
    n = sample_frames(video_path, out_folder, n_frames, jpeg_quality=jpeg_quality)
    return (video_path.name, n, "extract")


def extract_all(video_dir: Path, output_dir: Path, n_frames: int,
                force: bool = False, workers: int = 1, jpeg_quality: int = 90) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = list_videos(video_dir)

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    print(f"Found {len(videos)} video(s) in {video_dir}; using {workers} worker(s)")
    jobs = [(v, output_dir / v.stem, n_frames, force, jpeg_quality) for v in videos]

    extracted = 0
    skipped = 0
    if workers <= 1:
        for job in jobs:
            name, n, status = _extract_one(job)
            if status == "skip":
                print(f"  SKIP {name} (already extracted)")
                skipped += 1
            else:
                print(f"  DONE {name}: saved {n} frames")
                extracted += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one, job) for job in jobs]
            for fut in as_completed(futures):
                name, n, status = fut.result()
                if status == "skip":
                    print(f"  SKIP {name} (already extracted)")
                    skipped += 1
                else:
                    print(f"  DONE {name}: saved {n} frames")
                    extracted += 1

    print(f"\nDone. Extracted from {extracted} new video(s); skipped {skipped} already-extracted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from bacterial culture videos.")
    parser.add_argument("--video_dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--n_frames", type=int, default=1000)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the output folder already has enough frames.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of parallel video decoders.")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    args = parser.parse_args()

    extract_all(args.video_dir, args.output_dir, args.n_frames,
                force=args.force, workers=args.workers, jpeg_quality=args.jpeg_quality)


if __name__ == "__main__":
    main()
