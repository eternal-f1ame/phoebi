"""
Rename augmented images to clean zero-padded sequential names.

Input names:  frame_0001_aug_000.jpg, frame_0001_aug_001.jpg, frame_0002_aug_000.jpg, ...
Output names: 00001.jpg, 00002.jpg, 00003.jpg, ...

Sort order: (frame_num, aug_idx) — preserves temporal order within each combo folder.
"""
import os
import re
import argparse
from pathlib import Path


_AUG_PAT = re.compile(r"^frame_(\d+)_aug_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def rename_folder(folder: Path, dry_run: bool = False) -> int:
    entries = []
    for f in folder.iterdir():
        m = _AUG_PAT.match(f.name)
        if m:
            frame_num = int(m.group(1))
            aug_idx = int(m.group(2))
            entries.append((frame_num, aug_idx, f))

    if not entries:
        print(f"  {folder.name}: no augmented files found, skipping")
        return 0

    entries.sort(key=lambda t: (t[0], t[1]))

    # Two-pass rename through a temp name to avoid collisions
    renamed = 0
    for new_idx, (frame_num, aug_idx, src) in enumerate(entries, start=1):
        dst_name = f"{new_idx:05d}.jpg"
        dst = folder / dst_name
        tmp = folder / f"__tmp_{new_idx:05d}.jpg"
        if not dry_run:
            src.rename(tmp)
        renamed += 1

    if not dry_run:
        for new_idx in range(1, renamed + 1):
            tmp = folder / f"__tmp_{new_idx:05d}.jpg"
            dst = folder / f"{new_idx:05d}.jpg"
            tmp.rename(dst)

    print(f"  {folder.name}: renamed {renamed} files → 00001.jpg … {renamed:05d}.jpg")
    return renamed


def main():
    parser = argparse.ArgumentParser(
        description="Rename frame_XXXX_aug_NNN.jpg files to 00001.jpg, 00002.jpg, ..."
    )
    parser.add_argument("--aug_dir", required=True,
                        help="Root augmented directory (contains one sub-folder per combo)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would happen without renaming anything")
    args = parser.parse_args()

    root = Path(args.aug_dir)
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    combos = sorted(d for d in root.iterdir() if d.is_dir())
    if not combos:
        raise SystemExit(f"No sub-folders found in {root}")

    print(f"Renaming in {'DRY-RUN mode' if args.dry_run else 'LIVE mode'}: {root}")
    total = 0
    for combo in combos:
        total += rename_folder(combo, dry_run=args.dry_run)

    print(f"\nDone. Total files renamed: {total}")


if __name__ == "__main__":
    main()
