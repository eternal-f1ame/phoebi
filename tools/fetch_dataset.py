#!/usr/bin/env python3
"""Download the PHOEBI dataset from Hugging Face and lay it out as the code expects.

    python tools/fetch_dataset.py                    # phoebi6 images and every manifest
    python tools/fetch_dataset.py --subset all       # also the four-species subset

writes, under the repository root,

    data/images/<combo>/<n>.jpg          phoebi6: 120,000 images, about 13 GB
    data/images_4class/<combo>/<n>.jpg   phoebi4: 14,000 images (--subset phoebi4 or all)
    data/splits.json                     random 80/10/10 split
    data/splits_lco.json                 leave-combinations-out partition (seed 1337)
    data/splits_4class.json              four-species subset
    data/lco_dev_folds.json              development folds

Each Parquet row stores the original JPEG bytes and the path the manifests list for
that image; the bytes are written unchanged to that path (``data/images/...`` is
relative to the repository root, ``images_4class/...`` to ``data/``, which is where
the default ``--data_dir`` puts both). Shards are
fetched one at a time and deleted after export unless ``--keep_parquet`` is given,
and images already on disk are skipped, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ["splits.json", "splits_lco.json", "splits_4class.json", "lco_dev_folds.json"]
SUBSET_MANIFEST = {"phoebi6": "splits.json", "phoebi4": "splits_4class.json"}


def target(data_dir: Path, rel: str) -> Path:
    return data_dir / rel.removeprefix("data/")


def export_shard(parquet: Path, data_dir: Path) -> tuple[int, int]:
    written = skipped = 0
    pf = pq.ParquetFile(parquet)
    for i in range(pf.num_row_groups):
        image = pf.read_row_group(i, columns=["image"]).column("image").combine_chunks()
        for rel, data in zip(image.field("path").to_pylist(), image.field("bytes").to_pylist()):
            out = target(data_dir, rel)
            if out.exists() and out.stat().st_size == len(data):
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            written += 1
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default="sochastic/PHOEBI")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--subset", choices=["phoebi6", "phoebi4", "all"], default="phoebi6")
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data")
    ap.add_argument("--keep_parquet", type=Path, default=None,
                    help="keep the downloaded Parquet shards in this directory")
    args = ap.parse_args()
    subsets = ["phoebi6", "phoebi4"] if args.subset == "all" else [args.subset]
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for name in MANIFESTS:
        src = hf_hub_download(args.repo_id, name, repo_type="dataset", revision=args.revision)
        shutil.copyfile(src, args.data_dir / name)
    print(f"manifests -> {args.data_dir}")

    files = HfApi().list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision)
    for sub in subsets:
        manifest = json.loads((args.data_dir / SUBSET_MANIFEST[sub]).read_text())
        rels = [r["path"] for rows in manifest["splits"].values() for r in rows]

        def missing() -> list[str]:
            return [r for r in rels if not target(args.data_dir, r).exists()]

        if missing():
            shards = sorted(f for f in files if f.startswith(f"data/{sub}/") and f.endswith(".parquet"))
            for k, shard in enumerate(shards, 1):
                with tempfile.TemporaryDirectory(dir=args.data_dir) as tmp:
                    local = Path(hf_hub_download(args.repo_id, shard, repo_type="dataset",
                                                 revision=args.revision, local_dir=args.keep_parquet or tmp))
                    written, skipped = export_shard(local, args.data_dir)
                print(f"{sub} [{k}/{len(shards)}] {shard}: {written} written, {skipped} already present")
        left = missing()
        if left:
            raise SystemExit(f"{sub}: {len(left)} of {len(rels)} images missing, e.g. {left[0]}")
        print(f"{sub}: all {len(rels)} images present")


if __name__ == "__main__":
    main()
