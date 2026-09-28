#!/bin/bash
#SBATCH --job-name=phoebi_none_illum
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/none_illum_models.out
#SBATCH --error=slurm/logs/none_illum_models.err

# Matched no-illumination-correction decoders, for the fair arm of the
# illumination ablation: what the correction does to classification performance.
#
# The corruption sweep's `none` columns strip the correction at test time only,
# against prototypes calibrated WITH it -- a train/test mismatch. To claim
# anything about what the correction buys under acquisition shift we need models
# that were trained and calibrated without it. This produces them.
#
# Method B is trained first because it is closed-form (no epochs); its feature
# caches are then HARDLINKED into the A and C run dirs so the 108k-image
# extraction happens once instead of three times. The cache key is
# (paths, tile_config, backbone, illum_method, illum_sigma), which is identical
# across the three here, so the reuse is sound.

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

B_DIR=outputs/prototype_matching/none_illum
A_DIR=outputs/simplex_unmixing/none_illum
C_DIR=outputs/mc_channel/none_illum

echo "=== Matched none-illumination decoders ==="
echo "JOB=${SLURM_JOB_ID:-none} HOST=$(hostname) START=$(date)"
nvidia-smi -L || true

echo "--- [1/3] Method B (closed-form; performs the one feature extraction) ---"
conda run --no-capture-output -n phoebi python -u -m src.prototype_matching.train \
    --output_dir "$B_DIR" --illumination none --frame_batch_size 8 --num_workers 8

echo "--- hardlinking feature caches into the A and C run dirs ---"
mkdir -p "$A_DIR" "$C_DIR"
for f in train_features_cache.pt val_features_cache.pt; do
    if [ -f "$B_DIR/$f" ]; then
        ln -f "$B_DIR/$f" "$A_DIR/$f"
        ln -f "$B_DIR/$f" "$C_DIR/$f"
        echo "    linked $f ($(du -h "$B_DIR/$f" | cut -f1))"
    else
        echo "    WARN: $B_DIR/$f absent; A and C will re-extract"
    fi
done

echo "--- [2/3] Method A (30 epochs, reuses cache) ---"
conda run --no-capture-output -n phoebi python -u -m src.simplex_unmixing.train \
    --output_dir "$A_DIR" --illumination none --epochs 30 \
    --frame_batch_size 8 --num_workers 8

echo "--- [3/3] Method C (30 epochs, reuses cache) ---"
conda run --no-capture-output -n phoebi python -u -m src.mc_channel.train \
    --output_dir "$C_DIR" --illumination none \
    --frame_batch_size 8 --num_workers 8

echo "=== Done $(date) ==="
