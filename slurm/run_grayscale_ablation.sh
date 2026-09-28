#!/bin/bash
#SBATCH --job-name=phoebi_gray_abl
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/grayscale_ablation.out
#SBATCH --error=slurm/logs/grayscale_ablation.err

# Grayscale ablation across all six species.
#
# A bs-only pass showed that removing colour costs 0.069 mean similarity and drops the
# pass rate from 100% to 13%. Two readings of that are possible and they have opposite
# implications:
#
#   uniform across species  -> grayscale is simply off the manifold DINOv2 and the
#                              prototypes occupy. Colour carries no species information,
#                              the margin survives, recalibrated thresholds recover it.
#   species-dependent       -> colour carries discriminative signal. Recalibration cannot
#                              restore it, and the model leans on an acquisition cue.
#
# The warm cast is a property of the lamp and camera and is identical across species, so
# the first is more likely a priori. This measures it rather than assuming it.
#
# Deliberately the CPU feature path, matching the bs-only run so the two are directly
# comparable, hence a CPU partition with many cores rather than a GPU.

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

GA_N="${GA_N:-40}"
GA_OUT="${GA_OUT:-outputs/grayscale_ablation_all}"

echo "=== grayscale ablation, all six species ==="
echo "JOB=${SLURM_JOB_ID:-none} HOST=$(hostname) START=$(date)  cores=$(nproc)"
echo "n_images=$GA_N per species, out=$GA_OUT"

conda run --no-capture-output -n phoebi python -u experiments/run_grayscale_ablation.py \
    --species bs bt fj ka mx pf \
    --n_images "$GA_N" \
    --num_workers 8 \
    --output_dir "$GA_OUT"

echo "=== Done $(date) ==="
