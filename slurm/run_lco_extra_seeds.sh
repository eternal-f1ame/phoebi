#!/bin/bash
#SBATCH --job-name=phoebi_lco_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/lco_seeds_%a.out
#SBATCH --error=slurm/logs/lco_seeds_%a.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
set -o pipefail

SEEDS=(1338 1339)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTDIR="outputs/phoebi_heldout_seed${SEED}"

echo "=== LCO seed=$SEED -> $OUTDIR ==="
conda run -n phoebi python experiments/run_phoebi_heldout.py \
    --seed "$SEED" \
    --output_dir "$OUTDIR"

echo "=== seed $SEED complete ==="
