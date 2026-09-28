#!/bin/bash
#SBATCH --job-name=phoebi_mil
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/mil_attention.out
#SBATCH --error=slurm/logs/mil_attention.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

echo "=== Attention-MIL baseline (random + heldout) ==="
conda run -n phoebi python baselines/mil_attention.py \
    --protocol both \
    --output_dir outputs/mil_attention \
    --epochs 30
echo "=== Done ==="
