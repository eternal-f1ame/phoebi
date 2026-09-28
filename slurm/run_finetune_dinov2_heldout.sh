#!/bin/bash
#SBATCH --job-name=phoebi_ft_dino_lco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/logs/finetune_dinov2_heldout.out
#SBATCH --error=slurm/logs/finetune_dinov2_heldout.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

echo "=== DINOv2 e2e fine-tune, leave-combinations-out ==="
conda run -n phoebi python baselines/finetune_dinov2_phoebi_heldout.py \
    --output_dir outputs/finetune_dinov2_phoebi_heldout/dinov2_s14 \
    --epochs 6 \
    --frame_batch_size 32 \
    --num_workers 8 \
    --resume
echo "=== Done ==="
