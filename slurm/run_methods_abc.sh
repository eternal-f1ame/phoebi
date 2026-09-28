#!/bin/bash
#SBATCH --job-name=phoebi_abc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/methods_abc.out
#SBATCH --error=slurm/logs/methods_abc.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

echo "=== Method A — Simplex Unmixing ==="
conda run -n phoebi python -m src.simplex_unmixing.train \
    --output_dir outputs/simplex_unmixing/default \
    --epochs 30 --frame_batch_size 8 --num_workers 2

echo "=== Method B — Prototype Matching ==="
conda run -n phoebi python -m src.prototype_matching.train \
    --output_dir outputs/prototype_matching/default \
    --frame_batch_size 8 --num_workers 2

echo "=== Method C — MC Channel (train) ==="
conda run -n phoebi python -m src.mc_channel.train \
    --output_dir outputs/mc_channel/default

echo "=== Method C — MC Channel (eval) ==="
conda run -n phoebi python -m src.mc_channel.test_eval \
    --model_dir outputs/mc_channel/default

echo "=== Methods A/B/C complete ==="
