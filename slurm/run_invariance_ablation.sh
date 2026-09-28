#!/bin/bash
#SBATCH --job-name=phoebi_invariance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/invariance_ablation.out
#SBATCH --error=slurm/logs/invariance_ablation.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(conda shell.bash hook)"
conda activate phoebi
set -o pipefail

echo "=== Test-time invariance ablation (Method B) ==="
python -u experiments/run_invariance_ablation.py \
    --splits_path data/splits.json \
    --method_b_dir outputs/prototype_matching/default \
    --output_dir  outputs/ablations/invariance \
    && echo "=== Invariance ablation complete ===" \
    || echo "FAILED (rc=$?)"
