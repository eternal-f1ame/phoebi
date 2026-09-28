#!/bin/bash
#SBATCH --job-name=phoebi_presence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/presence_test.out
#SBATCH --error=slurm/logs/presence_test.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
set -o pipefail

echo "=== Score test split — Method A (simplex) ==="
conda run -n phoebi python experiments/run_presence_detection.py \
    --method simplex \
    --model_dir outputs/simplex_unmixing/default \
    || echo "WARN: simplex presence test failed (rc=$?)"

echo "=== Score test split — Method B (prototype) ==="
conda run -n phoebi python experiments/run_presence_detection.py \
    --method prototype \
    --model_dir outputs/prototype_matching/default \
    || echo "WARN: prototype presence test failed (rc=$?)"

echo "=== Presence test scoring complete ==="
