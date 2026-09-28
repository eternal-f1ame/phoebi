#!/bin/bash
#SBATCH --job-name=phoebi_loocv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/loocv.out
#SBATCH --error=slurm/logs/loocv.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

echo "=== LOOCV open-set detection ==="
conda run -n phoebi python experiments/run_openset_detection.py

echo "=== LOOCV discovery (SK K=1, canonical) ==="
conda run -n phoebi python experiments/run_discovery.py \
    --output_dir outputs/discovery_loocv_sk_k1 \
    --cluster_method sinkhorn --sinkhorn_k 1 --residual_threshold 0.15

echo "=== 5-way OSR score sweep ==="
conda run -n phoebi python experiments/run_osr_score_sweep.py

echo "=== LOOCV complete ==="
