#!/bin/bash
#SBATCH --job-name=phoebi_ncd_cmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/ncd_comparators.out
#SBATCH --error=slurm/logs/ncd_comparators.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(conda shell.bash hook)"
conda activate phoebi

set -o pipefail

echo "===== NCD comparator methods (supplementary) ====="

# Greedy baseline (no Sinkhorn balancing)
python -u experiments/run_discovery.py \
    --output_dir outputs/discovery_loocv \
    --cluster_method greedy \
    --residual_threshold 0.15 \
    || echo "WARN: greedy failed (rc=$?)"

# UNO comparators (K=1 and K=4 prototypes per novel class)
python -u experiments/run_discovery.py \
    --output_dir outputs/discovery_loocv_uno_k1 \
    --cluster_method uno_lite \
    --sinkhorn_k 1 \
    --residual_threshold 0.15 \
    || echo "WARN: uno_k1 failed (rc=$?)"

python -u experiments/run_discovery.py \
    --output_dir outputs/discovery_loocv_uno_k4 \
    --cluster_method uno_lite \
    --sinkhorn_k 4 \
    --residual_threshold 0.15 \
    || echo "WARN: uno_k4 failed (rc=$?)"

# SimGCD comparator
python -u experiments/run_discovery.py \
    --output_dir outputs/discovery_loocv_simgcd_k1 \
    --cluster_method simgcd \
    --sinkhorn_k 1 \
    --residual_threshold 0.15 \
    || echo "WARN: simgcd failed (rc=$?)"

echo "===== NCD comparators complete ====="
