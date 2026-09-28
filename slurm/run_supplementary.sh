#!/bin/bash
#SBATCH --job-name=phoebi_supp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/supplementary.out
#SBATCH --error=slurm/logs/supplementary.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

echo "=== Per-class learnable temperature ==="
conda run -n phoebi python experiments/run_learned_tau.py

echo "=== Inter-prototype repulsion ==="
conda run -n phoebi python experiments/run_repulsion.py

echo "=== Per-order F1 breakdown ==="
conda run -n phoebi python experiments/run_per_order_breakdown.py

echo "=== Reliability diagrams ==="
conda run -n phoebi python experiments/run_calibration_analysis.py

echo "=== Boundary-tile robustness ==="
conda run -n phoebi python experiments/run_boundary_tile_check.py

echo "=== Isotonic recalibration ==="
conda run -n phoebi python experiments/run_isotonic_ablation.py \
    --proto_dir outputs/prototype_matching/default

echo "=== Supplementary experiments complete ==="
