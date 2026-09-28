#!/bin/bash
#SBATCH --job-name=phoebi_lco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/lco.out
#SBATCH --error=slurm/logs/lco.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

echo "=== LCO compositional generalization (Methods A/B/C, seeds 1337/1338/1339) ==="
conda run -n phoebi python experiments/run_phoebi_heldout.py \
    --output_dir outputs/phoebi_heldout

echo "=== LCO complete ==="
