#!/bin/bash
#SBATCH --job-name=phoebi_ablations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/ablations.out
#SBATCH --error=slurm/logs/ablations.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

echo "=== Ablation sweeps (run each subcommand) ==="
for sweep in tile_count tile_size illumination projection threshold proto_init; do
    echo "--- $sweep ---"
    conda run -n phoebi python experiments/run_ablations.py "$sweep" \
        || echo "WARN: ablation '$sweep' failed (rc=$?)"
done

echo "=== Ablations complete ==="
