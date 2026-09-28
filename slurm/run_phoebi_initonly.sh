#!/bin/bash
#SBATCH --job-name=phoebi_initonly
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/logs/phoebi_initonly.out
#SBATCH --error=slurm/logs/phoebi_initonly.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

echo "=== Init-only LCO (epochs=0 for A and C, shows pure-culture init baseline) ==="

# Reuse feature caches from the canonical phoebi_heldout (same seed=1337 split)
mkdir -p outputs/phoebi_heldout_initonly/features
SHARED_FEATS=$(realpath outputs/phoebi_heldout/features)
for SPLIT in train val test; do
    TARGET="outputs/phoebi_heldout_initonly/features/${SPLIT}.pt"
    [ ! -e "$TARGET" ] && ln -sf "$SHARED_FEATS/${SPLIT}.pt" "$TARGET"
done

conda run -n phoebi python -u experiments/run_phoebi_heldout.py \
    --method_a_epochs 0 \
    --method_c_epochs 0 \
    --output_dir outputs/phoebi_heldout_initonly

echo "=== Init-only done ==="
