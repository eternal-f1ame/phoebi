#!/bin/bash
#SBATCH --job-name=phoebi_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/encoder_probe.out
#SBATCH --error=slurm/logs/encoder_probe.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
set -o pipefail

echo "=== 13-encoder linear probe (9 general backbones) ==="
conda run -n phoebi python -m baselines.multilabel_probe --epochs 10 --render_tex

echo "=== 13-encoder linear probe (4 biomedical backbones) ==="
conda run -n phoebi python -m baselines.multilabel_probe_bio --epochs 10

echo "=== Encoder probe complete ==="
