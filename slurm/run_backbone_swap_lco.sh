#!/bin/bash
#SBATCH --job-name=phoebi_lco_gigapath
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/lco_gigapath.out
#SBATCH --error=slurm/logs/lco_gigapath.err

# Does the LCO result depend on the backbone? The encoder-capacity control of the
# paper's appendix: the full LCO protocol re-run with the probe's top encoder.
#
# Prov-GigaPath (ViT-G, 1.13B params, 1536-d) tops the 13-encoder probe at 0.699
# per-sample F1 vs 0.676 for the parameter-matched DINOv2-S/14 (22.1M). If the LCO
# ordering and the absence of compositional collapse survive a 51x larger,
# top-ranked, pathology-pretrained encoder, the backbone choice is not load-bearing.
#
# run_phoebi_heldout.py derives D from the feature tensor, so 1536-d needs no code
# change (Method C splits it into K=6 groups of 256 channels).
# Same canonical seed 1337 -> same 9 held-out combinations as the paper.

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

echo "=== LCO with Prov-GigaPath features (backbone justification) ==="
echo "JOB=$SLURM_JOB_ID HOST=$(hostname) START=$(date)"
nvidia-smi -L || true

conda run --no-capture-output -n phoebi python -u experiments/run_phoebi_heldout.py \
    --backbone hf-hub:prov-gigapath/prov-gigapath \
    --output_dir outputs/phoebi_heldout_gigapath \
    --seed 1337 \
    --frame_batch_size 8 \
    --num_workers 8

echo "=== Done $(date) ==="
