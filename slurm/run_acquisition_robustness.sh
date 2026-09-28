#!/bin/bash
#SBATCH --job-name=phoebi_acq_robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/acquisition_robustness.out
#SBATCH --error=slurm/logs/acquisition_robustness.err

# Acquisition-shift robustness, illumination attribution, and a label-noise
# upper bound: the corruption battery, matched illumination arms and label-noise
# table of the paper's appendix.
#
# Decoders are frozen (paper prototypes + val-calibrated thresholds); this is a
# deployment-time robustness measurement, not a retraining study.
#
# --no-capture-output so progress reaches the log live; plain `conda run` buffers.

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

# Overridable so a smoke run and the real run share one code path:
#   ACQ_OUT=outputs/..._smoke ACQ_N=120 ACQ_CORRUPTIONS=illum_gradient \
#   ACQ_SEVERITIES=3 ACQ_FULL_CLEAN=0 sbatch slurm/run_acquisition_robustness.sh
ACQ_OUT="${ACQ_OUT:-outputs/acquisition_robustness}"
ACQ_N="${ACQ_N:-4000}"
ACQ_CORRUPTIONS="${ACQ_CORRUPTIONS:-}"
ACQ_SEVERITIES="${ACQ_SEVERITIES:-1,2,3}"
ACQ_FULL_CLEAN="${ACQ_FULL_CLEAN:-1}"

# Decoder set + illumination arms. Defaults are the paper's divide-trained decoders.
# For the matched no-correction arm, point all three at the none_illum models AND set
# ACQ_ILLUMINATIONS=none, so the models are evaluated under the correction they were
# trained with rather than against prototypes calibrated on a different front-end.
ACQ_SIMPLEX_DIR="${ACQ_SIMPLEX_DIR:-outputs/simplex_unmixing/default}"
ACQ_PROTO_DIR="${ACQ_PROTO_DIR:-outputs/prototype_matching/default}"
ACQ_MC_DIR="${ACQ_MC_DIR:-outputs/mc_channel/default}"
ACQ_ILLUMINATIONS="${ACQ_ILLUMINATIONS:-divide,none}"

echo "=== Acquisition robustness / illumination / label noise ==="
echo "JOB=${SLURM_JOB_ID:-none} HOST=$(hostname) START=$(date)"
echo "out=$ACQ_OUT n=$ACQ_N corruptions='${ACQ_CORRUPTIONS:-ALL}' sev=$ACQ_SEVERITIES full_clean=$ACQ_FULL_CLEAN"
echo "illuminations=$ACQ_ILLUMINATIONS"
echo "decoders: A=$ACQ_SIMPLEX_DIR  B=$ACQ_PROTO_DIR  C=$ACQ_MC_DIR"
nvidia-smi -L || true

EXTRA=()
[ "$ACQ_FULL_CLEAN" = "1" ] && EXTRA+=(--full_clean)
[ -n "$ACQ_CORRUPTIONS" ] && EXTRA+=(--corruptions "$ACQ_CORRUPTIONS")

conda run --no-capture-output -n phoebi python -u experiments/run_acquisition_robustness.py \
    --output_dir "$ACQ_OUT" \
    --n_sweep "$ACQ_N" \
    --severities "$ACQ_SEVERITIES" \
    --illuminations "$ACQ_ILLUMINATIONS" \
    --simplex_dir "$ACQ_SIMPLEX_DIR" \
    --proto_dir "$ACQ_PROTO_DIR" \
    --mc_dir "$ACQ_MC_DIR" \
    --num_workers 8 \
    --frame_batch_size 8 \
    "${EXTRA[@]}"

echo "=== Done $(date) ==="
