#!/bin/bash
#SBATCH --job-name=comp_stage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere"
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/comp_%x_%j.out
#SBATCH --error=slurm/logs/comp_%x_%j.err
# One stage of the compositional-generalisation chain (experiments/run_comp_*.py).
#   COMP_STAGE   stage0 | stage1 | stage2 | stage3 | mosaic   (required)
#   COMP_ROOT    output root (default outputs/compositional)
#   COMP_ENV     conda env (default phoebi)
#   COMP_ARGS    extra driver arguments
# Pascal nodes are excluded via --constraint (torch >= 2.8 cu128 has no sm_61 kernels).
# Chain submission with dependencies: slurm/run_comp_chain.sh
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
: "${COMP_STAGE:?set COMP_STAGE}"
COMP_ROOT=${COMP_ROOT:-outputs/compositional}
COMP_ENV=${COMP_ENV:-phoebi}
COMP_ARGS=${COMP_ARGS:-}
case "$COMP_STAGE" in
  stage0) DRIVER=experiments/run_comp_stage0_granularity.py ;;
  stage1) DRIVER=experiments/run_comp_stage1_patch_simplex.py ;;
  stage2) DRIVER=experiments/run_comp_stage2_mixing_subspace.py ;;
  stage3) DRIVER=experiments/run_comp_stage3_calibration.py ;;
  mosaic) DRIVER=experiments/run_comp_mosaic.py ;;
  *) echo "unknown COMP_STAGE=$COMP_STAGE"; exit 2 ;;
esac
echo "=== $COMP_STAGE  driver=$DRIVER  root=$COMP_ROOT  env=$COMP_ENV  node=$(hostname)  $(date) ==="
nvidia-smi -L || echo "WARN: no GPU visible"
conda run -n "$COMP_ENV" python -c "import torch; print('cuda available:', torch.cuda.is_available())"
mkdir -p "$COMP_ROOT/$COMP_STAGE"
# shellcheck disable=SC2086
conda run --no-capture-output -n "$COMP_ENV" python "$DRIVER" --output_dir "$COMP_ROOT/$COMP_STAGE" $COMP_ARGS
echo "=== $COMP_STAGE done $(date) ==="
[ -f "$COMP_ROOT/$COMP_STAGE/summary.md" ] && cat "$COMP_ROOT/$COMP_STAGE/summary.md"
