#!/bin/bash
#SBATCH --job-name=phoebi_abl_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/ablation_sweep_%j.out
#SBATCH --error=slurm/logs/ablation_sweep_%j.err
# One ablation sweep of experiments/run_ablations.py on a GPU node, into a fresh output
# dir. The CPU and GPU illumination paths give slightly different features, so any
# sweep compared against the headline numbers has to run on the GPU path.
#   ABL_SWEEP  subcommand (default projection)
#   ABL_OUT    output root (default outputs/ablations_gpu); results land in $ABL_OUT/$ABL_SWEEP/
#   ABL_ENV    conda env (default phoebi)
# Pascal nodes are excluded via --constraint: torch >= 2.8 cu128 ships no sm_61 kernels.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
ABL_SWEEP=${ABL_SWEEP:-projection}
ABL_OUT=${ABL_OUT:-outputs/ablations_gpu}
ABL_ENV=${ABL_ENV:-phoebi}
echo "=== sweep=$ABL_SWEEP out=$ABL_OUT env=$ABL_ENV node=$(hostname) ==="
nvidia-smi -L || echo "WARN: no GPU visible; the A numbers will be on the CPU path and NOT comparable"
conda run -n "$ABL_ENV" python -c "import torch; print('cuda available:', torch.cuda.is_available())"
mkdir -p "$ABL_OUT"
conda run --no-capture-output -n "$ABL_ENV" python experiments/run_ablations.py "$ABL_SWEEP" --output_dir "$ABL_OUT"
echo "=== done ==="; ls -la "$ABL_OUT/$ABL_SWEEP/"; cat "$ABL_OUT/$ABL_SWEEP/results.csv" 2>/dev/null || true
