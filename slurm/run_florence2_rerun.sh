#!/bin/bash
#SBATCH --job-name=phoebi_florence2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/florence2_rerun.out
#SBATCH --error=slurm/logs/florence2_rerun.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(conda shell.bash hook)"
conda activate phoebi
set -o pipefail

PY=python
BACKBONE="davit_base_fl.msft_florence2"
TAG="florence2_davit_b"
LR="1e-4"
EPOCHS=5
BATCH=32
WORKERS=8

echo "=== florence2_davit_b rerun (ampere 48 GB) ==="
echo "host=$(hostname)  gpu=$(nvidia-smi -L 2>/dev/null | head -1)"

for expt in random_split heldout; do
  case "$expt" in
    random_split)
      script="baselines/supervised_multilabel.py"
      outdir="outputs/supervised_multilabel/$TAG"
      ;;
    heldout)
      script="baselines/supervised_multilabel_heldout.py"
      outdir="outputs/supervised_multilabel_heldout/$TAG"
      ;;
  esac

  if [ -f "$REPO/$outdir/results.json" ]; then
    echo "[$TAG|$expt] skip: results.json exists"
    continue
  fi

  echo "[$TAG|$expt|lr=$LR] starting  $(date)"
  $PY -u "$script" \
    --backbone "$BACKBONE" \
    --output_dir "$outdir" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --num_workers "$WORKERS" \
    && echo "[$TAG|$expt] done  $(date)" \
    || echo "[$TAG|$expt] FAILED (rc=$?)  $(date)"
done

echo "=== florence2_davit_b rerun complete ==="
