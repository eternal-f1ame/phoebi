#!/bin/bash
#SBATCH --job-name=phoebi_resnet50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/resnet50_rerun.out
#SBATCH --error=slurm/logs/resnet50_rerun.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(conda shell.bash hook)"
conda activate phoebi
set -o pipefail

PY=python
BACKBONE="resnet50.a1_in1k"
TAG="resnet50"
LR="1e-4"
EPOCHS=5
BATCH=64
WORKERS=8

echo "=== resnet50 rerun ==="
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

echo "=== resnet50 rerun complete ==="
