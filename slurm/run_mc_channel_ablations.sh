#!/bin/bash
#SBATCH --job-name=phoebi_mc_abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/mc_channel_ablations.out
#SBATCH --error=slurm/logs/mc_channel_ablations.err

set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p slurm/logs

echo "=== Method C ablations (ASL / Mixup / ASL+Mixup) ==="

SHARED_TRAIN=$(realpath outputs/mc_channel/default/train_features_cache.pt)
SHARED_VAL=$(realpath outputs/mc_channel/default/val_features_cache.pt)
SHARED_TEST=$(realpath outputs/mc_channel/default/test_features_cache.pt)

# Train each ablation variant, reusing the baseline's feature caches
for CONFIG in "6class_asl:asl:0.0" "6class_mixup:bce:0.4" "6class_asl_mixup:asl:0.4"; do
    IFS=':' read -r NAME LOSS MIXUP <<< "$CONFIG"
    OUT="outputs/mc_channel/$NAME"
    mkdir -p "$OUT"
    [ ! -e "$OUT/train_features_cache.pt" ] && ln -sf "$SHARED_TRAIN" "$OUT/train_features_cache.pt"
    [ ! -e "$OUT/val_features_cache.pt" ]   && ln -sf "$SHARED_VAL"   "$OUT/val_features_cache.pt"
    echo "--- Training $NAME (loss=$LOSS, mixup=$MIXUP) ---"
    conda run -n phoebi python -u -m src.mc_channel.train \
        --output_dir "$OUT" \
        --loss "$LOSS" \
        --mixup_alpha "$MIXUP" \
        --epochs 30 \
        --batch_size 4096 \
        --lr 1e-2 \
        --weight_decay 1e-4 \
        --cra_drop_prob 0.5 \
        --frame_batch_size 8 \
        --num_workers 4
done

# Test eval: symlink test cache for each ablation dir, then score
for NAME in 6class_asl 6class_mixup 6class_asl_mixup; do
    OUT="outputs/mc_channel/$NAME"
    [ ! -e "$OUT/test_features_cache.pt" ] && ln -sf "$SHARED_TEST" "$OUT/test_features_cache.pt"
    echo "--- Test eval: $NAME ---"
    conda run -n phoebi python -u -m src.mc_channel.test_eval \
        --model_dir "$OUT" \
        --num_workers 4
done

echo "=== Method C ablations complete ==="
