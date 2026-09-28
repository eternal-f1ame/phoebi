#!/bin/bash
#SBATCH --job-name=phoebi_sup_lco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/supervised_baselines.out
#SBATCH --error=slurm/logs/supervised_baselines.err

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

eval "$(conda shell.bash hook)"
conda activate phoebi

set -o pipefail

echo "===== supervised baselines (9 paper backbones × random+heldout) ====="
echo "host=$(hostname)  gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo 'cpu')"

# Nine backbones, each fine-tuned on the random split and on the LCO partition
# (5 epochs, lr 1e-4, batch 64); a run whose results.json exists is skipped.
BACKBONES=(
  "resnet50|resnet50.a1_in1k"
  "convnext_b|convnext_base.fb_in22k_ft_in1k"
  "vit_b_in21k|vit_base_patch16_224.augreg_in21k_ft_in1k"
  "dinov2_s14|vit_small_patch14_dinov2.lvd142m"
  "dinov3_s16|vit_small_patch16_dinov3.lvd1689m"
  "clip_b16|vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"
  "siglip_b16|vit_base_patch16_siglip_224.webli"
  "eva02_b16|eva02_base_patch16_clip_224.merged2b"
  "florence2_davit_b|davit_base_fl.msft_florence2"
)
for entry in "${BACKBONES[@]}"; do
  IFS='|' read -r tag backbone <<<"$entry"
  for split in random heldout; do
    if [ "$split" = random ]; then
      script=baselines/supervised_multilabel.py; outdir=outputs/supervised_multilabel/$tag
    else
      script=baselines/supervised_multilabel_heldout.py; outdir=outputs/supervised_multilabel_heldout/$tag
    fi
    if [ -f "$outdir/results.json" ]; then echo "[$tag|$split] skip: results exist"; continue; fi
    echo "[$tag|$split] starting  $(date)"
    python -u "$script" --backbone "$backbone" --output_dir "$outdir" \
        --epochs 5 --batch_size 64 --lr 1e-4 --num_workers 4 \
        || echo "WARN: $tag $split failed (rc=$?)"
  done
done

echo "===== DINOv2 end-to-end fine-tune (heldout) ====="
python -u baselines/finetune_dinov2_phoebi_heldout.py \
    --output_dir outputs/finetune_dinov2_phoebi_heldout \
    || echo "WARN: finetune_dinov2_phoebi_heldout failed (rc=$?)"

echo "===== Attention-MIL baseline (random + heldout via --protocol both) ====="
python -u baselines/mil_attention.py \
    --output_dir outputs/mil_attention \
    --protocol both \
    || echo "WARN: mil_attention failed (rc=$?)"

echo "===== Supervised baselines complete ====="
