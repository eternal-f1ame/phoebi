#!/bin/bash
#SBATCH --job-name=phoebi_supp_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/supp_resume.out
#SBATCH --error=slurm/logs/supp_resume.err

set -eo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

echo "=== Isotonic recalibration (resume — only step missing from phoebi_supp) ==="
conda run -n phoebi python experiments/run_isotonic_ablation.py \
    --proto_dir outputs/prototype_matching/default

echo "=== Done ==="
