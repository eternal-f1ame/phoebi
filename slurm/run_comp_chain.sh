#!/bin/bash
# Submit the compositional-generalisation chain with SLURM dependencies:
#   stage0 -> stage1 -> stage2 -> stage3 -> mosaic
# Each stage reads the previous stage's best.json; a failed stage stops the chain
# (afterok). Re-running this script after a fix resubmits from the first stage whose
# summary.md is missing. Logs: slurm/logs/comp_comp_<stage>_<jobid>.{out,err}
#   COMP_ROOT   output root (default outputs/compositional)
#   COMP_FROM   first stage to (re)submit (default: auto)
#   COMP_DEP    job id the first submitted stage must wait for (e.g. a stage0 submitted by hand)
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
COMP_ROOT=${COMP_ROOT:-outputs/compositional}
STAGES=(stage0 stage1 stage2 stage3 mosaic)
mkdir -p slurm/logs "$COMP_ROOT"
prev="${COMP_DEP:-}"
started=0
for s in "${STAGES[@]}"; do
  if [ "$started" -eq 0 ]; then
    if [ -n "${COMP_FROM:-}" ]; then
      if [ "$s" != "$COMP_FROM" ]; then echo "$s: before COMP_FROM=$COMP_FROM, skipping"; continue; fi
    elif [ -f "$COMP_ROOT/$s/summary.md" ]; then
      echo "$s: summary present, not resubmitting"; continue
    fi
  fi
  started=1
  dep=""; [ -n "$prev" ] && dep="--dependency=afterok:$prev"
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable $dep --job-name "comp_$s" --export=ALL,COMP_STAGE=$s,COMP_ROOT=$COMP_ROOT slurm/run_comp_stage.sh)
  echo "$s: job $jid ${dep:+($dep)}"
  echo "$jid" > "$COMP_ROOT/$s.jobid"
  prev=$jid
done
squeue -u "$USER" -h -o '%.10i %.14j %.2t %R' | grep comp_ || true
