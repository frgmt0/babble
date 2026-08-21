#!/bin/bash
# Capacity sweep (booper run-20260821): shrink the model from the current
# default (3.3M params) toward genuinely tiny, 3 seeds per config, fixed
# lr/dropout/cosine at the current defaults (config.py) so only capacity
# varies. Never writes to checkpoints/ -- results land in
# experiments/results/capacity/. See CAPACITY_TOKENIZER_REPORT.md.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/results/capacity
LOG=$OUT/sweep.log
mkdir -p "$OUT"

# name : n_layer : n_head : n_embd : block_size : steps : eval_every
CONFIGS=(
  "cap-3.3M:4:4:256:512:600:50"
  "cap-1.4M:3:4:192:256:800:50"
  "cap-460K:2:4:128:256:1200:100"
  "cap-132K:2:2:64:256:1500:100"
  "cap-37K:2:2:32:128:2000:100"
  "cap-25K:1:2:32:128:2000:100"
  "cap-9K:1:1:16:128:2000:100"
  "cap-3K:1:1:8:64:2000:100"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r name nl nh ne bs steps ev <<< "$cfg"
  for seed in 1 2 3; do
    run="$name-s$seed"
    outfile="$OUT/$run.jsonl"
    if [ -f "$outfile" ] && tail -1 "$outfile" | grep -q '"summary": true'; then
      echo "[driver] skipping $run (already complete)" >> "$LOG"
      continue
    fi
    echo "[driver] starting $run" >> "$LOG"
    nice -n 19 $PY -m experiments.sweep --name "$run" --steps "$steps" --lr 3e-4 \
      --n-layer "$nl" --n-head "$nh" --n-embd "$ne" --block-size "$bs" \
      --dropout 0.2 --cosine --seed "$seed" --threads 2 --eval-every "$ev" \
      --out-dir "$OUT" >> "$LOG" 2>&1
    echo "[driver] finished $run" >> "$LOG"
  done
done
echo "[driver] CAPACITY_SWEEP_DONE" >> "$LOG"
