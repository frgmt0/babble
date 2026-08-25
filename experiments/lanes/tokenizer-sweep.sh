#!/bin/bash
# Tokenizer swap (booper run-20260821): byte vs. BPE (a few vocab sizes) vs.
# word-level, all at the same architecture and step budget, 3 seeds each.
# Off the production path -- babble/subword.py, never imported by core.py /
# generate.py / trainer.py / bot.py. Never writes to checkpoints/ -- results
# land in experiments/results/tokenizer/. See docs/reports/CAPACITY_TOKENIZER_REPORT.md.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/results/tokenizer
LOG=$OUT/sweep.log
mkdir -p "$OUT"

# name : tokenizer : vocab-size (full id space; ignored for byte)
CONDITIONS=(
  "byte-ref:byte:260"
  "bpe-512:bpe:512"
  "bpe-1024:bpe:1024"
  "bpe-2048:bpe:2048"
  "word-1024:word:1024"
)
ARCH="--n-layer 4 --n-head 4 --n-embd 256 --block-size 256 --dropout 0.2"
STEPS=600
EVAL_EVERY=50

for cond in "${CONDITIONS[@]}"; do
  IFS=: read -r name kind vocab <<< "$cond"
  for seed in 1 2 3; do
    run="$name-s$seed"
    outfile="$OUT/$run.jsonl"
    if [ -f "$outfile" ] && tail -1 "$outfile" | grep -q '"summary": true'; then
      echo "[driver] skipping $run (already complete)" >> "$LOG"
      continue
    fi
    echo "[driver] starting $run" >> "$LOG"
    nice -n 19 $PY -m experiments.tokenizer_sweep --name "$run" --steps "$STEPS" --lr 3e-4 \
      $ARCH --tokenizer "$kind" --vocab-size "$vocab" --cosine \
      --seed "$seed" --threads 2 --eval-every "$EVAL_EVERY" \
      --out-dir "$OUT" >> "$LOG" 2>&1
    echo "[driver] finished $run" >> "$LOG"
  done
done
echo "[driver] TOKENIZER_SWEEP_DONE" >> "$LOG"
