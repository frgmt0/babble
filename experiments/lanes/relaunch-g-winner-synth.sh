#!/bin/bash
# Lane G: the decision table for defaults — winner config (lr 3e-4, dropout
# 0.2, cosine) with and without the leak-free 3x synthetic mix, matched
# 1600-step budgets, 3 seeds each. This is the "synthetic +/- on winner
# config" comparison the ticket requires.
#   $1 = sub-lane: g1 | g2
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
COMMON="--lr 3e-4 --dropout 0.2 --cosine --steps 1600 --threads 2 --eval-every 200"
SYN="--synthetic-corpus experiments/synthetic_corpus_3x_trainonly.jsonl"
case "$1" in
  g1)
    nice -n 19 $PY -m experiments.sweep --name win-nosynth-s1 --seed 1 $COMMON >> experiments/results/win-nosynth-s1.log 2>&1
    nice -n 19 $PY -m experiments.sweep --name win-synth-s1   --seed 1 $COMMON $SYN >> experiments/results/win-synth-s1.log 2>&1
    nice -n 19 $PY -m experiments.sweep --name win-nosynth-s3 --seed 3 $COMMON >> experiments/results/win-nosynth-s3.log 2>&1
    echo LANE_G1_DONE
    ;;
  g2)
    nice -n 19 $PY -m experiments.sweep --name win-synth-s2   --seed 2 $COMMON $SYN >> experiments/results/win-synth-s2.log 2>&1
    nice -n 19 $PY -m experiments.sweep --name win-nosynth-s2 --seed 2 $COMMON >> experiments/results/win-nosynth-s2.log 2>&1
    nice -n 19 $PY -m experiments.sweep --name win-synth-s3   --seed 3 $COMMON $SYN >> experiments/results/win-synth-s3.log 2>&1
    echo LANE_G2_DONE
    ;;
esac
