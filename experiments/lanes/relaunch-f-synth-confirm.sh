#!/bin/bash
# Lane F: synthetic result confirmation, leak-free. The headline synth-3x-4k
# run (best val 2.468) was sourced from a chain that saw val rows; these runs
# use the train-only corpora from experiments/gen_synth_corpora.py.
#   $1 = sub-lane: s1 | s2 | s3
# s1: seed-1 3x confirm, then the word-shuffled confound control.
# s2: seed-2 3x confirm, then the 5x scale test.
# s3: seed-3 3x confirm.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
case "$1" in
  s1)
    nice -n 19 $PY -m experiments.sweep --name synth3xT-s1 --steps 3000 --lr 1e-3 --seed 1 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_3x_trainonly.jsonl >> experiments/results/synth3xT-s1.log 2>&1
    nice -n 19 $PY -m experiments.sweep --name synth3x-shuf --steps 3000 --lr 1e-3 --seed 1 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_3x_shuffled.jsonl >> experiments/results/synth3x-shuf.log 2>&1
    echo LANE_F1_DONE
    ;;
  s2)
    nice -n 19 $PY -m experiments.sweep --name synth3xT-s2 --steps 3000 --lr 1e-3 --seed 2 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_3x_trainonly.jsonl >> experiments/results/synth3xT-s2.log 2>&1
    nice -n 19 $PY -m experiments.sweep --name synth5xT-s1 --steps 3000 --lr 1e-3 --seed 1 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_5x_trainonly.jsonl >> experiments/results/synth5xT-s1.log 2>&1
    echo LANE_F2_DONE
    ;;
  s3)
    nice -n 19 $PY -m experiments.sweep --name synth3xT-s3 --steps 3000 --lr 1e-3 --seed 3 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_3x_trainonly.jsonl >> experiments/results/synth3xT-s3.log 2>&1
    echo LANE_F3_DONE
    ;;
esac
