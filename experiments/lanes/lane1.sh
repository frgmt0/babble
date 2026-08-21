#!/bin/bash
# Lane 1: waits for lr3e-4-10k, then synthetic-mix + wd runs (default shape, lr 1e-3, 4k steps)
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q '"summary"' experiments/results/lr3e-4-10k.jsonl 2>/dev/null; do sleep 30; done
nice -n 19 $PY -m experiments.sweep --name synth-1x-4k --steps 4000 --lr 1e-3 --threads 2 --eval-every 200 --synthetic-corpus data/synthetic_corpus.jsonl >> experiments/results/synth-1x-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name synth-3x-4k --steps 4000 --lr 1e-3 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_3x.jsonl >> experiments/results/synth-3x-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name wd0-4k --steps 4000 --lr 1e-3 --weight-decay 0.0 --threads 2 --eval-every 200 >> experiments/results/wd0-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name wd0.1-4k --steps 4000 --lr 1e-3 --weight-decay 0.1 --threads 2 --eval-every 200 >> experiments/results/wd0.1-4k.log 2>&1
echo LANE1_DONE
