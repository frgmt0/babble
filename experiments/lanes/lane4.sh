#!/bin/bash
# Lane 4: phase-2 confirmation runs -- cosine/dropout/small-model configs from
# the design review, 3 seeds each, short budgets (val turns <1k steps).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=experiments/results/lane4.log
for seed in 1 2 3; do
  nice -n 19 $PY -m experiments.sweep --name b-drop0.2-lr3e-4-s$seed --steps 1000 --lr 3e-4 --dropout 0.2 --cosine --seed $seed --threads 2 --eval-every 50 >> $LOG 2>&1
done
for seed in 1 2 3; do
  nice -n 19 $PY -m experiments.sweep --name c-small-drop0.1-s$seed --steps 1000 --lr 1e-3 --n-layer 3 --n-embd 128 --dropout 0.1 --weight-decay 0.05 --cosine --seed $seed --threads 2 --eval-every 50 >> $LOG 2>&1
done
nice -n 19 $PY -m experiments.sweep --name cosine-only-s1 --steps 1000 --lr 1e-3 --cosine --seed 1 --threads 2 --eval-every 50 >> $LOG 2>&1
echo LANE4_DONE
