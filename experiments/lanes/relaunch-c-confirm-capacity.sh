#!/bin/bash
# Relaunch lane C, highest priority first: the 3-seed confirmation runs that
# decide the recommended defaults, then the capacity comparisons.
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
echo CONFIRM_DONE
nice -n 19 $PY -m experiments.sweep --name cap-1M-4k --steps 4000 --lr 1e-3 --n-layer 3 --n-embd 160 --threads 2 --eval-every 200 >> experiments/results/cap-1M-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name cap-400k-4k --steps 4000 --lr 1e-3 --n-layer 2 --n-embd 112 --threads 2 --eval-every 200 >> experiments/results/cap-400k-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name cap-150k-4k --steps 4000 --lr 1e-3 --n-layer 2 --n-embd 64 --threads 2 --eval-every 200 >> experiments/results/cap-150k-4k.log 2>&1
echo LANE_C_DONE
