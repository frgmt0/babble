#!/bin/bash
# Relaunch lane A: the long convergence run. Pass 1's copy died with the
# session at step 1750; this restarts it from scratch under systemd --user.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
nice -n 19 $PY -m experiments.sweep --name baseline-lr1e-3-50k --steps 50000 --lr 1e-3 --threads 2 --eval-every 250 >> experiments/results/baseline-lr1e-3-50k.log 2>&1
echo LANE_A_DONE
