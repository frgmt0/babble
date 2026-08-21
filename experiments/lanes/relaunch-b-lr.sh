#!/bin/bash
# Relaunch lane B: LR sweep companions. Budget trimmed 10k -> 2400: every
# config's val minimum lands well before step 1000 and the post-turn
# divergence tail is already demonstrated by the lane-A curve; 2400 keeps a
# val@2k column while freeing the cores that actually decide the defaults.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
nice -n 19 $PY -m experiments.sweep --name lr3e-3-2k4 --steps 2400 --lr 3e-3 --threads 2 --eval-every 200 >> experiments/results/lr3e-3-2k4.log 2>&1
nice -n 19 $PY -m experiments.sweep --name lr3e-4-2k4 --steps 2400 --lr 3e-4 --threads 2 --eval-every 200 >> experiments/results/lr3e-4-2k4.log 2>&1
echo LANE_B_DONE
