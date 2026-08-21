#!/bin/bash
# Relaunch lane E: dropout grid then weight-decay comparisons. Budget 2400
# (best-val for every swept config lands <1000 steps; see relaunch-b note).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
nice -n 19 $PY -m experiments.sweep --name drop0.1-2k4 --steps 2400 --lr 1e-3 --dropout 0.1 --threads 2 --eval-every 200 >> experiments/results/drop0.1-2k4.log 2>&1
nice -n 19 $PY -m experiments.sweep --name drop0.2-2k4 --steps 2400 --lr 1e-3 --dropout 0.2 --threads 2 --eval-every 200 >> experiments/results/drop0.2-2k4.log 2>&1
nice -n 19 $PY -m experiments.sweep --name drop0.3-2k4 --steps 2400 --lr 1e-3 --dropout 0.3 --threads 2 --eval-every 200 >> experiments/results/drop0.3-2k4.log 2>&1
nice -n 19 $PY -m experiments.sweep --name wd0-2k4 --steps 2400 --lr 1e-3 --weight-decay 0.0 --threads 2 --eval-every 200 >> experiments/results/wd0-2k4.log 2>&1
nice -n 19 $PY -m experiments.sweep --name wd0.1-2k4 --steps 2400 --lr 1e-3 --weight-decay 0.1 --threads 2 --eval-every 200 >> experiments/results/wd0.1-2k4.log 2>&1
echo LANE_E_DONE
