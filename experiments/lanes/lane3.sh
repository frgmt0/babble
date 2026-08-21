#!/bin/bash
# Lane 3: waits for the post grid, then the dropout runs (default shape, lr 1e-3, 4k steps)
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until test -s experiments/results/post_grid.jsonl; do sleep 30; done
nice -n 19 $PY -m experiments.sweep --name drop0.1-4k --steps 4000 --lr 1e-3 --dropout 0.1 --threads 2 --eval-every 200 >> experiments/results/drop0.1-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name drop0.2-4k --steps 4000 --lr 1e-3 --dropout 0.2 --threads 2 --eval-every 200 >> experiments/results/drop0.2-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name drop0.3-4k --steps 4000 --lr 1e-3 --dropout 0.3 --threads 2 --eval-every 200 >> experiments/results/drop0.3-4k.log 2>&1
echo LANE3_DONE
