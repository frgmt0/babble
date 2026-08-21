#!/bin/bash
# Lane 2: waits for lr3e-3-10k, then capacity runs (lr 1e-3, 4k steps)
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q '"summary"' experiments/results/lr3e-3-10k.jsonl 2>/dev/null; do sleep 30; done
nice -n 19 $PY -m experiments.sweep --name cap-1M-4k --steps 4000 --lr 1e-3 --n-layer 3 --n-embd 160 --threads 2 --eval-every 200 >> experiments/results/cap-1M-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name cap-400k-4k --steps 4000 --lr 1e-3 --n-layer 2 --n-embd 112 --threads 2 --eval-every 200 >> experiments/results/cap-400k-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name cap-150k-4k --steps 4000 --lr 1e-3 --n-layer 2 --n-embd 64 --threads 2 --eval-every 200 >> experiments/results/cap-150k-4k.log 2>&1
echo LANE2_DONE
