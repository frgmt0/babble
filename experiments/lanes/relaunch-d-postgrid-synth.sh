#!/bin/bash
# Relaunch lane D: the full 9-cell post-train guardrail grid under the
# continuation layout (the serving layout, now the code default), then the
# synthetic-mix comparisons at the default config.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
BABBLE_POST_LAYOUT=continuation nice -n 19 $PY -m experiments.post_grid --pretrained /tmp/pretrained-live.pt --steps 200 --out experiments/results/post_grid_continuation.jsonl >> experiments/results/post_grid_continuation.log 2>&1
echo POST_GRID_DONE
nice -n 19 $PY -m experiments.sweep --name synth-1x-4k --steps 4000 --lr 1e-3 --threads 2 --eval-every 200 --synthetic-corpus data/synthetic_corpus.jsonl >> experiments/results/synth-1x-4k.log 2>&1
nice -n 19 $PY -m experiments.sweep --name synth-3x-4k --steps 4000 --lr 1e-3 --threads 2 --eval-every 200 --synthetic-corpus experiments/synthetic_corpus_3x.jsonl >> experiments/results/synth-3x-4k.log 2>&1
echo LANE_D_DONE
