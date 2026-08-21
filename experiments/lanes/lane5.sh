#!/bin/bash
# Lane 5: after the old-(pair)-layout post grid finishes, rerun it under the
# new continuation layout (now the code default) so the report can show the
# same guardrail grid on the layout the bot actually serves.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until test -s experiments/results/post_grid.jsonl; do sleep 30; done
mv experiments/results/post_grid.jsonl experiments/results/post_grid_pairlayout.jsonl
nice -n 19 $PY -m experiments.post_grid --pretrained /tmp/pretrained-live.pt --steps 200 --out experiments/results/post_grid_continuation.jsonl >> experiments/results/post_grid_continuation.log 2>&1
echo LANE5_DONE
