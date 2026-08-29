#!/usr/bin/env bash
# Launch an SFT run detached from the SSH session (survives disconnect; keeps the
# Mac awake). Usage: sft/train.sh <run-name> [extra sft_longform.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:?usage: sft/train.sh <run-name> [args]}"; shift
mkdir -p "runs/$name"
[ -f .env.sft ] && set -a && . ./.env.sft && set +a   # BABBLE_RUNS_URL / BABBLE_RUNS_TOKEN
py=.venv/bin/python
launcher=""
command -v caffeinate >/dev/null && launcher="caffeinate -i"
nohup $launcher $py sft/sft_longform.py --name "$name" "$@" >> "runs/$name/nohup.out" 2>&1 &
echo $! > "runs/$name/pid"
echo "started run '$name' pid $(cat runs/$name/pid); watch with: sft/monitor.sh $name"
