#!/usr/bin/env bash
# Launch an SFT run detached from the SSH session (survives disconnect; keeps the
# Mac awake). Usage: sft/train.sh <run-name> [extra sft_longform.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:?usage: sft/train.sh <run-name> [args]}"; shift
mkdir -p "runs/$name"
# One training process per machine: a second run on a 16GB laptop pages it to death.
if pgrep -f "python.*sft/sft_longform.py" >/dev/null; then
  echo "refusing: an SFT run is already alive:"; pgrep -fl "python.*sft/sft_longform.py"; echo "stop it with: sft/stop.sh"; exit 1
fi
[ -f .env.sft ] && set -a && . ./.env.sft && set +a   # BABBLE_RUNS_URL / BABBLE_RUNS_TOKEN
py=.venv/bin/python
launcher=""
command -v caffeinate >/dev/null && launcher="caffeinate -i"
# macOS QoS so the laptop stays responsive: SFT_QOS=background (default,
# `taskpolicy -b`: lowest CPU/IO priority), utility (`taskpolicy -c utility`),
# or none. taskpolicy execs python directly, so the policy applies to it.
qos="${SFT_QOS:-background}"
if command -v taskpolicy >/dev/null; then
  case "$qos" in
    background) launcher="$launcher taskpolicy -b" ;;
    utility) launcher="$launcher taskpolicy -c utility" ;;
    none) ;;
    *) echo "unknown SFT_QOS=$qos (background|utility|none)"; exit 1 ;;
  esac
fi
echo "[$(date '+%F %T')] launch: qos=$qos $launcher $*" >> "runs/$name/nohup.out"
nohup $launcher $py sft/sft_longform.py --name "$name" "$@" >> "runs/$name/nohup.out" 2>&1 &
echo $! > "runs/$name/pid"
echo "started run '$name' pid $(cat runs/$name/pid); watch with: sft/monitor.sh $name"
