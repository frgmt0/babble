#!/usr/bin/env bash
# Stop every SFT trainer on this machine (the run can be resumed with --resume).
pgrep -fl "sft_longform.py" || { echo "nothing running"; exit 0; }
pkill -f "sft_longform.py"; sleep 2; pgrep -f "sft_longform.py" >/dev/null && pkill -9 -f "sft_longform.py"; echo "stopped"
