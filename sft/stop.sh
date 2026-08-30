#!/usr/bin/env bash
# Stop every SFT trainer on this machine (the run can be resumed with --resume).
pgrep -fl "python.*sft/sft_longform.py" || { echo "nothing running"; exit 0; }
pkill -f "python.*sft/sft_longform.py"; sleep 2; pgrep -f "python.*sft/sft_longform.py" >/dev/null && pkill -9 -f "python.*sft/sft_longform.py"; echo "stopped"
