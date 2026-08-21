#!/bin/bash
# Runs the capacity sweep to completion, then the tokenizer sweep, strictly
# serially (never concurrently -- running both at once on this shared box is
# what forced the previous attempt to be killed). Both lane scripts are
# idempotent/resumable: they skip any run whose output jsonl already ends in
# a summary line, so this is safe to re-launch if interrupted again.
set -uo pipefail
cd "$(dirname "$0")/../.."
echo "[serial] capacity sweep starting $(date +%s)"
bash experiments/lanes/capacity-sweep.sh
echo "[serial] capacity sweep done, tokenizer sweep starting $(date +%s)"
bash experiments/lanes/tokenizer-sweep.sh
echo "[serial] ALL_SWEEPS_DONE $(date +%s)"
