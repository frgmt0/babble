#!/usr/bin/env bash
# Interleaved w4 benchmark rounds; each process holds the shared bench lock.
#   bench/extreme/w4_bench_rounds.sh ROUNDS OUT "w4|head2" "w4|head2" ...
# ("-" = off). Example: w4_bench_rounds.sh 5 /tmp/maxperf/w4/bench.jsonl "-|-" "-|256" "noh:64|256"
set -u
rounds=$1
out=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
py=${PY:-/home/jason/projects/babble/.venv/bin/python}
for r in $(seq 1 "$rounds"); do
  for cfg in "$@"; do
    w4=${cfg%%|*}
    h2=${cfg##*|}
    [ "$w4" = "-" ] && w4=""
    [ "$h2" = "-" ] && h2=""
    # W4_PAUSE: anchored pkill pattern of our own background jobs to freeze while timing
    BABBLE_NATIVE_W4=$w4 BABBLE_NATIVE_HEAD2=$h2 W4_REPS=${W4_REPS:-5} PY=$py HERE=$here \
      flock /tmp/babble-bench.lock bash -c '
        [ -n "${W4_PAUSE:-}" ] && pkill -STOP -f "$W4_PAUSE"
        "$PY" "$HERE/w4_bench.py"
        [ -n "${W4_PAUSE:-}" ] && pkill -CONT -f "$W4_PAUSE"
        true' >> "$out"
  done
done
