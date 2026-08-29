#!/usr/bin/env bash
# Watch a run: status line from metrics.jsonl, then follow train.log.
# Usage: sft/monitor.sh [run-name]   (default: most recent run)
#        sft/monitor.sh <run-name> --status   (one-shot summary, no tail)
cd "$(dirname "$0")/.."
name="${1:-$(ls -t runs 2>/dev/null | head -1)}"
[ -n "$name" ] && [ -d "runs/$name" ] || { echo "no such run: '$name' (have: $(ls runs 2>/dev/null | tr '\n' ' '))"; exit 1; }
dir="runs/$name"
pid=$(cat "$dir/pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then state="RUNNING pid $pid"; else state="not running"; fi
python3 - "$dir/metrics.jsonl" <<'PY'
import json, sys, time
recs = [json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
tr = [r for r in recs if "loss" in r]; ev = [r for r in recs if "val" in r]
if tr:
    r = tr[-1]
    print(f"step {r['step']}/{r.get('steps_total','?')}  loss {r['loss']:.4f}  {r['tok_s']:.0f} tok/s  eta {r.get('eta_s',0)/60:.0f}m  ({time.strftime('%H:%M:%S', time.localtime(r['t']))})")
if ev:
    e = ev[-1]; print(f"val {e['val']:.4f} @ step {e['step']}")
    s = e.get('samples') or []
    if s: print(f"sample: {s[0]['prompt']!r} -> {s[0]['reply'][:300]!r} ({s[0]['tokens']} tok)")
if any(r.get("event") == "done" for r in recs): print("run finished")
PY
echo "[$name] $state — following $dir/train.log (ctrl-c to stop watching; the run keeps going)"
[ "${2:-}" = "--status" ] && exit 0
tail -n 20 -f "$dir/train.log"
