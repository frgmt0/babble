"""Aggregate `prefix_sim.py` runs: per-config TTFT stats and a turn table.

    python bench/prefix_report.py RUNS_DIR SCENARIO [--table main default]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("scenario")
    ap.add_argument("--table", nargs="*", default=[])
    args = ap.parse_args()
    by = defaultdict(list)
    for f in sorted(Path(args.runs).glob(f"{args.scenario}-*-r*.json")):
        label = f.stem[len(args.scenario) + 1 : f.stem.rindex("-r")]
        by[label].append(json.loads(f.read_text()))
    print(f"### {args.scenario}\n")
    print("| config | runs | turns | TTFT median ms | p90 ms | mean ms | max ms | median per-run median | prefilled tok median | p90 | prewarm ms/turn |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, runs in by.items():
        ttft = [r["ttft_ms"] for run in runs for r in run["rows"]]
        pre = [r["prompt_tokens"] - r["reused"] for run in runs for r in run["rows"]]
        med_runs = statistics.median(run["summary"]["median_ms"] for run in runs)
        pw = [run["summary"].get("prewarm", {}).get("seconds", 0) * 1000 / len(run["rows"]) for run in runs]
        print(f"| {label} | {len(runs)} | {len(ttft)} | {statistics.median(ttft):.1f} | {pct(ttft, .9):.1f} | "
              f"{statistics.fmean(ttft):.1f} | {max(ttft):.1f} | {med_runs:.1f} | {statistics.median(pre):.0f} | "
              f"{pct(pre, .9):.0f} | {statistics.fmean(pw):.0f} |")
    if args.table:
        print("\nturn-by-turn (channel 0; median over runs): P = prompt tokens, pre = prefilled tokens\n")
        head = " | ".join(f"{l} P / pre / TTFT ms" for l in args.table)
        print(f"| turn | {head} |")
        print("|---" * (1 + len(args.table)) + "|")
        n = max(len([r for r in by[l][0]["rows"] if r["channel"] == 0]) for l in args.table)
        for t in range(n):
            cells = []
            for l in args.table:
                rows = [[r for r in run["rows"] if r["channel"] == 0][t] for run in by[l]]
                P = statistics.median(r["prompt_tokens"] for r in rows)
                pre = statistics.median(r["prompt_tokens"] - r["reused"] for r in rows)
                tt = statistics.median(r["ttft_ms"] for r in rows)
                cells.append(f"{P:.0f} / {pre:.0f} / {tt:.1f}")
            print(f"| {t + 1} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
