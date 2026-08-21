"""Aggregate a directory of experiments.sweep / experiments.tokenizer_sweep
run files (one JSONL per `name-sN` run) into a per-config markdown table:
mean/std of `best_val_bits_per_char` across seeds, plus params and whatever
other meta fields are asked for.

Both harnesses emit a `{"summary": true, ...}` line with a
`best_val_bits_per_char` field (see experiments/sweep.py and
experiments/tokenizer_sweep.py), which is what makes one aggregator work for
either directory -- and what makes a capacity-sweep row and a tokenizer-sweep
row sit in the same column.

Usage: python -m experiments.summarize_sweep --dir experiments/results/capacity
       python -m experiments.summarize_sweep --dir experiments/results/tokenizer
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_SEED_SUFFIX = re.compile(r"-s\d+$")


def load_summary(path: Path) -> dict | None:
    meta, summary = None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "meta" in raw:
            meta = raw["meta"]
        elif raw.get("summary"):
            summary = raw
    if meta is None or summary is None:
        return None
    return {"meta": meta, "summary": summary}


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--fields", nargs="*", default=[], help="extra meta fields to show, e.g. params tokenizer")
    args = ap.parse_args()

    by_config: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.dir.glob("*.jsonl")):
        run = load_summary(path)
        if run is None:
            continue
        config = _SEED_SUFFIX.sub("", run["meta"]["name"])
        by_config[config].append(run)

    if not by_config:
        print(f"no complete runs found under {args.dir}", file=sys.stderr)
        return 1

    extra = args.fields
    header = "| config | seeds | " + " | ".join(extra) + (" | " if extra else "") + "mean bits/char | std | best-seed bits/char |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for config, runs in sorted(by_config.items()):
        bpcs = [r["summary"]["best_val_bits_per_char"] for r in runs if r["summary"].get("best_val_bits_per_char") is not None]
        if not bpcs:
            continue
        m, s = mean_std(bpcs)
        best = min(bpcs)
        extra_vals = " | ".join(str(runs[0]["meta"].get(f, "—")) for f in extra)
        print(
            f"| {config} | {len(runs)} | "
            + (extra_vals + " | " if extra else "")
            + f"**{m:.4f}** | {s:.4f} | {best:.4f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
