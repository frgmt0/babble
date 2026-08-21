"""Summarise every sweep curve in experiments/results/ as one markdown table.

For each run: params, lr, dropout, wd, synthetic examples, best val (and its
step), final train_full, val at a few fixed steps -- the numbers the report's
tables are built from.

Usage: python -m experiments.tabulate [--dir experiments/results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict | None:
    meta, points, summary = None, [], None
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
        elif "step" in raw:
            points.append(raw)
    if meta is None or not points:
        return None
    best = min(points, key=lambda p: p["val"])
    return {"meta": meta, "points": points, "summary": summary, "best": best}


def val_at(points: list[dict], step: int) -> float | None:
    exact = [p for p in points if p["step"] == step]
    return exact[0]["val"] if exact else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("experiments/results"))
    args = ap.parse_args()

    runs = []
    for path in sorted(args.dir.glob("*.jsonl")):
        if path.name == "post_grid.jsonl":
            continue
        run = load_run(path)
        if run:
            runs.append(run)

    cols = "| run | params | lr | drop | wd | synth ex | best val@step | train@best | val@2k | val@4k | val@10k | final train | steps run |"
    print(cols)
    print("|" + "---|" * (cols.count("|") - 1))
    for run in runs:
        m, points, best = run["meta"], run["points"], run["best"]
        last = points[-1]
        print(
            f"| {m['name']} | {m['params']/1e6:.2f}M | {m['lr']:g} | {m['dropout']:g} "
            f"| {m['weight_decay']:g} | {m.get('synthetic_examples', 0)} "
            f"| **{best['val']:.3f}**@{best['step']} | {best['train_full']:.3f} "
            f"| {val_at(points, 2000) or '—'} | {val_at(points, 4000) or '—'} "
            f"| {val_at(points, 10000) or '—'} "
            f"| {last['train_full']:.3f} | {last['step']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
