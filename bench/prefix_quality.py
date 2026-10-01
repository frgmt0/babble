"""Side-by-side replies: full 8-turn window vs the trimmed window after a drop.

    PYTHONPATH=<tree> python bench/prefix_quality.py RUN.json --turns 12 21 26

Takes a `prefix_sim.py` run (its user messages and booper's replies form the
conversation), and for each chosen turn samples best-of-4 replies from the
same seed with (a) the last ``max_turns`` turns visible, as the sliding window
shows them, and (b) only the newest ``max_turns * keep`` turns, the least
context `conversation_overflow_keep` ever leaves (right after a trim).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from babble.conversation import ConversationTurn, conversation_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--turns", type=int, nargs="+", required=True)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import prefix_sim

    run = json.loads(Path(args.run).read_text())
    rows = [r for r in run["rows"] if r["channel"] == 0]
    ns = argparse.Namespace(root="/tmp/maxperf/prefix/simroot-quality", max_turns=8, max_tokens=1536,
                            max_new_tokens=509, cache_mb=0, set=None)
    _s, gen, _brain = prefix_sim.build(ns)
    low = max(1, int(args.max_turns * args.keep))
    for t in args.turns:
        turns = [ConversationTurn(r["user"], r["reply"]) for r in rows[: t - 1]]
        cur = rows[t - 1]["user"]
        print(f"\n=== turn {t}: user: {cur!r}")
        for label, hist in (("full", turns[-args.max_turns:]), (f"last{low}", turns[-low:])):
            prompt = conversation_prompt(hist, cur, max_turns=args.max_turns, max_chars=0)
            for s in range(args.samples):
                torch.manual_seed(1000 + s)
                text, _ = gen._generate(prompt, max_new_tokens=120, best_of=4, use_prefix_cache=False)
                print(f"  [{label:6s} s{s}] {text!r}")


if __name__ == "__main__":
    main()
