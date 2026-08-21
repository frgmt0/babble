"""Post-train option grid: which guardrails actually stop the fine-tune from
shipping a worse checkpoint, measured rather than argued.

Each cell copies the pretrained checkpoint into its own scratch checkpoint
dir, runs `post_train` with one configuration, and reports: pair val, corpus
val before/after (the promotion gate's own numbers), and the gate verdict.

Usage:
    python -m experiments.post_grid --pretrained /path/to/pretrained.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from babble.config import Settings
from babble.cpu_runtime import configure_cpu
from babble.posttrain import post_train


def run_cell(
    base: Settings,
    scratch_root: Path,
    pretrained: Path,
    name: str,
    *,
    lr: float,
    rehearsal: float,
    include_synthetic: bool,
    steps: int,
    gate_margin: float = 0.05,
) -> dict:
    ckpt_dir = scratch_root / name
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True)
    shutil.copyfile(pretrained, ckpt_dir / "latest.pt")

    settings = Settings.from_env()
    settings.checkpoint_dir = ckpt_dir
    settings.post_learning_rate = lr
    settings.post_rehearsal = rehearsal
    settings.post_gate_margin = gate_margin
    settings.post_min_pairs = 0
    settings.train_threads = 3

    result = post_train(
        settings,
        force=True,
        steps=steps,
        seed=1,
        echo=False,
        include_synthetic=include_synthetic,
    )
    cell = {
        "name": name,
        "lr": lr,
        "rehearsal": rehearsal,
        "include_synthetic": include_synthetic,
        "steps": steps,
        "pairs": result.pairs_trained,
        "synthetic_pairs": result.synthetic_pairs_trained,
        "final_step": result.final_step,
        "pair_val": round(result.val_loss, 4) if result.val_loss is not None else None,
        "corpus_val_before": round(result.corpus_val_before, 4)
        if result.corpus_val_before is not None
        else None,
        "corpus_val_after": round(result.corpus_val_after, 4)
        if result.corpus_val_after is not None
        else None,
        "promoted": result.promoted,
    }
    print(json.dumps(cell), flush=True)
    return cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--scratch", type=Path, default=Path("experiments/post-scratch"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/post_grid.jsonl"))
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    configure_cpu(args.threads)
    base = Settings.from_env()
    cells = []
    grid = [
        # The old live behaviour: pretrain LR, no rehearsal, human pairs only.
        ("old-lr1e-3", dict(lr=1e-3, rehearsal=0.0, include_synthetic=False)),
        ("lr1e-4", dict(lr=1e-4, rehearsal=0.0, include_synthetic=False)),
        ("lr1e-5", dict(lr=1e-5, rehearsal=0.0, include_synthetic=False)),
        ("lr1e-3+rehearse", dict(lr=1e-3, rehearsal=0.5, include_synthetic=False)),
        ("lr1e-4+rehearse", dict(lr=1e-4, rehearsal=0.5, include_synthetic=False)),
        ("lr1e-5+rehearse", dict(lr=1e-5, rehearsal=0.5, include_synthetic=False)),
        ("lr1e-4+synth", dict(lr=1e-4, rehearsal=0.0, include_synthetic=True)),
        ("lr1e-4+rehearse+synth", dict(lr=1e-4, rehearsal=0.5, include_synthetic=True)),
        ("lr1e-5+rehearse+synth", dict(lr=1e-5, rehearsal=0.5, include_synthetic=True)),
    ]
    for name, kw in grid:
        cells.append(
            run_cell(base, args.scratch, args.pretrained, name, steps=args.steps, **kw)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for cell in cells:
            fh.write(json.dumps(cell) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
