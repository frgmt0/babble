"""Regenerate the synthetic corpora used by the sweep, leak-free.

`generate_synthetic_corpus` now excludes val-side rows from the Markov chain
by default (babble/valsplit.py), so everything this script writes is sourced
from train-side rows only. It produces:

- data/synthetic_corpus.jsonl        -- the shipped 1x file (400 rows), rebuilt
- experiments/synthetic_corpus_3x_trainonly.jsonl  -- 1200 rows
- experiments/synthetic_corpus_5x_trainonly.jsonl  -- 2000 rows
- experiments/synthetic_corpus_3x_shuffled.jsonl   -- confound control: the 3x
  rows with words shuffled within each row. Same size, same vocabulary, same
  per-row unigram counts, no corpus word order. If this trains as well as the
  real 3x mix, the synthetic benefit is dilution/regularisation, not structure.

Run from the repo root: .venv/bin/python -m experiments.gen_synth_corpora
Deterministic for a given corpus; re-run after the corpus grows.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from babble.config import Settings
from babble.synthcorpus import generate_synthetic_corpus

ROOT = Path(__file__).resolve().parent.parent
OUT_3X = ROOT / "experiments" / "synthetic_corpus_3x_trainonly.jsonl"
OUT_5X = ROOT / "experiments" / "synthetic_corpus_5x_trainonly.jsonl"
OUT_3X_SHUF = ROOT / "experiments" / "synthetic_corpus_3x_shuffled.jsonl"


def _generate_to(settings: Settings, out: Path, count: int) -> None:
    result = generate_synthetic_corpus(settings, count=count, seed=0, rebuild=True)
    shutil.copyfile(settings.synthetic_corpus_path, out)
    print(
        f"{out.name}: {result.generated} rows from {result.source_rows} train-side "
        f"sources ({result.excluded_val_rows} val-side rows excluded)"
    )


def _shuffle_control(src: Path, out: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        words = row["text"].split()
        rng.shuffle(words)
        row["text"] = " ".join(words)
        row["method"] = row["method"] + "_shuffled_control"
        lines.append(json.dumps(row, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out.name}: {len(lines)} rows, word order destroyed")


def main() -> None:
    settings = Settings.from_env()
    # Biggest first, then the shipped 1x last so data/synthetic_corpus.jsonl
    # ends up holding the 400-row file the trainer actually mixes in.
    _generate_to(settings, OUT_5X, 2000)
    _generate_to(settings, OUT_3X, 1200)
    _shuffle_control(OUT_3X, OUT_3X_SHUF)
    result = generate_synthetic_corpus(settings, count=400, seed=0, rebuild=True)
    print(
        f"data/synthetic_corpus.jsonl: {result.generated} rows from "
        f"{result.source_rows} train-side sources "
        f"({result.excluded_val_rows} val-side rows excluded)"
    )


if __name__ == "__main__":
    main()
