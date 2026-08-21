"""Interpolated byte n-gram baseline on the trainer's exact split.

The floor any neural recipe has to beat: a count-based byte model with
interpolated backoff, fit on the train rows, scored on the held-out rows with
the trainer's own token accounting (<bos>-conditioned, <eos> a target,
token-weighted mean). Corpus-internal, external-data-free, fits in
milliseconds — if the transformer cannot beat this number, the recipe (not
the corpus) is the problem.

Usage: python -m experiments.ngram_baseline [--order 3]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

from babble.config import Settings
from babble.tokenizer import text_examples
from babble.trainer import corpus_rows, split_rows


def row_token_streams(rows, block_size: int):
    """The exact (context-conditioned) token streams the trainer scores:
    every example's tokens, loss on positions where mask==1."""
    streams = []
    for row in rows:
        for ex in text_examples(row.text, block_size):
            streams.append((ex.tokens, ex.mask))
    return streams


class InterpolatedNgram:
    """Counts for orders 1..N over token ids, Jelinek-Mercer interpolation."""

    def __init__(self, order: int, vocab: int = 260) -> None:
        self.order = order
        self.vocab = vocab
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.totals = [defaultdict(int) for _ in range(order)]

    def fit(self, streams) -> None:
        for tokens, mask in streams:
            for i in range(1, len(tokens)):
                if not mask[i]:
                    continue
                target = tokens[i]
                for n in range(self.order):
                    ctx = tuple(tokens[max(0, i - 1 - n) : i]) if n else tuple(tokens[i - 1 : i])
                    if len(ctx) != n + 1:
                        continue
                    self.counts[n][ctx][target] += 1
                    self.totals[n][ctx] += 1

    def prob(self, tokens, i, lambdas) -> float:
        """Interpolated P(tokens[i] | context), with a uniform floor."""
        p = 1.0 / self.vocab
        weight_left = 1.0
        # Highest order first; unseen contexts pass their weight down.
        acc = 0.0
        for n in reversed(range(self.order)):
            lam = lambdas[n]
            ctx = tuple(tokens[i - 1 - n : i])
            if len(ctx) != n + 1 or self.totals[n].get(ctx, 0) == 0:
                continue
            acc += lam * self.counts[n][ctx].get(tokens[i], 0) / self.totals[n][ctx]
        # Remaining mass on the uniform floor keeps every token possible.
        lam_used = sum(
            lambdas[n]
            for n in range(self.order)
            if self.totals[n].get(tuple(tokens[i - 1 - n : i]), 0) > 0
            and len(tuple(tokens[i - 1 - n : i])) == n + 1
        )
        return acc + max(0.0, 1.0 - lam_used) * p

    def mean_loss(self, streams, lambdas) -> float:
        total, count = 0.0, 0
        for tokens, mask in streams:
            for i in range(1, len(tokens)):
                if not mask[i]:
                    continue
                total += -math.log(max(self.prob(tokens, i, lambdas), 1e-12))
                count += 1
        return total / max(count, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    args = ap.parse_args()

    settings = Settings.from_env()
    rows = corpus_rows(settings)
    split = split_rows(rows, settings)
    train_streams = row_token_streams(split.train, settings.block_size)
    val_streams = row_token_streams(split.val, settings.block_size)

    model = InterpolatedNgram(args.order)
    model.fit(train_streams)

    # A small grid over interpolation weights, chosen on TRAIN (not val, to
    # keep the comparison honest); report the val loss of the train-best.
    grids = {
        1: [(1.0,)],
        2: [(a, 1 - a) for a in (0.1, 0.2, 0.3, 0.5)],
        3: [
            (a, b, 1 - a - b)
            for a in (0.05, 0.1, 0.2)
            for b in (0.2, 0.3, 0.4)
            if a + b < 1
        ],
    }[min(args.order, 3)]
    best = None
    for lambdas in grids:
        train_loss = model.mean_loss(train_streams, lambdas)
        if best is None or train_loss < best[1]:
            best = (lambdas, train_loss)
    lambdas, train_loss = best
    val_loss = model.mean_loss(val_streams, lambdas)
    print(
        json.dumps(
            {
                "order": args.order,
                "lambdas": lambdas,
                "train_rows": len(split.train),
                "val_rows": len(split.val),
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
