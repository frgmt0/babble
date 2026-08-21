"""How much of the val-loss movement that drives early stopping is split noise?

Holds a checkpoint fixed and recomputes val loss over many random holdouts of
the same corpus. If the spread across splits is comparable to the wobble that
fires the patience stop, the stop is reacting to which rows landed in val, not
to the model getting worse.

Usage:
    python -m experiments.val_noise --checkpoint /path/to/pretrained.pt [--splits 2000]

Reads the corpus via the normal Settings/consent path (BABBLE_DATA_DIR etc.).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

import torch

from babble.config import Settings
from babble.cpu_runtime import force_cpu_device
from babble.model import Babbler, ModelConfig, per_token_loss
from babble.tokenizer import PAD_ID, text_examples
from babble.trainer import corpus_rows, split_rows, val_holdout_size


def load_checkpoint(path: str) -> Babbler:
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload.get("step")


@torch.inference_mode()
def per_row_loss_sums(model: Babbler, rows) -> list[tuple[float, int]]:
    """(loss_sum, token_count) per corpus row, over all its examples."""
    out = []
    for row in rows:
        examples = text_examples(row.text, model.config.block_size)
        loss_sum, tokens = 0.0, 0
        for ex in examples:
            t = torch.tensor([ex.tokens], dtype=torch.long)
            m = torch.tensor([ex.mask], dtype=torch.long)[:, 1:]
            pt = per_token_loss(model, t)
            loss_sum += float((pt * m).sum())
            tokens += int(m.sum())
        out.append((loss_sum, tokens))
    return out


def split_val_loss(per_row: list[tuple[float, int]], idx: list[int]) -> float:
    s = sum(per_row[i][0] for i in idx)
    n = sum(per_row[i][1] for i in idx)
    return s / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = Settings.from_env()
    rows = corpus_rows(settings)
    model, step = load_checkpoint(args.checkpoint)
    print(f"checkpoint step={step} params={model.num_params():,} rows={len(rows)}", flush=True)

    per_row = per_row_loss_sums(model, rows)
    total_loss = sum(s for s, _ in per_row)
    total_tokens = sum(n for _, n in per_row)
    print(f"full-corpus loss {total_loss / total_tokens:.4f} over {total_tokens} tokens", flush=True)

    holdout = val_holdout_size(len(rows), settings.val_fraction)
    rng = random.Random(args.seed)
    losses = []
    for _ in range(args.splits):
        idx = rng.sample(range(len(rows)), holdout)
        losses.append(split_val_loss(per_row, idx))

    losses.sort()
    mean = statistics.mean(losses)
    std = statistics.stdev(losses)
    q = lambda p: losses[int(p * (len(losses) - 1))]
    print(json.dumps({
        "splits": args.splits,
        "holdout_rows": holdout,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(losses[0], 4),
        "p5": round(q(0.05), 4),
        "p25": round(q(0.25), 4),
        "median": round(q(0.5), 4),
        "p75": round(q(0.75), 4),
        "p95": round(q(0.95), 4),
        "max": round(losses[-1], 4),
        "iqr": round(q(0.75) - q(0.25), 4),
        "p95_minus_p5": round(q(0.95) - q(0.05), 4),
    }, indent=2), flush=True)

    # The split the live trainer actually uses, for reference.
    live = split_rows(rows, settings)
    if live.enabled:
        id_to_idx = {row.id: i for i, row in enumerate(rows)}
        live_idx = [id_to_idx[r.id] for r in live.val]
        print(f"hash-split val loss (the one the trainer sees): "
              f"{split_val_loss(per_row, live_idx):.4f} over {len(live_idx)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
