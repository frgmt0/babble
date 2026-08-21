"""One sweep run: train a config to a fixed step budget with early stop off,
logging train/val curves to a JSONL — the experiment behind the new defaults.

Deliberately reuses the trainer's own building blocks (same split, same batch
sampling, same loss) so a sweep number means the same thing as a live number.
Never writes to the repo's checkpoints/ — results land in experiments/results/.

Usage:
    python -m experiments.sweep --name baseline --steps 50000 [--lr 1e-3]
        [--n-layer 4 --n-embd 256] [--dropout 0.1] [--weight-decay 0.01]
        [--eval-every 250] [--synthetic-corpus data/synthetic_corpus.jsonl]

`--synthetic-corpus` mixes labelled synthetic rows (see babble/synthcorpus.py)
into the TRAIN side only. Val stays 100% real held-out corpus rows, so the
with/without comparison is always on the same real-text yardstick.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from babble.config import Settings
from babble.cpu_runtime import configure_cpu, force_cpu_device
from babble.generate import continue_text
from babble.model import Babbler, ModelConfig, per_token_loss
from babble.tokenizer import text_examples
from babble.trainer import corpus_rows, make_batch, split_rows, to_examples
from babble.model import sequence_loss

PROBES = ("hello", "the cat", "why is", "boop")


@torch.inference_mode()
def full_loss(model: Babbler, examples, chunk: int = 64) -> float:
    """Token-weighted mean loss over `examples`, batched in chunks."""
    was_training = model.training
    model.eval()
    try:
        total, tokens = 0.0, 0
        for i in range(0, len(examples), chunk):
            batch = examples[i : i + chunk]
            width = max(len(e) for e in batch)
            t = torch.full((len(batch), width), 256, dtype=torch.long)  # PAD_ID
            m = torch.zeros((len(batch), width), dtype=torch.long)
            for j, ex in enumerate(batch):
                t[j, : len(ex)] = torch.as_tensor(ex.tokens, dtype=torch.long)
                m[j, : len(ex)] = torch.as_tensor(ex.mask, dtype=torch.long)
            pt = per_token_loss(model, t)
            tm = m[:, 1:].to(pt.dtype)
            total += float((pt * tm).sum())
            tokens += int(tm.sum())
        return total / max(tokens, 1)
    finally:
        model.train(was_training)


def load_synthetic_rows(path: Path):
    """Labelled synthetic corpus rows: only the text matters here."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if raw.get("text"):
            rows.append(raw["text"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--cosine", action="store_true", help="cosine-anneal LR to lr/10 over the budget")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--synthetic-corpus", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/results"))
    args = ap.parse_args()

    configure_cpu(args.threads)
    device = force_cpu_device()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    settings = Settings.from_env()
    rows = corpus_rows(settings)
    split = split_rows(rows, settings)

    config = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = Babbler(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=True
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr * 0.1
        )
        if args.cosine
        else None
    )

    train_examples = to_examples(split.train, config.block_size)
    val_examples = to_examples(split.val, config.block_size)
    synth_examples = []
    if args.synthetic_corpus is not None:
        synth_texts = load_synthetic_rows(args.synthetic_corpus)
        for text in synth_texts:
            synth_examples.extend(text_examples(text, config.block_size))
    mixed = train_examples + synth_examples

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.out_dir / f"{args.name}.jsonl"
    meta = {
        "name": args.name,
        "steps": args.steps,
        "lr": args.lr,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "cosine": args.cosine,
        "batch_size": args.batch_size,
        "params": model.num_params(),
        "train_rows": len(split.train),
        "val_rows": len(split.val),
        "train_examples": len(train_examples),
        "synthetic_examples": len(synth_examples),
        "seed": args.seed,
    }
    with open(curve_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
    print(f"[{args.name}] {json.dumps(meta)}", flush=True)

    best = {"val": math.inf, "step": 0, "train_full": None, "state": None}
    window: list[float] = []
    started = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        tokens, mask, weights = make_batch(mixed, args.batch_size, rng)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        window.append(float(loss.detach()))

        if step % args.eval_every == 0 or step == args.steps:
            val = full_loss(model, val_examples)
            train_full = full_loss(model, train_examples)
            entry = {
                "step": step,
                "train_window": round(sum(window) / len(window), 4),
                "train_full": round(train_full, 4),
                "val": round(val, 4),
                "elapsed_s": round(time.time() - started, 1),
            }
            window = []
            if val < best["val"]:
                best = {
                    "val": val,
                    "step": step,
                    "train_full": train_full,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }
                entry["best"] = True
            with open(curve_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"[{args.name}] step {step} train {train_full:.4f} val {val:.4f}", flush=True)

    # Final summary + probe samples from the best-val weights.
    final_val = full_loss(model, val_examples)
    final_train = full_loss(model, train_examples)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    samples = {}
    for probe in PROBES:
        torch.manual_seed(args.seed)
        samples[probe] = continue_text(model, probe, max_new_tokens=64, temperature=0.5, top_k=40)
    summary = {
        "summary": True,
        "final_step": args.steps,
        "final_train_full": round(final_train, 4),
        "final_val": round(final_val, 4),
        "best_step": best["step"],
        "best_val": round(best["val"], 4) if best["val"] < math.inf else None,
        "best_train_full": round(best["train_full"], 4) if best["train_full"] else None,
        "elapsed_s": round(time.time() - started, 1),
        "samples": samples,
    }
    with open(curve_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    torch.save(
        {"config": config.to_dict(), "model": model.state_dict(), "step": best["step"]},
        args.out_dir / f"{args.name}-best.pt",
    )
    print(f"[{args.name}] DONE {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
