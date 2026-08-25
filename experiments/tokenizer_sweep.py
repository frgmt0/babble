"""One tokenizer-swap run: train+eval with byte / BPE / word tokenization,
reporting a normalised bits-per-character number so the three are comparable
on one axis -- see the measurement trap in docs/reports/CAPACITY_TOKENIZER_REPORT.md.

Mirrors experiments/sweep.py as closely as possible (same corpus/split via
babble.trainer, same optimiser loop, same early-stop-off + keep-best-val
discipline) so a number here means the same thing as a sweep.py number and
the same thing as babble/trainer.py's live number. The one thing that
necessarily differs is what "one token" is -- the reason this script exists
is to make that difference legible instead of accidentally comparing
per-token numbers that mean different things.

Usage:
    python -m experiments.tokenizer_sweep --name bpe1k --steps 800 \\
        --tokenizer bpe --vocab-size 1000 [--n-layer 4 --n-embd 256] [--seed 1]
    python -m experiments.tokenizer_sweep --name word1k --steps 800 \\
        --tokenizer word --vocab-size 1000
    python -m experiments.tokenizer_sweep --name byte-ref --steps 800 \\
        --tokenizer byte   # vocab-size ignored; same as babble/tokenizer.py

`--tokenizer byte` is the default -- this script changes nothing about the
live path even when imported, it only ever *reads* babble/subword.py's
ByteTokenizer adapter, which is a passthrough to babble/tokenizer.py.
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
import torch.nn.functional as F

from babble.config import Settings
from babble.cpu_runtime import configure_cpu, force_cpu_device
from babble.model import Babbler, ModelConfig, per_token_loss
from babble.subword import BPETokenizer, ByteTokenizer, WordTokenizer, text_context, text_examples
from babble.trainer import corpus_rows, split_rows

PROBES = ("hello", "the cat", "why is", "boop")

# Structural ids are inputs, never outputs -- same rule babble/generate.py
# applies for the byte tokenizer, generalised to whichever specials this
# tokenizer landed on.
_BANNED_KINDS = ("pad", "bos", "sep")


def make_tokenizer(kind: str, vocab_size: int, train_texts: list[str]):
    """Build and, for the learned kinds, train a tokenizer -- on TRAIN-side
    corpus text only. Fitting a BPE/word vocabulary on validation text would
    leak held-out phrasing into the vocabulary itself, before a single
    gradient step runs; that would be a subtler version of the same mistake
    as training an n-gram on the whole corpus."""
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe":
        # vocab_size counts the FULL id space (256 bytes + merges + 4
        # specials), matching how --vocab-size is described on the CLI.
        num_merges = max(0, vocab_size - 256 - 4)
        return BPETokenizer.train(train_texts, num_merges=num_merges)
    if kind == "word":
        max_words = max(0, vocab_size - 256 - 4)
        return WordTokenizer.train(train_texts, max_words=max_words)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def to_examples(tok, rows, block_size: int):
    return [example for row in rows for example in text_examples(tok, row.text, block_size)]


def char_count(rows) -> int:
    """Unicode character count of `rows`' raw text -- one row counts once no
    matter how many chunks the tokenizer's block-size splitting produces, so
    it means the same thing across tokenizers with very different chunk
    counts for the same text."""
    return sum(len(row.text) for row in rows)


def make_batch(examples, batch_size: int, rng: random.Random, pad_id: int):
    """Sample with replacement and right-pad -- same rule as
    babble.trainer.make_batch, parameterised by pad id since that id moves
    with the tokenizer's vocab size here."""
    chosen = [rng.choice(examples) for _ in range(batch_size)]
    width = max(len(e) for e in chosen)
    tokens = torch.full((len(chosen), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(chosen), width), dtype=torch.long)
    for i, ex in enumerate(chosen):
        tokens[i, : len(ex)] = torch.as_tensor(ex.tokens, dtype=torch.long)
        mask[i, : len(ex)] = torch.as_tensor(ex.mask, dtype=torch.long)
    weights = torch.as_tensor([e.weight for e in chosen], dtype=torch.float32)
    return tokens, mask, weights


def sequence_loss(model, tokens, mask, weights):
    per_token = per_token_loss(model, tokens)
    scale = mask[:, 1:].to(per_token.dtype) * weights[:, None]
    return (per_token * scale).sum() / scale.sum().clamp(min=1e-8)


@torch.inference_mode()
def full_loss_stats(model, examples, pad_id: int, chunk: int = 64) -> tuple[float, int]:
    """`(total nats, target-token count)` -- see experiments/sweep.py's
    identically-named function, which this mirrors so the two scripts can't
    quietly define "loss" two different ways."""
    was_training = model.training
    model.eval()
    try:
        total, tokens = 0.0, 0
        for i in range(0, len(examples), chunk):
            batch = examples[i : i + chunk]
            width = max(len(e) for e in batch)
            t = torch.full((len(batch), width), pad_id, dtype=torch.long)
            m = torch.zeros((len(batch), width), dtype=torch.long)
            for j, ex in enumerate(batch):
                t[j, : len(ex)] = torch.as_tensor(ex.tokens, dtype=torch.long)
                m[j, : len(ex)] = torch.as_tensor(ex.mask, dtype=torch.long)
            pt = per_token_loss(model, t)
            tm = m[:, 1:].to(pt.dtype)
            total += float((pt * tm).sum())
            tokens += int(tm.sum())
        return total, tokens
    finally:
        model.train(was_training)


def bits_per_char(nats_total: float, chars: int) -> float:
    return (nats_total / max(chars, 1)) / math.log(2)


@torch.inference_mode()
def continue_text(model, tok, prefix: str, *, max_new_tokens: int, temperature: float, top_k: int) -> str:
    """Sample a continuation after `prefix`, generic over any of the three
    tokenizers. A trimmed-down, cache-free version of
    babble/generate.py's `_decode_from` -- these runs are small enough
    (a handful of probes per experiment) that the KV cache's speed is not
    worth the extra surface, but the sampling rule (temperature, top-k,
    banned structural ids, stop on <eos>) is the same one the live bot uses.
    """
    was_training = model.training
    model.eval()
    try:
        banned = [getattr(tok.specials, k) for k in _BANNED_KINDS]
        context = text_context(tok, prefix, model.config.block_size)
        produced: list[int] = []
        for _ in range(max_new_tokens):
            window = context[-model.config.block_size :]
            logits = model(torch.tensor([window], dtype=torch.long))[:, -1].clone()
            logits[:, banned] = float("-inf")
            if temperature <= 0:
                nxt = int(logits.argmax(dim=-1)[0])
            else:
                logits = logits / temperature
                if top_k and 0 < top_k < logits.size(-1):
                    cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1)[0, 0])
            if nxt == tok.specials.eos:
                break
            produced.append(nxt)
            context.append(nxt)
        return tok.decode(produced)
    finally:
        model.train(was_training)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--tokenizer", choices=("byte", "bpe", "word"), default="byte")
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="full id-space size (256 bytes + merges/words + 4 specials); ignored for --tokenizer byte",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/results/tokenizer"))
    args = ap.parse_args()

    configure_cpu(args.threads)
    device = force_cpu_device()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    settings = Settings.from_env()
    rows = corpus_rows(settings)
    split = split_rows(rows, settings)

    tok = make_tokenizer(args.tokenizer, args.vocab_size, [r.text for r in split.train])
    pad_id = tok.specials.pad

    config = ModelConfig(
        vocab_size=tok.vocab_size,
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
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
        if args.cosine
        else None
    )

    train_examples = to_examples(tok, split.train, config.block_size)
    val_examples = to_examples(tok, split.val, config.block_size)
    val_chars = char_count(split.val)
    train_chars = char_count(split.train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.out_dir / f"{args.name}.jsonl"
    learned_vocab = getattr(tok, "merges", None) or getattr(tok, "vocab", None)
    meta = {
        "name": args.name,
        "tokenizer": args.tokenizer,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": tok.vocab_size,
        "steps": args.steps,
        "lr": args.lr,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "block_size": args.block_size,
        "dropout": args.dropout,
        "cosine": args.cosine,
        "params": model.num_params(),
        "train_rows": len(split.train),
        "val_rows": len(split.val),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_chars": train_chars,
        "val_chars": val_chars,
        "train_tokens": sum(len(e) - 1 for e in train_examples),
        "val_tokens": sum(len(e) - 1 for e in val_examples),
        "seed": args.seed,
    }
    with open(curve_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
    print(f"[{args.name}] {json.dumps(meta)}", flush=True)

    best = {"val_bpc": math.inf, "step": 0, "state": None}
    window: list[float] = []
    started = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        tokens, mask, weights = make_batch(train_examples, args.batch_size, rng, pad_id)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        window.append(float(loss.detach()))

        if step % args.eval_every == 0 or step == args.steps:
            val_nats, val_tokens = full_loss_stats(model, val_examples, pad_id)
            val_mean = val_nats / max(val_tokens, 1)
            val_bpc = bits_per_char(val_nats, val_chars)
            entry = {
                "step": step,
                "train_window": round(sum(window) / len(window), 4),
                "val_per_token": round(val_mean, 4),
                "val_bits_per_char": round(val_bpc, 4),
                "elapsed_s": round(time.time() - started, 1),
            }
            window = []
            if val_bpc < best["val_bpc"]:
                best = {
                    "val_bpc": val_bpc,
                    "val_per_token": val_mean,
                    "step": step,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }
                entry["best"] = True
            with open(curve_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(
                f"[{args.name}] step {step} val/token {val_mean:.4f} val_bits_per_char {val_bpc:.4f}",
                flush=True,
            )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    samples = {}
    for probe in PROBES:
        torch.manual_seed(args.seed)
        samples[probe] = continue_text(model, tok, probe, max_new_tokens=64, temperature=0.5, top_k=40)

    summary = {
        "summary": True,
        "tokenizer": args.tokenizer,
        "actual_vocab_size": tok.vocab_size,
        "best_step": best["step"],
        "best_val_per_token": round(best["val_per_token"], 4) if best["step"] else None,
        "best_val_bits_per_char": round(best["val_bpc"], 4) if best["val_bpc"] < math.inf else None,
        "best_val_nats_per_char": (
            round(best["val_bpc"] * math.log(2), 4) if best["val_bpc"] < math.inf else None
        ),
        "val_chars": val_chars,
        "elapsed_s": round(time.time() - started, 1),
        "samples": samples,
    }
    with open(curve_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "tokenizer_kind": args.tokenizer,
            "tokenizer": tok,
            "step": best["step"],
        },
        args.out_dir / f"{args.name}-best.pt",
    )
    print(f"[{args.name}] DONE {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
