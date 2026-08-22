"""Detached, throttled CPU pretrain on Discord-Dialogues.

This is the local multi-day run: same 42.5M architecture the bot serves
(vocab 16384, block 1024, 8/8/512), trained slowly on a subsample of
`mookiezi/Discord-Dialogues` so the box stays usable. Checkpoints land in a
scratch directory and are never auto-promoted.

The tokenizer is the served 16384 BPE sidecar, copied in — this script never
fits a new vocab. Before every checkpoint write it refuses if
`model.config.vocab_size` disagrees with that tokenizer (the fault that used
to let `babble-train` dump 260-id weights next to a 16384 tokenizer).

Resume: a restart loads `checkpoints/latest.pt` (step, tokens, docs, optim,
rng) and skips already-consumed stream documents.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import torch

from babble.cpu_runtime import configure_cpu, force_cpu_device
from babble.model import Babbler, ModelConfig, sequence_loss
from babble.subword import BPETokenizer, stack_examples, text_examples
from babble.util import utcnow_iso

DATASET_ID = "mookiezi/Discord-Dialogues"
SERVED_TOKENIZER = Path.home() / "babble-live" / "checkpoints" / "tokenizer.json"

# Architecture of the served checkpoint (confirmed on disk, not guessed).
SERVED_SHAPE = dict(
    vocab_size=16384,
    block_size=1024,
    n_layer=8,
    n_head=8,
    n_embd=512,
    dropout=0.1,
)


class VocabMismatch(RuntimeError):
    """Checkpoint write refused: model vocab ≠ tokenizer it was trained with."""


def row_to_text(row: dict) -> str:
    """Flatten one Discord-Dialogues record to a single training string."""
    for key in ("text", "content", "conversation", "conversations", "messages"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list) and val:
            parts = []
            for turn in val:
                if isinstance(turn, str) and turn.strip():
                    parts.append(turn)
                elif isinstance(turn, dict):
                    role = turn.get("role") or turn.get("author") or ""
                    body = turn.get("content") or turn.get("text") or turn.get("value") or ""
                    if body:
                        parts.append(f"{role}: {body}" if role else str(body))
            if parts:
                return "\n".join(parts)
    # Last resort: concatenate every string leaf we can find.
    blobs = [v for v in row.values() if isinstance(v, str) and v.strip()]
    return "\n".join(blobs)


def assert_vocab_matches(model: Babbler, tok: BPETokenizer) -> None:
    model_vocab = int(model.config.vocab_size)
    tok_vocab = int(tok.vocab_size)
    if model_vocab != tok_vocab:
        raise VocabMismatch(
            f"refusing to write checkpoint with vocab_size={model_vocab}: "
            f"training tokenizer has vocab_size={tok_vocab}"
        )


def save_checkpoint(
    ckpt_dir: Path,
    model: Babbler,
    optimizer: torch.optim.Optimizer,
    tok: BPETokenizer,
    step: int,
    tokens_consumed: int,
    docs_consumed: int,
    loss: float,
    keep: int,
) -> Path:
    """Atomic write of numbered ckpt + latest.pt. Never writes on vocab mismatch."""
    assert_vocab_matches(model, tok)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scratch = ckpt_dir / ".partial"
    scratch.mkdir(exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "tokens_consumed": tokens_consumed,
        "docs_consumed": docs_consumed,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "saved_at": utcnow_iso(),
        "dataset_id": DATASET_ID,
    }
    archive = ckpt_dir / f"ckpt-{step:08d}.pt"
    tmp = scratch / f"{archive.name}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, archive)
    latest_tmp = scratch / "latest.pt.tmp"
    shutil.copyfile(archive, latest_tmp)
    os.replace(latest_tmp, ckpt_dir / "latest.pt")
    archives = sorted(ckpt_dir.glob("ckpt-*.pt"))
    for old in archives[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)
    return archive


def open_stream(dataset_id: str, docs_to_skip: int = 0):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    if docs_to_skip:
        ds = ds.skip(docs_to_skip)
    return iter(ds)


def log_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def train(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / "loss.jsonl"
    tok_dest = output_dir / "tokenizer.json"

    threads = max(1, int(args.threads))
    try:
        os.nice(int(args.nice))
    except (OSError, PermissionError, AttributeError):
        pass
    force_cpu_device()
    configure_cpu(threads)

    if tok_dest.exists():
        tok = BPETokenizer.from_json(tok_dest)
    else:
        src = Path(args.tokenizer)
        if not src.is_file():
            raise FileNotFoundError(
                f"served tokenizer not found at {src} — copy it here first"
            )
        shutil.copy2(src, tok_dest)
        tok = BPETokenizer.from_json(tok_dest)
    print(
        f"[tokenizer] {tok.vocab_size}-token BPE from {tok_dest} "
        f"(source={args.tokenizer})",
        flush=True,
    )
    if tok.vocab_size != SERVED_SHAPE["vocab_size"]:
        raise VocabMismatch(
            f"tokenizer vocab_size={tok.vocab_size} does not match served "
            f"architecture vocab_size={SERVED_SHAPE['vocab_size']}"
        )

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    latest = ckpt_dir / "latest.pt"
    if latest.exists() and not args.fresh:
        payload = torch.load(latest, map_location=device, weights_only=False)
        model_cfg = ModelConfig.from_dict(payload["config"])
        model = Babbler(model_cfg)
        model.load_state_dict(payload["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(payload["optim"])
        step = int(payload["step"])
        tokens_consumed = int(payload.get("tokens_consumed", 0))
        docs_consumed = int(payload.get("docs_consumed", 0))
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"].cpu())
        print(
            f"[resume] step {step}, tokens {tokens_consumed:,}, docs {docs_consumed:,}",
            flush=True,
        )
    else:
        model_cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
        model = Babbler(model_cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        step = 0
        tokens_consumed = 0
        docs_consumed = 0
        print(f"[model] fresh init, {model.num_params():,} params", flush=True)

    assert_vocab_matches(model, tok)

    stop_requested = False

    def _handle(signum, _frame):
        nonlocal stop_requested
        print(f"[signal] {signum} — finish step then checkpoint", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    stream = open_stream(args.dataset, docs_to_skip=docs_consumed)
    buffer: list = []
    window: list[float] = []
    t0 = time.time()
    tokens_at_start = tokens_consumed
    steps_this_proc = 0
    print(
        f"[train] dataset={args.dataset} budget={args.token_budget:,} "
        f"batch={args.batch_size}x{args.block_size} threads={threads} nice={args.nice}",
        flush=True,
    )

    while tokens_consumed < args.token_budget and not stop_requested:
        while len(buffer) < args.batch_size:
            try:
                row = next(stream)
            except StopIteration:
                print("[train] stream exhausted", flush=True)
                tokens_consumed = args.token_budget
                break
            docs_consumed += 1
            text = row_to_text(row)
            if text:
                buffer.extend(text_examples(tok, text, args.block_size))
        if tokens_consumed >= args.token_budget:
            break
        if not buffer:
            break

        batch = buffer[: args.batch_size]
        buffer = buffer[args.batch_size :]
        tokens, mask, weights = stack_examples(batch, tok.specials.pad)

        model.train()
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        steps_this_proc += 1
        batch_tokens = int(mask[:, 1:].sum())
        tokens_consumed += batch_tokens
        window.append(float(loss.item()))

        elapsed = time.time() - t0
        run_tokens = max(1, tokens_consumed - tokens_at_start)
        tok_s = run_tokens / max(elapsed, 1e-6)
        mean_loss = sum(window) / len(window)

        if step % args.log_every == 0:
            entry = {
                "step": step,
                "loss": round(mean_loss, 6),
                "tokens_seen": tokens_consumed,
                "docs_consumed": docs_consumed,
                "elapsed_s": round(elapsed, 2),
                "tokens_per_s": round(tok_s, 2),
                "wall_clock": utcnow_iso(),
                "threads": threads,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
            }
            log_jsonl(loss_path, entry)
            print(
                f"[step {step:7d}] loss {mean_loss:7.4f} | "
                f"tokens {tokens_consumed:,} | {tok_s:7.1f} tok/s",
                flush=True,
            )
            window = []

        do_ckpt = (
            step % args.checkpoint_every == 0
            or tokens_consumed >= args.token_budget
            or stop_requested
        )
        if do_ckpt:
            try:
                save_checkpoint(
                    ckpt_dir,
                    model,
                    optimizer,
                    tok,
                    step,
                    tokens_consumed,
                    docs_consumed,
                    mean_loss,
                    args.keep_checkpoints,
                )
            except VocabMismatch as exc:
                print(f"[checkpoint] REFUSED: {exc}", file=sys.stderr, flush=True)
                raise
            print(f"[checkpoint] step {step} tokens {tokens_consumed:,}", flush=True)

        if args.max_steps is not None and steps_this_proc >= args.max_steps:
            break
        if args.bench_steps:
            if steps_this_proc >= args.bench_steps:
                print(
                    f"[bench] {tok_s:.1f} tok/s over {step} steps, "
                    f"{run_tokens} tokens in {elapsed:.1f}s, threads={threads}",
                    flush=True,
                )
                log_jsonl(
                    loss_path,
                    {
                        "event": "bench",
                        "step": step,
                        "loss": round(mean_loss, 6),
                        "tokens_seen": tokens_consumed,
                        "elapsed_s": round(elapsed, 2),
                        "tokens_per_s": round(tok_s, 2),
                        "wall_clock": utcnow_iso(),
                        "threads": threads,
                    },
                )
                break

    print(
        f"[done] step {step} tokens {tokens_consumed:,}/{args.token_budget:,} "
        f"elapsed {time.time() - t0:.1f}s",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=SERVED_TOKENIZER)
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--token-budget", type=int, default=80_000_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=SERVED_SHAPE["block_size"])
    p.add_argument("--n-layer", type=int, default=SERVED_SHAPE["n_layer"])
    p.add_argument("--n-head", type=int, default=SERVED_SHAPE["n_head"])
    p.add_argument("--n-embd", type=int, default=SERVED_SHAPE["n_embd"])
    p.add_argument("--dropout", type=float, default=SERVED_SHAPE["dropout"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--bench-steps", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="ignore existing latest.pt")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
