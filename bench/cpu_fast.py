"""CPU decode speed + bits/char ablations for the served checkpoint.

Anyone can run this on the box that holds `latest.pt` and the Discord corpus:

    BABBLE_DATA_DIR=/home/beckett/babble-live/data \\
    BABBLE_CHECKPOINT_DIR=/home/beckett/babble-live/checkpoints \\
    python bench/cpu_fast.py --tokens 64 --runs 5

Prints tok/s (cached greedy, full length) and bits/char on the Discord val
split for: fp32 baseline, int8 Linear, and the serving default. Also prints
same-seed completions before/after int8.

Does not train, does not write checkpoints, does not touch the live process.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from babble.config import Settings
from babble.cpu_runtime import configure_cpu, force_cpu_device, quantize_dynamic_linears
from babble.generate import continue_text, tokenizer_for_checkpoint
from babble.identity import Pseudonymiser
from babble.model import Babbler, ModelConfig, sequence_loss
from babble.subword import stack_examples, text_context, text_examples
from babble.trainer import corpus_rows, split_rows

PROBES = ["hola", "hello", "the cat", "why is"]


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def load_raw(path: Path) -> Babbler:
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    model.tokenizer = tokenizer_for_checkpoint(path, model.config.vocab_size)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def greedy_cached_tps(model: Babbler, ctx: list[int], n_new: int) -> tuple[float, float]:
    """Steady tok/s (tokens 2..N) and e2e tok/s including prefill. Never stops on eos."""
    cache = model.new_cache(1, max_len=min(model.config.block_size, len(ctx) + n_new))
    t0 = time.perf_counter()
    logits = model(torch.tensor([ctx], dtype=torch.long), cache=cache)[:, -1]
    tok = int(logits.argmax(dim=-1)[0])
    ttft = time.perf_counter() - t0
    step = torch.empty((1, 1), dtype=torch.long)
    body = 0.0
    n_body = 0
    for _ in range(n_new - 1):
        if cache.length >= cache.max_len:
            break
        s = time.perf_counter()
        step[0, 0] = tok
        logits = model(step, cache=cache)[:, -1]
        tok = int(logits.argmax(dim=-1)[0])
        body += time.perf_counter() - s
        n_body += 1
    total = time.perf_counter() - t0
    steady = (n_body / body) if body > 0 else float("nan")
    e2e = ((n_body + 1) / total) if total > 0 else float("nan")
    _ = ttft
    return steady, e2e


def bits_per_char(nats_total: float, chars: int) -> float:
    return (nats_total / max(chars, 1)) / math.log(2)


@torch.inference_mode()
def val_bpc(model: Babbler, examples, pad_id: int, val_chars: int) -> float:
    if not examples:
        return float("nan")
    # Score in modest batches so a 34M model does not try to pad every val
    # row into one giant tensor.
    nats = 0.0
    batch = 4
    for i in range(0, len(examples), batch):
        chunk = examples[i : i + batch]
        tokens, mask, weights = stack_examples(chunk, pad_id)
        per = sequence_loss(model, tokens, mask, weights)
        n = float((mask[:, 1:] * weights[:, None]).sum().item())
        nats += float(per) * n
    return bits_per_char(nats, val_chars)


def samples(model: Babbler, seed: int = 0) -> dict[str, str]:
    out = {}
    for p in PROBES:
        g = torch.Generator().manual_seed(seed)
        out[p] = continue_text(
            model, p, max_new_tokens=60, temperature=0.5, top_k=40, generator=g
        )
    return out


def time_load(path: Path, *, quantize: bool) -> float:
    t0 = time.perf_counter()
    m = load_raw(path)
    if quantize:
        m = quantize_dynamic_linears(m)
    _ = m(torch.tensor([[0]], dtype=torch.long))
    return time.perf_counter() - t0


def linear_nbytes(model: nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight
            n += w.numel() * (1 if w.dtype == torch.qint8 else w.element_size())
            if w.dtype == torch.qint8 and hasattr(mod, "scale"):
                pass
        elif type(mod).__name__.endswith("Linear"):
            w = getattr(mod, "weight", None)
            if w is None:
                continue
            # Packed quantized linear: report packed payload if present.
            packed = getattr(mod, "_packed_params", None)
            if packed is not None:
                try:
                    qw, _bias = packed.unpack()
                    n += qw.numel()
                    continue
                except Exception:
                    pass
            n += w.numel() * getattr(w, "element_size", lambda: 4)()
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--skip-compile", action="store_true", default=True)
    p.add_argument("--try-compile", action="store_true")
    args = p.parse_args(argv)

    settings = Settings.from_env()
    ckpt = args.checkpoint or settings.latest_checkpoint
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt}", file=sys.stderr)
        return 1

    configure_cpu(args.threads)
    os.environ["BABBLE_QUANTIZE"] = "0"  # samples go through continue_text, not load_model

    rows = corpus_rows(settings, Pseudonymiser.load(settings))
    split = split_rows(rows, settings)
    val_chars = sum(len(r.text) for r in split.val)

    print(f"checkpoint {ckpt}  threads={args.threads}  tokens={args.tokens}  "
          f"val_rows={len(split.val)} val_chars={val_chars}")

    fp32 = load_raw(ckpt)
    tok = fp32.tokenizer
    val_ex = [ex for row in split.val for ex in text_examples(tok, row.text, fp32.config.block_size)]
    ctx = text_context(tok, "hello", fp32.config.block_size)

    print(f"params={fp32.num_params():,}  config={fp32.config.to_dict()}")
    print(f"Linear+emb weight bytes fp32 ≈ {fp32.num_params() * 4 / 1e6:.1f} MB")

    bpc_fp32 = val_bpc(fp32, val_ex, tok.specials.pad, val_chars)
    samp_fp32 = samples(fp32)
    print(f"\nfp32 bits/char = {bpc_fp32:.4f}")
    print("fp32 samples (seed=0, temp=0.5, top_k=40, 60 tok):")
    for k, v in samp_fp32.items():
        print(f"  {k!r} -> {v!r}")

    for _ in range(args.warmup):
        greedy_cached_tps(fp32, ctx, args.tokens)
    fp32_steady, fp32_e2e = zip(*[greedy_cached_tps(fp32, ctx, args.tokens) for _ in range(args.runs)])
    print(f"fp32 cached greedy: steady {_median(list(fp32_steady)):.1f} tok/s  "
          f"e2e {_median(list(fp32_e2e)):.1f} tok/s")

    t_q0 = time.perf_counter()
    int8 = quantize_dynamic_linears(fp32)
    q_s = time.perf_counter() - t_q0
    print(f"\nquantize_dynamic wall {q_s:.2f}s")

    bpc_int8 = val_bpc(int8, val_ex, tok.specials.pad, val_chars)
    samp_int8 = samples(int8)
    print(f"int8 bits/char = {bpc_int8:.4f}  delta {bpc_int8 - bpc_fp32:+.4f}")
    print("int8 samples (same seed/prompts):")
    for k, v in samp_int8.items():
        print(f"  {k!r} -> {v!r}")

    for _ in range(args.warmup):
        greedy_cached_tps(int8, ctx, args.tokens)
    i8_steady, i8_e2e = zip(*[greedy_cached_tps(int8, ctx, args.tokens) for _ in range(args.runs)])
    print(f"int8 cached greedy: steady {_median(list(i8_steady)):.1f} tok/s  "
          f"e2e {_median(list(i8_e2e)):.1f} tok/s")

    if args.try_compile:
        raw = load_raw(ckpt)
        t0 = time.perf_counter()
        compiled = torch.compile(raw, backend="inductor", mode="reduce-overhead")
        compiled.eval()
        with torch.inference_mode():
            _ = compiled(torch.tensor([ctx], dtype=torch.long))
        compile_s = time.perf_counter() - t0
        print(f"\ntorch.compile first-forward {compile_s:.1f}s")
        try:
            st, e2 = zip(*[greedy_cached_tps(compiled, ctx, min(16, args.tokens)) for _ in range(2)])
            print(f"compiled greedy: steady {_median(list(st)):.1f} tok/s e2e {_median(list(e2)):.1f}")
        except Exception as exc:
            print(f"compiled decode failed: {type(exc).__name__}: {exc}")

    print("\n=== table ===")
    print(f"{'config':<12} {'tok/s':>8} {'bits/char':>10}")
    print(f"{'fp32':<12} {_median(list(fp32_steady)):8.1f} {bpc_fp32:10.4f}")
    print(f"{'int8':<12} {_median(list(i8_steady)):8.1f} {bpc_int8:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
