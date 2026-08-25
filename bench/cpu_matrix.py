"""The CPU decode matrix bench: weights x KV dtype x batch x threads x context.

This is the follow-up benchmark called for by
docs/reports/CPU_L3_EXTREME_OPTIMIZATION_REPORT_2026-08-25.md. It measures the
served checkpoint read-only across:

    weight mode: fp32 / int8 / int8-head / int4
    KV dtype:    fp32 / bf16 / fp16
    batch:       1 / 2 / 4 / 8 ...
    threads:     1 / 2 / 3 / 4 / 8 ...
    context:     prompt lengths in tokens

and records TTFT, steady and end-to-end tok/s, aggregate tok/s (batch),
p50/p95 completion latency, and RSS. When the data dir actually holds a
corpus it also reports bits/char and fixed-seed samples per weight mode, so
quality gates travel with the speed numbers.

Speed-only runs are safe anywhere:

    BABBLE_DATA_DIR=$(mktemp -d) python bench/cpu_matrix.py \\
        --checkpoint /path/to/latest.pt \\
        --weights fp32,int8,int8-head --threads 1,2,3,4,8

Does not train, does not write checkpoints, does not touch the live process.

`int4` uses torch's packed tinygemm CPU kernel
(`aten._weight_int4pack_mm_for_cpu`). On CPUs without AVX512 that kernel is a
scalar fallback and is dramatically *slower* than fp32 — that is a real
finding, not a bug in this bench; include `int4` in `--weights` only when you
want to document it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from babble.cpu_runtime import (
    configure_cpu,
    force_cpu_device,
    quantize_dynamic_linears,
    quantize_int8_head,
)
from babble.generate import tokenizer_for_checkpoint
from babble.model import Babbler, ModelConfig

KV_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class Int4Linear(nn.Module):
    """Weight-only int4 Linear on torch's packed tinygemm CPU kernel.

    Asymmetric group quantization in the tinygemm scales-and-zeros layout,
    verified against a dequantized fp32 matmul. Bench-only: the kernel is
    vectorized for AVX512 and falls back to scalar code on the AVX2 deploy
    box, where it measured ~25x slower than fp32 — see the 2026-08-25
    follow-up report before promoting this anywhere.
    """

    def __init__(self, linear: nn.Linear, group_size: int = 64) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("babble linears are bias-free")
        w = linear.weight.detach().to(torch.float32)
        n, k = w.shape
        if k % group_size:
            raise ValueError(f"in_features {k} not divisible by group {group_size}")
        self.out_features = n
        self.in_features = k
        self.group_size = group_size
        wg = w.reshape(n, k // group_size, group_size)
        mx = wg.amax(-1, keepdim=True)
        mn = wg.amin(-1, keepdim=True)
        scales = (mx - mn).clamp(min=1e-6) / 15
        zeros = mn + scales * 8
        q = wg.sub(mn).div(scales).round().clamp(0, 15).to(torch.int32).reshape(n, k)
        self.register_buffer(
            "packed", torch.ops.aten._convert_weight_to_int4pack_for_cpu(q, 1)
        )
        self.register_buffer(
            "scales_zeros",
            torch.stack([scales.squeeze(-1).t(), zeros.squeeze(-1).t()], dim=-1)
            .contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x.reshape(-1, self.in_features),
            self.packed,
            self.group_size,
            self.scales_zeros,
        )
        return y.view(*lead, self.out_features)


def quantize_int4_linears(model: Babbler, group_size: int = 64) -> Babbler:
    """Every Linear (including a tied lm_head, via clone) -> Int4Linear."""
    if model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.tok_emb.weight.detach().clone())

    def swap(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, Int4Linear(child, group_size))
            else:
                swap(child)

    swap(model)
    return model


def load_raw(path: Path) -> Babbler:
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    model.tokenizer = tokenizer_for_checkpoint(path, model.config.vocab_size)
    model.to(device)
    model.eval()
    return model


def build_model(ckpt: Path, weight_mode: str, group_size: int) -> Babbler:
    model = load_raw(ckpt)
    if weight_mode == "fp32":
        return model
    if weight_mode == "int8":
        return quantize_dynamic_linears(model)
    if weight_mode == "int8-head":
        return quantize_int8_head(model)
    if weight_mode == "int4":
        return quantize_int4_linears(model, group_size)
    raise ValueError(f"unknown weight mode {weight_mode!r}")


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


@torch.inference_mode()
def decode_once(
    model: Babbler,
    ctx: torch.Tensor,
    n_new: int,
    kv_dtype: torch.dtype,
) -> dict:
    """One batched cached greedy decode. Returns per-run timing facts."""
    batch, ctx_len = ctx.shape
    cache = model.new_cache(
        batch,
        max_len=min(model.config.block_size, ctx_len + n_new),
        dtype=kv_dtype,
    )
    t0 = time.perf_counter()
    logits = model(ctx, cache=cache)[:, -1]
    tok = logits.argmax(dim=-1)
    ttft = time.perf_counter() - t0
    body = 0.0
    n_body = 0
    step = torch.empty((batch, 1), dtype=torch.long)
    for _ in range(n_new - 1):
        if cache.length >= cache.max_len:
            break
        s = time.perf_counter()
        step[:, 0] = tok
        logits = model(step, cache=cache)[:, -1]
        tok = logits.argmax(dim=-1)
        body += time.perf_counter() - s
        n_body += 1
    total = time.perf_counter() - t0
    return {
        "ttft_s": ttft,
        "body_s": body,
        "n_body": n_body,
        "total_s": total,
        "generated": n_body + 1,
    }


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ordered = sorted(xs)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def bench_cell(
    model: Babbler,
    *,
    ctx_len: int,
    batch: int,
    tokens: int,
    kv_dtype: torch.dtype,
    runs: int,
    warmup: int,
    seed: int = 0,
) -> dict:
    block = model.config.block_size
    ctx_len = min(ctx_len, block - 1)
    n_new = min(tokens, block - ctx_len)
    gen = torch.Generator().manual_seed(seed)
    ctx = torch.randint(0, model.config.vocab_size, (batch, ctx_len), generator=gen)
    for _ in range(warmup):
        decode_once(model, ctx, n_new, kv_dtype)
    facts = [decode_once(model, ctx, n_new, kv_dtype) for _ in range(runs)]
    steady = [f["n_body"] / f["body_s"] for f in facts if f["body_s"] > 0]
    e2e = [f["generated"] / f["total_s"] for f in facts]
    totals = [f["total_s"] for f in facts]
    med_steady = _median(steady)
    return {
        "ctx": ctx_len,
        "new_tokens": n_new,
        "ttft_ms": _median([f["ttft_s"] for f in facts]) * 1e3,
        "steady_tps": med_steady,
        "e2e_tps": _median(e2e),
        "agg_steady_tps": med_steady * batch,
        "p50_s": _median(totals),
        "p95_s": _p95(totals),
        "rss_mb": rss_mb(),
    }


def quality(model: Babbler) -> dict | None:
    """bits/char + fixed-seed samples, only when a real corpus is present."""
    from babble.config import Settings
    from babble.identity import Pseudonymiser
    from babble.subword import text_examples
    from babble.trainer import corpus_rows, split_rows

    from bench.cpu_fast import samples, val_bpc

    settings = Settings.from_env()
    try:
        rows = corpus_rows(settings, Pseudonymiser.load(settings))
        split = split_rows(rows, settings)
    except Exception:
        return None
    if not split.val:
        return None
    tok = model.tokenizer
    val_ex = [
        ex
        for row in split.val
        for ex in text_examples(tok, row.text, model.config.block_size)
    ]
    val_chars = sum(len(r.text) for r in split.val)
    return {
        "bits_per_char": val_bpc(model, val_ex, tok.specials.pad, val_chars),
        "samples": samples(model),
    }


def _csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--weights", default="fp32,int8,int8-head")
    p.add_argument("--kv", default="fp32")
    p.add_argument("--threads", default="4")
    p.add_argument("--batch", default="1")
    p.add_argument("--context", default="16")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--quality", action="store_true",
                   help="also compute bits/char + samples (needs a corpus)")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    weight_modes = [w.strip() for w in args.weights.split(",") if w.strip()]
    kv_names = [k.strip() for k in args.kv.split(",") if k.strip()]
    for k in kv_names:
        if k not in KV_DTYPES:
            print(f"unknown KV dtype {k!r} (choose from {sorted(KV_DTYPES)})",
                  file=sys.stderr)
            return 1
    threads_list = _csv_ints(args.threads)
    batch_list = _csv_ints(args.batch)
    ctx_list = _csv_ints(args.context)

    configure_cpu(threads_list[0])
    rows: list[dict] = []
    header = (f"{'weights':<10} {'kv':<5} {'thr':>3} {'B':>2} {'ctx':>5} "
              f"{'ttft ms':>8} {'steady':>7} {'e2e':>7} {'agg':>8} "
              f"{'p95 s':>7} {'rss MB':>7}")
    print(f"checkpoint {args.checkpoint}  tokens={args.tokens} "
          f"runs={args.runs} warmup={args.warmup}")
    print(header)
    for weight_mode in weight_modes:
        model = build_model(args.checkpoint, weight_mode, args.group_size)
        for threads in threads_list:
            configure_cpu(threads)
            for kv_name in kv_names:
                for batch in batch_list:
                    for ctx_len in ctx_list:
                        cell = bench_cell(
                            model,
                            ctx_len=ctx_len,
                            batch=batch,
                            tokens=args.tokens,
                            kv_dtype=KV_DTYPES[kv_name],
                            runs=args.runs,
                            warmup=args.warmup,
                        )
                        cell.update(weights=weight_mode, kv=kv_name,
                                    threads=threads, batch=batch)
                        rows.append(cell)
                        print(f"{weight_mode:<10} {kv_name:<5} {threads:>3} "
                              f"{batch:>2} {cell['ctx']:>5} "
                              f"{cell['ttft_ms']:>8.1f} "
                              f"{cell['steady_tps']:>7.1f} "
                              f"{cell['e2e_tps']:>7.1f} "
                              f"{cell['agg_steady_tps']:>8.1f} "
                              f"{cell['p95_s']:>7.3f} {cell['rss_mb']:>7.0f}")
        if args.quality:
            q = quality(model)
            if q is None:
                print(f"{weight_mode}: no corpus/val rows here — quality skipped")
            else:
                print(f"{weight_mode}: bits/char = {q['bits_per_char']:.4f}")
                for prompt, sample_text in q["samples"].items():
                    print(f"  {prompt!r} -> {sample_text!r}")
                rows.append({"weights": weight_mode, "quality": q})
        del model

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
