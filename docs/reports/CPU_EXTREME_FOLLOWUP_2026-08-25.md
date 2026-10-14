# Extreme CPU optimisation follow-up: int4, KV precision, batching (2026-08-25)

Follow-up to
[`CPU_L3_EXTREME_OPTIMIZATION_REPORT_2026-08-25.md`](CPU_L3_EXTREME_OPTIMIZATION_REPORT_2026-08-25.md),
which ranked the next experiments (int4 weights, low-precision KV cache,
micro-batching) and asked for a matrix benchmark before any default change.
This report implements that benchmark (`bench/cpu_matrix.py`), runs the matrix
on the same box against the served checkpoint read-only, and settles each
question with measurements. **No serving default changed.** Two new opt-ins
exist: `BABBLE_QUANTIZE=head` and `BABBLE_KV_DTYPE=bf16|fp16`.

Box: Intel Core i7-4790 (Haswell, AVX2, **no AVX512/VNNI**), 4 cores / 8
threads, 8 MiB L3, dual-channel DDR3. torch 2.13.0+cpu. Model: the served
34,096,128-param BPE checkpoint (8×512, vocab 16384, block 1024), loaded from
`~/babble-live/checkpoints/latest.pt` read-only with `BABBLE_DATA_DIR`
pointed at an empty temp dir. Decoding: cached greedy, batch-shaped, median
over runs after warmup (`bench/cpu_matrix.py --json` for raw rows).

## Executive conclusions

1. **Weight-only int4 is dead on this hardware.** Both torch CPU int4 kernels
   are numerically correct but scalar fallbacks on AVX2: the packed tinygemm
   op (`aten._weight_int4pack_mm_for_cpu`) measured **25–30x slower** than
   fp32 per-GEMM at decode shapes, and the KleidiAI-style dynamic op
   (`aten._dyn_quant_matmul_4bit`) ~64x slower. End-to-end, an int4 model
   decodes at **28 tok/s vs 103 fp32** — a 3.7x regression — while saving RAM
   (289 MB RSS vs 365). Their vectorised paths require AVX512 (or ARM); this
   CPU has neither. Int4 on this box would need a hand-written AVX2 kernel
   (ggml-style, option 6 of the parent report). Not worth it while int8
   already delivers 150 tok/s.
2. **A low-precision KV cache loses end-to-end.** Haswell has no native
   half-precision compute; CPU SDPA in bf16/fp16 (or upcasting K/V each step)
   measured slower at every context length, worst exactly where the L3
   theory hoped it would win (int8 weights, 896-token context: 122.6 tok/s
   fp32-KV → **107.8** bf16-KV). Wired and tested behind `BABBLE_KV_DTYPE`
   for future hardware; do not enable on this box.
3. **New result — int8 on the output head only** (`BABBLE_QUANTIZE=head`):
   the 512→16384 lm_head is the single largest fp32 GEMM per token, and
   dynamic int8 makes that one matmul ~8x faster. Head-only decode is
   **119 tok/s vs 103 fp32** (+16%) while all eight transformer blocks stay
   bit-exact fp32 — logits are ~6x closer to fp32 than full int8 (mean KL
   0.0010 vs 0.0065), and all four fixed-seed probe samples are identical to
   fp32's. A middle rung on the speed/fidelity ladder; full int8 remains the
   throughput champion at ~150 tok/s.
4. **Micro-batching is the big aggregate lever, as predicted.** With int8
   weights at 4 threads, batch 8 delivers **925 aggregate tok/s** (6.2x
   batch 1) while per-stream speed only drops 150 → 116 tok/s. Even fp32 at
   `best_of=4`'s natural batch of 4 gets 337 aggregate tok/s. Each weight
   load is amortised across the batch, exactly the L3 reuse the parent
   report described.
5. **Threads: 3–4 confirmed again**; 8 regresses every mode (int8 drops
   150 → 116).

## Thread sweep (batch 1, ctx 16, 128 tokens, steady tok/s)

| threads | fp32 | int8 | int8-head |
|---:|---:|---:|---:|
| 1 | 82.4 | 136.0 | 94.6 |
| 2 | 100.4 | 147.7 | 113.5 |
| 3 | 102.4 | 149.1 | **119.7** |
| 4 | **102.9** | **149.6** | 119.0 |
| 8 | 87.1 | 115.5 | 97.7 |

Matches the parent report's fp32/int8 numbers run-for-run, so the new
harness measures the same thing `bench/cpu_fast.py` did.

## KV dtype × context (4 threads, batch 1, steady tok/s)

| weights | KV | ctx 16 | ctx 256 | ctx 896 |
|---|---|---:|---:|---:|
| fp32 | fp32 | **103.1** | **96.1** | **87.7** |
| fp32 | bf16 | 99.5 | 94.6 | 81.5 |
| fp32 | fp16 | 98.0 | 94.1 | 85.7 |
| int8 | fp32 | **146.4** | **139.9** | 122.6 |
| int8 | bf16 | 141.2 | 131.6 | 107.8 |
| int8 | fp16 | 142.9 | 137.1 | **123.5** |

fp32 KV wins or ties every cell (the one fp16 "win" is within run noise).
Conversion cost exceeds the bandwidth saved; the parent report's caution
("could make lower precision slower") is the measured outcome.

## Batch sweep (4 threads, fp32 KV, 128 tokens)

| weights | B | ctx | per-stream tok/s | aggregate tok/s | TTFT ms | p95 s |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 1 | 16 | 102.0 | 102.0 | 13 | 1.26 |
| fp32 | 4 | 16 | 84.3 | 337.1 | 25 | 1.54 |
| fp32 | 8 | 16 | 76.2 | 609.7 | 37 | 1.70 |
| int8 | 1 | 16 | 150.5 | 150.5 | 9 | 0.86 |
| int8 | 2 | 16 | 143.4 | 286.8 | 12 | 0.90 |
| int8 | 4 | 16 | 131.8 | 527.3 | 20 | 1.02 |
| int8 | 8 | 16 | 115.7 | 925.3 | 35 | 1.15 |
| int8 | 8 | 256 | 82.6 | 660.8 | 523 | 2.09 |

`best_of` already rides this curve (one batched decode per reply). A
cross-request scheduler batching independent Discord replies would trade a
bounded queueing delay for up to ~6x aggregate throughput; worth designing
only if the bot actually becomes decode-throughput-bound.

## Quality ladder (live checkpoint, 8 random 64-token contexts + fixed-seed samples)

| mode | mean KL vs fp32 | top-5 overlap | max │Δlogit│ | seed-0 samples changed |
|---|---:|---:|---:|---:|
| int8-head | **0.0010** | 97.5% | 0.24 | 0/4 |
| int8 | 0.0065 | 90.0% | 0.69 | 0/4 |
| int4 (g=64) | 0.0311 | 77.5% | 1.68 | 1/4 |

Logit-level fidelity only — this box holds no corpus, so bits/char must be
gated on a box with the Discord val split before any promotion:
`bench/cpu_matrix.py --quality` computes it when `BABBLE_DATA_DIR` has real
rows (per the parent report, do not infer quality from perplexity alone).

## What landed (all opt-in, defaults unchanged)

- `bench/cpu_matrix.py` — the matrix bench the parent report asked for:
  `--weights fp32,int8,int8-head,int4 --kv fp32,bf16,fp16 --threads …
  --batch … --context …`, reporting TTFT, steady/e2e/aggregate tok/s,
  p50/p95, RSS, and (with a corpus) bits/char + samples. Includes the
  kernel-backed `Int4Linear` used for the int4 measurements.
- `babble/cpu_runtime.py` — `quantize_int8_head()` and
  `BABBLE_QUANTIZE=head`; `kv_dtype_from_env()` / `BABBLE_KV_DTYPE`.
- `babble/model.py` — `new_cache(..., dtype=…)`; attention runs in the cache
  dtype when it differs from the model dtype.
- Tests for all of the above in `tests/test_cpu_model.py`.

## Recommendation

Serving stays fp32 by default (project bar: no default quality change). If
the deploy wants speed, the measured menu on this box is:

| option | per-stream | quality cost |
|---|---:|---|
| `BABBLE_QUANTIZE=head` | ~119 tok/s (+16%) | logit rounding in one matmul; KL 0.0010 |
| `BABBLE_QUANTIZE=1` | ~150 tok/s (+45%) | +0.0024 bits/char (prior measurement) |

Do not pursue int4 or low-precision KV further on this hardware. The next
real steps, if 150 tok/s ever stops being enough, are (a) a bounded
cross-request batching window, and (b) a packed AVX2 int8/int4 runtime
(ggml-style) — both per the parent report's options 4 and 6.
