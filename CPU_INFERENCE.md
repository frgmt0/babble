# CPU inference on the 34.1M BPE model (updated 2026-08-22, post-#26 rebase)

Measured on this box: Intel Core i7-4790 @ 3.60 GHz, 4 cores / 8 threads, 32 GB, **CPU only**, no AVX-512, no VNNI. torch 2.13.0+cpu. Served checkpoint: `~/babble-live/checkpoints/latest.pt` (34,096,128 params, 8×512, vocab 16384, block 1024).

This branch was originally cut from `cbea4f7` (repetition penalty / top-p, #24). Main has since landed `b40396e` (#26: live-tree reconciliation + frequency-scaled repetition/presence penalty and no-repeat-ngram, the fix for the "boop boop boop" → ~85 repeated "boop" tokens bug). This doc reflects the branch **rebased onto `b40396e`**, with main's sampler semantics kept byte-identical — see the reconciliation PR body for the per-file conflict resolution.

Reproduce:

```sh
BABBLE_DATA_DIR=/home/beckett/babble-live/data \
BABBLE_CHECKPOINT_DIR=/home/beckett/babble-live/checkpoints \
  python bench/cpu_fast.py --tokens 64 --runs 4 --threads 4
```

`bench/bench_inference.py` is still the longer TTFT / RAM / thread-sweep harness (BPE-aware). `bench/cpu_fast.py` is the tok/s + bits/char ablation anyone can re-run in under a minute.

## What changed in the rebase, and why the default flipped

The original (pre-rebase) version of this branch shipped dynamic int8 **on by default**, justified by a "well under a 0.05 bits/char budget" bar. That bar was this branch's own, not the project's. Re-measured on the post-rebase code, int8 costs **+0.0024 bits/char** on the current 109-row Discord val split — small, but a real, non-zero regression, and the ticket landing this work requires bits/char **unchanged or better than main by default**. So:

- **`BABBLE_QUANTIZE` now defaults to `0` (off).** Serving ships fp32 by default, so bits/char is exactly main's number, not a regressed one.
- Int8 is still fully wired and tested (`tests/test_cpu_model.py`) — set `BABBLE_QUANTIZE=1` to opt in to the extra decode speed at that small, now-explicit quality cost.
- `BABBLE_INFER_THREADS` (default `4`) and the KV-cache / decode-loop fixes stay on unconditionally: they cost nothing in bits/char (verified below — fp32 bits/char is identical to current `origin/main`, run for run).

## Bottleneck (measured, not assumed)

fp32 cached greedy decode: **~60-65 tok/s** at 4 threads (~15-16 ms/token) on the post-rebase code.

Weights are **136 MB** of fp32. At 4 threads that is roughly **8.5-9 GB/s** of weight traffic — what this dual-channel DDR3 box actually delivers. Single-stream decode is **memory-bandwidth bound**; the Python decode loop is noise next to the Linear traffic.

KV cache is used on the serving path (`_decode_from` / `_decode_many_from` via `model.new_cache`). `_can_cache` now uses the cache whenever the **prompt** fits in `block_size`, then slides the window if it fills, instead of the old `context + max_new_tokens <= block_size` test that could silently fall back to full-prefix attention for a long Discord prompt. `torch.inference_mode()` is on the whole decode path.

## tok/s: before (current main) vs after (this branch), fp32, no quantization

Both sides measured with the same greedy-cached-decode harness against the same checkpoint, so this isolates the branch's KV-cache/decode-loop changes from main's `_decode_from` (`git worktree add` against `origin/main`, same prompt `"hello"`, 64 tokens, median of 3 runs after 1 warmup):

| threads | main (`b40396e`) tok/s | this branch tok/s | speedup |
|---:|---:|---:|---:|
| 2 | 56.6 | 57.8 | 1.02x |
| 4 | 59.6 | 64.7 | 1.09x |

The KV-cache/decode-loop micro-optimizations (preallocated step tensors, no `torch.cat` per token, the fixed `_can_cache` window) are a real but modest ~2-9% win on this short prompt — main's decode path was already cached correctly for prompts this size, so there was little headroom left. The bulk of the speedup this branch set out to find is in quantization, below, which is now opt-in rather than default.

## Table — tok/s and bits/char, this branch, full thread sweep

Discord val split as of this run: **109 rows, 7456 chars**. bits/char is `(total nats / chars) / ln 2`, teacher-forced, unaffected by sampler settings.

Greedy cached decode, 64 new tokens, median of 3 runs after 1 warmup, `--tokens 64`.

| config | threads | tok/s (steady) | bits/char | Δ bits/char vs fp32 |
|---|---:|---:|---:|---:|
| fp32 (**served default**) | 1 | 45.7 | **2.4193** | — |
| fp32 (**served default**) | 2 | 57.8 | 2.4193 | 0 |
| fp32 (**served default**) | **4** | **64.7** | 2.4193 | 0 |
| fp32 (**served default**) | 8 | 52.8 | 2.4193 | 0 |
| int8 (`BABBLE_QUANTIZE=1`, opt-in) | 1 | 95.7 | 2.4217 | +0.0024 |
| int8 (`BABBLE_QUANTIZE=1`, opt-in) | 2 | 92.5 | 2.4217 | +0.0024 |
| int8 (`BABBLE_QUANTIZE=1`, opt-in) | 4 | 89.9 | 2.4216 | +0.0023 |
| int8 (`BABBLE_QUANTIZE=1`, opt-in) | 8 | 81.9 | 2.4218 | +0.0025 |
| torch.compile (inductor, reduce-overhead) | — | n/a | n/a | tens of seconds to first forward — not default |

Notes:

- `origin/main`'s own fp32 bits/char, measured the same way on this checkpoint/split, is **2.4193** — identical to this branch's default. `sequence_loss`, `config.py`'s Settings loading, and `trainer.py`'s corpus/split logic are byte-for-byte unchanged by this rebase, so this is not a coincidence.
- fp32 peaks at 4 threads on this box; int8 is fastest single-threaded here and degrades with more threads (dynamic-quant kernel dispatch overhead dominates once the per-token compute is this small). Both numbers are measured, not extrapolated from the earlier pre-rebase note about an "8-thread collapse" — that earlier 7.4 tok/s figure does not reproduce post-rebase and is retracted; 8 threads is simply *slower than 4*, not catastrophic, in this run.
- `BABBLE_INFER_THREADS` defaults to **4** (serving); training stays at 2.

## Sampler: main's #24 + #26 semantics survived unchanged

Every knob, default, and env var from `cbea4f7` (#24) and `b40396e` (#26) is present, byte-identical, and untouched by this rebase:

- `repetition_penalty` (default `1.15`, env `BABBLE_REPETITION_PENALTY`) — flat per-unique-token divisor.
- `top_p` (default `0.9`, env `BABBLE_TOP_P`) — nucleus sampling.
- `frequency_penalty` (default `0.12`, env `BABBLE_FREQUENCY_PENALTY`) — `logit -= frequency_penalty * count`, the fix for the induced "boop boop boop" loop.
- `presence_penalty` (default `0.0`, env `BABBLE_PRESENCE_PENALTY`).
- `no_repeat_ngram_size` (default `0`, env `BABBLE_NO_REPEAT_NGRAM_SIZE`).
- `_apply_repetition_penalty`, `_apply_frequency_presence_penalty`, `_apply_no_repeat_ngram`, `_apply_top_p` in `generate.py` are unmodified; `git diff origin/main -- babble/generate.py` shows zero changes to any of them — only the KV-cache/decode-loop plumbing around `_next_token` changed.

`config.py` and `trainer.py` have **zero diff** against `origin/main` after this rebase.

## Five-prompt adversarial + normal generation table (post-rebase, served defaults)

`babble sample -p "<prompt>" -n 60` against the live checkpoint, default settings (`temperature=0.5`, `top_k=40`, `top_p=0.9`, `repetition_penalty=1.15`, `frequency_penalty=0.12`, fp32/`BABBLE_QUANTIZE=0`):

| prompt | completion |
|---|---|
| `boop boop boop` | ` boop boop.\n- Duster Boop: A new rope, a small and large rope, is the best way to do it.\n- Tags: The ` |
| `beep beep` | `, and the way you can find a place for your next adventure!\nThe main reason why I have been with this is that it hasn't been possible!\nThis` |
| `hello hello hello` | ` hello\nThe Oddson of the Sauce, a leading manufacturer of Ryan Aaronans, has been developing a new technology for overcoming the world.\nWith ` |
| `hello` | ` is a popular choice for those looking to create an unforgettable experience.\nThe Best For: A Unique and Component\nThe best for any fans, like the "Pig` |
| `tell me a story` | ` about the world's life and how to make a difference in your life.\nWhen I was born, I realized that my life is very different; it doesnt mean` |

`boop boop boop` breaks out after two more "boop" tokens and derails into unrelated text — no wall of repeated "boop". Same for `beep beep` and `hello hello hello`: neither degenerates into a loop of the input token.

## Completions, verbatim, fp32 vs int8, same seed (from `bench/cpu_fast.py`)

- `hola` fp32 → ` is a vibrant and vibrant lifestyle for everyone.\n- The world of online casinos has a profound impact on the online gambling industry.\n- The world of online casinos is`
- `hola` int8 → ` is a vibrant and vibrant lifestyle for everyone.\n- The world of online casinos has a growing popularity in the online gambling industry.\n- The world of online casinos is`
- `hello` fp32 → ` is a critical component of the role of the Indian international media industry. This article delves into the world of online media and why it is essential to understand `
- `hello` int8 → ` is a popular choice for those who prefer to find the perfect fit for your needs. Whether you prefer a cozy wedding, a cozy dining room, or a cozy `
- `the cat` fp32 → ` of the world is a true and powerful cat of all the human body. The cat of the world has been the only person who has been a cat of`
- `the cat` int8 → ` of the world is a true and unapologetic, and the most recent people in the world are not just the only person who is not a cat. `
- `why is` fp32 → ` it important to understand the importance of security in your daily life? In this article, we will explore the various aspects of security in your daily life and how `
- `why is` int8 → ` it important to understand the importance of security in your daily life? In this article, we will explore the various aspects of security in your daily life and how `

Same register of English throughout. Sampling diverges on most prompts because int8 logits are not bit-identical to fp32; `why is` matched exactly. This is the tradeoff `BABBLE_QUANTIZE=1` accepts — it is off by default.

## vs 400 tok/s

**Not reached, and not the point of this rebase.** 400 tok/s was the older 3.35M byte model (~10x fewer params, ~13 MB of weights). On this 34M model, fp32 tops out around 60-65 tok/s on this box (bandwidth-bound), and dynamic int8 (opt-in) roughly 1.4-1.6x that. A runtime that runs int8/int4 GEMMs at actual memory speed (llama.cpp / ggml, or a packed torchao kernel) would close most of the remaining gap; that is out of scope here.

## Not touched

Correction collection code (`core.py` capture / `>>` marker), the training path, and the served checkpoint are unchanged by this work — inference path only, per the ticket constraints. `/home/beckett/babble-live` (the live serving install) was read from (checkpoint + Discord corpus, for benchmarking) but never written to by this branch; promoting these changes to the live install is a separate, deliberate step after merge.
