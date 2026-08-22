# CPU inference on the 34.1M BPE model (2026-08-22)

Measured on this box: Intel Core i7-4790 @ 3.60 GHz, 4 cores / 8 threads, 32 GB, **CPU only**. torch 2.13.0+cpu. Served checkpoint: `~/babble-live/checkpoints/latest.pt` (34,096,128 params, 8×512, vocab 16384, block 1024).

Reproduce:

```sh
BABBLE_DATA_DIR=/home/beckett/babble-live/data \
BABBLE_CHECKPOINT_DIR=/home/beckett/babble-live/checkpoints \
  python bench/cpu_fast.py --tokens 64 --runs 4 --threads 2
```

`bench/bench_inference.py` is still the longer TTFT / RAM / thread-sweep harness (now BPE-aware). `bench/cpu_fast.py` is the tok/s + bits/char ablation anyone can re-run in ~30s.

## Bottleneck (measured, not assumed)

fp32 cached greedy decode: **53 tok/s** at 2 threads (~19 ms/token).

Weights are **136 MB** of fp32. 53 tok/s × 136 MB ≈ **7.2 GB/s** of weight traffic, which is what this dual-channel DDR3 box actually delivers. Single-stream decode is **memory-bandwidth bound**. Python in the decode loop (tensor reuse, `.item()`, tokenizer) is noise next to 19 ms of Linear traffic.

KV cache **is** used on the serving path (`_decode_from` / `_decode_many_from` via `model.new_cache`). The old `_can_cache` test required `context + max_new_tokens <= block_size`; a long `max_new_tokens` could have forced the full-prefix fallback. Live Discord prompts plus 256 new tokens still fit in block_size 1024, so the live bot was already on the cached path. The helper now caches whenever the **prompt** fits and stops (then slides) if the window fills.

`torch.inference_mode()` was already on the whole decode path.

## Table — tok/s and bits/char

Discord val split as of this run: **98 rows, 7232 chars**. bits/char is `(total nats / chars) / ln 2`, same definition as `experiments/ssh_posttrain_measure.py`.

The promoted-checkpoint figure of **2.11 bits/char** was on an earlier snapshot of that split. On today's 98-row split the same fp32 weights score **2.41 bits/char**. Optimizations are judged against **this** measurement, not the historical 2.11.

Greedy cached decode, 64 new tokens, median of 4 runs after 1 warmup. Completions below used temp 0.5, top_k 40, 60 tokens, `Generator.manual_seed(0)`.

| config | threads | tok/s (steady) | bits/char | Δ bits/char vs fp32 |
|---|---:|---:|---:|---:|
| fp32 + KV cache (baseline) | 2 | **53.0** | **2.4096** | — |
| + decode tensor reuse | 2 | ~53 (inside noise) | 2.4096 | 0 (numerics untouched) |
| + dynamic int8 Linears | 2 | **110.9** | **2.4125** | **+0.0029** |
| int8 + KV cache | 1 | 122 | 2.4125 | +0.0029 |
| int8 + KV cache | 2 | 123 | 2.4125 | +0.0029 |
| int8 + KV cache | **4** | **143** | 2.4125 | +0.0029 |
| int8 + KV cache | 8 | 7.4 | 2.4125 | +0.0029 (oversubscription) |
| torch.compile (inductor, reduce-overhead) | 2 | n/a | n/a | **34.7 s** to first forward — not default |

Dynamic int8 costs **0.003 bits/char**, well under the 0.05 budget, so it is **on by default** (`BABBLE_QUANTIZE=0` turns it off). Serving threads default to **4** (`BABBLE_INFER_THREADS`); training stays at 2.

## Completions, verbatim, same seed

**fp32**

- `hola` → ` is a vibrant and vibrant lifestyle for everyone.\n- The world of online casinos has a profound impact on the online gambling industry.\n- The world of online casinos is`
- `hello` → ` is a critical component of the role of the Indian international media industry. This article delves into the world of online media and why it is essential to understand `
- `the cat` → ` of the world is a true and powerful cat of all the human body. The cat of the world has been the only person who has been a cat of`
- `why is` → ` it important to understand the importance of security in your daily life? In this article, we will explore the various aspects of security in your daily life and how `

**int8** (same prompts, same seed)

- `hola` → ` is a vibrant and vibrant lifestyle for everyone.\n- The world of online casinos has a growing popularity in the online gambling industry.\n- The world of online casinos is`
- `hello` → ` is a popular choice for those who prefer to find the perfect fit for your needs. Whether you prefer a cozy wedding, a cozy dining room, or a cozy `
- `the cat` → ` of the world is a true and unapologetic, and the most recent people in the world are not just the only person who is not a cat. `
- `why is` → ` it important to understand the importance of security in your daily life? In this article, we will explore the various aspects of security in your daily life and how `

Same register of Ultra-FineWeb-ish English. Sampling diverges on some prompts because int8 logits are not bit-identical; `why is` matched fp32 exactly. Loss delta is 0.003 bits/char.

## vs 400 tok/s

**Not reached.** 400 tok/s was the 3.35M byte model (~10× fewer params, ~13 MB of weights). On this 34M model:

- fp32 ceiling on this box ≈ **50–60 tok/s** (bandwidth).
- int8 pytorch dynamic quant ceiling ≈ **140 tok/s** at 4 threads.
- 8 threads is a **regression** (7 tok/s).

What 400 tok/s would take at 34M params on this CPU: a runtime that actually runs int8/int4 GEMMs at memory speed (llama.cpp / ggml, or torchao with a packed kernel that is not the current `quantize_dynamic` path), **or** a much smaller model, **or** a machine with several times this DRAM bandwidth. Do not claim 400; the benchmark says ~140.

## Live

`babble-bot` restarted on `/home/beckett/babble-live` after these serving changes (`systemctl --user restart babble-bot`, `bot.ready` at step 3118). Default path: `load_model` → `prepare_for_cpu_infer` (dynamic int8) at `infer_threads=4`.

This branch is rebased onto `cbea4f7` (repetition penalty / top-p sampler). Settings defaults are the landed ones (`top_p=0.9`, `repetition_penalty=1.15`); `_next_token` uses that PR's `_apply_top_p` / `_apply_repetition_penalty` helpers rather than a parallel reimplementation. `bot.generate` still logs both fields.

Batched decode (`_decode_many_from`, the live `best_of=4` path) continues on the sliding-window forward after the KV cache fills, matching the single-stream overflow fallback. Truncating there had been a silent short-reply bug.

Real completion from the live install (`babble sample -p hello --tokens 40` in `/home/beckett/babble-live` after restart):

`'hello' -> ' and the Giant is a very good one of the best things to do with this kind.\nThe '`

Correction collection code (`core.py` capture / `>>` marker) was not changed. After the restart, `capture.corpus` still fires (ambient rows 517–518) and `bot.ready` is step 3118 / 34,096,128 params. `tests/test_core.py` + `tests/test_collection_integration.py` still pass.

A restart at 01:27 let the in-process trainer write a 34-step **byte** `latest.pt` over the promoted 34.1M weights (`tokenizer.json` stayed BPE, so pings 500'd). Restored `artifacts/hf-booper-pretrain/latest.pt` (`sha256 f207a1d8…`, same file as the promotion).

Note: copying serving modules into `babble-live` leaves that tree dirty, so `update-live.sh` will refuse to ff-merge until Beckett publishes this repo and the live tree is reset to origin. Inference is live regardless.
