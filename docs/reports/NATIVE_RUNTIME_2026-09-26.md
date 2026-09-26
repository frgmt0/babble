# Native hf runtime (`BABBLE_HF_RUNTIME=native`), 2026-09-26

Branch `perf/nativeserve`, on top of `perf/leanserve`. The runtime is not
promoted: the default stays `transformers`. It is the production port of the
`perf/native-engine` prototype (`bench/extreme/native/`, 3da7b8d).

Model under test: `artifacts/hf-booper-multiturn-v1`, the live model
(`max_position_embeddings` 4096). Box: i7-4790 (AVX2/FMA/F16C), gcc 16.2,
4 engine threads.

## What it is

- `babble/_native/engine.cpp` is the C++ engine, with a plain C ABI.
  - Weights: the on-disk int8 weights are repacked at load into 16-row panels and dequantized in registers. Activations and accumulation are fp32 (W8A32).
  - Fusion: QKV is one GEMM, SwiGLU is fused into w1/w3, and the tied lm_head reuses the embedding panel.
  - MoE: top-1 routing, with rows grouped by expert so each expert's weights are read once per step.
  - Decode: best-of shares one prompt prefill, and the prompt KV is scored for all streams at once.
  - Threads: a spin/futex thread pool runs each reply as a single parallel region.
- Changes from the prototype:
  - **Geometry is read at runtime from `config.json`, not compiled in.** Any top-1 Mixtral with dimensions that are multiples of 16 runs on the same build. The tests use a tiny random Mixtral; the live model and the 4096-context SFT need no rebuild. Speed is unchanged: 387-395 tok/s decode-1, ~1020 tok/s best-of-4, 155 ms T512 TTFT, against the prototype's 380-390 / 1030 / 153.
  - **Prefill at an offset, plus prefix KV import/export.** After a reply, the prompt's K/V is exported as a snapshot. A later prompt that shares a prefix imports the snapshot and prefills only the suffix.
  - **Sampler.** Temperature is now applied before top-k, as in HF. Frequency/presence penalties are implemented. Top-k above 256 uses `nth_element`; the prototype had a fixed 256-entry buffer. When every token is banned, the engine emits `<eos>` instead of token 0.
  - **Robustness.** Argument and token-id validation, a destructor that frees everything, an ABI version check, and a `eng_warp_probs` sampler probe for tests.
  - Bench-only probes (membw, profiling, skip-lm-head) are removed. Prefetch distance and row block are now constants, at the prototype's tuned values.
- `babble/_native/__init__.py` builds the engine on first use.
  - It runs `g++ -O3 -march=haswell`. The cache is `$BABBLE_NATIVE_CACHE`, or by default `~/.cache/babble/native/`, outside the repo and outside /tmp.
  - The cache key covers the source hash, the compiler identity (`--version`), the flags, the CPU features and the ABI.
  - Builds take an exclusive `flock` and publish with an atomic rename. A cached library that fails to load is rebuilt once.
  - Before building, it checks that the CPU is x86-64 Linux with AVX2, FMA and F16C.
- `babble/nativeserve.py`: `NativeEngine` is the ctypes wrapper, with torch tensors as buffers and no numpy. `NativeGenerator` subclasses `LeanGenerator` and inherits from it:
  - the tokenizer and special ids
  - `_encode_prompt` and the prompt budget
  - `conversation_prompt`
  - `_sampling` (Settings to `SamplingConfig`, including the frequency-penalty gate)
  - `__call__` and `benchmark_sample` (which bypasses the prefix cache)

  It overrides `__init__`, `_generate` and `benchmark_metadata`. The prefix cache is lean's `PrefixKVCache`, holding engine snapshots.
- `hfserve.make_generator` selects the runtime: `native` -> `NativeGenerator`. On `NativeUnavailable` it logs `model.native_fallback` with the reason, prints the reason to stderr, and returns `LeanGenerator`.
- `pyproject.toml` ships `_native/*.cpp` as package data. `babble sample` prints the runtime that actually loaded, so a fallback is visible.

## Gates: `bench/extreme/reference.py check`, through `NativeGenerator`

The entry points in `bench/extreme/nativeserve_check.py` build the generator
with `make_generator` (`hf_runtime="native"`) and call its engine. The data is
985 positions across 5 synthetic cases, and the bar is top1 >= 0.995 with
|dNLL| < 0.002.

| target | max abs logit diff | top1 | dNLL/tok |
|---|---|---|---|
| `nativeserve_check:full` (one batched prefill) | 0.0078 | 1.0 | -7.43e-05 |
| `nativeserve_check:decode` (1-token prefill, then token by token) | 0.0078 | 1.0 | -7.41e-05 |
| `nativeserve_check:prefix` (half prefilled and exported, rest prefilled on the restored snapshot) | 0.0078 | 1.0 | -7.41e-05 |

- full: `{'max_abs_logit_diff': 0.007793426513671875, 'top1_agreement': 1.0, 'ref_nll_per_tok': 2.2098386212714813, 'cand_nll_per_tok': 2.20976433663998, 'delta_nll_per_tok': -7.428463150120381e-05, 'ref_ppl': 9.114245429013605, 'cand_ppl': 9.113568405797022, 'positions': 985}`
- decode: `{'max_abs_logit_diff': 0.0077991485595703125, 'top1_agreement': 1.0, 'ref_nll_per_tok': 2.2098386212714813, 'cand_nll_per_tok': 2.209764528574434, 'delta_nll_per_tok': -7.409269704758746e-05, 'ref_ppl': 9.114245429013605, 'cand_ppl': 9.113570155004961, 'positions': 985}`
- prefix: `{'max_abs_logit_diff': 0.007793426513671875, 'top1_agreement': 1.0, 'ref_nll_per_tok': 2.2098386212714813, 'cand_nll_per_tok': 2.2097645345723853, 'delta_nll_per_tok': -7.408669909591195e-05, 'ref_ppl': 9.114245429013605, 'cand_ppl': 9.113570209667714, 'positions': 985}`

The 0.0078 is the fp16 rounding step of the stored reference logits. The fp32
logits agree to about 5e-5, as the long-context check below shows. These
numbers are identical to the prototype's.

**Long context** (2048 tokens against transformers' fp32 forward, run live
rather than against the fixture):

- full prefill: max |dlogit| 4.7e-5, top1 1.0
- 1500-token prefill + 548 decode steps: 4.0e-5, top1 1.0

A 1536-token prompt with best-of-4 and 512 new tokens runs with no rebuild:
TTFT 596 ms, 408 aggregate tok/s. At that length the fp32 attention over the
KV dominates decode.

## `babble bench --json`

Protocol:

- 3 runs of each runtime in separate processes, interleaved native / lean / transformers, under `flock /tmp/babble-bench.lock`.
- Env: `BABBLE_DATA_DIR=/tmp/bbl-empty/data BABBLE_CHECKPOINT_DIR=/tmp/bbl-empty/ck BABBLE_SERVE_BACKEND=hf BABBLE_HF_MODEL_DIR=$PWD/artifacts/hf-booper-multiturn-v1 BABBLE_NO_REPEAT_NGRAM_SIZE=4 BABBLE_CONVERSATION_CONTEXT=1`.
- Workload: a 27-token prompt, 4 candidates, 64 tokens each. Every candidate ran the full 64 in every run.

| runtime | TTFT ms | steady agg tok/s | e2e agg tok/s | selected tok/s | wall ms | RSS / peak MB |
|---|---|---|---|---|---|---|
| **native** | 12.9 / 12.7 / 12.2 | 1034 / 1053 / 1061 | 996 / 1014 / 1024 | 249 / 254 / 256 | 257 / 252 / 250 | 401 / 531 |
| lean (int8) | 35.0 / 36.8 / 36.6 | 419 / 444 / 449 | 402 / 423 / 428 | 101 / 106 / 107 | 637 / 605 / 598 | 1075 / 1188 |
| transformers | 61.6 / 60.8 / 62.6 | 149 / 150 / 149 | 146 / 147 / 146 | 36.5 / 36.7 / 36.4 | 1751 / 1743 / 1760 | 995 / 1818 |

Median speedups:

| | end-to-end | TTFT |
|---|---|---|
| native over transformers | 6.9x | 4.9x |
| native over lean | 2.4x | 2.9x |

CPU use was 3.9 cores for all three runtimes.

## RSS (load, then one reply; `.scratch` harness, not committed)

| config | after load MB | after reply MB | peak MB | load s |
|---|---|---|---|---|
| native | 389 | 433 | 532 | 0.19 |
| lean int8 (from the lean report) | 1048 | 1065 | 1189 | 0.24 |
| transformers (from the lean report) | 924 | 938 | 1818 | 3.4 |

- Of the native total, 221 MB is the Python + torch + babble baseline. The engine's weights are about 150 MB of int8 panels plus scales.
- The peak is the mmapped safetensors file while tensors stream in; it is released after load.
- KV buffers grow with the request. At a 3584-token prompt plus 512 new × 4 streams, the process was 710 MB.
- A prefix snapshot costs 43 KB per prompt token (2 × 6 layers × 896 × 4 B), so the default 128 MB cache holds about 3000 tokens of snapshots.

## First-start build time

- A direct `_native.build()` into an empty cache took 4.4 s. The unit test run from an empty cache took 13 s in total, including the compile.
- Loading from a warm cache takes 0.19 s.
- Only the first start after an engine change compiles. `model.load` logs `native_build=compiled|cached` and `native_build_s`. `deploy/update-live.sh` waits 90 s for `bot.ready`, so the compile fits comfortably.

## Multi-turn TTFT: prefix KV reuse

Setup: a 495-token transcript on turn k+1, with the cache primed by turn k's
472-token prompt. Best-of-4, median of 9, measured through
`NativeGenerator._generate`.

| runtime | cold (no reuse) | warm (reuse) | reused / suffix tokens |
|---|---|---|---|
| native | 148.8 ms | 12.9 ms (11.5x) | 471 / 24 |
| lean | 174.3 ms | 42.3 ms | 471 / 24 |

Hits need a growing transcript. Once the history window is full
(`conversation_max_turns`, or the token budget), each turn drops its oldest
turn from the front, the prefix differs, and the cache misses. In the
steady-state sliding case, a 206-token prompt with `max_turns=6` reused 4
tokens (`<bos>user: `) and took 59 ms. This applies to lean in the same way.
If the win matters in steady state, drop old turns in larger chunks rather
than one at a time. That is a conversation-format change, so it is not made
here.

## Fallback behaviour

Every path below is covered by a test, and the missing-compiler path was also
run through the CLI.

| cause | result |
|---|---|
| CPU without AVX2/FMA/F16C, or not x86-64 Linux | `NativeUnavailable` → `model.native_fallback{reason, fallback: "lean"}` + a stderr line → `LeanGenerator` |
| no compiler (`CXX` not on PATH) | same fallback. Observed: `babble: native runtime unavailable (no C++ compiler ('no-such-compiler' not on PATH) to build the native engine); falling back to lean`, then `# hf backend (lean)` and a reply |
| compile error | same fallback, with the last 8 lines of compiler output in the reason. Nothing is published to the cache. |
| cached `.so` that does not load | rebuilt once, then the same fallback if it still fails |
| unsupported snapshot: GQA, top-k>1 routing, dims not multiples of 16, untied head, bias, sliding window, rope scaling, non-int8 matrices | same fallback. Lean then applies its own checks, so a snapshot neither runtime supports is a hard error, as before. |
| missing model dir, weights or tokenizer | hard `HFServeError`, not swallowed (as with every runtime) |

## Thread safety and threads

- The bot runs every generation, including `/bench`, under its `asyncio.Lock` via `asyncio.to_thread`. `NativeGenerator` holds its own `threading.Lock` around the engine and the prefix cache as well.
- ctypes releases the GIL for the whole call, so the event loop keeps running during a reply.
- The engine pool has `BABBLE_INFER_THREADS` threads, with the calling thread as thread 0. Workers spin briefly after a reply, then futex-sleep, so an idle bot uses no CPU.
- Torch is configured with the same thread count (`configure_cpu`), but no torch op runs in parallel during a native reply, so the two pools never compete.
- From the prototype's sweep: 2-3 threads already saturate memory bandwidth, and 8 threads (hyperthreads) or pinning are slower.

## Sampled replies (native vs transformers)

Settings: live sampling (T 0.5, top-k 40, top-p 0.9, repetition 1.15, no-repeat-4, best-of-4, max 256). Conversation prompts, seed `1000+i`.

| prompt | native | ms | transformers | ms |
|---|---|---|---|---|
| hey booper whats up | I gotta go back to my home and ask for a new one | 88 | I was playing with my friends in the park | 473 |
| do you like cats or dogs | I don't like dogs | 48 | I like dogs | 170 |
| can you tell me a short story about a robot | The robot was a small robot, with a big head and a long tail. It had been around for 3 years and it was the most powerful robot in the whole world. | 970 | Once upon a time there was a small robot. He was always looking for something to do. One day, he saw a very big robot. (3 more sentences) | 7736 |
| (whats your favorite game / minecraft probably) why minecraft | idk i just play it | 47 | i like to play with my friends | 445 |
| what should i eat for dinner tonight | Eat your dinner | 33 | Eat the whole meal | 334 |
| (i had a rough day / oh no what happened) my boss yelled at me for nothing | I was just joking | 47 | I don't know how to explain | 354 |
| recommend me a movie | I was talking about the movie | 41 | I like the first one | 316 |
| whats the best pizza topping | Pizza is a great topping for your family | 50 | Idk but i like to eat pizza | 766 |
| (do you sleep / only when nobody is pinging me) lol fair. what do you dream about | I don't think I'm going to sleep | 56 | I dreamed about being a girl | 651 |
| tell me a joke | Sure! I can help | 47 | I don't know how to spell | 7207 |

- The replies have the same register, length distribution and coherence.
- The text differs because the random streams differ (see below).
- The long outliers on both runtimes are best-of-4 waiting for its longest candidate, up to 256 tokens.

## Simulated live start

Env: the bench env above plus `BABBLE_HF_RUNTIME=native` and an empty
`BABBLE_NATIVE_CACHE`, never the live dirs. Command: `python -m babble sample
--prompt "hey booper, whats up" --count 3`, which goes through
`make_generator`.

- It compiled, loaded, and printed `# hf backend (native)` with 3 replies in 58-82 ms.
- A second start with a warm cache loaded the cached library.
- With `CXX=no-such-compiler` and an empty cache, it fell back to lean and replied.

## Semantic differences

- **Same seed, different text.** The engine draws from its own splitmix64 stream, seeded per call from torch's global generator. `torch.manual_seed` (and `/bench`'s fixed seed) still makes runs reproducible, but the tokens differ from lean's and transformers' for the same seed. The distribution is the same.
- **Top-k ties.** Exact ties at the k-th value are all kept, as in transformers. Lean keeps exactly k.
- **Top-p boundary.** The cumulative sum is in float32 over the survivors only. It matches transformers' probabilities to 2e-6 in the tests. A token sitting exactly on the top-p boundary could in principle land differently.
- **All tokens banned** (only possible with extreme penalties): the engine emits `<eos>`, where transformers' multinomial would raise.
- **Arithmetic.** Arithmetic is fp32 with a different summation order than torch. Logits agree to about 5e-5, and greedy decoding matches transformers token for token over 128 tokens (see Tests).
- **Row compaction and early stop.** These work as in lean: a candidate that emits `<eos>` leaves the batch, and the reply ends when every candidate has.
- **Benchmark metadata.** Backend `hf-native`, dtype `int8/fp32`, no transformers version.

## Tests

`tests/test_nativeserve.py` has 42 tests, needs no network, and runs in 9 s (13 s from an empty build cache). They cover:

- **Parity with MixtralForCausalLM** on a tiny random Mixtral (hidden 64, head_dim 16, 3 experts): full prefill; decode with prefill 1, 9 and T; atol 1e-4.
- **Prefix restore == full prefill**, max |d| <= 1e-5. This is checked at cuts 1, 7, 16, 23 and 40, for partial use of a longer snapshot, and with a negative control: a wrong snapshot does differ.
- **Consistency** (ported from `consistency.py`):
  - batch-4 greedy == batch-1 greedy
  - greedy == transformers greedy over 48 tokens (tiny model) and **128 tokens (real model)**
  - real-model greedy after prefix restore == cold greedy
  - sampled best-of invariants: no repeated n-gram, `<eos>` only last, early stop happens, best index and timing order
  - seed determinism
- **Sampler equivalence.** `eng_warp_probs` is compared with transformers' RepetitionPenalty and NoRepeatNGram processors plus `_CandidateTracker`. There are 6 configs, including the live one, frequency/presence and top-k 300, each run at V = 200 and V = 16384. The support must match exactly and the probabilities within 2e-6. A separate check draws 2560 times and requires total variation below 0.04 against the warped distribution.
- **Generator seam:**
  - `make_generator` selects native and logs `model.load`
  - prompt encoding and conversation formatting are identical to lean
  - the `Generation` contract holds, and `run_benchmark` reports `hf-native`
  - seed determinism
  - cross-turn cache hit, with the same tokens as a cache-bypassed run on the same seed
  - `/bench` never touches the cache
  - the frequency-penalty gate
- **Fallback:** build failure (monkeypatched source), no AVX2, no compiler, unsupported shape, GQA and top-2 refusals. Missing dirs still raise.
- **Build cache:** cached, keyed (an edited source rebuilds), atomic (no `.tmp` left behind), a truncated library is rebuilt, and the default dir is outside the repo and /tmp.

Full suite: 839 passed, 1 skipped.

## Promotion notes (for the parent session)

- Set `BABBLE_HF_RUNTIME=native` in the live env file and restart. The first start compiles for about 5 s.
- Check that `model.load` logs `runtime=native` and `native_build=compiled`, and that there is no `model.native_fallback`.
- To roll back, set `BABBLE_HF_RUNTIME=lean` or `transformers`.
- I checked, read-only, that the `babble-bot` user unit sets no Environment, and that the user manager's PATH includes `/usr/bin` (where `g++` lives).
- The cache key is content-based, and `~/.cache/babble/native` already holds a build of this exact source with this compiler and flags. So the live start may load that library rather than compile. `native_build=cached` is expected in that case.
- If the compiler or PATH ever changes, the fallback keeps the bot up on lean, and the event says why.
