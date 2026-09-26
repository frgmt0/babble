# Lean hf runtime (`BABBLE_HF_RUNTIME=lean`), 2026-09-26

Branch `perf/leanserve`. The runtime is not promoted: the default stays
`transformers`. The runtime is `babble/leanserve.py`, a production port of the
`bench/extreme/torch_lean` prototype.

Model under test: `artifacts/hf-booper-multiturn-v1`, the live model. Box:
i7-4790, 4 threads.

## What it is

- `LeanModel` reads `model-int8.safetensors` one tensor at a time. Nothing imports transformers.
  - `int8` mode: decode-sized matmuls (8 rows or fewer per call) run `aten._weight_int8pack_mm` on the on-disk int8 weights with bf16 activations. Prefill calls run every matmul on an fp32 copy, experts included.
  - `fp32` mode: the same dequantized weights as transformers.
  - The router is always fp32.
- In int8 mode, the tied head scores the vocab through the int8 kernel, then recomputes the top-k candidates exactly in fp32. Penalties, top-p, sampling and best-of scoring all see exact logits.
- `LeanGenerator` has the `HFGenerator` interface:
  - `__call__` returns `Generation`.
  - It provides `conversation_prompt`, `_encode_prompt`, `benchmark_sample` (returning `HFGenerationStats`), `benchmark_metadata` and `step`.
  - The `BABBLE_HF_FREQUENCY_PENALTIES` gate works the same way.
- Other runtime features:
  - static KV cache
  - one batch-1 prefill broadcast to the best-of rows
  - routed-only MoE
  - row compaction when a candidate emits `<eos>`
- `PrefixKVCache` is a content-addressed LRU of prompt K/V. It is bounded by entry count and bytes, and runs under the generator's lock.
- `make_generator` chooses the runtime from `BABBLE_HF_RUNTIME`. `babble sample` now goes through `make_generator` too.

## Correctness gate (`bench/extreme/reference.py check`)

A runtime passes the gate as "lossless" if top1 >= 0.995 and |dNLL| < 0.002.
There are 985 positions across 5 synthetic cases.

| target | max abs logit diff | top1 | dNLL/tok |
|---|---|---|---|
| `leanserve_check:int8_full` (prefill path) | 0.125 | 1.0 | -0.00038 |
| `leanserve_check:int8_decode` (token by token through int8) | 2.93 | 0.99594 | +0.00066 |
| `leanserve_check:fp32_full` | 0.0078 | 1.0 | -0.000074 |
| `leanserve_check:fp32_decode` | 0.0078 | 1.0 | -0.000074 |

The raw dicts:

- int8_full: `{'max_abs_logit_diff': 0.125, 'top1_agreement': 1.0, 'ref_nll_per_tok': 2.2098386, 'cand_nll_per_tok': 2.2094561, 'delta_nll_per_tok': -0.000382, 'positions': 985}`
- int8_decode: `{'max_abs_logit_diff': 2.9297, 'top1_agreement': 0.995939, 'ref_nll_per_tok': 2.2098386, 'cand_nll_per_tok': 2.2104970, 'delta_nll_per_tok': 0.000658, 'positions': 985}`
- fp32_full: `{'max_abs_logit_diff': 0.0078, 'top1_agreement': 1.0, 'cand_nll_per_tok': 2.2097646, 'delta_nll_per_tok': -7.4e-05, 'positions': 985}`
- fp32_decode: same as fp32_full to within 3e-10.

In `int8_full`, the whole vocab is scored with bf16 activations and the top-64 are rescored exactly, which is where the 0.125 comes from. The int8 decode path passes, but the margin is thin (0.9959 against a bar of 0.995), as in the prototype.

## `babble bench --json`

Protocol: 3 runs each, interleaved, under `flock /tmp/babble-bench.lock`. The env was the serving env from the brief (multiturn-v1, `NO_REPEAT_NGRAM_SIZE=4`, `CONVERSATION_CONTEXT=1`). Each run used a 27-token prompt, 4 candidates and 64 tokens per candidate; every candidate ran the full 64 tokens.

| runtime | TTFT ms | steady agg tok/s | e2e agg tok/s | wall ms | RSS / peak MB |
|---|---|---|---|---|---|
| transformers | 78 / 61 / 63 | 140.8 / 149.6 / 148.0 | 137.1 / 146.6 / 145.0 | 1868 / 1746 / 1766 | 967 / 1885 |
| lean int8 | 34 / 36 / 36 | 410.2 / 441.8 / 438.9 | 394.5 / 422.4 / 419.7 | 649 / 606 / 610 | 1075 / 1188 |
| lean fp32 (1 run) | 36 | 203.3 | 200.6 | 1276 | 843 / 1142 |

Median speedups over transformers:

- lean int8: 2.9x end-to-end, 1.8x TTFT
- lean fp32 (lossless): 1.4x end-to-end

## RSS (`leanserve_demo.py rss`: load, then one reply)

The Python + torch + babble baseline is 222 MB in all cases.

| config | after load MB | after reply MB | peak MB | load s |
|---|---|---|---|---|
| transformers | 924 | 938 | 1818 | 3.4 |
| lean int8 (default, fp32 prefill copy) | 1048 | 1065 | 1189 | 0.24 |
| lean int8, `BABBLE_LEAN_PREFILL_FP32=0` | 495 | 544 | 567 | 0.09 |
| lean fp32 | 816 | 849 | 1143 | 0.24 |

- Steady-state memory: lean int8 costs about 130 MB more than transformers.
- Peak memory: lean's load peak is about 650 MB lower, because tensors are streamed in rather than the whole state dict being built at once.
- Without the prefill copy, the process is about 400 MB smaller than transformers. The cost is TTFT: 92 ms against 36 ms on the bench prompt.

## Multi-turn TTFT (`leanserve_demo.py ttft`)

Setup:

- One conversation growing over 7 turns, best-of-4, measuring TTFT only.
- "Cold" means no prefix reuse. "Warm" means the cache was primed by the previous turn only.
- Each value is the median of 3.

The two runtimes' conversations differ slightly in length because their sampled replies differ.

| turn | lean prompt tok | transformers cold | lean cold | lean warm | reused tok |
|---|---|---|---|---|---|
| 1 | 44 | 75 ms | 34 ms | 10 ms¹ | 43 |
| 2 | 101 | 145 ms | 51 ms | 39 ms | 43 |
| 3 | 152 | 192 ms | 63 ms | 38 ms | 100 |
| 4 | 214 | 279 ms | 82 ms | 46 ms | 151 |
| 5 | 276 | 354 ms | 111 ms | 47 ms | 213 |
| 6 | 340 | 428 ms | 124 ms | 50 ms | 275 |
| 7 | 403 | 521 ms | 146 ms | 55 ms | 339 |

¹ Turn 1 re-sends an identical prompt, so only `<sep>` is recomputed. Real traffic doesn't do this.

At turn 7, TTFT is 9.5x lower than transformers and 2.7x lower than lean without the cache.

In every turn, the cache reused exactly the previous prompt minus its trailing `<sep>`. This confirms that the role transcript tokenizes stably at turn boundaries. The newest reply and user message are always recomputed. They follow `\nassistant: `, where the generated tokens followed `<sep>`, so their K/V differs.

## Sampled replies

Settings: seed `1000+i`, best-of-4, live sampling settings, one-turn transcript.

| prompt | lean int8 | transformers |
|---|---|---|
| hey booper whats up | He's been here since the first time | I was playing with my friends in the park |
| do you like cats or dogs | I like cats, but not all of them | I like dogs |
| what should i eat for dinner | Eat some of your favorite food | Eat the whole meal |
| lol that game last night was wild | It was a lot of fun | I just got back from the game |
| i cant sleep | I can't sleep | I think you should sleep |
| whats your favorite song rn | Idk but i like it | I love the lyrics |
| booper say something funny | I like to be funny | I don't know how to spell |
| im so bored in class | I don't even know how to do it | I'm not even bored |
| did you finish your homework | no, i finished my homework | No I finished my homework |
| good morning!! | i am good but i cant sleep | Good morning |

The two runtimes produce the same register, length and coherence. The replies differ word for word because sampling consumes random numbers differently (see below). Per-reply latency was 106–169 ms for lean against 189–789 ms for transformers.

## Semantic differences from the transformers path

- **Same seed, different text.** Lean draws `multinomial` over the top-k candidates, whereas transformers draws over the full vocab. The distribution is identical, but the random stream is not.
- **Top-k ties.** An exact floating-point tie at the k-th value keeps exactly k tokens. Transformers keeps every tied token.
- **int8 only: top-k membership.** Which tokens make the top-k cut is decided on bf16-accurate logits after penalties. A token within bf16 error of the k-th logit could be swapped. Every candidate's logit is then exact fp32.
- **int8 only: decode hidden states.** These carry bf16 activation rounding. This is the int8_decode gate above: top1 0.9959, dNLL +0.00066.
- **Row compaction.** Finished candidates stop decoding. The outputs are unaffected, and so are best-of scores (pads were already excluded).
- **Benchmark metadata.** `babble bench` reports backend `hf-lean`, dtype `int8/bf16` and no transformers version.

## Tests

`tests/test_leanserve.py` has 25 tests and needs no network. They cover:

- sampler masks, probabilities and log-probabilities against transformers' processors plus `_CandidateTracker` on random logits, across 5 configs including the live one;
- n-gram bans for n = 1 to 4;
- fp32 full, decode and prefix-resume parity against MixtralForCausalLM on a tiny random Mixtral (atol 1e-4);
- int8 tolerance on the same tiny model;
- exact candidate rescoring;
- KV compaction;
- a toy-model check that rows never mix histories under compaction;
- prefix-cache hit, miss, supersede, LRU and byte eviction;
- seed determinism in int8 and fp32;
- a cross-turn cache hit through the generator;
- runtime selection and env parsing.

Full suite: 797 passed, 1 skipped.

## Promotion notes

The parent session handles promotion. Set `BABBLE_HF_RUNTIME=lean` in the env file, restart, and check that `model.load` logs `runtime=lean`. To roll back, remove the line.

The conservative choice is `BABBLE_LEAN_PRECISION=fp32`: it is lossless and 1.4x faster than transformers.

The native C++ engine (`perf/native-engine`) measured faster and lossless. If it lands, this runtime becomes the pure-PyTorch fallback.
