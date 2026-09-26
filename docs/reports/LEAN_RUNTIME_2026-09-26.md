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

RESULTS_PLACEHOLDER
