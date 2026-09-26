# Extreme inference experiments (branch `perf/extreme`)

Goal: find how fast the live booper model can go on this box (i7-4790,
4C/8T Haswell, AVX2+FMA, no AVX-512/VNNI, dual-channel DDR3, CPU only).

Model under test: `artifacts/hf-booper-multiturn-v1` (Mixtral MoE: 6 layers,
hidden 896, 14 heads (MHA, head_dim 64), 7 experts top-1, expert FFN 1024,
vocab 16384 tied, RoPE theta 1e6, RMSNorm eps 1e-5). Weights ship as int8
per-output-channel + fp32 scale; ~150M total params, ~50M active per token.

## Correctness gate

`python bench/extreme/reference.py check module:fn` — `fn(ids) -> [T, vocab]`
logits. Reference = HF fp32 path (what serves live). Report the whole dict.
A candidate runtime is "lossless" if top1_agreement >= 0.995 and
|delta_nll_per_tok| < 0.002; anything else is a quality trade and must be
labelled as such.

## Benchmark protocol (so numbers are comparable)

Wrap every timed run in `flock /tmp/babble-bench.lock ...` (the box is shared
by parallel experiments and the live bot). 4 threads unless sweeping.
Median of >= 5 runs after 1 warmup.

1. **decode-1**: single stream, greedy, 128 new tokens after the 27-token
   benchmark prompt, EOS ignored. tok/s.
2. **best-of-4** (live shape): 4 sampled candidates, temperature 0.5, top-k 40,
   top-p 0.9, repetition penalty 1.15, no-repeat-ngram 4, 64 new tokens each,
   EOS ignored. Aggregate tok/s and wall time.
3. **TTFT**: prompt lengths 32, 128, 512 tokens -> time to first sampled
   token (prefill + 1 sample), batch 1 and batch 4.

## Baseline (HF generate, fp32, 4 threads) — `babble bench`, 3 runs

| metric | value |
| --- | --- |
| TTFT (27-token prompt, 4 candidates) | 65 ms |
| best-of-4 aggregate | ~140 tok/s (1.8 s for 4x64) |
| per-candidate | ~35 tok/s |
