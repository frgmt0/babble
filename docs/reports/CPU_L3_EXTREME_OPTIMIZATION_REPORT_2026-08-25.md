# Extreme CPU inference optimisation: making use of L3 cache (2026-08-25)

## Executive conclusion

The current box is an Intel Core i7-4790 with four physical cores, eight
logical CPUs, and only **8 MiB of shared L3 cache**. The served BPE checkpoint
has **34,096,128 parameters** and is approximately **136 MiB in fp32**. It
cannot fit in L3, and neither can the complete int8 runtime representation.

The best optimisation measured so far is dynamic int8 quantisation:

- fp32 peak: **~102 tokens/sec** at three threads;
- int8 peak: **~147 tokens/sec** at four threads;
- eight threads are slower because of cache, memory-bandwidth, and SMT
  contention.

The next serious experiments are weight-only int4, a smaller/quantised KV
cache, and micro-batching. None should become the default without a quality
(bits/character and sample) comparison.

## Hardware and model

| item | value |
|---|---:|
| CPU | Intel Core i7-4790 @ 3.60 GHz |
| physical/logical CPUs | 4 / 8 |
| L1 | 32 KiB instruction + 32 KiB data per core |
| L2 | 256 KiB unified per core |
| L3 | **8 MiB unified, shared by CPUs 0–7** |
| model | 8 layers, 8 heads, 512 embedding width |
| vocabulary | 16,384 BPE tokens |
| context window | 1,024 tokens |
| parameters | 34,096,128 |
| fp32 parameter payload | approximately 136 MiB |

The L3 and CPU topology were read from `lscpu` and
`/sys/devices/system/cpu/cpu0/cache/index*`.

## Measurement

The benchmark used the current served checkpoint read-only. It did not start,
stop, or restart `babble-bot`, and it pointed `BABBLE_DATA_DIR` at a temporary
empty directory rather than the live corpus. Empty validation data means the
benchmark reports no quality score; this test is only about decode speed.

Command shape:

```sh
BABBLE_DATA_DIR="$TMP_DATA" BABBLE_CHECKPOINT_DIR="$TMP_DATA/checkpoints" \
  .venv/bin/python bench/cpu_fast.py \
    --checkpoint /path/to/latest.pt \
    --threads N --tokens 128 --runs 15 --warmup 3
```

The measurement is cached greedy decoding, 128 generated tokens, median over
15 runs after three warmups. It measures both the serving-default fp32 model
and the existing opt-in dynamic int8 path.

### Thread sweep

| threads | fp32 steady tok/s | int8 steady tok/s |
|---:|---:|---:|
| 1 | 80.1 | 135.8 |
| 2 | 99.1 | 145.2 |
| 3 | **102.2** | 145.4 |
| 4 | 100.1 | **147.1** |
| 8 | 80.3 | 107.0 |

A separate cache comparison at three threads measured approximately **99.5
cached tok/s versus 34.4 uncached tok/s**, a **2.9x cache speedup**. This is
why the KV-cache path must remain enabled independently of any weight
quantisation work.

## What L3 can and cannot do for this model

### The whole model cannot fit

The fp32 parameter payload is about 17 times larger than the 8 MiB L3. Dynamic
int8 does not make the entire runtime fit either:

- the transformer linear weights are quantised;
- embeddings remain fp32;
- the tied output embedding is cloned and quantised as a decode-only output
  head;
- the resulting runtime representation is still tens of MiB, before packed
  metadata, activations, and KV state.

`mlock`, `mmap`, huge pages, or manually retaining Python tensors do not force
these bytes into L3. They only change paging, virtual memory, or TLB
behaviour. L3 placement is hardware-managed.

### A single int8 layer is cache-sized, approximately

For this 8x512 architecture, one transformer block has roughly 3.15 million
linear weight values. At int8 that is about 3.15 MiB before packing overhead.
That is small enough for one active block plus activations to make useful use
of L3. However, autoregressive decoding revisits all eight blocks for every
new token. By the time the next token reaches a block, other blocks have
streamed through the shared cache and may have evicted it.

This explains the measured int8 win without implying that the model is
resident in cache: int8 reduces the amount of RAM traffic and lets each GEMM
use cache blocking more effectively; it does not eliminate RAM traffic.

### The KV cache can consume the entire L3

With batch 1, eight layers, eight heads, head dimension 64, and fp32 K/V:

- 256 cached tokens require approximately **8 MiB**;
- the full 1,024-token window requires approximately **32 MiB**.

Therefore a long context can compete directly with the model's active weights.
The current cache uses the model's dtype, which is fp32. A lower-precision KV
cache is a plausible cache-residency optimisation, but CPU attention kernels
and conversion costs must be measured rather than assumed to be favourable.

## Optimisation options, ranked

### 1. Keep int8 as the first-line optimisation

This already exists in `babble/cpu_runtime.py` as
`quantize_dynamic_linears()` and is enabled with `BABBLE_QUANTIZE=1`. It gave
approximately 1.44x peak throughput in this run. The quality cost is not
zero, so it remains opt-in until a representative validation and completion
comparison says otherwise.

Dynamic quantisation also has activation-quantisation and dispatch overhead.
A static packed int8 runtime could be faster, but would require checkpoint
conversion and careful handling of the tied embedding/output head.

### 2. Test weight-only int4

Int4 or packed 4-bit weights would reduce linear-weight traffic again and may
be the largest remaining single-stream opportunity. The implementation needs a
real CPU kernel; storing int4 values in Python or unpacking them into fp32 for
every matmul would lose the benefit.

Required gates:

1. bits/character on the same held-out split;
2. fixed-seed completions and repetition/loop probes;
3. warm and cold latency;
4. tok/s at 1, 2, 3, 4, and 8 threads;
5. RSS and packed-weight size.

Do not infer quality from perplexity alone: the sampler is sensitive to small
logit changes.

### 3. Experiment with FP16/BF16 KV state

A lower-precision KV cache would halve its footprint and could leave more L3
for active weights. The experiment should compare:

- fp32 KV, fp32 weights;
- fp16 or BF16 KV, fp32 weights;
- fp16 or BF16 KV with int8/int4 weights;
- short and long contexts.

The important result is end-to-end tok/s, not just cache size. On this older
CPU, conversion or unsupported attention kernels could make lower precision
slower.

### 4. Micro-batch independent generations

Batching lets a layer's weights serve several sequences before the next layer
streams in. This is especially relevant to `best_of=4`, which already generates
multiple candidates in one batch. A request scheduler could batch separate
Discord replies too, trading queueing latency for aggregate throughput.

Benchmark batch sizes 1, 2, 4, and 8 with fixed-length generations and report
both:

- aggregate generated tok/s;
- p50/p95 time to first token and completion latency.

Batch throughput is not a substitute for single-user latency, and an
interactive bot should use a small bounded batching window.

### 5. Use physical cores, not SMT threads

The measured sweet spot is 3–4 threads. Eight logical threads reduced both
fp32 and int8 throughput. Pinning work to physical CPUs 0–3 may reduce jitter,
but it is not expected to change the fundamental bandwidth ceiling. Never
raise the serving thread count merely because `nproc` reports eight.

### 6. Fuse or replace the CPU runtime

The current model already uses a KV cache, inference mode, oneDNN-friendly
linear layouts, and the CPU torch build. Further gains probably require a
runtime with specialised packed kernels rather than Python-level changes:

- FBGEMM/oneDNN static packed int8;
- a torchao-style CPU weight-only kernel;
- a small C++/ggml-style runtime supporting the exact BPE model;
- operator fusion for layer norm, projections, and activation.

This is more invasive than the current dynamic quantisation switch and must
preserve checkpoint compatibility and the existing tokenizer sidecar contract.

### 7. Reduce model size if cache residency is the actual goal

To fit an entire fp32 model in 8 MiB, the parameter budget is roughly 2.1M
parameters before runtime state. At int8, the theoretical parameter budget is
roughly 8.4M. The current 34M model is far beyond both limits.

A distilled or separately trained smaller model is the only robust way to make
the complete model cache-resident. That is a quality/architecture decision,
not a serving tweak. A smaller model may also underperform the current model on
quality even if its tok/s is much higher.

## Things unlikely to help

- `mlock()` and huge pages: prevent paging or reduce TLB misses, but do not
  increase L3 capacity.
- Increasing PyTorch threads to eight: measured regression.
- Copying weights into another RAM buffer: still RAM, not L3.
- Reducing `block_size` alone: reduces positional/KV state but does not reduce
  model weights.
- `torch.compile` as a first move: compile startup is expensive and does not
  magically make the weight set cache-resident. It should be tested only for a
  long-lived process with a separate warmup/latency budget.

## Recommended next benchmark

Implement no default change yet. Add an isolated benchmark covering:

```text
weight mode: fp32 / int8 / int4
KV dtype:    fp32 / bf16 or fp16
batch:       1 / 2 / 4 / 8
threads:     1 / 2 / 3 / 4 / 8
context:     short / 256 / 1024 tokens
```

Record steady tok/s, end-to-end tok/s, TTFT, p95 latency, RSS, and quality
metrics. The likely winning interactive configuration is int8 or int4,
small-precision KV, three or four physical CPU threads, and bounded batch size;
the likely winning raw-throughput configuration may use a larger batch.

## Bottom line

The 8 MiB L3 is too small for the complete 136 MiB fp32 model, but it is still
useful. Quantisation makes the active matrix blocks more cache-friendly, the
KV cache can be made less intrusive, and batching can reuse each block across
multiple sequences. The measured int8 result—**147 tok/s versus 102 tok/s
fp32**—is already the concrete payoff. The next extreme optimisation should
be a real packed int4/int8 kernel and a KV-dtype/batch sweep, not a paging or
thread-count trick.
