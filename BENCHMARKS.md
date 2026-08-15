# babble performance, measured

Real numbers from `bench/bench_inference.py`, not estimates. Every figure below is
a **median over many runs** on the box described here; spreads are given where they
matter.

**tl;dr — how fast is babble actually?** On this CPU-only i7-4790, a warm babble
decodes at **~480–500 tokens/sec** (2–4 torch threads) with a **~3 ms
time-to-first-token**, and a full Discord reply (best-of-4, up to 256 tokens) lands
in **~0.9 s**. The KV cache from PR #7 makes steady decode **2.7–3.8× faster** than
recomputing attention every step. The model's weights are **12.8 MB**; the Python
process around them is **~300 MB**, almost all of it torch's CPU runtime.

## What was measured on

| | |
|---|---|
| **Commit** | `a2250af` (`babble/` is byte-for-byte this commit; the `bench/` harness is added on top and touches no model or decode code) |
| **Date** | 2026-08-15 |
| **Box** | Intel Core i7-4790 @ 3.60 GHz · 4 cores / 8 threads · 32 GB RAM · **CPU only, no GPU** · Linux |
| **Runtime** | torch 2.13.0+cpu · Python 3.12.13 (torch default thread count: 4) |
| **Model** | `checkpoints/base.pt` — 3,347,968 params · byte vocab (260) · 4 layers × 256 dim · 4 heads · block_size 512 |
| **Prompts** | real rows from the live corpus (82 rows), e.g. `"idk where you live"` (median length, 19 context tokens); a short→long spread for the TTFT-vs-length curve |

Two honesty notes up front:

* **Which checkpoint.** The live bot serves `checkpoints/latest.pt`, which **does not
  exist yet** — the corpus is at 82 rows and the stage-2 voice pass only fires at 100
  (`voice_trigger_rows`), so the bot currently answers from a random init. `base.pt`
  (the frozen stage-1 checkpoint) is the only *trained* weights on the box, so that is
  what these numbers load. It does not matter for timing: decode speed, TTFT and RAM
  are functions of the model **geometry**, not the weight values — a random init and
  `base.pt` at the same shape time out identically.
* **`max_new_tokens`.** Pure-decode numbers use **96** tokens per generation (the
  figure in this task). The live bot's own default is **256** (`BABBLE_MAX_NEW_TOKENS`),
  which is why the realistic-reply section below is measured at 256 — that is what a
  Discord user actually waits for.

The box is a **shared desktop** and had ~2–3 of load average from other work (a
browser, a terminal) during the run. Medians over 40–50 runs are robust to that, but
the absolute TPS — especially at 4 and 8 threads — would be a little higher on an idle
machine. The 8-thread row in particular fights both the background load *and*
hyperthread oversubscription. Where a single run looked contended (e.g. an 8-thread
sample dipping to 33 tps) the median still holds; re-run if you need a clean number.

## Reproduce

```sh
# from a checkout, with the trained checkpoint and corpus staged locally
# (data/ and checkpoints/ are gitignored; nothing here touches the live bot)
mkdir -p checkpoints data
cp ~/babble-live/checkpoints/base.pt checkpoints/base.pt
cp ~/babble-live/data/corpus.jsonl ~/babble-live/data/consent.json ~/babble-live/data/.salt data/

BABBLE_DATA_DIR=data BABBLE_CHECKPOINT_DIR=checkpoints BABBLE_EXTERNAL_DIR=external \
  python bench/bench_inference.py all --runs 50 \
    --checkpoint checkpoints/base.pt --json-out bench/results.json
```

`bench/results.json` holds the full raw output (every percentile). Sub-commands:
`infer` (one thread count), `train` (one voice-pass), `all` (the whole report).

Harness choices that make the numbers mean what they say:

* Decode is **greedy** (argmax). This makes the cached and uncached paths emit the
  *identical* byte sequence, so their speed ratio is a like-for-like comparison and
  not two different generations; it also removes sampling RNG from the timing. The
  realistic sampled path (`best_of=4`, temperature 0.5) is measured separately as an
  end-to-end latency, so its cost is not hidden.
* The harness always runs the **full** `max_new_tokens` and never stops on `<eos>`, so
  TPS is a property of the decoder, not of what today's weights happen to say.

---

## 1–5. Inference: TTFT, TPS, cached vs uncached, cold vs warm, thread scaling

### Thread-scaling table (cached greedy decode, 96 tokens, 50 runs each)

| threads | TTFT (ms) | steady TPS | end-to-end TPS | uncached steady TPS | **cache speedup** | cores busy | peak RSS |
|--------:|----------:|-----------:|---------------:|--------------------:|------------------:|-----------:|---------:|
| 1 | 4.5 | 425 | 420 | 111 | **3.82×** | 0.98 | 319 MB |
| 2 | 3.3 | 483 | 480 | 179 | **2.69×** | 1.99 | 319 MB |
| **4** | **2.7** | **498** | **497** | 216 | **2.31×** | 3.98 | 320 MB |
| 8 | 3.9 | 293 | 291 | 105 | 2.77× | 7.59 | 320 MB |

*Steady TPS = tokens 2…96 (first token excluded). End-to-end TPS = all 96 tokens
including TTFT — nearly identical here because TTFT (~3 ms) is a rounding error against
a ~190 ms, 96-token generation. Spread at 2 threads: e2e median 480, p25 471, p75 485
(n=50).*

**Thread-scaling finding — small models peak below max threads, and hyperthreads
hurt.** TPS is essentially flat from 2 to 4 threads (483 → 498, +3%) and **collapses at
8 threads** (293, *worse than a single thread*). The matmuls in a 3.3M / 256-dim model
are too small to feed 4 physical cores' worth of threading, and oversubscribing all 8
logical threads on 4 physical cores adds scheduling and cache-contention overhead that
swamps the gain. **2 threads is the sweet spot for throughput-per-core; 4 buys a last
few percent at the cost of the whole machine; 8 is a regression.** The bot's default of
`train_threads=2` is already a good, polite choice — near-peak TPS while leaving cores
free for everything else on the box.

### TTFT (time to first token)

t=0 is defined as **process already warm, model already loaded**: TTFT is the prefill
forward over the prompt plus the first sample. For a realistic short Discord prompt
(19 context tokens) it is **~2.7 ms** at 4 threads (3.3 ms at 2). It scales with prompt
length because prefill is O(prompt):

| prompt | context tokens | TTFT |
|---|---:|---:|
| 1 char | 2 | 2.2 ms |
| 9 chars | 10 | 2.6 ms |
| 18 chars | 19 | 2.7 ms |
| 34 chars | 35 | 3.5 ms |
| 289 chars | 290 | 14.1 ms |

### Cached vs uncached (the PR #7 headline)

Same prompt, same greedy sequence, KV cache on vs off. The cache turns each new byte
from a full-prefix attention redo into a single-token forward:

* **At the bot's operating point (2 threads): 2.7× faster** steady decode (483 vs 179
  tps).
* **Single-threaded: 3.8× faster** (425 vs 111 tps) — the cleanest read of the cache's
  own contribution, with no thread-scaling effects mixed in.
* The *ratio* shrinks as threads rise (3.8× → 2.3× from 1 → 4 threads) because the
  uncached path's larger full-window matmuls parallelize better than the cache's tiny
  single-token ones — so more threads help the slow path more. The cache still wins at
  every thread count. The absolute win also **grows with sequence length**: at 96
  tokens the uncached window only reaches ~115 tokens, so this understates the cache on
  the bot's 256-token replies.

### Cold vs warm

| | time |
|---|---:|
| **Cold — restart → ready for first token** | **~1.36 s** |
|  ↳ torch import | 1.04 s |
|  ↳ model load (`base.pt`, 40 MB) | 0.09 s |
|  ↳ first generation (96 tokens) | 0.20 s |
| **Warm — steady-state generation (96 tokens)** | 0.19 s (~493 e2e tps) |

A restart costs **~1.4 s before the first reply**, and it is almost entirely the
`import torch` tax — the model itself loads in under 100 ms. The first *generation* is
only ~10% slower than a warm one (lazy allocator/init), so once the process is up
there is no meaningful warm-up ramp. The bot is long-lived, so **warm is the number
that matters day to day**; cold is a one-off ~1.4 s cost per restart.

---

## 6. RAM

Measured by reading `/proc/self/status` (`VmRSS`, and `VmHWM` for the peak).

| | size | what it is |
|---|---:|---|
| **Model parameters** | **12.8 MB** | 3,347,968 params × 4 bytes (fp32). This is "the model". |
| torch import baseline (RSS) | 257 MB | interpreter + torch's CPU runtime, before the model exists |
| Model idle (RSS) | 296 MB | baseline + weights + first working tensors |
| **Peak during decode (RSS)** | **320 MB** | high-water mark across a full generation |

The gap is the story: **the process is ~25× the model.** babble's weights are 12.8 MB;
the ~300 MB resident is torch's CPU kernels (oneDNN/MKL), Python, and shared libraries,
essentially fixed regardless of this tiny model. Decode adds only ~24 MB of working
set on top of idle.

**Live bot cross-check (read-only, process untouched):** the running `babble-bot` unit
reports **VmRSS 301 MB** — in line with the idle figure above. (systemd's cgroup
accounting shows ~183 MB for the same process; that counts unique memory, whereas
`VmRSS` counts shared library pages in full. Both are "correct", they measure different
things.)

## 7. CPU

Cores busy = process CPU-seconds ÷ wall-seconds over a sustained decode burst:

| threads | cores busy | ≈ % of one core |
|--------:|-----------:|----------------:|
| 1 | 0.98 | 98% |
| 2 | 1.99 | 199% |
| 4 | 3.98 | 398% |
| 8 | 7.59 | 759% |

Decode fully saturates whatever thread count it is given (1 thread → 1 core pinned,
4 threads → 4 cores pinned). Read alongside the thread table: at 8 threads babble burns
7.6 cores' worth of CPU to produce *fewer* tokens than 2 threads at 2 cores — the
oversubscription is pure waste.

## 8. Training — a stage-2 voice pass

The voice pass is the training that actually fires in normal operation (every 100
corpus rows; the endless base-training loop is retired). Measured by running
`babble voice-pass --force` exactly as the bot launches it — as a low-priority
(`nice 19`), `train_threads=2` subprocess — and watching its `/proc` while it ran:

| | value |
|---|---:|
| Corpus | 82 consented rows → step 400, final loss 0.216 |
| Threads | 2 |
| **Wall time** | **75.1 s** |
| **Steps/sec** | **5.3** |
| **Peak RSS** | **632–647 MB** |
| CPU | 1.92 cores busy |

Training's peak RSS (~640 MB) is about **2× the inference process**, because the
optimizer (AdamW) carries two moment buffers per parameter and the backward pass holds
activations — the classic ~3× parameter-memory rule, on top of the same ~300 MB torch
baseline. It stays a two-core, minute-scale job: a full voice pass is ~75 s and leaves
6 of 8 threads free, which is the point of running it at `nice 19`.

---

## Recommendations (not implemented — this is measurement, not optimization)

Cheap wins spotted while measuring, written down rather than changed:

* **Default inference/train threads to 2, not 4.** torch defaults to 4 (physical
  cores); babble peaks at 2–4 and the extra threads buy ~3% for the whole machine.
  The bot already sets `train_threads=2` — this just confirms it is the right call and
  argues against ever raising it toward the logical-thread count. **Never run babble at
  8 threads**: it is slower than 1.
* **The ~1 s cold start is entirely `import torch`.** If restart latency ever matters,
  that is the only lever — the model load is <100 ms. Nothing to do today for a
  long-lived bot.
* **A quantized (int8) or `torch.compile`'d decode** could push steady TPS higher, but
  the model is already sub-3-ms-TTFT and ~500 tps; the user-visible latency is
  dominated by `best_of=4` drawing four 256-token candidates (~0.9 s), not by
  per-token speed. If reply latency is the goal, lowering `best_of` or
  `max_new_tokens` would move the needle far more than any kernel-level change.
```
