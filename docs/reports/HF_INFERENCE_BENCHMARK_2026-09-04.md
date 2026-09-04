# HF inference benchmark and serving optimization (2026-09-04)

## Scope

Measurements used the isolated `/tmp/babble-review.0Pbb08` source tree on
jason's i7-4790 desktop. The model was read from
`/home/jason/babble-live/artifacts/hf-booper-story-v2`; no live source, data,
service, or environment file was changed. Runtime: torch 2.13.0+cpu,
Transformers 5.16.1, four inference threads.

The fixed synthetic prompt was:

> Write two sentences about a robot learning why people laugh.

Each comparison drew four candidates with a 64-token ceiling, temperature 0.5,
top-k 40, top-p 0.9, repetition penalty 1.15, and 4-gram repetition blocking.

## Proven default-device gain

`torch.set_default_device("cpu")` installs a global Python dispatch hook even
though CPU is already PyTorch's native default and babble passes explicit CPU
devices at model/tensor boundaries. Alternating the hook on and off around the
same loaded model and same seed produced identical candidate token counts
(`64/53/47/64`, 228 aggregate):

| Runtime mode | Wall runs | Median wall | Aggregate throughput |
| --- | ---: | ---: | ---: |
| global CPU hook | 1.921 s, 1.900 s | 1.911 s | 118.7, 120.0 tok/s |
| native CPU default | 1.698 s, 1.703 s | 1.701 s | 134.2, 133.9 tok/s |

Removing the redundant hook reduced median wall time by 11.0% and raised the
median aggregate throughput by 12.3%. `configure_cpu()` now removes an earlier
hook with `torch.set_default_device(None)` while retaining the explicit CPU
device, thread, oneDNN, and denormal configuration.

## Best-of score retention

The old HF path requested `output_scores=True`, retained one
`candidates × vocabulary` tensor per output step, then stacked the history to
recover one log probability for each sampled token. The new tracker applies the
same temperature/top-k/top-p distribution inline and retains one current score
tensor plus sampled-token scalars.

At the live geometry (four candidates, vocabulary 16,384), retained full-score
payload falls from 16 MiB to about 0.26 MiB at 64 steps. At the configured
512-token ceiling it falls from 128 MiB to about 0.27 MiB. These are tensor
payload calculations; allocator and model memory are separate.

Within one process, resetting torch to the same seed produced identical four
candidate sequences and the same best candidate between Transformers' original
score-history path and the streamed path. Selection remains mean normalized
log probability over generated tokens, including EOS and excluding forced pad.

The optimized `/bench` workload measured one run at:

- TTFT: 54 ms
- steady aggregate throughput: 132.9 tok/s
- end-to-end: 131.0 aggregate tok/s, 36.8 selected-candidate tok/s
- actual tokens: 228 aggregate (`36/64/64/64`), 64 in the selected candidate
- elapsed: 1.741 s; average CPU: 3.9 cores; current RSS after the run: 1,142 MiB

This is a single diagnostic run, not a median performance claim.

## Thread sweep

The optimized generator was also swept from one through four torch threads.
Each setting used one warmup plus three timed runs with the identical fixed
prompt, seed, and sampling configuration above. Every run generated the same
`36/64/64/64` candidate lengths (228 tokens aggregate).

| Threads | Median wall | Median aggregate TPS | TPS range | Median TTFT | Median steady TPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.577 s | 88.5 | 88.2–88.5 | 124.2 ms | 91.3 |
| 2 | 1.921 s | 118.7 | 118.6–119.0 | 73.4 ms | 121.3 |
| 3 | 1.756 s | 129.9 | 129.5–130.0 | 61.2 ms | 132.2 |
| 4 | 1.723 s | 132.3 | 132.0–132.6 | 52.7 ms | 134.1 |

Four threads remained the winner: 1.9% higher median aggregate throughput and
1.8% lower median wall time than three threads, with only 0.4% spread between
the slowest and fastest four-thread runs. The existing four-thread inference
default is retained.

## Behavior controls and deferred work

The HF backend previously ignored `frequency_penalty` and `presence_penalty`.
The tracker implements both with prompt-and-completion token counts, but support
is gated behind `BABBLE_HF_FREQUENCY_PENALTIES=1`. It remains off by default,
because applying the native backend's non-zero frequency default would change
the promoted model's output distribution without an HF-specific quality gate.
The generation metadata reports the values actually applied.

Runtime INT8 was not enabled. The artifact's INT8 matrices are currently
expanded to fp32, while Transformers 5.16.1 already dispatches only the MoE
experts hit by routing. Replacing those expert matmuls with dynamically
quantized kernels would add activation quantization error and needs a held-out
loss and output-quality gate before it can become a serving default.

## Benchmark interface

`babble.benchmark.run_benchmark(generator)` runs one fixed, bounded synthetic
sample against the already-loaded HF generator. It records actual candidate
lengths, aggregate and selected throughput, TTFT, steady decode, process CPU,
RSS, model/runtime identity, dtype, and active optimizations. It saves and
restores torch RNG state. Multi-turn serving uses the backend's role-transcript
formatter and exact tokenizer budget for the synthetic prompt.

`format_benchmark(result)` renders a Discord-safe report under 2,000 characters.
Generators without step-level instrumentation raise `BenchmarkUnavailable`
instead of returning guessed TTFT or candidate metrics.
