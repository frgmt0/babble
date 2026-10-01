# Max-perf native runtime promoted to live (jason's box, 2026-09-30)

## What shipped
PR #49 (`integrate/max-perf`) landed on main at `4c17588`. It merges five tracks, all of which ran in parallel and were measured on the i7-4790:

| track | default? | result |
|---|---|---|
| kv: q16 KV cache (int16 K + fp16 V), flash/split-K decode, tiled prefill attention | on | L1536 TTFT 600 -> 451 ms (with prepack). Best-of-4 steady at L1536: 590 -> 785 tok/s. |
| a8: fp32 prepacked prefill GEMM | on | Bit-identical logits. T512 TTFT -13%. int8 activations failed the gate (top1 0.98) and are opt-in only. |
| prefix: chunked overflow trim, background prewarm, 512 MB cache, LRU fix | on | 30-turn chat simulation, median TTFT: 126 -> 11 ms with short messages, 467 -> 61 ms with long pastes. |
| spec: n-gram speculative decoding with exact rejection sampling | **enabled live** via `.env` | 1.23x end to end on live-shaped replies. Output distribution unchanged; greedy output identical. |
| w4: int4 decode weights, two-stage lm_head | off | int4 is a quality trade (top1 0.92-0.96). head2 sampling was identical, but the gain is small. |

Gates:
- `reference.py check` full / decode / prefix / verify: top1 1.0, |dNLL| <= 6e-5.
- 2048-token parity against transformers fp32.
- pytest: 969 passed.

Independent review verdict was SHIP:
- ASan/TSan clean, and output is bit-exact across thread counts.
- One caveat: q16 flips near-tie MoE routes on some prompts. That is 0.08% argmax flips over 38.9k real-text positions, which is accepted and documented.

## Live change
- `git pull --ff-only` to `4c17588`.
- `.env` additions:
  - `BABBLE_NATIVE_SPEC=1`
  - `BABBLE_NATIVE_SPEC_TABLE=/home/jason/babble-live/artifacts/spec-ngram-longctx-v1.pt`
- The spec table is sha256 `b1ed5dceb52c7fcc33bbe916a4497b5a62424beb50f7186acc9ea70768c0f758`. It is built by `bench/extreme/spec_ngram.py` from a public Discord-Dialogues sample plus the model's own samples. It contains no user data.
- The new defaults (q16 KV, 512 MB prefix cache, overflow keep 0.5) need no `.env` change.
- Backup of the previous `.env`: `~/babble-live/backups/maxperf-promote-2026-09-30/.env`.

## Verified
- `model.load`: runtime native, `native_build=compiled` (9.5 s), `native_kv=q16`, `native_spec=on k=1`, `prefix_cache_mb=512` (15 full snapshots).
- `bot.ready`: booper#9024, 5 guilds.
- Live-env `babble sample`: 52-58 ms short replies.
- Live-env `babble bench --json` (27-token prompt, best-of-4 x64):

| | before (native, 09-26) | now |
|---|---|---|
| TTFT | 12.5 ms | 12.5 ms |
| steady agg tok/s | ~1050 | **1364** |
| e2e agg tok/s | ~1020 | **1296** |
| selected tok/s | ~254 | 324 |

## Rollback
| what | how |
|---|---|
| spec only | Remove the two `BABBLE_NATIVE_SPEC*` lines and restart. |
| exact KV | Set `BABBLE_NATIVE_KV=fp32` and restart. |
| prefix trim | Set `BABBLE_CONVERSATION_OVERFLOW_KEEP=1.0` and restart. |
| everything | Restore the backed-up `.env`, `git checkout d5320bb` in `~/babble-live`, and restart. |

## Next levers (measured, not yet built)
- **int4 weights x int8 activations via vpmaddubsw.** Bandwidth halves and the kernel would no longer be compute-bound. Only worth doing paired with a QAT/fine-tune that recovers int4 quality: plain int4 already gives +67% decode-1 and +30% best-of-4, but at top1 0.92.
- **Distilled draft model for spec.** Ceiling acceptance is 0.72 per position, against 0.52 for the n-gram.
- **Re-warm on Discord typing events.** Keep the last conversation's KV resident in the engine.
