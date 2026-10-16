# longctx-v2 promoted to live (jason's box, 2026-10-15)

## What was trained

- **Starting point:** continued from `hf-booper-longctx-v1` on the MacBook (M2 Pro, MPS). Code and recipe: PR frgmt0/babble#54 (merge `220f0f8`), preset `configs/sft/unhinged-mac.json`.
- **Context:** unchanged from longctx-v1: 8 turns of history, a 1536-token prompt budget, sequence length 2048.
- **Data mix** (details in `docs/reports/UNHINGED_DATA_2026-10-14.md`):

| source | share / size |
| --- | --- |
| `discord_vibe` | 52% (315,877 conversations from Discord-Dialogues rows 233,990+, never seen by earlier runs) |
| `discord` rehearsal | 42.5% |
| persona, ultrachat, oasst, no_robots, smoltalk | small rehearsal slices |
| `booper_local` | 7 consented approvals/corrections, x3, train-only |

- **Schedule:** 8,600 steps, about 281.8M nominal tokens, duty 1.0. Launched 2026-10-14 16:59 PDT, finished 2026-10-15 21:03 PDT.
- **Discarded launches:** two earlier attempts were thrown away. One was restarted at step 10 to enable dashboard reporting. The other crashed with an MPS OOM at about step 55; the fix was activation checkpointing for micro-batches 1536 or wider (`--grad-checkpoint-min-width`). It does not change what is trained (same losses at steps 10-40).

## Gates

- The export gate passed: best = final step 8600, aggregate val **2.0460** vs step-0 baseline 2.1359.
- The source gate passed: guard sources discord, persona and ultrachat held, and `discord_vibe` multi-turn improved.
- **Export:**
  - `model-int8.safetensors` sha256 `94d1b2fd311beed111c5597c79066ac89447d171a64b9a367035999b86409413`.
  - `tokenizer.json` sha256 `b1e1233481d5b2c3637fe8bcef81ec696fec835341680f0a0a72449bd699717a`, identical to longctx-v1's.
  - The Mac and desktop copies match. The `babble_*` serving fields in `config.json` are identical to longctx-v1.

## Vibe eval

`sft/vibe_eval.py`, the same 200 seeded real user prompts from the consented snapshot, live sampling settings. longctx-v1 to longctx-v2:

| metric | v1 | v2 |
| --- | ---: | ---: |
| vibe mean | 0.18 | 0.72 |
| replies with vibe >= 0.75 | 6% | 58% |
| coherent | 66% | 77% |
| engaged | 66.5% | 77% |
| lowercase | 31.5% | 61% |
| slang | 6% | 41.5% |
| laughter | 0.5% | 7% |
| caps bursts | 1.5% | 13% |
| assistant-speak | 0% | 0.5% |
| mean reply length (tokens) | 14.4 | 14.3 |

- **Caveat:** the vibe score is the same heuristic that selected the data, so it is partly circular. Coherence and engagement are the more independent signals.
- **Observed:** the model is louder, with occasional all-caps outbursts and occasional profanity. The blocklist covers slurs, not ordinary profanity.
- **Live probes** (`babble sample` on the live box after restart, about 50 ms each):
  - "hey booper" -> "Hey dude, how are you"
  - "good clanker" -> "clanker is mid"
  - "do you like dario or sam" -> "sam is goated"

## Live change

- Live tree fast-forwarded `0f6239e` -> `220f0f8` (2026-10-15 ~21:13 PDT / 04:13 UTC). Nothing changed under `babble/`, `deploy/` or `pyproject.toml`.
- Model copied to `~/babble-live/artifacts/hf-booper-longctx-v2/`.
- `.env`: only `BABBLE_HF_MODEL_DIR` changed, `hf-booper-longctx-v1` -> `hf-booper-longctx-v2`.
- **Verified:** after restarting `babble-bot`, `model.load source=hf runtime=native threads=4 native_kv=q16`, then `bot.ready booper#9024 guilds=7` (latency 84 ms).
- **Backup:** `~/babble-live/backups/longctx-v2-promote-2026-10-15/` holds `env.before`, `checkpoints/hf-booper-longctx-v1/`, `checkpoints/spec-ngram-longctx-v1.pt` and `sha256-longctx-v1.txt` (model `c543cb9d…1e35`, tokenizer `b1e12334…717a`; verified with `sha256sum -c`). `artifacts/hf-booper-longctx-v1/` is also left in place.

## Follow-ups

- **Spec table:** `BABBLE_NATIVE_SPEC_TABLE` still points at `spec-ngram-longctx-v1.pt` (sha256 `b1ed5dce…f758`). It only affects speed (acceptance rate), not output. Rebuilding it for longctx-v2 with `bench/extreme/spec_ngram.py` is still to do.
- **Dashboard gap:** booper.frgmt.xyz/runs is missing longctx-v2 records for steps 5490-7400 (and one at 2190). The worker's 2 KV writes per record exceeded the Workers KV free-tier limit of 1,000 writes/day, so POSTs got HTTP 500 from 10:53 to 16:59 PDT, until the midnight-UTC reset. The local `runs/longctx-v2/metrics.jsonl` is complete, and best-checkpoint selection is local and unaffected.

## Rollback

```
cp ~/babble-live/backups/longctx-v2-promote-2026-10-15/env.before ~/babble-live/.env && systemctl --user restart babble-bot
```

Then wait for `bot.ready` in `logs/babble.log`.
