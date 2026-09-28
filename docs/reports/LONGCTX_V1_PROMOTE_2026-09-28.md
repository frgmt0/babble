# longctx-v1 promoted to live (jason's box, 2026-09-28)

## What was trained

- **Starting point:** continued from `hf-booper-multiturn-v1` on the MacBook (M2 Pro, MPS). Code and recipe: `train/longctx-gif` (PR #46), preset `configs/sft/longctx-mac.json`.
- **Context:** 8 completed exchanges of history, a 1536-token prompt budget, and sequence length 2048.
- **Data mix:**

| source | share |
| --- | --- |
| Discord-Dialogues | 46% |
| ultrachat_200k (multi-turn) | 28% |
| SmolTalk (multi-turn) | 10% |
| WritingPrompts / TinyStories / no_robots rehearsal | the remainder |

- **GIF tags:** a synthetic slice of `[gif: ...]` tags on 0.64% of Discord targets. Upstream stripped all real GIF URLs; only 1 real send survived.
- **Schedule:** planned for 12,969 steps. It was shortened on 2026-09-27 to finish overnight: 4,800 steps and 114.6M tokens (about 0.19 of an epoch). The first 2,850 steps ran at duty 0.5, the rest at duty 1.0 (~2,250 tok/s).

## Gates

- The run's own export gate passed: best = final step 4800, aggregate val **1.611** vs Story-v2 baseline 2.418.
- Per-source change vs baseline, in nats (negative = better); the limit is +0.05:

| view | discord | ultrachat | smoltalk | writingprompts | no_robots | tinystories |
| --- | --- | --- | --- | --- | --- | --- |
| role-transcript | −0.024 | −1.146 | −1.010 | −0.107 | −0.011 | +0.032 |
| multi-turn | −0.022 | −1.419 | −1.373 | | | |

- **Export:** `model-int8.safetensors` sha256 `c543cb9d70bf30f1e0f501d15a8e24b7dad155cbe91d1542d4a8ed6a2abf1e35`. The Mac and desktop copies match.
- **Side-by-side check:** 8 prompts, single-turn and 4-turn history, native runtime, same seed.
  - Replies are in the same register as multiturn-v1.
  - They stay somewhat more on-topic with history, e.g. "why is the sky blue then" after a rainbow turn → "rainbows are made up of rainbow particles".
  - It is still a small, silly model.
- **GIF tags:** 0 tags in 120 sampled replies to reaction-style prompts, with both best-of-4 and best-of-1. The synthetic slice was too small and too briefly seen to teach the behavior. `BABBLE_GIFS` stays off.

## Live change

- Model copied to `~/babble-live/artifacts/hf-booper-longctx-v1/`.
- `.env` changes:

| setting | before | after |
| --- | --- | --- |
| `BABBLE_HF_MODEL_DIR` | `…/hf-booper-multiturn-v1` | `…/hf-booper-longctx-v1` |
| `BABBLE_CONVERSATION_MAX_TURNS` | 3 | 8 |
| `BABBLE_CONVERSATION_MAX_TOKENS` | 512 | 1536 |
| `BABBLE_MAX_NEW_TOKENS` | 510 | 509 |

- Runtime stays `BABBLE_HF_RUNTIME=native`.
- **Verified:** `model.load` (runtime native, cached build, 0.21 s), then `bot.ready` (booper#9024, 5 guilds). A live-env `babble sample` replied in 56 ms.
- **Backup:** previous `.env` in `~/babble-live/backups/longctx-v1-promote-2026-09-28/`. `hf-booper-multiturn-v1` is untouched.

## Rollback

Restore `~/babble-live/backups/longctx-v1-promote-2026-09-28/.env`, then `systemctl --user restart babble-bot`.
