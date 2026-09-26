# Native runtime promoted to live (jason's box, 2026-09-26)

## What changed

- Code: `~/babble-live` fast-forwarded from `cd5b801` to `4257ec4` (PR #46).
- `.env`: added `BABBLE_HF_RUNTIME=native`.
- Unchanged:
  - Model: `artifacts/hf-booper-multiturn-v1`.
  - Sampling settings.
  - `BABBLE_GIFS`, which stays unset (off). The served model was never trained to emit GIF tags. GIFs get enabled together with the long-context model that learns them.
- Backup of the previous `.env` and commit: `~/babble-live/backups/native-runtime-2026-09-26/`.

## Verification

- `model.load` logged `runtime=native`, `native_build=cached`, `load_s=4.2`. No `model.native_fallback` event, then `bot.ready` (booper#9024, 5 guilds).
- `babble sample` with the live env, through the native backend:

| prompt | reply | time |
| --- | --- | --- |
| `hey booper whats up` | "I just woke up from the hospital" | 62 ms |
| `do you like cats or dogs` | "I don’t like dogs" | 66 ms |

For comparison, replies of this length took 400–550 ms on the transformers runtime in the live logs.
- Gates (see `NATIVE_RUNTIME_2026-09-26.md`):
  - reference-logit parity: top-1 agreement 1.0, ΔNLL −7e-5;
  - greedy output identical to transformers over 128 tokens;
  - `babble bench`: 1014 vs 146 tok/s best-of-4 aggregate, TTFT 12.7 vs 62 ms.

## Rollback

Set `BABBLE_HF_RUNTIME=transformers` (or `lean`) in `~/babble-live/.env`, then `systemctl --user restart babble-bot`. No model files moved, so there is nothing else to restore.
