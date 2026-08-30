# Promotion: booper-story-v1 live on jason's box (2026-08-29)

booper#9024 on jason's box (`~/babble-live`) now serves
**`typer-org/booper-story-v1`** through the hf backend, replacing
`ProCreations/Booper-Big-Chat-INT8`. Same architecture, same tokenizer, same
INT8 layout — a directory swap in `.env`, not a code change.

## Why

Big-Chat was SFT'd only on `mookiezi/Discord-Dialogues` (one-line replies), so
"write me a story" got a one-liner or nothing. story-v1 continues that SFT on
a long-form mix so the model has actually seen `<sep>` followed by paragraphs.

## What was promoted

- Base: `ProCreations/Booper-Big-Chat-INT8` (Mixtral MoE, 6 layers, hidden 896,
  7 experts top-1, 149.6M total / ~50M active, vocab 16384, context 4096).
- SFT: `sft/sft_longform.py`, pair layout, loss on the response only, on
  jason's MacBook (M2 Pro 16 GB, MPS, ~2.5k tok/s). 30M tokens / 915 steps,
  lr 4e-5 cosine, seq len 1024, ~3.5 h.
- Data mix: `roneneldan/TinyStoriesInstruct` 54,000 · `HuggingFaceH4/no_robots`
  8,387 · `mookiezi/Discord-Dialogues` (last user→assistant pair) ~48,000 rehearsal.
- Val loss (400 held-out examples of the mix): 2.438 → 1.335.
  Curve and samples: https://booper.frgmt.xyz/runs#story-v1
- Published: https://huggingface.co/typer-org/booper-story-v1 (public);
  local snapshot `~/babble-live/artifacts/hf-booper-story-v1/`.

sha256 (identical on the MacBook export and the live snapshot):

```
cf82891ce3b2bd0c1cb4726f0ba9316143064e9ff5710c82321b2868eb7d904b  model-int8.safetensors
b1e1233481d5b2c3637fe8bcef81ec696fec835341680f0a0a72449bd699717a  tokenizer.json   (unchanged from Big-Chat)
a76eb9459f850e5cd8c0e312719540b6d2f303c0b7db2551d0e6f37a3a056d6a  config.json
```

## Live changes

`.env` (previous copy: `~/babble-live/backups/story-v1-promote-2026-08-29/.env`):

```
BABBLE_HF_MODEL_DIR=/home/jason/babble-live/artifacts/hf-booper-story-v1
BABBLE_MAX_NEW_TOKENS=512        # new; was the 256 default, which cut stories mid-sentence
```

Code in `~/babble-live` unchanged (`a10f9a4`); training triggers stay zeroed.

## Verification

- Preflight through the live venv (`BABBLE_SERVE_BACKEND=hf … babble sample`):
  "write me a short story about a dragon who is afraid of fire" → a 5-paragraph
  story (12.0 s, best-of-4, CPU); "hey booper whats up" → "i was playing with
  my friends" (0.8 s). Chat replies stay short; story requests get stories.
- `model.load source=hf … hf-booper-story-v1 params=149602432` then
  `bot.ready guilds=5` at 2026-08-30T04:23:18Z.

## Known limits

- Register is TinyStories (children's-book prose, simple vocab). Adult-register
  stories need `euclaise/writingprompts` / `HuggingFaceTB/smoltalk` in the mix (v2).
- Story replies cost ~7–12 s on CPU at best-of-4; `BABBLE_BEST_OF=2` halves that
  if it feels slow.

## Rollback

Restore the backed-up `.env` (or set `BABBLE_HF_MODEL_DIR` back to
`…/hf-booper-big-chat-int8` and drop `BABBLE_MAX_NEW_TOKENS`) and
`systemctl --user restart babble-bot`. Nothing else was touched.
