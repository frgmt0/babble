# Promotion: booper-story-v2 live on jason's box (2026-08-30)

booper#9024 on jason's box now serves **`typer-org/booper-story-v2`**,
replacing `typer-org/booper-story-v1` (see `STORY_V1_PROMOTE.md`). Same
architecture, tokenizer, INT8 layout and serving knobs — a directory swap.

## What was promoted

- Start: story-v1's exported weights (`--base runs/story-v1/export`), i.e. a
  continuation, not a fresh SFT of Big-Chat.
- 40M tokens / 1220 steps on jason's MacBook (M2 Pro, MPS, ~2.6k tok/s, ~5 h),
  lr 4e-5 cosine, seq len 1024. Dashboard: https://booper.frgmt.xyz/runs#story-v2
- Data mix (by example count; Discord kept high because its examples are ~30×
  shorter than a story): `roneneldan/TinyStoriesInstruct` 32,000 ·
  `euclaise/writingprompts` 40,000 (adult-register r/WritingPrompts fiction,
  detokenised, >3.5k-char stories skipped) · `HuggingFaceH4/no_robots` 16,000
  (the 8k set ×2) · `HuggingFaceTB/smoltalk` 16,000 (`smol-magpie-ultra` +
  `everyday-conversations`, first turn) · `mookiezi/Discord-Dialogues` 56,000.
- Val loss on this mix: 2.034 → 1.800 (v1's 1.335 was on an easier mix; not comparable).
- Published: https://huggingface.co/typer-org/booper-story-v2 (public);
  local snapshot `~/babble-live/artifacts/hf-booper-story-v2/`.

sha256 (identical on the MacBook export and the live snapshot):

```
06e68c59f090b81d712374394cc71a05d0ae4d3c94a8dd567c87b6ed79a12596  model-int8.safetensors
a76eb9459f850e5cd8c0e312719540b6d2f303c0b7db2551d0e6f37a3a056d6a  config.json      (unchanged)
b1e1233481d5b2c3637fe8bcef81ec696fec835341680f0a0a72449bd699717a  tokenizer.json   (unchanged)
```

## Live changes

`.env` (previous copy: `~/babble-live/backups/story-v2-promote-2026-08-30/.env`):

```
BABBLE_HF_MODEL_DIR=/home/jason/babble-live/artifacts/hf-booper-story-v2
```

`BABBLE_MAX_NEW_TOKENS=512` and `BABBLE_NO_REPEAT_NGRAM_SIZE=4` from the v1
promotion stay. Code unchanged (`~/babble-live` on `main`).

## Verification

- Preflight through the live venv (`BABBLE_SERVE_BACKEND=hf … babble sample`):
  dragon story → 5 paragraphs, on-prompt ("He was very afraid to fire"),
  10.4 s best-of-4 CPU; "hey booper whats up" → "Hey boo" (0.4 s).
- `model.load … hf-booper-story-v2 params=149602432` → `bot.ready guilds=5`
  at 2026-08-30T16:07:34Z.

## Honest assessment

- Chat voice survived (rehearsal did its job: "i gotta go on a walk", "Hey boo").
- Stories are longer and more on-prompt than v1, but **still children's-book
  register**: every story still opens "Once upon a time" and the detective
  prompt never got noir. 25% writingprompts on top of v1's weights was not
  enough to move the style. A v3 that wants adult prose should either start
  from Big-Chat (not v1) with WP ≥ 40% and TinyStories ≤ 10%, or drop
  TinyStories entirely.

## Rollback

Restore the backed-up `.env` (or point `BABBLE_HF_MODEL_DIR` back at
`…/hf-booper-story-v1`) and `systemctl --user restart babble-bot`.
