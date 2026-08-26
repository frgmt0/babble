# Promotion: Booper-Big-Chat-INT8 live on jason's box (2026-08-26)

booper#9024 on jason's box (`~/babble-live`) now serves
**`ProCreations/Booper-Big-Chat-INT8`** through the new hf backend
(`babble/hfserve.py`, `BABBLE_SERVE_BACKEND=hf`) instead of the native
booper-chat checkpoint. This is a backend switch, not a checkpoint swap:
`checkpoints/latest.pt` + `tokenizer.json` (booper-chat, pair layout) are
untouched and remain the rollback.

## What was promoted

- Model: Mixtral-architecture MoE, 6 layers, hidden 896, 7 experts (top-1),
  vocab 16384 (the shared booper BPE tokenizer), context 4096. 149,602,432
  params total, ~50M active per token. INT8 on disk, re-expanded to fp32 in
  memory for CPU serving.
- Chat SFT on `mookiezi/Discord-Dialogues`, pair layout
  (`<bos> prompt <sep> response <eos>`) — same layout booper-chat used.
- Local snapshot: `~/babble-live/artifacts/hf-booper-big-chat-int8/`
  (serving never touches the Hub).

sha256:

```
9836b5329122c940448e0a35f503a576b9466adf58dc08bc24827b17f0249583  model-int8.safetensors
b1e1233481d5b2c3637fe8bcef81ec696fec835341680f0a0a72449bd699717a  tokenizer.json
50be36ecdccb967a7ac7e70ae12eee3afa49a8cdb6ec1466a2be222f04fd7110  config.json
```

## Live changes

- `pip install -e ".[hf]"` into `~/babble-live/.venv`
  (transformers 5.16.1, tokenizers 0.23.1, safetensors).
- `.env` (previous copy backed up to
  `~/babble-live/backups/big-chat-int8-promote-2026-08-26/.env`):

```
BABBLE_SERVE_BACKEND=hf
BABBLE_HF_MODEL_DIR=/home/jason/babble-live/artifacts/hf-booper-big-chat-int8
```

`BABBLE_SERVE_LAYOUT=pair` stays in the .env but is not read by the hf
backend, which always serves pair layout. Training triggers remain zeroed;
`babble train`/post-train still write `latest.pt`, which the hf backend never
reads — flipping the backend back is the only way a native checkpoint serves
again.

## Verification

- `model.load source=hf … params=149602432 device=cpu` then `bot.ready` in
  `logs/babble.log` at 2026-08-26T17:09:54Z.
- Live-venv preflight through the same generator class:
  `BABBLE_SERVE_BACKEND=hf BABBLE_HF_MODEL_DIR=… babble sample --prompt
  "hey booper, what's up?"` → coherent replies at ~0.5–0.6s per generation
  (best-of-4, CPU).

## Rollback

Delete the two `BABBLE_SERVE_*` lines above from `~/babble-live/.env` (or
restore the backed-up copy) and `systemctl --user restart babble-bot`. The
native checkpoint pair in `checkpoints/` was not modified.
