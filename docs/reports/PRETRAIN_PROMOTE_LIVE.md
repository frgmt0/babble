# Promote SSH pretrain to live booper (2026-08-22)

Promoted **ProCreations/booper-pretrain** (`latest.pt` + `tokenizer.json`) into
`/home/beckett/babble-live`. Did **not** promote the 150-step post-train
checkpoint (loop collapse / pair memorization).

## Serving tokenizer

`babble.generate` now loads the tokenizer that ships **beside** a checkpoint:

- `checkpoints/tokenizer.json` present → `BPETokenizer.from_json`, refused if
  `vocab_size` disagrees with the weights.
- sidecar absent and `vocab_size == 260` → historical byte tokenizer (rollback).
- sidecar absent and a larger vocab → hard error, not a silent byte decode.

`CheckpointGenerator` reloads when either `latest.pt` or `tokenizer.json`
changes.

## Backup and rollback

Backup (survives in the live tree):

`/home/beckett/babble-live/backups/pre-ssh-pretrain-promote-2026-08-22/checkpoints/`

Previous live `latest.pt` sha256: `8519c57218d4466e93ea6cd927ceb9807078806b2339cc3a73d29894abfa48d9`

One-command rollback (restore the byte checkpoint, drop the BPE sidecar, restart):

```
cp /home/beckett/babble-live/backups/pre-ssh-pretrain-promote-2026-08-22/checkpoints/latest.pt /home/beckett/babble-live/checkpoints/latest.pt && rm -f /home/beckett/babble-live/checkpoints/tokenizer.json && systemctl --user restart babble-bot
```

## Live after promotion

- `babble-bot` systemd user unit: **active (running)**
- served file `/home/beckett/babble-live/checkpoints/latest.pt` sha256
  `f207a1d821f195c8c99133b3eaddf5d8021731c66dc8516305e3a07302f39327`
  (same as HF `artifacts/hf-booper-pretrain/latest.pt`, 409 MB)
- `step=3118`, **34,096,128** params, `vocab_size=16384`, `block_size=1024`,
  tokenizer `BPETokenizer` from `checkpoints/tokenizer.json`
- `babble-train` was inactive; not started. `pretrained.pt` left as the old
  snapshot (not the post-train candidate).

## Completions from the live install

`continue_text` on live `Settings.from_env()`, temp 0.5, top_k 40, 60 new
tokens, `torch.Generator().manual_seed(0)` per prompt. Prefix is included in
the model's context; printed text is **the continuation only**.

**hola** →
` is a vibrant and vibrant lifestyle for everyone.\n- The world of online casinos has a profound impact on the online gambling industry.\n- The world of online casinos is`

**hello** →
` is a critical component of the role of the Indian international media industry. This article delves into the world of online media and why it is essential to understand `

**the cat** →
` of the world is a true and powerful cat of all the human body. The cat of the world has been the only person who has been a cat of`

**why is** →
` it important to understand the importance of security in your daily life? In this article, we will explore the various aspects of security in your daily life and how `

These are Ultra-FineWeb-ish English continuations, not the post-train loop
(`hellohellohello` / `the cat is the cat is`). Left live.

## Collection

Same live serving code + promoted checkpoint: FakeDiscord ping then
`>> hey!` banked a `correction` row with `chosen == "hey!"` (`COLLECTION_OK`).
Core collection path was not changed; a byte-checkpoint rollback still works
because a missing `tokenizer.json` selects the byte tokenizer.
