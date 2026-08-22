---
license: apache-2.0
datasets:
- openbmb/Ultra-FineWeb-L1
language:
- en
tags:
- babble
- booper
- language-model
- pretraining
library_name: pytorch
---

This model was asked to be published under my account, not the creators. The compute came from https://huggingface.co/posts/ProCreations/855858308074329

# Booper pretrain

A small from-scratch transformer matching the [babble / booper](https://github.com/kowo-co/babble) architecture, pretrained on a streamed slice of [openbmb/Ultra-FineWeb-L1](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1) (filtered English web text, Apache-2.0).

This is **stage-1 English pretraining only**. It is not the Discord-tuned chatbot. The architecture, tokenizer scheme, and training script come from [`kowo-co/babble`](https://github.com/kowo-co/babble) (`pretrain_hf.py` + the default 34.1M config).

## What it is

| | |
|---|---|
| Parameters | 34,096,128 |
| Layers / width / heads | 8 / 512 / 8 |
| Context | 1024 tokens |
| Tokenizer | byte-level BPE, 16,384 tokens |
| Data | `openbmb/Ultra-FineWeb-L1` |
| Train split | `CC-MAIN-2025-51` |
| Val split | `CC-MAIN-2025-47` (disjoint crawl) |
| Tokens trained | 600,206,202 |
| Hardware | 1× NVIDIA H200 (Hugging Face Jobs) |
| Wall clock | ~51 minutes end-to-end (~46 min of training at ~217k tok/s) |
| Final train loss | 2.531 |
| Final val loss | 2.472 |

Val loss fell steadily from 4.23 (step 200) to 2.47 (step 3118). Loss is **nats per BPE token**, not nats per byte, so it is not comparable to babble's older byte-level numbers without a bits-per-character conversion.

## Files

- `latest.pt` — checkpoint (`model` state dict, `config`, optimizer, step/token counts)
- `tokenizer.json` — BPE merge list, loadable with `babble.subword.BPETokenizer.from_json`
- `loss.jsonl` — per-checkpoint train/val loss, throughput, and samples

## End-of-run samples

Prompts used by the training script (temperature 0.7, top-k 40):

- `the cat` → `the catastrophic system and its adaptability to manage the catastrophic system is the case with a significant surge`
- `In the beginning` → `In the beginning of the New Jersey Law and Law, “law enforcement of the Law and Law and Law in the law`
- `Scientists have discovered` → `Scientists have discovered that the current market is expected to take a long way to see how we look at the new market`
- `The weather today is` → `The weather today is about to create a more sustainable and sustainable future. It’s a way to make the most of your time and`

These are expected to be clumsy: 34M params and 600M tokens is a short English pretrain, not a finished assistant.

## Training

Run on Hugging Face Jobs with the repo's self-contained `pretrain_hf.py` (bf16 AMP, AdamW, cosine LR after warmup). Job: [ProCreations/6a893d3e7c5c7dd37923450f](https://huggingface.co/jobs/ProCreations/6a893d3e7c5c7dd37923450f).

Source: [kowo-co/babble](https://github.com/kowo-co/babble), including [PR #21](https://github.com/kowo-co/babble/pull/21).
