# GPU pretrain (continue the 600M-token checkpoint on Discord-Dialogues)

This folder is a drop-in job for a single RTX Pro 6000 (96GB). It does **not** need access to anyone else's machine.

**Architecture:** 16,384 BPE vocab, context 1024, 8 layers / 8 heads / 512 width, dropout 0.1. Tied embeddings: **34,096,128** unique parameters (counting the tied `lm_head` a second time looks like ~42M — that is the same weights twice). Tokenizer is `tokenizer.json` in this folder — do not refit it.

**This is a continue-pretrain, not a new random init.** The weights you already produced were pretrained on [`openbmb/Ultra-FineWeb-L1`](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1) (English web), **not** on Discord. They have seen **600,206,202 tokens** (step 3118, 408,581 Ultra-FineWeb docs, loss ~2.53). The default command **loads that checkpoint** (model + AdamW + RNG) and keeps counting those tokens toward the budget (default **1B**).

**This run's corpus is Discord chat:** [`mookiezi/Discord-Dialogues`](https://huggingface.co/datasets/mookiezi/Discord-Dialogues), from **row 0**. The 408,581 `docs_consumed` on the start checkpoint indexes the *web* stream. It is **not** a skip into the Discord parquet. Using it as one would throw away ~408k Discord threads nobody has trained on. Mid-run restarts from `run-output/checkpoints/latest.pt` *do* skip Discord rows already consumed on this job (that file records `dataset_id`).

**What you send back:** `run-output/checkpoints/latest.pt` **and** `run-output/tokenizer.json`.

## Checkpoint you will be sent

You will get a ~409MB file named `latest.pt` out of band (payload: `step`, `tokens_consumed`, `docs_consumed`, `loss`, `config`, `model`, `optim`, `torch_rng`, `saved_at` — no `dataset_id` on that original file). It is **not** in git.

```bash
cd gpu-pretrain
# put the file here:
cp /path/to/the/latest.pt ./start.pt
chmod +x run.sh
./run.sh --smoke          # ~20 steps on a tiny bundled slice — no start.pt needed
./run.sh                  # load start.pt, train Discord from row 0 → 1B tokens total
```

Or pass the path: `./run.sh /path/to/latest.pt`

If `start.pt` (or `--checkpoint`) is missing, the job **exits with an error** instead of silently training from scratch. Random init is opt-in only: `./run.sh --from-scratch`.

A reboot mid-run: run `./run.sh` again. It prefers `run-output/checkpoints/latest.pt` over `start.pt`. Those mid-run files carry Discord `docs_consumed` and resume the parquet skip correctly.

Architecture is checked on load (`vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`). A mismatch is a hard error.

## Token budget (why 1B)

Hoffmann et al. compute-optimal is ~20 tokens/param. **34,096,128 × 20 ≈ 682M**. You are already at 600M on *web* text (~17.6 tokens/param). Default budget **1,000,000,000** is ~29 tokens/param on the *combined* token count: past compute-optimal on the original web run, plus ~400M tokens of Discord (the domain the bot actually talks in).

| `--token-budget` | New Discord tokens after 600M web | Guess on a 6000 (unverified) |
|---|---|---|
| `800000000` | +200M | ~15–60 min at 50–200k tok/s |
| **`1000000000` (default)** | **+400M** | **~30 min – 2 h** |
| `1360000000` | +760M (~20 tok/param of Discord *alone*) | ~1–4 h |
| `2000000000` | +1.4B | several hours |

Examples:

```bash
./run.sh                              # 1B total (600M web already in the ckpt + 400M Discord)
./run.sh --token-budget 1360000000    # hungrier Discord continue
```

These wall-clock numbers were **not** measured on a GPU while this package was written.

Default batch **64**, **bf16**. 96GB is far more than this model needs; `--batch-size 128` or `256` is fair game. Disk: leave **~10GB** (347MB parquet, `start.pt` 409MB, a few 409MB checkpoints, venv).

## How data is ordered

The pinned parquet has **7,300,966** rows.

- **Start from the Ultra-FineWeb checkpoint:** Discord is read in **file order from row 0**. We do not skip 408,581 Discord rows. Those web-doc counts do not correspond to this file (and even if they did, that original run used a *different dataset*).
- **Mid-run resume** (`run-output/checkpoints/latest.pt`): skip the first `docs_consumed` Discord rows in file order (epoch 0).
- **Later epochs** (only if you set a budget huge enough to finish the file): `torch.randperm` with seed `1337 + epoch`. That loads the parquet into RAM. The default 1B budget should **not** wrap.

Epoch-1+ shuffle is **new** to this package. The original 600M web run's shuffle (if any) is irrelevant here because we are not continuing that corpus.

## Logs

- `run-output/loss.jsonl` — keys `step`, `loss`, `tokens_seen`, `docs_consumed`, `elapsed_s`, `tokens_per_s`, `wall_clock`, `threads`, `batch_size`, `block_size`. After the corpus switch, `docs_consumed` counts **Discord** rows (starts at 0). `tokens_seen` keeps the 600M web total and climbs from there.
- `run-output/train.log` — console copy

## Dataset download

- repo: `mookiezi/Discord-Dialogues`
- revision: `a8b2294bd5b4acfe4ce537b688e7eee111c50fe2`
- file: `data/train.parquet` (~347MB)
- SHA256: `241e350e7f651085c5c2cb4d5274f7cb671b84b3d5fba091101823678da454ec`

Cached under `gpu-pretrain/.cache/`. `--smoke` never downloads it. Smoke: `./run.sh --smoke` or `BABBLE_PRETRAIN_SMOKE=1 ./run.sh`.

## If CUDA wheels mismatch

`run.sh` installs PyTorch from the **cu128** index when `nvidia-smi` exists. If import fails:

```bash
source .venv/bin/activate
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python -u train.py --config config.json --output-dir run-output --checkpoint ./start.pt
```
