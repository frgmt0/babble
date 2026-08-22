# Stage-2 post-train on SSH's HF pretrain (2026-08-22)

One CPU pass of `babble post-train-from-checkpoint` against
[ProCreations/booper-pretrain](https://huggingface.co/ProCreations/booper-pretrain)
and the **real** Discord corpus copied out of `/home/beckett/babble-live`.
Nothing was written back to the live install.

## What was actually in the HF repo

Inspected with `list_repo_files` before downloading. Files:

| file | role |
|---|---|
| `latest.pt` | checkpoint (409 MB) |
| `tokenizer.json` | BPE merges, loads with `BPETokenizer.from_json` |
| `pretrain_config.json` | run hyperparams |
| `loss.jsonl` | stage-1 curve |
| `README.md` | card |

Tokenizer and config were both present. No substitution.

Checkpoint payload: `step=3118`, `tokens_consumed=600,206,202`, `config={vocab_size:16384, block_size:1024, n_layer:8, n_head:8, n_embd:512, dropout:0.1}`, **34,096,128 params**. Vocab matches tokenizer (16,384 = 256 bytes + 16,124 merges + 4 specials).

SSH's relayed numbers (step 1600, val 2.6088, sample `'the cat, and the pet is a human…'`) were **mid-run**. The published `latest.pt` is the **end of the 600M-token budget**: train 2.531 / val **2.472** (nats/BPE token). Card sample for `the cat` is `'the catastrophic system and its adaptability…'`.

## Stage-2 code

Already on this tree (`post_train_from_checkpoint`, merged as #21). Not reimplemented. Guardrails left as-is: `post_learning_rate=1e-4`, `post_rehearsal=0.5`, `post_min_pairs=100`, `post_gate_margin=0.05`. **`--force` was required**: 53 consented trainable pairs < 100 floor. No synthetic pairs, no pair augmentation.

## Data (copy-out, not in-place)

From `/home/beckett/babble-live` into this worktree's `data/` (gitignored):

- 484 corpus rows / 28,245 chars (train 387 / val 97 / **7,215 val chars**)
- 53 trainable correction pairs / 1,843 chars prompt+chosen
- live `latest.pt` copied to `experiments/results/ssh-posttrain/live_served.pt` for scoring only

Live hashes after the run match the pre-run snapshot (`8519c572…` `latest.pt`, `43358490…` `pretrained.pt`).

## Post-train curve (CPU, ~5.5 min)

Default budget 200 steps, checkpoint every 50, best-**corpus-val** kept. LR 1e-4, 50% corpus rehearsal.

| step | train loss (window) | pair val | corpus val |
|-----:|--------------------:|---------:|-----------:|
| 50 | 3.1400 | 4.8276 | 3.3864 |
| 100 | 1.9670 | 4.4814 | 3.3327 |
| **150** | **1.7361** | **3.6628** | **3.3092** ← best, restored |
| 200 | 1.5799 | 3.6190 | 3.3432 |

Train still falling at 200; corpus val ticked **up** 0.034 from 150→200, so the run did start to overfit the tiny Discord set. No second pass: the best-val restore already undid that last interval, which is what the existing early-select/gate is for.

## Discord val: same tokenizer, before vs after

Held-out real corpus rows, BPE 16k, nats/token and bits/char (scheme-independent).

| | nats / BPE token | bits / char |
|---|---:|---:|
| SSH pretrain, **before** post-train | **3.7705** | 2.407 |
| Post-train candidate (step 150) | **3.3092** | 2.113 |
| Δ | **−0.461** | −0.294 |

This is **not** the previous failure mode (train 0.32 / val 5.56, worse than the pretrain it started from). Corpus val **improved**. Workspace promotion gate therefore said `promoted=True` (candidate within margin of — in fact better than — the supplied pretrain). That write went to **this worktree's** `checkpoints/latest.pt` only.

## Samples (verbatim, temp 0.5, top_k 40, 60 new tokens, SHA-256-seeded)

Same three seeds per prompt.

### `hola`

- **pretrain:** `hola can be a valuable resource for anyone looking to enhance their fitness and fitness skills. With its intuitive interface…`
- **pretrain:** `hola and the other is the most important part of the process. This is why we should not use the following methods for the future. The best way to get the`
- **pretrain:** `hola and the “pale of the human being,” the “pale of the human being,” …`
- **post:** `hola senorhola senorhola senorhola`
- **post:** `hola is a stupid clanker`
- **post:** `hola is the best of the repo`

### `hello`

- **pretrain:** `hello in the world of the world of digital life. The journey of digital life is a journey that will be both transformative and transformative. At the heart of this `
- **pretrain:** `hello and the first two of the three is the first of the second half of the two is the second half of the two is the second half of the`
- **pretrain:** `hello and the other thing that was made to be a great deal with the fact that the company was able to deliver the same thing as a result of the`
- **post:** `hellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohello`
- **post:** `hello there`
- **post:** `hellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohello`

### `do you want to enter giveway`

- **pretrain:** `do you want to enter giveway and have a good chance of having a great deal of money. If you are a member of the brand new casino you can also get a great deal of`
- **pretrain:** `do you want to enter giveway and you will be able to get the most out of the internet to be able to play the newest online gambling enterprises…`
- **pretrain:** `do you want to enter giveway for the new version of the game you are playing in any way that you want to play in the game you are playing in the game. In this case,`
- **post:** `do you want to enter givewayobviously of course. gimme`
- **post:** `do you want to enter giveway`
- **post:** `do you want to enter giveway`

### `why is`

- **pretrain:** `why is it important to remember that the fact that we do not know what it means to be what we do? You can get the best out of the world `
- **pretrain:** `why is it important to remember that your dream is not a good idea but it is not a good idea for your dream to be a good idea for your dream`
- **pretrain:** `why is it important to know what I do to do in a good way? I believe that you have a great time to start your journey with the help of `
- **post:** `why is the sky` (all three seeds)

### `the cat`

- **pretrain:** `the cat. It is a big part of the cat that says the cat is not being seen as a cat. It is a cat that is not seen as a`
- **pretrain:** `the cat of the world. The cat of the world is a human expression of the human body. …`
- **pretrain:** `the catalyze of the human body. The catalyze of the human body is a form of a human body. …`
- **post:** `the cat is the cat is the cat is the cat` (repeated / looped)

### `where`

- **pretrain:** `where to make the most of the most effective and effective solutions for your business. …`
- **pretrain:** `where it was a big deal to the WiFi market on the NATO market. …`
- **pretrain:** `where you are going to be able to find a new one that you are looking for. …`
- **post:** `where are you live` / `where is the sky` / `where is the sky`

Honest qualitative read: post-train **did** pull the model off generic web-SEO English toward short Discord-ish replies, and it also **collapsed** — repetition loops (`hellohello…`, `the cat is the cat is…`) and pair-memorisation (`stupid clanker`, `senor`, `why is the sky`). Compression got better; generation got narrower.

## vs currently-served babble-live

Live `latest.pt` is the **byte-level 3.3M** model (`vocab 260`, 4×256, step 150). Raw nats/token are **not comparable** to BPE. Bits/char on the **same 97 val rows / 7,215 chars**:

| checkpoint | bits / char | nats / its own token | tokens on val |
|---|---:|---:|---:|
| **SSH pretrain** (BPE 34.1M) | 2.407 | 3.770 | 3,193 |
| **This post-train** (BPE 34.1M) | **2.113** | 3.309 | 3,193 |
| **Live served** (byte 3.3M) | 3.659 | 2.492 | 7,344 |

On held-out Discord text, the post-trained SSH checkpoint **beats what is served today by 1.546 bits/char** (2.113 vs 3.659). Live samples on the same probes are still byte-level garble (`l he wed the ber st is blent…`). That gap is mostly **English pretraining + BPE**, not the 150-step pair pass (the pretrain alone already beats live: 2.407 vs 3.659).

**Not promoted to `/home/beckett/babble-live`.** Even though it clears the numeric bar, serving a 34M BPE checkpoint is a follow-up decision (tokenizer mismatch with `generate.py`'s byte specials, 409 MB vs 40 MB, collapsed samples). Workspace `checkpoints/latest.pt` / `post_candidate.pt` exist for that decision; live hashes unchanged.

## Assumptions

- `--force` despite `post_min_pairs=100` because the ticket asked to run stage 2 on the real corpus as it exists (53 pairs).
- One pass, default hyperparameters. No architecture change, no sweep, no synthetic data.
- Comparability with live is bits/char only.

## Artifacts (this repo)

- `experiments/results/ssh-posttrain/post_loss.jsonl` — stage-2 curve
- `experiments/results/ssh-posttrain/hf_pretrain_loss.jsonl` — SSH stage-1 curve
- `experiments/results/ssh-posttrain/measure.json` — numbers + verbatim probes
- `experiments/results/ssh-posttrain/train.log`
- `experiments/ssh_posttrain_measure.py` — replayable measurement
- HF weights stay in gitignored `artifacts/hf-booper-pretrain/`
