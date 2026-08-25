# HF pretrain (SSH's GPU) + Discord post-train: design, deliverable, and status

## TL;DR

- **Stage 1 is a deliverable script, not a run we did.** `pretrain_hf.py` (repo root, self-contained)
  streams `openbmb/Ultra-FineWeb-L1` and pretrains booper's architecture at a chosen size. It is
  smoke-tested end-to-end on this CPU box (tiny budget) but has **not** been run at real scale --
  that happens on SSH's rented GPU via Hugging Face Jobs. See "Exact command for SSH" below.
- **Default model: 34.1M params** (measured), trained on a default budget of 600M tokens
  (17.6 tokens/param -- see "Model size vs token budget"). Cheaper (14.1M / 350M tokens) and bigger
  (50.5M / 1B tokens) presets ship alongside it.
- **Tokenizer changed: byte-level -> BPE (16,384 tokens)**, fit on the pretrain corpus itself. This
  makes stage 1's loss numbers **not comparable** to the old 2.23 trigram / 2.4681 (see correction
  below) byte-level numbers without a conversion -- see "Comparability" below for the exact fix
  (bits-per-character) and what to run once a real checkpoint exists.
- **Stage 2 (`babble post-train-from-checkpoint`) is built, tested, and proven against a small
  locally-pretrained stand-in checkpoint** (not a real HF pretrain -- none exists yet). It is ready
  to run the moment SSH's checkpoint + tokenizer come back.
- **Nothing was promoted.** `~/babble-live` was not touched. There is no real pretrained checkpoint
  to promote yet -- promotion happens after stage 2 runs against SSH's output and clears the gate.

## Why this document exists, and what changed mid-run

The original brief for this ticket assumed the HF pretrain would run locally, on this box, over a
CPU-budgeted (50-200MB) slice of an HF dataset, overnight. Partway through the run, SSH (who
actually owns compute -- rented GPUs via Hugging Face Jobs) said he would run the pretrain himself,
and asked for `openbmb/Ultra-FineWeb-L1` specifically and a 20-50M parameter model. That changes
what "stage 1" means: it is no longer a run we execute and report on, it is **a script we hand to
someone else to execute**, and it has to work the first time with zero follow-up questions. This
document, and `pretrain_hf.py`, were built for that target. Stage 2 (the Discord post-train) is
still run locally, still CPU, still keeps every guardrail from the original brief.

Two things the original brief stated as fact turned out to be stale or wrong once checked against
the repo, corrected here rather than silently carried forward:

- **Corpus size**: the brief said "~423 rows / ~24,000 characters." The live corpus
  (`~/babble-live/data/corpus.jsonl`) is **463 rows** as of this run -- it grew since the brief was
  written. Directionally nothing changes (still three orders of magnitude too small for even a 3.3M
  param model, let alone 20-50M), but the exact number matters for anyone computing ratios later.
- **"Previous best: 2.4681"**: `PIPELINE_REVAMP_2026-08-20.md` §7 documents that this number was
  **leaky** -- the synthetic-corpus generator built its Markov chain over the whole corpus
  including held-out val rows, so 2.4681 partly measured memorized val phrasing, not
  generalization. That same report's own §7.2, after the leak was fixed, puts the honest number at
  **2.581** (synthetic-mix best val) vs **2.647** (clean baseline, no synthetic mix). **The real
  number to beat is 2.581**, not 2.4681. Both are byte-level nats/token, same scale as the 2.23
  trigram baseline; see "Comparability" for how the new BPE-based pipeline's numbers relate to any
  of these.

## Dataset: openbmb/Ultra-FineWeb-L1

Checked directly against the dataset's HF API metadata (`cardData`), not just the card prose:

- **License: `apache-2.0`.** Permissive, no attribution-only or non-commercial restriction.
- **Size: ~1.14 billion documents, "1T+ tokens", 2.83TB total** across six Common Crawl snapshot
  configs (`CC-MAIN-2025-30`, `-33`, `-38`, `-43`, `-47`, `-51`; 171M-215M rows each). This is the
  **L1 filtered layer** of openbmb's UltraData pipeline -- built on the FineWeb processing recipe
  with additional filtering/deduplication on top, i.e. already-cleaned web English, not raw Common
  Crawl. That's the choice being made here over something like raw C4 or a from-scratch scrape: a
  tiny model gets more signal per token from already-filtered data, and SSH specifically asked for
  this dataset ("giving extremely high quality data like this to a tiny model would be
  interesting").
- **Format**: Parquet, columns `uid` (string), `content` (string, the document text), `meta`
  (JSON-encoded string: URL, language score, WARC id, source file), `dataset_index` (int64).
  `pretrain_hf.py` reads `content` (`PretrainConfig.text_field`).
- **Streaming**: confirmed working end-to-end in the smoke test below via
  `datasets.load_dataset(..., name=config, split="train", streaming=True)` -- no full download,
  parquet shards are fetched and decoded lazily. `IterableDataset.skip(n)` is used for resume (see
  "Robustness to a job dying midway").
- **Train/val split is two different snapshot configs**, not a held-out slice of one
  (`train_config="CC-MAIN-2025-51"`, `val_config="CC-MAIN-2025-47"` by default): different Common
  Crawl crawl dates, same processing pipeline. This is a stronger separation than shuffling one
  stream and holding out a tail -- there is no shared crawl-date structure between train and val
  that a leaky split could quietly exploit.

## Model size vs token budget

`pretrain_hf.py --config <preset>` picks the model shape from JSON, not code. Three presets ship
under `configs/pretrain/`; every dimension (layers, width, heads, context, batch, lr, token budget)
is a field, so a fourth is a copy-and-edit away. Params were measured directly (not hand-computed)
by instantiating each config's `ModelConfig`/`Babbler`:

| preset  | vocab | block | n_layer | n_head | n_embd | params (measured) | token budget | tokens/param | ~steps |
|---------|------:|------:|--------:|-------:|-------:|-------------------:|-------------:|--------------:|-------:|
| cheap   | 8,192 |   768 |       6 |      6 |    384 | **14.07M**          |  350,000,000 |          24.9 |  ~7,120 |
| default |16,384 | 1,024 |       8 |      8 |    512 | **34.10M**          |  600,000,000 |          17.6 |  ~9,155 |
| big     |16,384 | 1,024 |       8 |      8 |    640 | **50.48M**          | 1,000,000,000 |         19.8 | ~10,172 |

Reasoning: SSH asked for "20-50m params ... a large amount but not an insane amount [of tokens]."
**Default lands at 34.1M** (middle of the requested band) against **600M tokens**, chosen with a
Chinchilla-style tokens-per-param sanity check -- the empirical compute-optimal ratio from
Hoffmann et al. 2022 is ~20 tokens/param, and 600M/34.1M = 17.6, i.e. close to but slightly under
that ratio (deliberately: over-training a small model a bit past compute-optimal is a good trade
here since inference cost is what SSH actually cares about downstream, not raw pretraining FLOPs,
and 600M tokens is a fraction of a percent of the >1T-token dataset -- "large but not insane" per
the ask). **cheap** dips to 14.1M params, intentionally below the requested band, as the genuinely
cheap fallback if `default` proves too slow/expensive; **big** sits at the top of the band (50.5M)
with a near-canonical 19.8:1 ratio.

Every one of these is far, far larger than the ~24KB/463-row Discord corpus can ever inform on its
own -- which is the entire point of a two-stage pipeline: the model learns English structure from
~350M-1B real tokens where it's cheap to do so (SSH's GPU), then a much smaller, much slower,
heavily-guarded stage 2 nudges the *voice* on our tiny corpus without touching what stage 1 taught
it about English.

## Tokenizer: byte-level -> BPE, and why

`babble/tokenizer.py`'s raw-UTF-8-byte scheme (`VOCAB_SIZE=260`) was chosen for a from-scratch
model on a 24KB corpus, where spelling one byte at a time is a real but survivable cost and a
learned vocabulary has nothing real to fit on. Neither justification holds at HF-pretrain scale:
- A byte tokenizer spends ~4-5x more sequence positions per English sentence than a decent BPE
  vocab does, which at a *fixed* `block_size` context window means byte-level sees roughly a
  quarter as much text per example, and at a *fixed* token budget spends a large multiple of the
  compute reproducing spelling rather than learning structure.
- There is finally a real corpus (600M+ tokens) to fit a vocabulary on without the leakage risk
  that ruled it out for the 24KB Discord corpus (`babble/subword.py`'s docstring: "fitting on
  validation text would leak held-out phrasing into the vocabulary itself" -- moot at this scale,
  and `pretrain_hf.py` still fits strictly on train-config documents, matching that same
  discipline).

`pretrain_hf.py` fits its own byte-level BPE tokenizer (16,384 tokens by default: 256 raw bytes +
16,124 learned merges + 4 specials) on a bounded sample (`tokenizer_fit_docs`, default 20,000
documents) of the **train** config only, before any training step, and saves it to
`tokenizer.json` immediately so a resumed run reuses the exact same tokenizer rather than refitting
(refitting would silently change what every previously-trained id means). This reuses
`babble/subword.py`'s exact BPE scheme (byte-level, GPT-2-style whitespace/non-whitespace
pretokenization, deterministic merge replay) -- `pretrain_hf.py` deliberately duplicates the
*training loop* implementation (see its module docstring for why: it must run with zero dependency
on this repo being installed), but the **artifact format is shared, not duplicated**:
`babble.subword.BPETokenizer.from_json()`/`.to_json()` (added alongside this work) read and write
the exact same `{"merges": [...]}` schema, so `tokenizer.json` from a `pretrain_hf.py` run loads
directly into the real, tested `BPETokenizer` class stage 2 uses -- no translation step, no second
implementation to keep in sync for anything except the training algorithm itself.

### Comparability: these numbers are NOT the same scale as 2.23 / 2.581

This is the exact trap the brief warned about, so being explicit: a BPE-tokenized model's
cross-entropy loss is **nats per BPE token**, not nats per byte. Because each BPE token spans
several bytes on average, a BPE model's raw loss number will look *numerically lower* than a byte
model's even at equal or worse actual compression -- comparing them directly ("3.1 beats 2.23!")
would be comparing different units and drawing a false conclusion.

The fix already exists in this repo: `bits_per_char()` (used identically in `experiments/sweep.py`
and `experiments/tokenizer_sweep.py`, and central to `CAPACITY_TOKENIZER_REPORT.md`'s BPE-vs-byte
comparison) converts any tokenizer's loss to **bits per character of the original text**, a scheme-
independent unit: `bits_per_char = (total_nats / total_chars) / ln(2)`. The trigram baseline
(2.23 nats/byte-token = 3.2655 bits/char) and the corrected transformer best (2.581 nats/byte-token
= ~3.72 bits/char, computed the same way) are already reported on this scale in
`CAPACITY_TOKENIZER_REPORT.md`.

**What to run once a real checkpoint exists**: `babble post-train-from-checkpoint` already computes
`corpus_val_before` -- the loss of the *raw pretrained checkpoint*, before any Discord fine-tuning,
on our held-out real corpus rows, in the pretrain's own token units. Run it with a minimal step
budget (e.g. `--steps 1`) purely to get that one number cheaply, then convert it to bits/char using
the corpus's known held-out character count (`sum(len(row.text) for row in corpus_split.val)`) the
same way `bits_per_char()` does, and compare *that* number to 3.2655 (trigram) and ~3.72
(transformer best). Do not compare the raw nats numbers across tokenizers.

## Exact command for SSH

```bash
# one-time: install the CLI
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login

# the actual run -- ~/.cache is never touched with a full dataset copy, streaming only.
hf jobs uv run \
  --flavor l4x1 \
  --timeout 8h \
  --secrets HF_TOKEN \
  https://raw.githubusercontent.com/kowo-co/babble/main/pretrain_hf.py \
  -- --config https://raw.githubusercontent.com/kowo-co/babble/main/configs/pretrain/default.json \
     --output-dir /data/pretrain-default
```

Notes:
- `pretrain_hf.py` reads its own dependencies (`torch`, `datasets`, `huggingface_hub`, `pyarrow`)
  from the `# /// script` PEP 723 header at the top of the file -- `hf jobs uv run` (which wraps
  `uv run`) installs exactly those on the job's container. **No repo checkout, no `pip install -e
  .`, nothing else has to exist on the machine that runs this** -- that's why the deliverable is
  one file, fetched straight off `raw.githubusercontent.com` once this branch is on `main`
  (`kowo-co/babble`, public, per this project's normal publish flow). `--config` accepts either a
  local path or, as above, another raw GitHub URL -- `argparse.FileType`/`Path.read_text` both
  handle a URL string fine on `pathlib` >=... actually: **if `--config <URL>` does not resolve on
  SSH's box** (this hasn't been tested against a real `hf jobs` container), the safe fallback is to
  `curl` the three preset JSONs down first and pass a local path -- flagging this explicitly rather
  than asserting it definitely works, since it wasn't possible to test inside an actual `hf jobs`
  container from here.
- `--flavor l4x1` (1x Nvidia L4, 24GB VRAM, $0.80/hr as of this writing per HF's published Spaces
  GPU pricing table) is the recommended default: 24GB is enormous headroom for a 34M-param model
  (the whole model + AdamW states + activations at batch 64 / block 1024 is on the order of a few
  hundred MB), so there is no reason to pay for more VRAM than that. `a10g-large` ($1.50/hr, more
  compute) or `a100-large` ($2.50/hr) are the step-ups if `l4x1` proves too slow once real
  throughput is measured (see "Wall clock and cost" below) -- change `--flavor` only, nothing else.
- `--timeout 8h` is comfortably above the "wall clock" estimate below with margin; `hf jobs`
  defaults to 30 minutes and silently kills the job past that, so this flag is not optional.
- `--secrets HF_TOKEN` passes SSH's own logged-in HF token into the job as an env var, needed only
  if `hub_checkpoint_repo` is set in the config (see next section) to push results out before the
  job's ephemeral filesystem is deleted. Not needed for `default.json`/`cheap.json`/`big.json` as
  shipped (`hub_checkpoint_repo: ""`) -- but see "Persist results" below for why you almost
  certainly want to set it.
- Swap `configs/pretrain/default.json` for `cheap.json` or `big.json` to change the point on the
  size/cost curve; nothing else in the command changes.

### Persist results: set `hub_checkpoint_repo`, or the job's output vanishes

HF Jobs' own docs are blunt about this: *"A Job's filesystem is deleted when the Job ends."*
`--output-dir /data/pretrain-default` above only matters while the job is running. Before SSH
launches for real, he should edit the chosen config's `"hub_checkpoint_repo"` field to a repo name
under his account (e.g. `"ssh-username/babble-pretrain-default"`) and pass `--secrets HF_TOKEN`
(already in the command above) -- `pretrain_hf.py` then pushes `latest.pt`, `tokenizer.json`, and
`loss.jsonl` to that Hub repo after every checkpoint (`maybe_push_to_hub`, best-effort: a push
failure is logged and never kills the run, so a rate limit or network blip mid-job doesn't lose
hours of compute). Without this set, SSH must manually pull `/data/pretrain-default` off the job
container before it's torn down, which `hf jobs` does not make easy to do after the fact.

### What SSH sends back

Exactly three files (whichever way he gets them off the job -- Hub repo or manual pull):

1. **`latest.pt`** -- the checkpoint. `{"step", "tokens_consumed", "docs_consumed", "loss",
   "config", "model", "optim", "torch_rng", "saved_at"}`; `config` is enough on its own to
   reconstruct the exact `ModelConfig`/`Babbler` shape (`babble.model.ModelConfig.from_dict`),
   `model` is the plain `state_dict`.
2. **`tokenizer.json`** -- `{"merges": [[a, b, new_id], ...]}`. Loads via
   `babble.subword.BPETokenizer.from_json(path)`.
3. **`loss.jsonl`** -- one line per checkpoint: `step`, `tokens_consumed`, `docs_consumed`,
   `train_loss`, `val_loss`, `lr`, `elapsed_s`, `tokens_per_s`, `samples` (four fixed prompts,
   generated fresh at that checkpoint), `at`. This is the loss curve and the qualitative "did it
   learn English" evidence in one file -- no separate report needed from SSH's side.

## Wall clock and cost

**Honest uncertainty up front**: there is no GPU on this box, so nothing here is a direct GPU
measurement. What was actually measured is CPU throughput for a much smaller smoke-test model (see
below); the GPU numbers are an order-of-magnitude estimate from that, general knowledge of small-
transformer throughput on modern GPUs, and the fact that `pretrain_hf.py` is plain eager PyTorch
(no `torch.compile`, no fused kernels) -- so treat these as planning numbers, not commitments.
**SSH should watch `hf jobs stats`/the `tokens_per_s` field in the first few log lines and adjust
`--timeout` or the token budget if real throughput is far off.**

Measured on this box (2 CPU threads, `configs/pretrain/smoke.json`: 0.14M params, block_size 128,
batch 4): steady-state throughput reached **~15,000-17,500 tok/s** after warmup (see "Smoke test"
below for the full transcript) -- that's a model roughly 240x smaller than the default 34.1M
config, on CPU, so it is not a stand-in for GPU throughput at real size; it only proves the
mechanics run.

| preset  | params | tokens | est. tok/s (L4, eager fp32/bf16 mix) | est. wall clock | flavor | $/hr | est. cost |
|---------|-------:|-------:|--------------------------------------:|-----------------:|--------|-----:|----------:|
| cheap   | 14.1M  | 350M   | 60,000-150,000                        | 0.6-1.6h          | l4x1   | $0.80 | ~$0.50-1.30 |
| default | 34.1M  | 600M   | 30,000-100,000                        | 1.7-5.6h          | l4x1   | $0.80 | ~$1.35-4.50 |
| big     | 50.5M  | 1.0B   | 25,000-70,000                         | 4.0-11.1h         | a10g-large | $1.50 | ~$6-16.7 |

If `default`'s measured throughput on the actual job lands near the low end of that range, moving
`--flavor` up to `a10g-large` ($1.50/hr, more SMs than L4) or `a100-large` ($2.50/hr) is the single
lever to pull -- no other config change is needed, since `pretrain_hf.py` auto-detects CUDA and
uses whatever GPU it's given.

## Robustness to a job dying midway

- **Checkpointing**: every `checkpoint_every_steps` (default 200), `pretrain_hf.py` writes
  `checkpoints/latest.pt` (atomic: temp file + `os.replace`, so a mid-write kill leaves the
  previous checkpoint intact -- mirrors `babble/trainer.py`'s exact discipline), appends one line
  to `loss.jsonl`, and (if configured) pushes both plus `tokenizer.json` to the Hub.
- **Resume**: on startup, if `checkpoints/latest.pt` exists, `pretrain_hf.py` loads `step`,
  `tokens_consumed`, and `docs_consumed` from it, reopens the train stream, and calls
  `IterableDataset.skip(docs_consumed)` before continuing -- **verified working** in the smoke test
  below (a second invocation with a higher `--token-budget` resumed from step 82 exactly, at the
  exact document offset, and kept training). The tokenizer is loaded once from `tokenizer.json`
  rather than refit, so ids mean the same thing across the interruption.
  - This does depend on the stream *not* being shuffled (`pretrain_hf.py` deliberately does not
    call `.shuffle()` on the `IterableDataset`, precisely so `.skip(n)` gives a reproducible
    resume point) -- documented as a real, deliberate simplicity/shuffling tradeoff, not an
    oversight: Common Crawl document order within one snapshot isn't sorted by anything the model
    would key on, so the loss from not shuffling is small next to the resume complexity a shuffle
    buffer would add.
- **Signal handling**: SIGINT/SIGTERM set a flag checked once per step; the current step finishes,
  a checkpoint is written, then the process exits -- so a `hf jobs cancel`, a preemption, or an
  operator's Ctrl+C all leave a loadable checkpoint rather than a torn one. This does **not** cover
  a hard `kill -9` or an OOM crash -- that's what periodic checkpointing + resume-by-skip is for.

## Smoke test (this run, real, on CPU)

Ran end-to-end twice against `configs/pretrain/smoke.json` (140K params, vocab 512, block 128,
token budget 40K then resumed to 60K) with real network access to `openbmb/Ultra-FineWeb-L1`:

```
$ python pretrain_hf.py --config configs/pretrain/smoke.json --output-dir /tmp/pretrain-smoke --device cpu
[tokenizer] fitting 252 BPE merges on 40 docs from openbmb/Ultra-FineWeb-L1/CC-MAIN-2025-51 ...
[tokenizer] fit 512-token vocab, saved to /tmp/pretrain-smoke/tokenizer.json
[val] collecting 8 docs from CC-MAIN-2025-47 (disjoint snapshot)...
[val] 377 held-out examples
[model] fresh init, 139,904 params, config=ModelConfig(vocab_size=512, block_size=128, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
[train] budget 40,000 tokens (~78 steps at batch 4 x block 128), device=cpu
[step      10] loss  5.6817 | lr 9.93e-04 | tokens        5,040/40,000 |    1435.2 tok/s
...
[checkpoint] step 10 | val_loss 5.520649087778429 | sample('the cat') -> 'the catevglThe enhkinsyoutim  withliAyoudisddthe e wofofin outcwisywkof '
...
[checkpoint] step 82 | val_loss 4.471759265426836 | sample('the cat') -> 'the cat     a  eingc the \n   toiile   re     \n and  le   '
[done] step 82, tokens_consumed 40,243/40,000, elapsed 7.7s

$ python pretrain_hf.py --config configs/pretrain/smoke.json --output-dir /tmp/pretrain-smoke --device cpu --token-budget 60000
[tokenizer] loaded 512-token vocab from /tmp/pretrain-smoke/tokenizer.json
[resume] step 82, 40,243/60,000 tokens, 22 docs already consumed
[train] budget 60,000 tokens (~117 steps at batch 4 x block 128), device=cpu
...
[checkpoint] step 123 | val_loss 4.3656324519743155 | sample('the cat') -> 'the cat s   h ed .      i     n .   the le      ic  '
[done] step 123, tokens_consumed 60,294/60,000, elapsed 4.6s
```

What this proves: streaming from the real dataset works (both train and val configs), tokenizer
fitting and persistence works, val loss is computed on a genuinely disjoint snapshot, checkpointing
and **resume from a killed/restarted process works exactly as designed** (picked up at step 82,
correct token/doc counts, continued training, val loss kept falling: 4.4718 -> 4.3656), and sample
generation runs at every checkpoint. Val loss falling steadily and samples starting to show
word-like fragments (`the`, spaces, punctuation) rather than uniform noise is the expected shape
at this toy scale/budget -- 140K params on 60K tokens is nowhere near enough to speak English, and
isn't meant to be; it's a mechanism proof, not a quality result. The real quality question is
answered once SSH's `default.json` run lands.

## Stage 2: `babble post-train-from-checkpoint`, proven against a local stand-in

`babble/posttrain.py` already had a stage 2 (`post_train`), but it hardcodes
`babble.tokenizer`'s raw-byte layout end to end (`build_example`, `build_continuation_example`,
and critically `_stack_examples`'s **hardcoded `PAD_ID=256`**, which would silently collide with a
real learned BPE merge id -- BPE merge ids start at 256 too, so padding with a bare `256` under a
BPE tokenizer pads with the *first learned merge*, not a pad token, corrupting every batch without
ever raising). That's the reuse-vs-fork call documented in `babble/posttrain.py`'s new
`post_train_from_checkpoint()` docstring: extended, not forked -- it imports and reuses
`Babbler`/`ModelConfig`/`sequence_loss`, `save_checkpoint`, `pair_split`, `trainable_pairs`,
`corpus_rows`/`split_rows`, `_build_optimizer`, `be_polite`, and the entire promotion-gate/best-val
selection logic verbatim from the existing pipeline. Only the tokenizer-specific example-building
and batch-stacking calls are swapped for the new generic versions
(`babble/subword.py`'s `build_continuation_example`/`text_examples`/`stack_examples`, all added
alongside this work, parameterized over any `Tokenizer` instead of hardcoding byte ids). The
original `post_train()` is completely untouched -- zero risk to the existing local pretrain -> post
-train path or its test suite (612 tests still pass).

Guardrails carried over exactly, all still config-flippable via the same `Settings` fields:
own (lower) learning rate, corpus rehearsal (half of every batch is plain corpus text, so a
fine-tune can't drift the weights off-corpus unopposed), a minimum-pairs floor (`--force`
overrides, same as before), best-**val**-checkpoint selection with early stopping, and a promotion
gate that scores the candidate against the checkpoint it started from on held-out real corpus rows
and **never writes `latest.pt`** if the candidate is worse by more than the noise-band margin.

**New**: `checkpoint_path` and `tokenizer_path` are now explicit inputs (`--checkpoint`/
`--tokenizer` on the CLI: `babble post-train-from-checkpoint --checkpoint <ckpt> --tokenizer
<tok.json> [--force] [--steps N]`), and a mismatched pair raises immediately (`ValueError`,
checked before a single batch is built) rather than silently training garbage:

```python
if model_cfg.vocab_size != tok.vocab_size:
    raise ValueError(f"checkpoint vocab_size {model_cfg.vocab_size} does not match tokenizer "
                      f"vocab_size {tok.vocab_size} ({tokenizer_path}) -- wrong tokenizer.json "
                      f"for this checkpoint")
```

**Proven against a real (if tiny) stand-in** -- the smoke-test checkpoint above, plus a fresh local
data dir seeded with `babble fake-data` (39 correction pairs, 48 corpus rows):

```
$ babble post-train-from-checkpoint \
    --checkpoint /tmp/pretrain-smoke/checkpoints/latest.pt \
    --tokenizer /tmp/pretrain-smoke/tokenizer.json \
    --force --steps 40
train.polite           nice=19 threads=2 cpus=8 device=cpu mkldnn=True
post_from_ckpt.start   pairs=39 train_examples=31 val_examples=8 rehearsal=0.5 ... lr=0.0001 block_size=128 ...
post_from_ckpt.checkpoint step=10 loss=5.1435 val_loss=5.1007 corpus_val=4.9544
post_from_ckpt.checkpoint step=20 loss=4.9195 val_loss=5.0098 corpus_val=4.8741
post_from_ckpt.checkpoint step=30 loss=4.8698 val_loss=4.9522 corpus_val=4.7979
post_from_ckpt.checkpoint step=40 loss=4.8973 val_loss=4.8950 corpus_val=4.7356
post_from_ckpt.done    corpus_val_before=5.0280 corpus_val_after=4.7356 promoted=True
stage 2 (post-train-from-checkpoint): fine-tuned 39 correction pair(s) ... after 4 checkpoint(s)
promoted (corpus val 4.7356 vs supplied checkpoint 5.0280) -> latest.pt
```

Loaded the external checkpoint and tokenizer, trained, evaluated, and gated correctly (this
particular run happened to clear the gate and promote -- corpus val improved, 5.028 -> 4.736 -- but
the run is a mechanism proof, not a quality claim: 39 fake pairs against a 140K-param toy pretrain
proves nothing about voice). Automated regression coverage: `tests/test_posttrain.py` gained three
tests (`test_post_train_from_checkpoint_loads_and_trains_an_external_pretrain`,
`test_post_train_from_checkpoint_rejects_a_mismatched_tokenizer`,
`test_post_train_from_checkpoint_refuses_below_the_min_pairs_floor`) pinning this path the same
way the existing suite pins `post_train()`. `tests/test_subword.py` also gained three tests for the
new generalized `build_continuation_example`/`stack_examples`/`BPETokenizer.to_json`/`from_json`
helpers. Full suite: **612 passed** before this work, **618 passed** after (6 new: 3 in
`test_posttrain.py`, 3 in `test_subword.py` -- no test was removed, weakened, or skipped).

## Promotion: not attempted, correctly

No real HF-pretrained checkpoint exists yet -- SSH has not run the job. There is nothing to promote
and no reason to touch `~/babble-live`. When his checkpoint (`latest.pt` + `tokenizer.json`) comes
back:

1. Back up `~/babble-live/checkpoints` (as `PIPELINE_REVAMP_2026-08-20.md` §10 already did once --
   copy the whole directory, timestamped, before touching anything).
2. `BABBLE_DATA_DIR=~/babble-live/data BABBLE_CHECKPOINT_DIR=<scratch dir, not the live one> babble
   post-train-from-checkpoint --checkpoint <ssh's latest.pt> --tokenizer <ssh's tokenizer.json>
   --force` -- run against a **scratch** checkpoint dir first, not `~/babble-live/checkpoints`
   directly, so a bad run never touches what the bot serves.
3. Compare `corpus_val_before`/`corpus_val_after` from that run (already gated automatically) and
   the bits-per-char conversion above against 3.2655 (trigram) and ~3.72 (transformer best) on the
   live corpus's actual held-out rows.
4. Only if it demonstrably beats what's currently live (same gate logic, same bits-per-char
   comparison, and eyeballed samples that stay Discord-shaped -- lowercase, fragments, slang, not
   clean assistant prose, per the brief's explicit "voice replication is the point" requirement) --
   copy the scratch run's `latest.pt` into `~/babble-live/checkpoints/latest.pt` and sample the
   live install to confirm it actually came up on the new weights.
5. If it does not beat live, leave live exactly as is and say so plainly, same discipline the
   original brief asked for and `post_train`'s own gate already enforces automatically for the
   local-only path.

## Re-running on other hardware (the 96GB card, or anything else)

`pretrain_hf.py --device {auto,cpu,cuda}` autodetects CUDA and needs nothing else changed to run on
a different machine -- including the RTX Pro 6000 96GB mentioned in the original brief, if that
becomes the compute of record instead of (or in addition to) SSH's HF Jobs run. To use it: `python
pretrain_hf.py --config configs/pretrain/big.json --output-dir <path> --device cuda` directly on
that box (no `hf jobs` involved at all when running on hardware you already have shell access to).
With 96GB of VRAM, the real lever worth pulling is **batch size**, not model size -- none of the
three presets come close to saturating that much memory even at `big.json`'s 50.5M params; raising
`batch_size` (and, if throughput data suggests it's worth it, `block_size`) in a copied config
would use the extra memory instead of leaving it idle, but that's a tuning decision for whoever's
running it against real measured throughput, not something to guess at here.
