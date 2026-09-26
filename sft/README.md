# sft/ — long-form SFT for Booper-Big-Chat on a laptop

Everything runs from the repo root with `.venv` set up (`uv venv --python 3.12 && uv pip install -e ".[dev,hf]" datasets`).

```bash
sft/train.sh story-v1 --tokens 30e6      # detached (nohup + caffeinate); one run per machine
sft/train.sh story-v2 --base runs/story-v1/export --tokens 40e6 \
    --mix-story 0.25 --mix-wp 0.25 --mix-norobots 0.10 --repeat-norobots 2 --mix-smoltalk 0.10 --mix-discord 0.30   # v2 mix, continue from v1
sft/monitor.sh story-v1                  # status line + follow train.log   (--status for one-shot)
sft/stop.sh                              # stop the trainer; sft/train.sh story-v1 --resume continues
sft/train.sh smoke --smoke               # 12-step end-to-end check first
.venv/bin/python sft/sft_longform.py --name story-v1 --export                     # re-pack ckpt -> export/ (INT8)
.venv/bin/python sft/sft_longform.py --name story-v1 --export --push <your-hf-namespace>/booper-story-v1   # ...and publish it
```

## Multi-turn Mac run

The checked-in `configs/sft/multiturn-mac.json` preset continues from Story-v2
with 55% Discord examples, 12M input tokens, a lower `2e-5` learning rate, and
three completed exchanges of history. Each assistant turn in a Discord row is
a separate response-only target. All targets from one conversation are kept on
the same side of the split; repeated `no_robots` rows are added only after the
split.

```bash
sft/train.sh multiturn-smoke --base runs/story-v2/export --config configs/sft/multiturn-mac.json --smoke
sft/train.sh multiturn-v1 --base runs/story-v2/export --config configs/sft/multiturn-mac.json
```

Validation is reported separately for every source. A checkpoint becomes the
export candidate only when its aggregate validation loss improves on Story-v2
and no source regresses by more than `0.05` nats. Held-out multi-turn loss must
also improve. If no checkpoint clears these conditions, the run keeps its
resumable checkpoint but creates no export. The
best passing checkpoint, rather than the final step, is exported.

The export records `babble_prompt_format=role_transcript_v1`,
`babble_history_turns=3`, and `babble_prompt_budget=512` in `config.json`.
Promotion must keep Story-v2 available for rollback and activate the matching
runtime contract together with the new model directory:

```bash
BABBLE_CONVERSATION_CONTEXT=1
BABBLE_CONVERSATION_MAX_TURNS=3
BABBLE_CONVERSATION_MAX_TOKENS=512
```

The validation report compares candidate `*_single` role-transcript prompts
against Story-v2's raw-prompt `*_legacy` loss on the exact same retained
single-turn targets. Examples that do not fit both tokenized forms are removed
from both views. It also reports a separate `*_multiturn` view and applies the
same regression ceiling. These paired views are part of the export gate;
aggregate validation alone is not evidence that old behavior was preserved.

At Story-v2's observed 2.6k input tokens/second, 12M tokens is about 77 minutes
of gradient work. Source-specific evaluation and longer histories make roughly
1.5-2 hours a realistic M2 Pro wall-clock estimate.

## Long-context run (`longctx-mac.json`)

Continues from `runs/multiturn-v1/export` (the live model). Changes from multi-turn v1:

- **Context**: `seq_len 2048`, `prompt_budget 1536`, `history_turns 8`. The export's `config.json` records
  `babble_history_turns=8` / `babble_prompt_budget=1536`; serving it needs the matching
  `BABBLE_CONVERSATION_MAX_TURNS=8`, `BABBLE_CONVERSATION_MAX_TOKENS=1536` and `BABBLE_MAX_NEW_TOKENS<=509`.
  A long response now trims the oldest history turns to fit the sequence instead of being skipped.
- **Data**: `HuggingFaceH4/ultrachat_200k` (MIT, pinned) and every assistant turn of SmolTalk's multi-turn subsets
  (`--smoltalk-multiturn`), each turn a response-only target grouped per conversation for the val split.
  Discord-Dialogues stays the largest share (46%); TinyStories/WritingPrompts/no_robots stay in as rehearsal.
- **GIF tags** (`--gif-tags`): the model learns to emit `[gif: <2-5 lowercase search words>]` as the whole reply or
  at its end (contract shared with the bot). Gif URLs (tenor/giphy/`*.gif`) and Discord attachment filenames
  (`SipsBubble.gif`) are rewritten into tags from their slug; slug-less gif URLs are dropped, and targets still
  holding any raw URL are not trained on. Discord-Dialogues had its URLs stripped upstream, so real sends are rare;
  a budgeted synthetic slice turns short reactions ("lmao", "bruh", "no way") into tags, hard-capped by
  `--gif-synth-max-frac` (2.5% of Discord targets). Counts and examples are logged at data build (`gif:` lines).
- **Throttle**: `--duty-cycle 0.5` sleeps as long as each optimizer step computed; `--pause-on-battery` idles while
  `pmset` reports battery power. `sft/train.sh` launches under `taskpolicy -c utility` (`SFT_QOS=background|none`
  to change). `taskpolicy -b` was measured and rejected: it cut MPS throughput from ~2000 to ~750 tok/s, so the
  gentleness comes from the duty cycle instead. Metrics carry `duty_cycle`, `tok_s` (wall), `tok_s_active`
  (compute only) and `idle_s`. `tokens`/`tok_s` count real (non-padding) tokens; `--tokens` is nominal
  (`steps = tokens / (tokens_per_batch * accum)`).
- **Memory on a 16 GB Mac** (all measured with `vmmap`; the first launch hit a 19 GB footprint and 9 GB of swap):
  - `--expert-bucket 128`: on MPS the HF eager Mixtral experts loop, and `grouped_mm` too, leak about 35 MB of
    CPU heap per micro-batch whenever routing changes, because MPS caches a graph per per-expert token count.
    Padding each expert's token count to a multiple of 128 is mathematically identical, just as fast, and flat.
  - `--pad-multiple 128 --fixed-rows`: a small fixed set of batch shapes. `--mps-high-watermark 0.6` caps the MPS
    pool at about 7 GiB; at 0.45 a 2048-token micro-batch OOMs. `--grad-checkpoint` exists but costs about 55% of
    throughput, so the preset leaves it off. Steady footprint is about 7.5 GB.
- **Resume**: the tokenized split is cached in `runs/<name>/data-cache.pkl`, and `--resume` skips the batches the
  checkpoint already consumed, so a reboot mid-run costs minutes, not a data rebuild or replayed data.

```bash
sft/train.sh longctx-smoke --base runs/multiturn-v1/export --config configs/sft/longctx-mac.json --smoke
sft/train.sh longctx-v1 --base runs/multiturn-v1/export --config configs/sft/longctx-mac.json
sft/stop.sh    # then, to continue (same flags + --resume):
sft/train.sh longctx-v1 --base runs/multiturn-v1/export --config configs/sft/longctx-mac.json --resume
```

Sizing (M2 Pro 16 GB, measured on the launched run): 16 micro-batches of 2048 tokens per step, about 23.9k real
tokens per step, 25.2 s per step wall at duty 0.5 (about 950 tok/s wall, 1875 active). Eval covers about 2.5k val
views, which is 7.3 min of compute and about 15 min wall with the duty nap, every 300 steps. `tokens 425e6` gives
12,969 steps: 12,969 × 25.2 s + 43 evals × ~890 s + checkpoints ≈ 368k s ≈ 4.3 days, covering about 0.87 of an
epoch (615k train targets, mean 579 tokens). Lid-closed sleep or battery pauses push the finish out
(`idle_s` in metrics).

Live dashboard: put `BABBLE_RUNS_URL=https://booper.frgmt.xyz` and `BABBLE_RUNS_TOKEN=<the worker's RUNS_TOKEN secret>`
in `.env.sft` (gitignored) and every metrics record is also POSTed to `/api/runs/<name>` → https://booper.frgmt.xyz/runs.

Output of a run: `runs/<name>/{train.log,metrics.jsonl,ckpt/,export/}`. `export/` is a drop-in for
`BABBLE_HF_MODEL_DIR` on the live box (same INT8 layout `babble.hfserve` reads; the script round-trips it through
that loader before finishing).
