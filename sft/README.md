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

## Q&A / knowledge run (`qa-mac.json`)

Continues from `runs/longctx-v1/export` (the live model). longctx-v1 chats fine but treats a question as something
to echo ("what's 2+2" -> "2+2"): almost none of its data pairs a short question with a short correct answer. qa-v1
keeps every longctx setting (seq 2048, prompt budget 1536, 8 history turns, gif tags, the memory settings, duty 0.5,
pause on battery) and changes only the data mix, the LR and the gate.

**Sources.** Every new source has a `--mix-*` fraction, a pinned `--*-revision` in the preset, response-only targets,
its own grouped val split and its own val entry (plus the usual `_single`/`_legacy`/`_multiturn` views). They are
appended after `ultrachat`, so older presets keep their split seeds and data signatures (`_data_extras` records qa
settings only when a qa source is active).

| source | flag | licence | what goes in |
| --- | --- | --- | --- |
| `databricks/databricks-dolly-15k` | `--mix-dolly`, `--repeat-dolly 2` | CC-BY-SA-3.0 | open_qa, general_qa, classification, brainstorming, plus closed_qa / information_extraction / summarization **with the passage in the user turn** (13.4k usable rows). closed_qa answers come from the passage, so stripping it would teach confident recall of facts a 45M-active model cannot know. creative_writing is left out because story rehearsal already covers it. |
| `OpenAssistant/oasst2` | `--mix-oasst`, `--repeat-oasst 2` | Apache-2.0 | English trees rebuilt from the message table. At each prompter node the best-ranked reply is the target, with its true root path as history. Lower-ranked replies that passed review can appear as history, never as targets. Deleted, review-failed and synthetic messages are dropped. 4.5k trees give 10.6k targets, 6.5k of them with history. |
| `HuggingFaceTB/smoltalk` smol-magpie-ultra | `--mix-smol-short` | Apache-2.0 | First answers of 900 chars or less, read from the **last three** train shards. longctx-v1 only reached the head of shard 0, so the two sets are disjoint by construction. Coding, role-play, editing, creative and data-analysis rows are skipped, leaving 8.5k. everyday-conversations was fully used by longctx and stays in only through the existing `--mix-smoltalk` rehearsal. |
| `google-research-datasets/nq_open` | `--mix-nq` | CC-BY-SA-3.0 | Train split only, 85k usable. Each answer span is templated into a short casual reply ("that'd be New Zealand", "in 1950", "Matthew Broderick", "pretty sure it's ..."). Template pools depend on the question type, "in X" is only used for years, and both template and prompt shape are picked deterministically per item. |
| synthetic arithmetic | `--mix-arith` | n/a (generated) | + − × ÷ on small numbers, with division always exact. Many phrasings ("whats 7+5", "what is 9 times 3", "subtract 4 from 12?", number words). Replies are short and always correct ("12", "that's 12", "7 + 5 = 12"). The stream is deterministic from `--seed`. A problem is its own split group, so a val problem never shows up in train under another phrasing. |
| `sft/data/booper_persona.jsonl` | `--mix-persona`, `--repeat-persona 3` | project-authored | 155 hand-written identity pairs in booper's lowercase voice: name, bot vs human, not chatgpt/claude/siri, "people on this server built me" (no invented names), what it can and can't do, memory, and the real `!babble` consent commands. Each pair adds 3-4 surface variants. A normalized question is one split group. The source is small, so it ends up about 0.8% of examples, not the nominal 2%. |
| `wikimedia/wikipedia` `20231101.simple` | `--mix-wiki` | CC-BY-SA-3.0 / GFDL | Simple English lead paragraphs, 1-3 sentences, framed as "tell me about X". List, disambiguation and year pages are skipped. |

`mandarjoshi/trivia_qa` was considered and left out: its HF licence is `unknown`, and nq_open covers the same need
under CC-BY-SA. Targets that claim another assistant's identity ("I am Open Assistant", "as an AI language model",
"ChatGPT") are dropped from oasst, dolly and smol-short so they do not fight the persona.

**Mix** (fractions of `examples=232000`): QA/knowledge 53%. nq 15, dolly 10, oasst 10 (capped by supply at about
9.1%), arith 5, wiki 4, smoltalk 4, smol-short 3, persona 2 (about 0.8% in practice). Rehearsal 47%. discord 30,
ultrachat 12, no_robots 3 (x2), writingprompts 2. QA items are short (nq 38 tokens, arith 20, persona 40, wiki 75 on
average) while ultrachat/smoltalk average about 1.2k, so QA is about half the *examples* but well under a quarter of
the *tokens*. That trade is deliberate: rehearsal is what holds the voice. No source repeats more than 3x.

**LR 2e-5, warmup 100, cosine to 10%.** longctx used 1.5e-5 over 13k steps. This run is 3.6x shorter and has to move
a behaviour the base doesn't have (answering), so it uses the multiturn-v1 LR, which already proved safe on this model
family. It stays well under the 4e-5 story runs because rehearsal regression is the main risk. If the first eval
(step 400) shows `discord`/`ultrachat` already near the 0.05 ceiling, restart at 1.5e-5.

**Gate.** The export rule is unchanged: aggregate val must beat the base, and no source may regress by more than
0.05 nats (`*_role` and `*_migration`). Two changes:
- `--guard-sources discord ultrachat` (`--guard-max-regression 0.05`): the rehearsal sources must be **present**
  in val (a missing guard fails the gate instead of silently passing it), and both their role and multi-turn views
  must stay within the ceiling. These show up as `discord_guard`, `discord_multiturn_guard`, etc. in
  `source_regression`.
- `--multiturn-must-improve oasst`: longctx required every `*_multiturn` view to strictly improve, because
  multi-turn was that run's objective. Here only oasst, the new multi-turn QA source, must improve. The other
  `*_multiturn` views (discord, ultrachat, smoltalk) are held to the 0.05 ceiling. Requiring strict improvement on
  conversations the base was already trained on would very likely gate a run whose goal is QA.

Caveat: the rehearsal val groups for discord/ultrachat/smoltalk are drawn from the same streams longctx-v1 trained
on, so some may be seen data for the base. For those sources the guard measures retention, not generalization. All
the new sources' val sets are unseen by the base.

**Sizing** (measured on a 10% build, then scaled): ~23.1k real tokens per step at 0.82 fill (longctx: 23.9k), and
~64 examples per step, so `examples 232000` is about one epoch. Taking longctx's measured 25.2 s/step wall at duty
0.5: `tokens 118e6` -> 3,601 steps x 25.2 s = 90.7k s. Add 11 evals (3.6k views, 1.2M tokens each, ~750 s) and
samples/checkpoints, and the total is about 100k s ≈ **28 h** wall. Battery pauses and sleep add to that.

**Disk.** Every source streams (`streaming=True`), so the HF datasets cache barely grows. The only big local file is
`runs/<name>/data-cache.pkl`, about 0.2 GB of uint16 token arrays. Checkpoints (`ckpt/` with optimizer, `best/`) are
about 1.8 GB, plus the export at 165 MB.

```bash
sft/train.sh qa-smoke --base runs/longctx-v1/export --config configs/sft/qa-mac.json --smoke
sft/train.sh qa-v1 --base runs/longctx-v1/export --config configs/sft/qa-mac.json
sft/stop.sh    # then, to continue (same flags + --resume):
sft/train.sh qa-v1 --base runs/longctx-v1/export --config configs/sft/qa-mac.json --resume
```

At build time, `log_examples: 3` logs three rendered train examples per source: the model input and the exact
loss-bearing target (`toks[n_prompt:]`, i.e. the response plus `<eos>`).

**Promotion contract.** The prompt contract is the same as longctx-v1. The export records
`babble_prompt_format=role_transcript_v1`, `babble_history_turns=8` and `babble_prompt_budget=1536`, so the live
conversation env (`BABBLE_CONVERSATION_CONTEXT=1`, `..._MAX_TURNS=8`, `..._MAX_TOKENS=1536`,
`BABBLE_MAX_NEW_TOKENS=509`, `BABBLE_SERVE_LAYOUT=pair`) needs **no change**. Only `BABBLE_HF_MODEL_DIR` moves to the
new export directory. The native runtime (`BABBLE_HF_RUNTIME=native`) reads the same `model-int8.safetensors`. One
model-specific artifact is affected: live sets `BABBLE_NATIVE_SPEC_TABLE=.../spec-ngram-longctx-v1.pt`, an n-gram
draft table partly built from longctx-v1's own samples. The model verifies every draft, so the old table changes
speed (acceptance rate), not output. Rebuild it for qa-v1 with `bench/extreme/spec_ngram.py` as part of promotion.
Keep longctx-v1 in place for rollback.

Live dashboard: put `BABBLE_RUNS_URL=https://booper.frgmt.xyz` and `BABBLE_RUNS_TOKEN=<the worker's RUNS_TOKEN secret>`
in `.env.sft` (gitignored) and every metrics record is also POSTed to `/api/runs/<name>` → https://booper.frgmt.xyz/runs.

Output of a run: `runs/<name>/{train.log,metrics.jsonl,ckpt/,export/}`. `export/` is a drop-in for
`BABBLE_HF_MODEL_DIR` on the live box (same INT8 layout `babble.hfserve` reads; the script round-trips it through
that loader before finishing).
