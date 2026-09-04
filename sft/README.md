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

Live dashboard: put `BABBLE_RUNS_URL=https://booper.frgmt.xyz` and `BABBLE_RUNS_TOKEN=<the worker's RUNS_TOKEN secret>`
in `.env.sft` (gitignored) and every metrics record is also POSTed to `/api/runs/<name>` → https://booper.frgmt.xyz/runs.

Output of a run: `runs/<name>/{train.log,metrics.jsonl,ckpt/,export/}`. `export/` is a drop-in for
`BABBLE_HF_MODEL_DIR` on the live box (same INT8 layout `babble.hfserve` reads; the script round-trips it through
that loader before finishing).
