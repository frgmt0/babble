# sft/ — long-form SFT for Booper-Big-Chat on a laptop

Everything runs from the repo root with `.venv` set up (`uv venv --python 3.12 && uv pip install -e ".[dev,hf]" datasets`).

```bash
sft/train.sh story-v1 --tokens 30e6      # detached (nohup + caffeinate); one run per machine
sft/monitor.sh story-v1                  # status line + follow train.log   (--status for one-shot)
sft/stop.sh                              # stop the trainer; sft/train.sh story-v1 --resume continues
sft/train.sh smoke --smoke               # 12-step end-to-end check first
.venv/bin/python sft/sft_longform.py --name story-v1 --export                     # re-pack ckpt -> export/ (INT8)
.venv/bin/python sft/sft_longform.py --name story-v1 --export --push <your-hf-namespace>/booper-story-v1   # ...and publish it
```

Live dashboard: put `BABBLE_RUNS_URL=https://booper.frgmt.xyz` and `BABBLE_RUNS_TOKEN=<the worker's RUNS_TOKEN secret>`
in `.env.sft` (gitignored) and every metrics record is also POSTed to `/api/runs/<name>` → https://booper.frgmt.xyz/runs.

Output of a run: `runs/<name>/{train.log,metrics.jsonl,ckpt/,export/}`. `export/` is a drop-in for
`BABBLE_HF_MODEL_DIR` on the live box (same INT8 layout `babble.hfserve` reads; the script round-trips it through
that loader before finishing).
