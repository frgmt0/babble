# Benchmark babble: TPS, TTFT, CPU and RAM on this box
> run: run-20260815-benchmark-babble-tps-ttft-cpu-and-ram-on · branch: beckett/run-benchmark-babble-tps-ttft-cpu-and-ram-on · created: 2026-08-15T23:03:04.585Z

## Goal
ro (user 1151230208783945818) asked, verbatim:

  "what's the TPS of babble? I know it's small but I want a wholistic CPU usage, RAM usage, etc.
   and how that coalesces into TPS, TTFT, and whatnot"

So: a real, measured performance profile of babble on this machine. Numbers, not estimates. The
deliverable is a benchmark that anyone can re-run and get the same shape of answer.

## The machine and the model — measure on these, don't assume

- Box: i7-4790, 4 cores / 8 threads, 32GB RAM, **CPU only, no GPU**, Linux.
- Model: ~3.3M params, byte vocab, 4 layers x 256 dim, block size 512, `max_new_tokens` 96.
- Repo `~/Projects/babble`, current `origin/main` is `a2250af`, which includes the KV-cached
  decode and CPU-first work from PR #7.
- The live bot runs from `~/babble-live` under the systemd user unit `babble-bot`, currently on
  `a2250af`. Its checkpoint is `checkpoints/latest.pt`, with a `base.pt` from stage-1 pretraining.

## What to measure

Inference, per generated sequence, using the **real current checkpoint** and realistic prompts
(pull actual prompts from the corpus, don't invent a synthetic 4-token prompt):

1. **TTFT** — time to first token, ms. Measure it separately from the rest of decode; state
   explicitly what you're counting as t=0 (process already warm, model already loaded — say so).
2. **TPS** — tokens per second during steady-state decode, and separately the **end-to-end**
   tokens/sec for a whole `max_new_tokens=96` generation including TTFT. Those two numbers differ
   and both are worth having.
3. **Cached vs uncached decode** — the KV cache is the headline claim of PR #7, so report TPS and
   TTFT both ways on the same prompts, same seed, and give the actual speedup ratio.
4. **Cold vs warm** — first generation after process start (model load, any lazy init) versus
   steady state. The bot is long-lived so warm is the number that matters day to day, but cold is
   what a restart costs.
5. **Thread scaling** — this is a 4-core/8-thread box, so sweep `torch` thread count (1, 2, 4, 8)
   and report TPS at each. Small models often peak below max threads; if so, that's a concrete
   config finding and worth saying out loud.

Resources, measured while the above runs:

6. **RAM** — RSS of the bot process at idle and peak during generation, and separately the
   model's own parameter footprint (params x dtype) so the gap between "the model" and "the python
   process" is visible. Note torch's own baseline.
7. **CPU** — utilization during decode (% of one core, and across cores), and how it changes with
   the thread sweep.

Training, since ro asked holistically:

8. **Steps/sec and wall time** for a `voice-pass` run on the current corpus (81 rows), plus peak
   RSS and CPU during it. Don't kick off a full base pretrain — measure the stage-2 voice pass,
   which is the one that actually fires in normal operation.

## How to deliver it

- Write the benchmark as a **committed script** in the repo (e.g. `bench/bench_inference.py` or a
  `babble bench` subcommand if that fits the CLI's existing shape better) so these numbers can be
  re-measured after future changes instead of being a one-off.
- Report **medians over multiple runs**, not a single sample, and say how many runs and what the
  spread was. A single timing on a desktop under load is noise.
- Put the results in a `BENCHMARKS.md` in the repo root: the numbers, the exact machine, the
  commit measured, the date, and the command to reproduce. Include the thread-sweep table.
- Where a number is surprising, say why in one line rather than leaving it bare.

## Constraints

- **Do not stop, restart, or disturb the live `babble-bot` unit.** Benchmark in your own worktree
  with its own process. Measuring the live bot's RSS by reading `/proc` is fine; killing it is not.
- Note in the results that the box is shared and may have other load; if you see contention,
  re-run rather than reporting a contended number.
- `babble-train.service` is intentionally disabled — the endless training loop was retired in
  favour of the bot invoking `voice-pass` every 100 corpus rows. Do not re-enable it.
- Don't change model architecture or hyperparameters to make numbers look better. This is
  measurement, not optimization. If you spot an obvious cheap win, write it down as a
  recommendation in BENCHMARKS.md instead of implementing it.
- Do not point freebuff or any ad-funded / free-tier tool at this repo.
- `beckett gh` for anything on GitHub. Never raw `gh`, never raw `git push`.

## Done means

- A committed, re-runnable benchmark script.
- `BENCHMARKS.md` answering all eight items above with real medians, on commit `a2250af`, on this
  box, including the cached-vs-uncached ratio and the thread-scaling table.
- A short summary in the PR body: TTFT, warm TPS, peak RSS, and the one-line answer to "how fast
  is babble actually."

## Checklist
- [ ] (worker fills this in as its FIRST action: concrete, verifiable items)

## Notes
(worker scratch: decisions, blockers, handoff notes)
