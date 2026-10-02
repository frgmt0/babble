# qa-v1 promoted to live (jason's box), 2026-10-02

## What
`qa-v1` is a knowledge and Q&A SFT that continues from `longctx-v1`. It was trained with `configs/sft/qa-mac.json` (PR #52) on the M2 Pro.
- Data: 3,601 steps and about 118M nominal tokens. 53% of the mix is QA/knowledge (nq_open, dolly, oasst2, synthetic arithmetic, Simple Wikipedia, smol-short, persona). The other 47% is rehearsal (discord, ultrachat, no_robots, writingprompts).
- Run: launched 2026-10-01 10:53 PT and finished 2026-10-02 00:41 PT.
- Speed: it ran at duty 0.5 until step 50, then at duty 1.0 with `SFT_QOS=none`.

## Gate (training-side)
The best checkpoint was the final step. The source gate passed: val went from 1.5977 to 1.5275.

| source | step 0 | step 3601 | Δ |
| --- | ---: | ---: | ---: |
| discord (guarded) | 2.452 | 2.460 | +0.008 |
| discord_multiturn (guarded) | 2.440 | 2.445 | +0.005 |
| ultrachat (guarded) | 1.415 | 1.365 | −0.050 |
| ultrachat_multiturn (guarded) | 1.398 | 1.344 | −0.054 |
| nq | 3.214 | 2.082 | −1.132 |
| arith | 3.010 | 1.304 | −1.706 |
| persona | 2.456 | 1.734 | −0.722 |
| wiki | 1.895 | 1.666 | −0.229 |
| dolly | 1.725 | 1.641 | −0.084 |
| oasst | 1.604 | 1.546 | −0.058 |

## Held-out QA eval (`bench/qa`, greedy, 250 items)
| model | overall | fact | math | definition | about_bot | commonsense | followup | echo | clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longctx-v1 (previous live) | 13.3% | 7.1 | 2.5 | 6.7 | 4.0 | 28.6 | 44.0 | 24.0% | 65.6% |
| **qa-v1** | **24.0%** | 10.0 | 2.5 | 63.3 | 32.0 | 31.4 | 32.0 | 6.4% | 92.0% |

Full results are in `bench/qa/results/qa-v1.{json,md}`.

### What changed, honestly
- **It answers instead of echoing.** Echo went from 24% to 6.4%, and "idk"-style non-answers from 10.4% to 1.6%. Definitions jumped from 6.7% to 63.3%.
- **The shape of an answer is learned, the facts mostly aren't.** It replies "it's X" or "pretty sure it's X", but X is often wrong ("capital of france" → "it's France", "minutes in an hour" → "it's 5"). That is the expected ceiling for about 45M active params.
- **Math has the right format and the wrong numbers** ("what's 7+5" → "that'd be 9"), even though the arith val loss halved. Getting correct arithmetic would need much more arithmetic data, or digit-level formatting.
- **Identity is better but leaky.** It now says "no, i'm a bot. i'm just a small model". Some "who are you" prompts still get nq-style name answers ("pretty sure it's Kristen Barton"). Persona was only about 0.8% of the mix in practice.
- **Followup dropped from 44% to 32%.** This is a real regression on multi-turn recall, and the next run should weight multi-turn QA.
- **Caveat on two categories.** The eval's math and about_bot items are close in kind to the synthetic arith and persona training data, so the fact and definition numbers are the fairer generalisation signal.

## Promotion
- Copied `runs/qa-v1/export` to `~/babble-live/artifacts/hf-booper-qa-v1`.
- `.env`: `BABBLE_HF_MODEL_DIR=/home/jason/babble-live/artifacts/hf-booper-qa-v1`. That was the only change: the prompt format, history turns and budget are identical to longctx-v1.
- Backup: `~/babble-live/backups/qa-v1-promote-2026-10-02/` (`env.before` and sha256s of both models). The longctx-v1 directory is untouched.
- `babble-bot` was restarted. `model.load` shows `model_dir=.../hf-booper-qa-v1`, `runtime=native`, and `bot.ready` arrived at 2026-10-02T08:21:23Z.
- `BABBLE_NATIVE_SPEC_TABLE` still points at `spec-ngram-longctx-v1.pt`. Speculation is verified, so a stale table costs only acceptance rate, not output. It should be rebuilt for qa-v1 with `bench/extreme/spec_ngram.py`.

| file | sha256 |
| --- | --- |
| qa-v1 `model-int8.safetensors` | `35fc139c6cb824fe6f9646b2c1a8df3aacbcbc8e304b9eaeeda233d772c8c6b7` |
| qa-v1 `tokenizer.json` (identical to longctx-v1) | `b1e1233481d5b2c3637fe8bcef81ec696fec835341680f0a0a72449bd699717a` |
| qa-v1 `config.json` | `f399dc36fac72c955d58f404f7e5be9bd1522a5575438bf9948fabae42694f5c` |

## Rollback
```bash
sed -i 's|^BABBLE_HF_MODEL_DIR=.*|BABBLE_HF_MODEL_DIR=/home/jason/babble-live/artifacts/hf-booper-longctx-v1|' ~/babble-live/.env
systemctl --user restart babble-bot
```

## Rolled back, 2026-10-02 15:51 UTC
Users reported that qa-v1 lost longctx-v1's personality and said they much preferred the old one. Live was rolled back to `hf-booper-longctx-v1` using the rollback steps above. `model.load` confirmed `model_dir=.../hf-booper-longctx-v1`, and `bot.ready` arrived at 15:51:41Z. The qa-v1 `.env` is kept as `backups/qa-v1-promote-2026-10-02/env.qa-v1`, and the model directory stays on disk.

**Lesson:** a gain on the QA eval plus a passing discord val-loss guard did not protect the voice. A guard on loss over Discord targets cannot see a register shift on *new* prompts. Next: curate knowledge data in longctx-v1's register, and gate on a voice fingerprint.
