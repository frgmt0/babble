# Reports

Dated, one-off experiment and status reports. Not living documentation — each
records the outcome of a specific investigation or a specific promotion event
at the time it was written. For current reference docs (benchmarks, CPU
inference tuning), see [`BENCHMARKS.md`](../../BENCHMARKS.md) and
[`CPU_INFERENCE.md`](../../CPU_INFERENCE.md) at the repo root.

- [BOOPER_REGRESSION_2026_08_20.md](BOOPER_REGRESSION_2026_08_20.md) (2026-08-20) — root-cause investigation into an apparent quality regression in booper's samples, traced to the corpus-only pretrain pivot rather than "more data made it worse."
- [STORY_V1_PROMOTE.md](STORY_V1_PROMOTE.md) (2026-08-29) — promotion record for typer-org/booper-story-v1 (long-form SFT of Big-Chat) going live on jason's box, with data mix, val curve, sha256s, .env changes, rollback.
- [BIG_CHAT_INT8_PROMOTE.md](BIG_CHAT_INT8_PROMOTE.md) (2026-08-26) — promotion record for ProCreations/Booper-Big-Chat-INT8 going live on jason's box via the new hf serving backend, with sha256s, .env changes, and rollback.
- [CAPACITY_TOKENIZER_REPORT.md](CAPACITY_TOKENIZER_REPORT.md) (2026-08-21) — capacity sweep (shrinking the model) plus a byte-vs-BPE-vs-word tokenizer swap; shrinking didn't help but switching to BPE beat the trigram baseline.
- [HF_PRETRAIN_PIPELINE.md](HF_PRETRAIN_PIPELINE.md) — design and status writeup for `pretrain_hf.py`, the Hugging Face Jobs GPU pretrain script and its Discord post-train follow-on, including model-size/token-budget presets and tokenizer change.
- [PAIR_AUGMENT_REPORT.md](PAIR_AUGMENT_REPORT.md) (2026-08-21) — with/without measurement of LLM-paraphrased correction-pair augmentation; the generator works but augmentation doesn't earn its keep at any tested multiplier, so it stays off by default.
- [PIPELINE_REVAMP_2026-08-20.md](PIPELINE_REVAMP_2026-08-20.md) (2026-08-20) — evidence report for a training-pipeline overhaul (early-stop fix, post-train guardrails) and a corpus-internal synthetic data generator, including the leaked-validation finding behind the guardrails.
- [POST_TRAIN_EXPERIMENT.md](POST_TRAIN_EXPERIMENT.md) (2026-08-18) — before/after measurement of post-training on corrections only; corrections shortened and degraded responses rather than improving them (overfitting).
- [PRETRAIN_PROMOTE_LIVE.md](PRETRAIN_PROMOTE_LIVE.md) (2026-08-22) — promotion record for SSH's HF pretrain checkpoint going live on `beckett`'s box, with backup location, rollback command, and post-promotion verification.
- [PRETRAIN_SYNTHETIC_REPORT_2026-08-20.md](PRETRAIN_SYNTHETIC_REPORT_2026-08-20.md) (2026-08-20) — fresh pretrain on the grown corpus plus a synthetic in-voice correction generator and combined post-train run; synthetic pairs work mechanically but the model still overfits at this scale.
- [SSH_POSTTRAIN_REPORT.md](SSH_POSTTRAIN_REPORT.md) (2026-08-22) — stage-2 CPU post-train run against SSH's HF-hosted pretrain checkpoint and the real Discord corpus, isolated from the live install.
