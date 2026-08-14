# Validation loop: held-out split + val loss in the feed
> run: run-20260814-validation-loop-held-out-split-val-loss · branch: beckett/run-validation-loop-held-out-split-val-loss · created: 2026-08-14T06:11:42.315Z

## Goal
ro (user 1151230208783945818) asked, about the babble/booper bot: "for booper i think we need a validation loop as well".

Context: babble is a tiny from-scratch char-level model (3.28m params) in kowo-co/babble that learns from Discord reply-corrections. It trains in a continuous loop (BABBLE_STEPS_PER_CYCLE steps, then BABBLE_REST_SECONDS idle) over every consented correction row in data/interactions.jsonl. Right now there are only 10 rows, and training loss has already been driven to ~0.02, which tells us nothing — the model is memorizing the corpus. There is currently no held-out set and no way to tell learning from overfitting.

Build a validation loop in the kowo-co/babble repo.

What it has to do:

1. **Held-out split.** Deterministically hold out a fraction of correction rows from training and use them only for evaluation. Split on a stable hash of the row id, NOT on position or a shuffle, so the same row always lands on the same side of the split as the corpus grows and as the trainer restarts. Fraction configurable via env (something like BABBLE_VAL_FRACTION, default 0.2).
2. **Small-corpus guard.** With a handful of rows a 20% holdout is 2 rows and the split can starve training entirely. Below a configurable minimum corpus size (env, e.g. BABBLE_VAL_MIN_ROWS, default something like 20), skip the holdout, train on everything, and log/report explicitly that validation is disabled and why. Never silently report a meaningless val number.
3. **Eval at checkpoint.** Every time the trainer writes a checkpoint, compute loss over the held-out rows in eval mode (no gradient updates, no optimizer step, no state mutation — the val pass must not train the model) and log it alongside train loss.
4. **Surface it.** Include val loss in the `train.checkpoint` log line and in the Discord training feed post, next to train loss. If validation is disabled per (2), say so instead of printing a number.
5. **Make overfitting visible.** Track the gap between train and val loss over checkpoints and report when val loss has been rising while train loss falls — a plain, honest signal in the log and the feed, e.g. a flag or a short note on the checkpoint post. Keep it simple; do not build early stopping, do not halt training, do not change the learning rate or any training hyperparameter. Reporting only.

Constraints:
- Python, existing dependencies only. Do NOT add new dependencies.
- Do not change the training math, the optimizer, the LR, the batch size, or the checkpoint format in any way that breaks resuming from an existing checkpoint. The live bot is mid-run at step ~18471 and MUST resume cleanly from its existing checkpoints after this lands.
- Do not touch the consent flow, the blocklist/content filter, the HuggingFace export, or the bot's message handling.
- Held-out rows must still respect consent and the blocklist exactly as training rows do — a row bounced by the filter is bounced everywhere, val included.
- Keep the feed's existing throttle (BABBLE_LOG_EVERY_N) and its mention suppression untouched, and keep feed failures best-effort so they can never stall training.
- Document the new env vars in README.md and .env.example, matching the style already there.

Done means: tests green (the suite is at 159 and all of them must still pass), new tests covering the deterministic split (same row → same side across runs), the small-corpus guard, and that the eval pass does not mutate model weights or optimizer state; `train.checkpoint` log lines and feed posts carry val loss (or an explicit "validation disabled, corpus too small" note); and the running trainer resumes from its existing checkpoint without a reset.

## Checklist
- [x] `Settings` gains `val_fraction` (BABBLE_VAL_FRACTION, default 0.2) and `val_min_rows` (BABBLE_VAL_MIN_ROWS, default 20)
- [x] `trainer.py`: deterministic hash-of-row-id split into train/val (`split_rows`), same row -> same side regardless of position/order
- [x] Small-corpus guard: below `val_min_rows`, val disabled, all rows train, reason recorded
- [x] `eval_loss`: eval-mode loss over held-out examples, no grad, no optimizer step, restores `model.training`
- [x] Overfit signal: pure function comparing (train_loss, prev_train_loss, val_loss, prev_val_loss)
- [x] `train.checkpoint` log line carries val_loss/val_enabled/val_disabled_reason/val_rows/overfit_signal
- [x] `TrainingFeed.checkpoint()` carries the same info in the Discord post, backward compatible (no val kwargs -> unchanged post)
- [x] Checkpoint format (`save_checkpoint` payload) untouched -- resuming from existing checkpoint must still work
- [x] README.md + .env.example document BABBLE_VAL_FRACTION / BABBLE_VAL_MIN_ROWS
- [x] New tests: deterministic split, small-corpus guard, eval pass does not mutate weights/optimizer, checkpoint log + feed carry val info
- [x] Full suite green (159 existing + 26 new = 185)
- [x] Resume-from-existing-checkpoint sanity check (test_a_second_run_resumes_where_the_first_stopped, plus new test_a_second_run_with_validation_enabled_still_resumes_cleanly)

## Notes
- `split_rows()` operates on the output of `consented_rows()`, which already applies consent +
  blocklist filtering -- so held-out rows automatically respect both, with no extra plumbing
  needed. Covered by `test_held_out_rows_are_drawn_only_from_already_consented_rows`.
- The split hash salts the row id with a fixed string (`babble-val-split`) via sha256 and takes
  the top 32 bits as a uniform float in [0, 1) -- deterministic across process restarts (unlike
  Python's `hash()`, which is salted per-process) and independent of row order/position.
- `TrainingFeed.checkpoint()`'s new val kwargs all default such that omitting them (as the
  pre-existing `test_discord_feed.py` calls do) reproduces the exact old post -- `val_enabled`
  defaults to `None`, meaning "no validation state supplied," not "disabled."
- `save_checkpoint()` / the checkpoint `.pt` payload were not touched at all -- val loss lives
  only in the log line, the loss.jsonl append was intentionally left alone too, and the feed
  post, never in the resumable state.
- Confirmed pre-existing flaky test `test_killing_the_trainer_mid_run_leaves_a_loadable_checkpoint`
  (SIGKILL timing race, unrelated to this change) is unaffected: diffed `save_checkpoint` byte-for-byte
  against the pre-change version, identical. Fails intermittently (~1/3 runs) on the base commit too.
- Did not touch `stats.py` / `babble curve` / `babble summary` / `loss.jsonl` -- out of the
  ticket's explicit scope (log line + feed post only), and touching the checkpoint-format-adjacent
  loss curve file felt like unnecessary risk for zero required benefit.
