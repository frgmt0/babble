<<<<<<< ours
# Land the probe work on top of the validation loop
> run: run-20260814-land-the-probe-work-on-top-of-the-valida · branch: beckett/run-land-the-probe-work-on-top-of-the-valida · created: 2026-08-14T06:37:47.801Z

## Goal
This is a MERGE RECONCILIATION job, not a feature job. Two finished, reviewed pieces of work
collided on the same lines and someone has to land them together. Do not add anything new.

## The situation

Repo: kowo-co/babble (clone at ~/Projects/babble).

- `main` is at `ca596b3` — "Validation loop: held-out split + val loss in the feed (#3)".
- Branch `beckett/run-probe-the-real-dataset-full-training-inf` holds ONE commit, `9a2a6a9`,
  "Probe the real dataset + full training info in the channel". It was branched off `7bac62b`,
  which is one commit behind main, and it has already passed its own review (full 174-test suite
  green, 12 new tests).

Cherry-picking `9a2a6a9` onto `main` conflicts in three files: `babble/trainer.py` (7 conflict
hunks, one of them ~170 lines), `babble/discord_feed.py` (2 hunks) and `spec.md` (1 hunk, the
whole file). The conflicts are because BOTH commits edited the same regions — the training loop
in `train()`, the `_checkpoint()` helper and its signature, and the Discord feed's checkpoint
message formatter.

## What each side is trying to do — keep BOTH

**main / validation loop (`ca596b3`)**: splits the consented rows into a train set and a held-out
validation set, computes validation loss, and reports val loss alongside train loss in the feed.

**the probe commit (`9a2a6a9`)**: the checkpoint probe prompt used to be hardcoded to two strings
("hello" / "how are you"), so the training channel looked frozen no matter how much data arrived.
It now rotates deterministically through the real consented dataset prompts, shows the dataset's
expected `chosen` answer next to what the model generated, and adds full drop accounting to the
feed (how many interactions are stored vs how many are actually training vs how many were dropped
by consent or by the blocklist), plus cycle-start/cycle-end info (examples, batch size, learning
rate, steps, seconds).

Neither feature is optional and neither gets simplified away to make the merge easier. The merged
result must have: held-out split + val loss AND rotating real-dataset probes with expected answers
AND the drop accounting.

## Reconciliation decisions you have to make, and how to make them

1. **The probe must rotate over the TRAIN split, not the full row set.** After the validation
   commit, rows are split; probing with a held-out row would be probing something the model never
   saw. Probe from the training split. If the merged code makes it natural to ALSO probe one
   held-out prompt per checkpoint, that is welcome but strictly optional — do not invent it if it
   complicates the merge.
2. **One coherent feed message, not two bolted together.** Both sides added fields to the same
   checkpoint line. The result must read as one compact human-readable Discord message with train
   loss, val loss, the probe prompt, the generated sample and the expected answer — not two
   half-merged formats stacked. Rarely-changing fields (batch size, lr, split sizes) belong on the
   cycle-start line rather than repeated every checkpoint.
3. **`_checkpoint()` signature**: both sides added parameters. Merge them into one signature; do
   not keep two variants or a `**kwargs` escape hatch.
4. **`spec.md` conflicted whole-file.** It is the run's own scratch checklist, not product docs.
   Take whatever leaves an accurate description of the merged state; do not agonize over it.

## Constraints

- Do NOT change training behaviour: row selection, consent logic, blocklist logic, the split
  itself, the loss/optimizer path, duty cycle, nice level, checkpoint cadence. This merge changes
  no semantics that either side didn't already ship.
- Do NOT weaken filtering. Probe text, expected answers and samples all still go through the
  blocklist withholding and `neuter_sample` before they reach Discord.
- Python, existing repo style, no new dependencies.
- Both sides shipped tests (`tests/test_trainer.py`, `tests/test_discord_feed.py`) and both test
  files also conflict at the working-tree level. Keep BOTH sets of tests and make both pass; if a
  test asserts on an exact feed string that the merged format changed, update the assertion to the
  merged format — never delete a test to make it green.

## Done means

- A branch off current `main` containing the reconciled work, with no git conflict markers
  anywhere (a recursive grep for the merge marker sequences returns nothing).
- The full test suite passes, including both sides' tests.
- A short manual sanity check that the merged checkpoint path actually runs and produces one
  well-formed feed message containing both val loss and a real rotating dataset probe with its
  expected answer.

## Checklist
- [x] Cherry-pick `9a2a6a9` (probe) onto current `main` (`ca596b3`, validation loop) and resolve
      every conflict. Test files (`tests/test_trainer.py`, `tests/test_discord_feed.py`) applied
      cleanly because validation shipped its tests in a separate `tests/test_validation.py`.
- [x] `babble/trainer.py`: keep BOTH the validation helpers (`split_rows`, `eval_loss`,
      `overfit_signal`, `Split`) and the probe helpers (`dataset_stats`, `distinct_prompts`,
      `probe_prompt`, `DatasetStats`).
- [x] `_checkpoint()`: one merged signature — `rows: list[Interaction]` + `probe_index` (probe)
      AND the keyword-only `val_*` params (validation). Returns `(mean_loss, val_loss)`.
- [x] The probe rotates over the TRAIN split: `train()` passes `split.train` into `_checkpoint`
      (decision #1). Held-out rows are never probed.
- [x] One coherent checkpoint feed message: train-loss line, optional val line, one
      `prompt → sample` line, optional `expected:` line. Rarely-changing fields (stored/trained/
      dropped, examples, batch, lr) live on the cycle-start line, not repeated per checkpoint.
- [x] Filtering intact: probe prompt, expected answer, and sample all pass through blocklist
      withholding in `_checkpoint` and `neuter_sample` in the feed before reaching Discord.
- [x] Updated the one direct `_checkpoint` call in `test_validation.py` to the merged signature
      (adds `probe_index`, `rows=[]`); no test deleted, no assertion weakened.
- [x] No conflict markers anywhere; full suite green (200 passed) — both sides' tests included.
- [x] Manual sanity check: merged checkpoint path emits one well-formed feed message carrying
      both val loss and a real rotating dataset probe with its expected answer.

## Notes
- Merge only; no training semantics changed. Row selection, consent, blocklist, the split, the
  loss/optimizer path, duty cycle, nice level, and checkpoint cadence are all as either side
  shipped them.
- The per-checkpoint `{rows} rows` count now reflects the train-split size passed into
  `_checkpoint` (`len(split.train)`), consistent with the `examples` count on the cycle-start
  line. The full stored/trained/dropped accounting lives on the cycle-start line.
=======
# Auto-publish dataset to HF every 20 checkpoints
> run: run-20260814-auto-publish-dataset-to-hf-every-20-chec · branch: beckett/run-auto-publish-dataset-to-hf-every-20-chec · created: 2026-08-14T06:36:15.365Z

## Goal
ro (user 1151230208783945818) asked, in the babble channel: how long until Hugging Face reflects the up-to-date data in the dataset? Right now the answer is "never, unless I run the export by hand" — nothing pushes automatically. I offered him a cadence (every 20 checkpoints, or every 15 minutes) and he picked: "yeah every 20 checkpoints is fine".

Build that: the trainer auto-publishes the corrections dataset to Hugging Face every 20 checkpoints.

Repo: kowo-co/babble (this repo). Relevant existing code: `babble/export_hf.py` is the existing Hugging Face export path, `babble/cli.py` has the commands, `babble/core.py` has the training loop and checkpointing. The dataset lives at https://huggingface.co/datasets/kowo-co/babble-corrections under the kowo-co org (free plan). The trainer posts a feed of checkpoints into a Discord channel already.

What to build:

1. Auto-publish hook in the training loop: every 20 checkpoints (counted by checkpoints written, not steps), export the corrections dataset and push it to kowo-co/babble-corrections using the existing export path. Reuse `export_hf.py` — do not write a second exporter.
2. The cadence is configurable, defaulting to 20. Put it in the existing config module (`babble/config.py`) alongside the other knobs, with an env override, and make 0/None mean "off". Do not change any other default.
3. The consent gate and the blocklist/redaction filter that already guard which rows are exportable MUST still run on every automatic publish, exactly as they do on the manual export. This is the whole safety story for going automatic — ro was told the tradeoff is that a bad row becomes public within minutes, so the existing filters must not be bypassed or weakened. Do not add a new filter, just make absolutely sure the automatic path goes through the same one.
4. Failure isolation: an HF push that fails (network, rate limit, bad token, HF down) must NOT kill the training run. Log the failure, post it in the trainer feed so it's visible, and keep training. The next scheduled publish just tries again. No retry storms — one attempt per scheduled publish is fine.
5. Report it in the feed: when an auto-publish happens, the feed line says so, with the row count that went live and the dataset URL. Keep it to one short line, matching how the existing feed lines are written.
6. Skip a publish if nothing changed since the last one (same row count and same content hash) so we're not pushing identical commits to HF every 20 checkpoints.

Constraints:
- Python, matching the existing style in this repo. No new dependencies — the HF client is already a dependency.
- Don't touch the training math, the validation/held-out split logic, or the model config.
- Don't change the manual export command's behavior; the automatic path is additive.
- The HF token comes from the environment the same way the existing export gets it. Never print or log the token.

Done means:
- Training for 20 checkpoints with the feature on results in a real push to kowo-co/babble-corrections, and the trainer feed shows one line saying it published with the row count.
- Consent-gated / blocklisted rows are provably excluded from the auto-published dataset — there is a test that asserts this specifically.
- A simulated HF failure at publish time leaves training running and produces a visible error line, with a test covering it.
- `pytest` is green (185 tests currently pass — keep them all passing) and the new behavior has tests.
- README documents the cadence setting and how to turn it off.

Ceiling: this is one feature in the trainer loop plus config and tests. Don't refactor the exporter, the feed, or the training loop beyond what's needed to hook this in.

## Checklist
- [ ] (worker fills this in as its FIRST action: concrete, verifiable items)

## Notes
(worker scratch: decisions, blockers, handoff notes)
>>>>>>> theirs
