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
