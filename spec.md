# Make the corpus pivot work on the live data
> run: run-20260814-make-the-corpus-pivot-work-on-the-live-d · branch: beckett/run-make-the-corpus-pivot-work-on-the-live-d · created: 2026-08-14T21:31:39.170Z

## Goal
The unlabelled-corpus pivot (`2a9860c`, now on `kowo-co/babble` main) is correct in tests and
BROKEN against real live data. I deployed it to the live bot, watched it go inert, and rolled the
live checkout back to `de65544`. Main still carries the pivot; the live box does not. Your job is to
make the pivot survive contact with the existing data so it can actually ship.

## What I observed on the live box

Live install is `/home/beckett/babble-live` (its own clone + `.venv`, systemd units `babble-bot`
and `babble-train`). Its data dir has `interactions.jsonl` (29 stored correction rows, 30 trainable
including the 👍), `consent.json` (5 users, all `{"decision": "granted", "notice_version": 1}`),
`exchanges.json`, and 5 checkpoints at step ~113,000.

On `de65544` (pre-pivot) `babble summary` reports:

    checkpoints 5 · corrections 29 · 👍 1 · trainable rows 30
    people opted in 5 of 5 asked

and the trainer runs normally: `train.cycle.start ... rows=30 examples=24 ... dropped_consent=0`.

On `2a9860c` (the pivot), same data dir, same venv, full suite green (356 passed), `babble summary`
reports:

    corpus 0 rows (0 training, 0 chars) · checkpoints 5
    corrections 29 · 👍 1 · trainable pairs 30
    people opted in 0 of 5 asked

and the trainer immediately goes idle:

    train.idle    reason=no_consented_rows rows=0

So on real data the pivot trains on nothing at all. Two distinct defects:

1. **No migration.** 29 existing correction rows never become corpus rows. `babble/corpus.py` has
   `CorpusRow`/`make_corpus_id`/`append`/`all`, but nothing walks the existing `interactions.jsonl`
   into it. The PR claimed a `capture→backfill→train` path was verified end-to-end — whatever that
   backfill was, it does not run against an install that already has pair data on disk.
2. **Consent stops resolving.** The same `consent.json` that reads as "5 of 5 opted in" before the
   pivot reads as "0 of 5" after it, and the trainer's own gate agrees (`no_consented_rows`).
   Nothing in that file changed between the two runs — I only changed the checked-out commit. Find
   out why the pivot's consent lookup misses rows the old one matched; my guess is the corpus rows
   are keyed by an author identifier that is pseudonymised or salted differently from the one
   `consent.json` is keyed by, or `notice_version` 1 no longer satisfies a bumped expected version.
   Prove it rather than guessing.

## The job

1. Fix the consent resolution first — it is the more dangerous of the two. A user who granted
   consent under the pair model must still read as granted under the corpus model, and a user who
   never granted must still read as not granted. Get this wrong in the permissive direction and we
   are training on text nobody agreed to, so add a test for BOTH directions.
2. Migrate the existing data. Every stored correction row that a consented user contributed should
   become corpus text under the new model. Decide and state whether that is a one-shot migration
   run at startup, an explicit CLI command, or a lazy read-through, and make it idempotent —
   running it twice must not duplicate rows (`make_corpus_id` looks like it already gives you the
   dedupe key). Rows from users who did not consent must NOT migrate.
3. Prove it against a copy of the real data, not just fixtures. A snapshot of the live data dir as
   it is right now is at `/home/beckett/babble-live/data.bak-preppivot-20260814` — copy it, run the
   migration against the copy, and report the actual before/after numbers in the PR: rows in,
   corpus rows out, consented vs dropped. "Tests pass" is not evidence for this one; the tests
   already passed while the live box sat idle.
4. Keep the full suite green and add regression tests: an install that already has
   `interactions.jsonl` and `consent.json` and no corpus file must end up training, not idle.
   That exact scenario is what nothing covered.

Done means: on a copy of the real live data dir, the pivot code reports a non-zero corpus, resolves
all 5 consented users, and the trainer starts a real cycle instead of logging
`train.idle reason=no_consented_rows`. Report those numbers.

Ceiling: make the landed pivot work on existing data. Don't redesign the corpus model, don't touch
the blocklist, don't change what the bot says to people.

## Checklist
- [x] 1. Reproduce both defects against a copy of the real live data dir
- [x] 2. Prove the actual root cause of the consent miss (not salt, not notice_version)
- [x] 3. Fix consent resolution: provenance-scoped gate shared by trainer/stats/export/backfill
- [x] 4. Regression test: legacy grant → migrated rows train (permissive direction)
- [x] 5. Regression test: no grant / declined / withdrawn → nothing trains (restrictive direction)
- [x] 6. Run the migration automatically + idempotently (decide & state the mechanism)
- [x] 7. Regression test: interactions.jsonl + consent.json + no corpus file → trains, not idle
- [x] 8. Verify on a copy of the real data: non-zero corpus, 5/5 users, a real train cycle
- [x] 9. Full suite green
- [x] 10. README/docs updated to match the shipped rule

## Notes

### Root cause (proven, 2026-08-14)

Both guesses in the ticket are wrong, and I checked each directly against the real snapshot:

- **Not the salt.** All 5 consent ids pseudonymise to the 5 author hashes actually present in
  `interactions.jsonl` — `authors that map to a known consent id: 5 of 5`, unmatched set empty.
- **Not `notice_version`.** `ConsentStore.decision()` never reads `notice_version` at all.

The real cause is **scope**. The pivot split one grant into two (`corrections`, `corpus`).
`consent.py:_read_grants` deliberately loads a legacy flat record as `{corrections: granted}` and
leaves `corpus` **unknown**, so people get re-asked before ordinary messages are collected. But
every gate that decides what *trains* asks for `corpus` (`trainer.py:112`, `trainer.py:352`,
`stats.py:66`, `export_hf.py:148`) — hence `0 of 5` and `no_consented_rows`.

The fatal part is a **disagreement inside the pivot**: `backfill.py` gates the migration on the
*corrections* grant, so it happily writes rows that the trainer then refuses to train on. Proven on
the real-data copy: `backfill-corpus` added **54** corpus rows and the very next `summary` still
said `corpus 54 rows (0 training …) · people opted in 0 of 5`. So defect 1 was never the whole
story — running the missing migration by hand still leaves the box inert.

### Decisions

- **Consent rule: provenance-scoped.** The grant that governs a corpus row is decided by where its
  text came from. `prompt`/`correction` rows were flattened out of pairs collected under the
  corrections notice → governed by the `corrections` grant. `mention`/`reply`/`dm`/`ambient` rows
  are live corpus capture → governed by the `corpus` grant. An explicit no/withdrawal on `corpus`
  suppresses rows of *either* provenance, and an unrecognised source falls back to `corpus` (the
  stricter one). This is the one rule, expressed once in `consent.CorpusConsent`, and used by the
  trainer, the summary, the export and the backfill, so those four can no longer disagree.
  - Rejected: carrying a legacy grant wholesale onto `corpus`. It would report 5/5 and train, but
    it would also silently authorise future capture of people's ordinary messages under a notice
    that only ever mentioned corrections — exactly the permissive failure the ticket warns about.
- **Migration mechanism: automatic and idempotent, at the top of every training cycle**, plus at
  bot startup, with `babble backfill-corpus` kept for operators. Startup-only is not enough: both
  live units are long-lived (`--loop`), so corrections captured after boot would never reach the
  corpus until someone restarted the unit. Per-cycle is cheap and dedupes on `make_corpus_id`.

### Verified against the real data

Fresh copy of `/home/beckett/babble-live/data.bak-preppivot-20260814` each time, worktree code on
the live venv's torch. No migration command run — the trainer picks the data up itself.

| | shipped pivot (`2a9860c`) | this branch |
|---|---|---|
| `people opted in` | 0 of 5 | **5 of 5** |
| corpus rows | 0 | **54** (1,092 chars) |
| corpus training | 0 | **54** |
| trainer | `train.idle reason=no_consented_rows rows=0` | `train.cycle.start rows=54 examples=43 tokens=890` |

Migration on the real snapshot: **30 interaction rows in → 59 pieces considered → 54 corpus rows
out**, `skipped_consent=0`, `skipped_blocklist=0`, `skipped_empty=1` (a blank prompt),
`skipped_duplicate=4` (the same text said twice collapses by content-addressed id). Re-running adds
**0** and the file stays at 54 lines / 54 distinct ids.

Restrictive direction, same real data with 2 users removed from `consent.json` and 1 set to
`withdrawn`: **56 of 59 pieces skipped for consent, 2 rows migrated, 0 banned authors in the
corpus**, summary `2 of 3 asked`.

Export still clean: 54 corpus + 30 correction rows, no raw Discord id anywhere in the output. The
54 corpus rows are the same words already published under the corrections config, so this is not a
new exposure.
