# Finish and land the unlabelled-corpus pivot
> run: run-20260814-finish-and-land-the-unlabelled-corpus-pi · branch: beckett/run-finish-and-land-the-unlabelled-corpus-pi · created: 2026-08-14T21:02:46.440Z

## Goal
Finish and land work that is already ~95% done. Its worker crashed before it could report a
verdict, so nothing reviewed it and nothing landed it. Do NOT restart the design — read what is
there and finish it.

Repo: kowo-co/babble. The work is on branch `beckett/run-collect-an-unlabelled-pretraining-corpus`,
tip `fc16922` (a crash-time WIP commit), cut from main at `de65544`. Start from that branch, not
from main.

The original ask, from ro (user 1151230208783945818, the project owner): "make it so booper/babble
now just collects pretraining unlabelled corpus rather than training against the inputs. So people
can just talk to booper to collect that corpus" — because "in general right now we are collecting
english corpus and overfitting it to a pair. which is fine for now but ideally itd just be a pool of
unlabelled data to be picked at". Correction pairs stop being the training objective; an unlabelled
text corpus becomes it. The crashed worker marked all 20 of its checklist items done and wrote ~4000
lines across 31 files, including a new `tests/test_corpus.py`.

**It is not actually finished — the test suite is red.** I ran it on that branch just now:
`352 passed, 4 failed`. All four failures are in `tests/test_trainer.py`, all in the probe path:

- `test_probe_prefix_walks_the_whole_dataset_in_order_then_wraps`
- `test_probe_prefix_never_repeats_two_checkpoints_in_a_row`
- `test_probe_prefix_dedupes_a_row_whose_text_repeats`
- `test_a_row_with_nothing_usable_in_it_is_probed_as_a_fallback_not_as_trained`

The first one fails like this:

    assert walked == [(f"row-text-{i}", PROBE_TRAIN) for i in range(4)]
    E  At index 0 diff: ('row-text-', 'trained') != ('row-text-0', 'trained')

so the probe prefix is losing the last character of each row's text — an off-by-one in whatever
slices the probe prefix out of a corpus row. Diagnose it properly rather than editing the test to
match the buggy output: decide what the probe prefix is SUPPOSED to be under the new corpus model,
fix the code if the code is wrong, and only change a test if you can say in one sentence why the
test encodes the old pair-based model and is genuinely obsolete.

The job:
1. Get the full suite green on that branch. All 356 tests, no skips added, no tests deleted to make
   red go away.
2. Rebase onto current `origin/main` if it has moved, and make sure no conflict markers survive
   anywhere.
3. Re-read the branch's own `spec.md` checklist and verify each item against the code as it stands.
   The crashed worker ticked all 20 while leaving four tests red, so its self-assessment is not
   trustworthy — check the claims, and fix or honestly un-tick anything that does not hold.
4. Sanity-check the end-to-end story yourself: a message to the bot lands in the unlabelled corpus,
   the trainer trains on corpus text rather than on prompt/chosen pairs, and the export path still
   produces something coherent. Say in the PR description what you actually ran to confirm it.

Done means: the whole suite green on top of current main, the checklist honest, and a PR
description that states what the pivot changed (storage, training objective, export) and anything
about existing stored correction rows that a human needs to decide — the live bot has ~30 stored
rows under the old pair model and it matters whether they migrate, get reused as plain text, or get
left alone.

Ceiling: finish this pivot and land it. No new features beyond what the branch already does, no
architecture rewrite, don't touch the consent gate or the blocklist.

## Checklist

Every item below was verified against the code as it now stands (file:line evidence in the PR
notes), and cross-checked against a from-scratch end-to-end run — not taken on the crashed
worker's word.

- [x] **Bug that made the suite red is fixed.** `leading_words` (trainer.py:387-426) no longer
      drops the last byte of a row that fits the budget: it holds back the last *word* and hands a
      single spaceless word back whole. The four `test_probe_prefix_*` failures pass; no test was
      edited to match buggy output (`git diff fc16922 HEAD -- tests/` is empty).
- [x] **Full suite green on top of current main.** `356 passed` via `uv run --extra dev pytest`.
      No skips added, no tests deleted. `origin/main` has not moved past the branch's base
      `de65544` (merge-base == origin/main tip), so no rebase was needed; no conflict markers
      anywhere.
- [x] `babble/corpus.py` — `CorpusRow` + `CorpusStore` over `data/corpus.jsonl`: atomic append,
      dedupe by content-addressed id, `purge_author`, `counts_by_source`, `approx_tokens`.
- [x] Content-addressed id over `(text, author)` only, so the same words by the same person
      collapse to one row however they arrived (`make_corpus_id`, corpus.py:81-92).
- [x] Capture: any message addressed to booper (mention / reply / DM) from a corpus-consented
      person becomes a corpus row, with `source` provenance and pseudonymous guild+channel
      (core.py:165-179, 559, 465-466).
- [x] `IncomingMessage.is_dm` set explicitly by `bot.py` (bot.py:141), not inferred from a missing
      guild id.
- [x] Widening `!babble all` / `!babble pings`: per person per channel, off by default, revocable
      immediately, requires the corpus grant first (core.py:768-830, consent.py:187-198, 253-265).
- [x] Ambient capture never replies and never prompts anyone for consent (core.py:345-358).
- [x] Correction capture still works and also files both human halves (prompt + chosen) into the
      corpus (core.py:646-686).
- [x] `babble/backfill.py` + `babble backfill-corpus` — idempotent (content-addressed dedupe),
      blocklist + consent re-checked at backfill time (backfill.py:80-110). Verified idempotent by
      running it twice against the same store: 0 rows added the second time.
- [x] Trainer: plain next-token LM over corpus text; prompt-masking and per-row weighting gone from
      the corpus path (`to_examples` trainer.py:117-125, `build_text_example` tokenizer.py:124-154).
      The old masked layout survives only to *score* correction pairs in `generate.py`.
- [x] Held-out split, val loss, checkpoint cadence, nice level, duty cycle, atomic checkpoints and
      kill-safety all preserved (trainer.py:78-92, 175-218, 298-314, 467-501, 603-626).
- [x] Consent + blocklist re-checked at training time, not only at capture (`corpus_rows`
      trainer.py:95-114, called every cycle).
- [x] Feed: cycle line has corpus rows / tokens / train+val split / batch / lr; checkpoint line has
      train + val loss with deltas; probe is an honest labelled continuation with no fake expected
      field (discord_feed.py:143-242).
- [x] Two consent scopes tracked distinctly; legacy `consent.json` loads as corrections-only and
      the person is re-asked once before anything of theirs is collected (consent.py:45-72, 125-135;
      core.py:509-514).
- [x] Legacy "no" carries across both scopes; a legacy "yes" does not (consent.py:132-135).
- [x] `!babble forget` purges interactions and corpus, and clears widened channels
      (core.py:733-745, consent.py:248-251, 277-285).
- [x] Export: corpus as the `default` config, corrections kept as the `corrections` config; consent
      (per scope) and blocklist re-checked; guild/channel not published (export_hf.py:113-162,
      220-234, 350-358). Verified: export README declares `default`→corpus / `corrections`→train,
      and exported corpus rows carry no guild/channel keys.
- [x] `rescan-blocklist` purges both stores; `fake-data` fills both; `sample` continues
      (cli.py:117-134, 258-284; fakedata.py:96-108).
- [x] README, spec.md and .env.example rewritten for what babble now is.
- [x] Tests: corpus append/dedupe/purge, consent boundary, idempotent backfill, blocklist at
      capture + training + export + rescan, unmasked objective, and the full ping→correct→train→
      reload→export flow end to end (tests/test_corpus.py and across the suite).

## Notes

### What the pivot changed (PR description)

**The ask.** ro: booper/babble should "just collect pretraining unlabelled corpus rather than
training against the inputs" — correction pairs stop being the training objective, an unlabelled
text corpus becomes it.

**Storage.** New `data/corpus.jsonl` (`babble/corpus.py`): one unlabelled `CorpusRow` per piece of
human writing — `id, text, author (pseudonym), source, guild?, channel?, created_at`. Ids are
content-addressed over `(text, author)` only, so the same words by the same person collapse to one
row however they arrive. The old `data/interactions.jsonl` correction store is untouched and still
records `(prompt, rejected, chosen)` triples — corrections remain a real artifact worth keeping;
they're just no longer the training target.

**Training objective.** The trainer now does plain next-token prediction over corpus text: every
token of every row is a target, every row counts the same, nothing is paired. The old
prompt-masking + thumbs-up upweighting is gone from the training path (`to_examples`), and the
masked/weighted example machinery survives only to *score* correction pairs in `generate.py`. The
`model.py` `sequence_loss` docstring was corrected to describe the corpus objective (commit
cc6df10). Generation moved to a continuation family (`continue_text`) because `<sep>`'s embedding
is now never trained — feeding `<bos> prompt <sep>` at inference would put an untrained token in
front of every generation.

**Export.** `babble export` now writes a two-config HuggingFace dataset: `default` = the corpus
(`data/corpus.jsonl`), `corrections` = the pairs (`data/train.jsonl`). Both re-check consent (per
scope) and blocklist at export time; guild/channel are never published.

**Probe fix (the four red tests).** `leading_words` had an off-by-one — `raw[: min(budget,
max(1, len(raw)-1))]` chopped the final byte of every row that fit the budget, so a row `row-text-0`
probed as `row-text-`. Under the corpus model the probe prefix should be a genuine byte-prefix of a
real trained row with the last *word* held back (so the model has something to continue), and a
single spaceless word handed back whole rather than truncated. Fixed in the code; no test changed.

**What I ran to confirm end-to-end** (fresh `BABBLE_DATA_DIR`, salt set):
- `babble fake-data` → seeds both stores and flattens correction halves into the corpus (24 rows).
- `babble train --steps 60` → trains on corpus rows (`examples=19 tokens=306`, held-out
  `val_rows=5`); checkpoint probe reads a real corpus row and labels it `trained` (`prefix=hello
  probe_side=trained`) — the off-by-one is gone.
- `babble sample -p hello` → continues from the latest checkpoint.
- `babble export` → writes `data/corpus.jsonl` (config `default`) + `data/train.jsonl` (config
  `corrections`); exported corpus rows carry only `author (pseudonym) / created_at / id / source /
  text` — no raw Discord ids, no guild/channel.
- `babble backfill-corpus` twice → 0 rows added the second time (idempotent).

### Human decision needed: the ~30 existing correction rows on the live bot

Those rows were collected under the old pair model, under the **corrections** consent notice, and
have already been published under it. The prompt and the human-typed `chosen` of each are real
human writing; `rejected` and 👍-approval responses are *not* filed (a 👍 response is the bot's own
output, not a person's writing). Three options for a human to pick:

- **Reuse as plain text (recommended):** run `babble backfill-corpus` once against the live data
  dir. It flattens each consented correction into corpus rows, gated on the corrections grant and
  skipping anyone who explicitly said no to the corpus, re-checking the blocklist. It is idempotent,
  so re-running is safe. This is what `fake-data` already does in tests.
- **Leave them alone:** do nothing; the corpus then builds only from new messages going forward.
  The correction rows stay exactly where they are and still export under the `corrections` config.
- **Migrate/discard:** not built and not recommended — there is no separate migration path, and
  nothing deletes the correction store. "Leave alone" already covers not-reusing.

I did **not** run the backfill against live data — that's ro's call and touches real stored user
text. Everything above was verified on throwaway fake data only.

### Scope

Stayed inside the ceiling: finished and verified the branch's own pivot, no new features, no
architecture rewrite, did not touch the consent gate or the blocklist logic.
