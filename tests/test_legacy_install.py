"""Upgrading a box that already has data: the case that shipped broken.

The corpus pivot was green on 356 tests and trained on nothing at all against
the live install, because every test built its fixtures *after* the pivot. Not
one of them started from what a real upgrade actually starts from: an
`interactions.jsonl` full of correction pairs, a `consent.json` written under
the old single-grant notice, and no corpus file at all.

Everything here starts from exactly that, and the shape of `_legacy_consent` is
copied from the real file the live bot went inert on -- a flat `decision`, a
`notice_version` of 1, no `scopes` key.
"""

from __future__ import annotations

import json

import pytest

from babble.backfill import backfill_corpus
from babble.consent import (
    SCOPE_CORPUS,
    SCOPE_CORRECTIONS,
    ConsentStore,
    CorpusConsent,
    scope_for_source,
)
from babble.corpus import (
    SOURCE_AMBIENT,
    SOURCE_CORRECTION,
    SOURCE_DM,
    SOURCE_MENTION,
    SOURCE_PROMPT,
    SOURCE_REPLY,
    CorpusStore,
)
from babble.stats import snapshot
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.trainer import corpus_rows, dataset_stats, train

# Raw Discord-shaped ids, because a legacy consent.json is keyed by raw ids
# while every stored row is keyed by the pseudonym derived from one.
GRANTED = "1151230208783945818"
ALSO_GRANTED = "1272933865752887317"
NEVER_ASKED = "134295609287901184"
DECLINED = "151760826771046401"
WITHDRAWN = "220354473212641293"


def _legacy_consent(settings, *user_ids: str, decision: str = "granted") -> None:
    """Write consent.json in the pre-pivot shape: one flat grant, notice 1."""
    existing = {}
    if settings.consent_path.exists():
        existing = json.loads(settings.consent_path.read_text(encoding="utf-8"))
    for user_id in user_ids:
        existing[user_id] = {
            "decision": decision,
            "notice_version": 1,
            "updated_at": "2026-08-14T03:55:05+00:00",
        }
    settings.consent_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _pair(settings, ids, asker: str, helper: str, prompt: str, chosen: str) -> Interaction:
    """Store one correction pair the way the pre-pivot bot stored them."""
    row = Interaction(
        id=make_row_id(CORRECTION, prompt, chosen, ids.user(asker), ids.user(helper)),
        signal=CORRECTION,
        prompt=prompt,
        rejected="whatever the model babbled",
        chosen=chosen,
        prompt_author=ids.user(asker),
        signal_author=ids.user(helper),
        weight=1.0,
        created_at="2026-08-14T03:55:20+00:00",
    )
    InteractionStore(settings.interactions_path).append(row)
    return row


@pytest.fixture
def legacy(settings, ids):
    """A box mid-upgrade: pairs on disk, an old consent file, no corpus."""
    _legacy_consent(settings, GRANTED, ALSO_GRANTED)
    _pair(settings, ids, GRANTED, ALSO_GRANTED, "hallo", "Hi")
    _pair(settings, ids, ALSO_GRANTED, GRANTED, "what's the weather", "Idfk")
    assert not settings.corpus_path.exists()
    return settings


# --- the consent gate, both directions ------------------------------------


def test_a_legacy_grant_still_reads_as_granted_for_its_own_correction_text(legacy, ids):
    """The permissive direction: yes under the pair model is still yes."""
    gate = CorpusConsent(ConsentStore(legacy.consent_path), ids)

    assert gate.allows_author(ids.user(GRANTED), SOURCE_PROMPT)
    assert gate.allows_author(ids.user(GRANTED), SOURCE_CORRECTION)
    assert len(gate) == 2


@pytest.mark.parametrize("source", [SOURCE_PROMPT, SOURCE_CORRECTION, SOURCE_MENTION, SOURCE_DM])
def test_somebody_who_never_answered_is_never_allowed_anything(legacy, ids, source):
    """The restrictive direction, which is the one that must never regress."""
    gate = CorpusConsent(ConsentStore(legacy.consent_path), ids)

    assert not gate.allows_author(ids.user(NEVER_ASKED), source)


@pytest.mark.parametrize("decision", ["declined", "withdrawn"])
@pytest.mark.parametrize("source", [SOURCE_PROMPT, SOURCE_CORRECTION, SOURCE_MENTION])
def test_an_explicit_no_beats_every_scope_and_every_source(settings, ids, decision, source):
    """A no is stronger than a yes, whichever notice it was said to."""
    _legacy_consent(settings, DECLINED, decision=decision)
    gate = CorpusConsent(ConsentStore(settings.consent_path), ids)

    assert not gate.allows_author(ids.user(DECLINED), source)
    assert len(gate) == 0


def test_a_legacy_grant_does_not_authorise_collecting_ordinary_messages(legacy, ids):
    """The whole reason this is scoped per source rather than carried wholesale.

    Reading the old yes as a yes to *everything* would make the live box train
    again, and would also start collecting these people's ordinary messages on
    the strength of a notice that only ever mentioned corrections.
    """
    gate = CorpusConsent(ConsentStore(legacy.consent_path), ids)

    for source in (SOURCE_MENTION, SOURCE_REPLY, SOURCE_DM, SOURCE_AMBIENT):
        assert not gate.allows_author(ids.user(GRANTED), source), source


def test_answering_the_new_notice_unlocks_live_capture_too(legacy, ids):
    consent = ConsentStore(legacy.consent_path)
    consent.grant(GRANTED, SCOPE_CORPUS)

    gate = CorpusConsent(consent, ids)
    assert gate.allows_author(ids.user(GRANTED), SOURCE_MENTION)
    assert gate.allows_author(ids.user(GRANTED), SOURCE_PROMPT)


def test_an_unrecognised_source_is_held_to_the_stricter_grant():
    assert scope_for_source("something-invented-later") == SCOPE_CORPUS
    assert scope_for_source(SOURCE_PROMPT) == SCOPE_CORRECTIONS
    assert scope_for_source(SOURCE_MENTION) == SCOPE_CORPUS


# --- the migration ---------------------------------------------------------


def test_the_migration_turns_legacy_pairs_into_trainable_corpus_rows(legacy, ids, log):
    result = backfill_corpus(legacy, log=log, ids=ids)

    assert result.added == 4  # two pairs, each a prompt and a correction
    assert result.skipped_consent == 0
    # And critically: what it wrote is what the trainer will accept.
    assert len(corpus_rows(legacy, ids)) == 4


def test_rows_from_people_who_did_not_consent_never_migrate(settings, ids, log):
    _legacy_consent(settings, GRANTED)
    _legacy_consent(settings, DECLINED, decision="declined")
    _pair(settings, ids, GRANTED, GRANTED, "mine", "also mine")
    _pair(settings, ids, NEVER_ASKED, DECLINED, "not mine", "definitely not")

    backfill_corpus(settings, log=log, ids=ids)

    authors = {r.author for r in CorpusStore(settings.corpus_path).all()}
    assert authors == {ids.user(GRANTED)}
    assert ids.user(NEVER_ASKED) not in authors
    assert ids.user(DECLINED) not in authors


def test_running_the_migration_twice_adds_nothing_the_second_time(legacy, ids, log):
    first = backfill_corpus(legacy, log=log, ids=ids)
    second = backfill_corpus(legacy, log=log, ids=ids)

    assert first.added == 4
    assert second.added == 0
    assert second.skipped_duplicate == first.considered - first.skipped_empty
    rows = CorpusStore(legacy.corpus_path).all()
    assert len(rows) == len({r.id for r in rows}) == 4


def test_a_quiet_migration_only_logs_when_it_actually_added_rows(legacy, ids, log, read_log):
    backfill_corpus(legacy, log=log, ids=ids, log_noop=False)
    assert len(read_log("corpus.backfill")) == 1

    backfill_corpus(legacy, log=log, ids=ids, log_noop=False)
    assert len(read_log("corpus.backfill")) == 1  # nothing added, nothing logged


# --- end to end ------------------------------------------------------------


def test_a_legacy_install_trains_instead_of_going_idle(legacy, ids, log, read_log):
    """The exact failure that shipped: a real cycle, not `no_consented_rows`.

    Nobody runs a migration command in this test, because nobody ran one on the
    live box either -- the trainer has to pick the data up by itself.
    """
    result = train(legacy, steps=2, echo=False, seed=1, log=log)

    assert not read_log("train.idle"), "trainer went idle on an install that has data"
    cycles = read_log("train.cycle.start")
    assert cycles and cycles[0]["rows"] == 4
    assert cycles[0]["dropped_consent"] == 0
    assert result.steps_run == 2
    assert result.stopped_because != "no_data"


def test_the_summary_resolves_legacy_users_and_counts_the_migrated_corpus(legacy, ids, log):
    backfill_corpus(legacy, log=log, ids=ids)

    snap = snapshot(legacy)

    assert snap.consented_users == 2
    assert snap.known_users == 2
    assert snap.corpus_rows == 4
    assert snap.corpus_trainable == 4


def test_dataset_stats_agrees_with_what_actually_trains(legacy, ids, log):
    """These two are read side by side in the feed; they must not diverge."""
    _legacy_consent(legacy, NEVER_ASKED, decision="declined")
    _pair(legacy, ids, NEVER_ASKED, NEVER_ASKED, "excluded", "also excluded")
    backfill_corpus(legacy, log=log, ids=ids)

    stats = dataset_stats(legacy, ids)

    assert stats.trained == len(corpus_rows(legacy, ids)) == 4
    assert stats.stored == 4  # the declined pair never made it to disk at all
