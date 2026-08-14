"""The two files everything else trusts: who agreed, and what was kept."""

from __future__ import annotations

import json

from babble.consent import (
    DECLINED,
    GRANTED,
    PENDING,
    SCOPE_CORPUS,
    SCOPE_CORRECTIONS,
    UNKNOWN,
    WITHDRAWN,
    ConsentStore,
)
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.identity import Pseudonymiser
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id


# --- consent ------------------------------------------------------------


def test_an_unknown_user_has_not_consented(settings):
    store = ConsentStore(settings.consent_path)

    assert store.decision("nobody") == UNKNOWN
    assert not store.may_capture("nobody")


def test_silence_is_not_consent(settings):
    store = ConsentStore(settings.consent_path)
    store.mark_prompted("alice")

    assert store.decision("alice") == PENDING
    assert not store.may_capture("alice")


def test_capture_needs_every_participant(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.decline("bob")

    assert store.may_capture("alice")
    assert not store.may_capture("alice", "bob")


def test_decisions_survive_a_restart(settings):
    ConsentStore(settings.consent_path).grant("alice")

    assert ConsentStore(settings.consent_path).decision("alice") == GRANTED


def test_withdrawal_is_remembered_not_forgotten(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.withdraw("alice")

    assert store.decision("alice") == WITHDRAWN
    assert not store.may_capture("alice")
    assert "alice" in store.known_ids()


def test_a_corrupt_consent_file_means_nobody_consented(settings):
    settings.consent_path.write_text("{ this is not json", encoding="utf-8")

    store = ConsentStore(settings.consent_path)

    assert store.granted_ids() == []
    assert not store.may_capture("alice")


# --- two scopes -----------------------------------------------------------


def test_a_legacy_grant_loads_as_corrections_granted_and_corpus_unknown(settings):
    """A pre-corpus consent.json only ever answered the corrections notice."""
    settings.consent_path.write_text(
        json.dumps(
            {
                "alice": {
                    "decision": GRANTED,
                    "updated_at": "2025-01-01T00:00:00+00:00",
                    "notice_version": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    store = ConsentStore(settings.consent_path)

    assert store.decision("alice", SCOPE_CORRECTIONS) == GRANTED
    assert store.decision("alice", SCOPE_CORPUS) == UNKNOWN


def test_a_legacy_decline_loads_as_declined_for_both_scopes(settings):
    """A "no" needs no re-asking to stay a no, so it carries across scopes."""
    settings.consent_path.write_text(
        json.dumps({"bob": {"decision": DECLINED, "updated_at": "2025-01-01T00:00:00+00:00"}}),
        encoding="utf-8",
    )

    store = ConsentStore(settings.consent_path)

    assert store.decision("bob", SCOPE_CORRECTIONS) == DECLINED
    assert store.decision("bob", SCOPE_CORPUS) == DECLINED


def test_grant_with_no_scopes_named_grants_both(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")

    assert store.decision("alice", SCOPE_CORRECTIONS) == GRANTED
    assert store.decision("alice", SCOPE_CORPUS) == GRANTED


def test_grant_with_one_scope_named_grants_only_that_one(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice", SCOPE_CORPUS)

    assert store.decision("alice", SCOPE_CORPUS) == GRANTED
    assert store.decision("alice", SCOPE_CORRECTIONS) == UNKNOWN


def test_withdraw_clears_widened_channels(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.widen("alice", "chan-1")

    store.withdraw("alice")

    assert store.wide_channels("alice") == []


def test_decline_clears_widened_channels(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.widen("alice", "chan-1")

    store.decline("alice")

    assert store.wide_channels("alice") == []


def test_may_capture_channel_is_false_without_the_corpus_grant_even_if_widened(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice", SCOPE_CORRECTIONS)
    store.widen("alice", "chan-1")

    assert not store.may_capture_channel("alice", "chan-1")


def test_may_capture_channel_is_false_in_a_different_channel(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.widen("alice", "chan-1")

    assert not store.may_capture_channel("alice", "chan-2")


def test_may_capture_channel_is_true_only_for_the_exact_person_and_channel(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.grant("bob")
    store.widen("alice", "chan-1")

    assert store.may_capture_channel("alice", "chan-1")
    assert not store.may_capture_channel("bob", "chan-1")


def test_widen_is_idempotent(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")

    assert store.widen("alice", "chan-1") is True
    assert store.widen("alice", "chan-1") is False


def test_narrow_returns_false_if_the_channel_was_never_widened(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")

    assert store.narrow("alice", "chan-1") is False


def test_scopes_and_wide_channels_round_trip_through_a_restart(settings):
    store = ConsentStore(settings.consent_path)
    store.grant("alice")
    store.widen("alice", "chan-1")

    reloaded = ConsentStore(settings.consent_path)

    assert reloaded.decision("alice", SCOPE_CORRECTIONS) == GRANTED
    assert reloaded.decision("alice", SCOPE_CORPUS) == GRANTED
    assert reloaded.wide_channels("alice") == ["chan-1"]


def test_a_corrupt_consent_file_fails_closed_for_both_scopes(settings):
    settings.consent_path.write_text("{ this is not json", encoding="utf-8")

    store = ConsentStore(settings.consent_path)

    assert store.decision("alice", SCOPE_CORRECTIONS) == UNKNOWN
    assert store.decision("alice", SCOPE_CORPUS) == UNKNOWN
    assert not store.may_capture("alice", scope=SCOPE_CORPUS)


# --- store --------------------------------------------------------------


def _row(prompt="hi", chosen="hey", asker="u_" + "a" * 16, helper="u_" + "b" * 16):
    return Interaction(
        id=make_row_id(CORRECTION, prompt, chosen, asker, helper),
        signal=CORRECTION,
        prompt=prompt,
        rejected="junk",
        chosen=chosen,
        prompt_author=asker,
        signal_author=helper,
        weight=1.0,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_rows_round_trip_through_the_file(settings):
    store = InteractionStore(settings.interactions_path)
    store.append(_row())

    (loaded,) = InteractionStore(settings.interactions_path).all()
    assert loaded == _row()


def test_identical_rows_collapse(settings):
    store = InteractionStore(settings.interactions_path)

    assert store.append(_row()) is True
    assert store.append(_row()) is False
    assert store.count() == 1


def test_purge_removes_rows_from_either_side(settings):
    store = InteractionStore(settings.interactions_path)
    asker, helper, other = "u_" + "a" * 16, "u_" + "b" * 16, "u_" + "c" * 16
    store.append(_row(prompt="one", asker=asker, helper=helper))
    store.append(_row(prompt="two", asker=other, helper=asker))
    store.append(_row(prompt="three", asker=other, helper=other))

    removed = store.purge_author(asker)

    assert removed == 2
    assert [r.prompt for r in store.all()] == ["three"]


def test_a_torn_line_does_not_take_the_corpus_down(settings):
    store = InteractionStore(settings.interactions_path)
    store.append(_row(prompt="good"))
    with open(settings.interactions_path, "a", encoding="utf-8") as fh:
        fh.write('{"id": "truncated", "sig\n')  # killed mid-write

    assert [r.prompt for r in store.all()] == ["good"]


def test_unicode_survives_storage(settings):
    store = InteractionStore(settings.interactions_path)
    store.append(_row(chosen="🫠 https://tenor.com/view/x こんにちは"))

    (loaded,) = store.all()
    assert loaded.chosen == "🫠 https://tenor.com/view/x こんにちは"


# --- corpus store ---------------------------------------------------------


def _corpus_row(text="hello there", author="u_" + "a" * 16, source=SOURCE_MENTION):
    return CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=source)


def test_identical_corpus_rows_collapse(settings):
    store = CorpusStore(settings.corpus_path)

    assert store.append(_corpus_row()) is True
    assert store.append(_corpus_row()) is False
    assert store.count() == 1


def test_purge_author_removes_exactly_that_authors_corpus_rows(settings):
    store = CorpusStore(settings.corpus_path)
    alice, bob = "u_" + "a" * 16, "u_" + "b" * 16
    store.append(_corpus_row(text="one", author=alice))
    store.append(_corpus_row(text="two", author=bob))
    store.append(_corpus_row(text="three", author=alice))

    removed = store.purge_author(alice)

    assert removed == 2
    assert [r.author for r in store.all()] == [bob]


def test_a_torn_corpus_line_is_skipped_rather_than_fatal(settings):
    store = CorpusStore(settings.corpus_path)
    store.append(_corpus_row(text="good"))
    with open(settings.corpus_path, "a", encoding="utf-8") as fh:
        fh.write('{"id": "truncated", "tex\n')  # killed mid-write

    assert [r.text for r in store.all()] == ["good"]


# --- identity -----------------------------------------------------------


def test_hashes_are_stable_and_unlike_the_id(settings):
    ids = Pseudonymiser.load(settings)

    assert ids.user("123") == ids.user("123")
    assert ids.user("123") != ids.user("124")
    assert "123" not in ids.user("123")


def test_a_different_salt_gives_a_different_pseudonym():
    assert Pseudonymiser("salt-a").user("123") != Pseudonymiser("salt-b").user("123")


def test_the_generated_salt_persists_so_forget_keeps_working(settings):
    settings.salt = None

    first = Pseudonymiser.load(settings).user("123")

    assert settings.salt_path.exists()
    assert Pseudonymiser.load(settings).user("123") == first
