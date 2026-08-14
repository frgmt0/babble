"""The two files everything else trusts: who agreed, and what was kept."""

from __future__ import annotations

import json

from babble.consent import DECLINED, GRANTED, PENDING, UNKNOWN, WITHDRAWN, ConsentStore
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
