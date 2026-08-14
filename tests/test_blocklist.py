"""The content filter: bounces a whole row, never edits it, never logs the text.

A blocklist is a speed bump, not a guarantee -- these tests are about the
normalisation defeating the lazy/obvious evasions and the three enforcement
points (capture, train, export) actually refusing a matching row, not about
having a "complete" list of terms.
"""

from __future__ import annotations

import pytest

from babble.blocklist import Blocklist, normalise, row_fingerprint
from babble.consent import ConsentStore
from babble.core import Babble
from babble.exchanges import Exchange
from babble.export_hf import DATA_FILE, build_export
from babble.identity import Pseudonymiser
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.trainer import consented_rows

ALICE = "111111111111111111"
BOB = "222222222222222222"


def _blocklist(*terms: str) -> Blocklist:
    return Blocklist(frozenset(normalise(t) for t in terms))


def _store_row(
    settings, author_id, helper_id=None, prompt="hi", chosen="hey", signal=CORRECTION, rejected=None
):
    ids = Pseudonymiser.load(settings)
    asker = ids.user(author_id)
    helper = ids.user(helper_id or author_id)
    if rejected is None and signal == CORRECTION:
        rejected = "junk"
    row = Interaction(
        id=make_row_id(signal, prompt, chosen, asker, helper),
        signal=signal,
        prompt=prompt,
        rejected=rejected,
        chosen=chosen,
        prompt_author=asker,
        signal_author=helper,
        weight=1.0 if signal == CORRECTION else 0.25,
        created_at="2026-01-01T00:00:00+00:00",
    )
    InteractionStore(settings.interactions_path).append(row)
    return row


# --- normalisation -----------------------------------------------------


def test_normalisation_defeats_spacing_leetspeak_and_diacritic_evasion():
    bl = _blocklist("badword")

    assert bl.hit("b a d w o r d") is not None
    assert bl.hit("b.a.d.w.o.r.d") is not None
    assert bl.hit("b-a-d_w-o-r-d") is not None
    assert bl.hit("b4dw0rd") is not None
    assert bl.hit("bàdwörd") is not None
    assert bl.hit("BADWOOOORD") is not None


def test_a_clean_word_with_an_awkward_substring_is_not_rejected():
    bl = _blocklist("ass")

    assert bl.hit("that classy assassin wore glass armour in class") is None


def test_matching_is_whole_word_not_substring():
    bl = _blocklist("cat")

    assert bl.hit("the category was concatenated") is None
    assert bl.hit("the cat sat down") is not None


def test_row_fingerprint_is_stable_and_reveals_nothing():
    fp1 = row_fingerprint("hello", "world")
    fp2 = row_fingerprint("hello", "world")

    assert fp1 == fp2
    assert "hello" not in fp1 and "world" not in fp1


def test_an_unconfigured_blocklist_matches_nothing():
    bl = Blocklist(frozenset())

    assert not bl.enabled
    assert bl.hit("anything at all, even badword") is None


def test_the_shipped_blocklist_loads_and_is_not_empty():
    bl = Blocklist.load()

    assert bl.enabled
    assert len(bl.terms) > 0


def test_blocklist_path_is_overridable_via_env(tmp_path, monkeypatch):
    custom = tmp_path / "custom.txt"
    custom.write_text("wobblefrog\n# a comment\n\nsecondterm\n", encoding="utf-8")
    monkeypatch.setenv("BABBLE_BLOCKLIST_PATH", str(custom))

    bl = Blocklist.load()

    assert bl.hit("a wobblefrog appeared") is not None
    assert bl.hit("secondterm here") is not None


# --- capture time (core.py) ---------------------------------------------


@pytest.fixture
def brain(settings, generator, log):
    """Overrides conftest's `brain` fixture: this one has a live blocklist."""
    return Babble(
        settings, generator=generator, log=log, blocklist=_blocklist("badword"), bot_user_id="bot-9999"
    )


def test_a_correction_matching_the_blocklist_is_never_stored(fake, brain, settings, read_log):
    fake.onboard("alice")
    generation = fake.ping("alice")[-1]

    fake.ping("alice", "that response had a badword in it", reply_to=generation.id)

    assert InteractionStore(settings.interactions_path).all() == []
    entries = read_log("capture.blocked")
    assert len(entries) == 1
    assert entries[0]["stage"] == "correction"
    assert "badword" not in str(entries[0])


def test_being_told_the_correction_was_not_accepted_without_repeating_it(fake, brain):
    fake.onboard("alice")
    generation = fake.ping("alice")[-1]

    reply = fake.ping("alice", "definitely a badword here", reply_to=generation.id)[-1]

    assert "badword" not in reply.content
    assert "not accepted" in reply.content or "content filter" in reply.content


def test_a_clean_correction_with_an_awkward_substring_is_still_stored(fake, brain, settings):
    fake.onboard("alice")
    generation = fake.ping("alice")[-1]

    fake.ping("alice", "a classy assassin in glass armour", reply_to=generation.id)

    assert len(InteractionStore(settings.interactions_path).all()) == 1


def test_an_approval_whose_exchange_matches_the_blocklist_is_never_stored(fake, brain, settings, read_log):
    fake.onboard("alice")
    # Simulate a legacy remembered exchange predating the blocklist entry.
    brain.exchanges.record(
        "legacy-msg", Exchange(prompt="hi", response="a real badword", prompt_author_id="alice")
    )

    fake.react("alice", "legacy-msg")

    assert InteractionStore(settings.interactions_path).all() == []
    entries = read_log("capture.blocked")
    assert entries and entries[0]["stage"] == "approval"


def test_a_match_in_the_models_own_output_is_caught_before_send(fake, brain, generator, settings):
    generator.text = "you are a real badword, honestly"
    fake.onboard("alice")

    sent = fake.ping("alice")

    assert "badword" not in sent[-1].content
    assert brain.exchanges.get(sent[-1].id) is None  # nothing rememberable was ever sent


# --- training time (trainer.py) -----------------------------------------


def test_a_row_stored_before_the_term_was_blocked_is_excluded_at_training_time(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="a real badword")

    rows = consented_rows(settings, blocklist=_blocklist("badword"))

    assert rows == []


def test_a_blocked_term_hiding_in_the_rejected_field_is_also_caught(settings):
    """The blocklist can be extended after a generation was sent and remembered;
    the row it produced must not survive on the strength of a clean prompt and
    chosen text alone if the old, rejected answer is now blocked too."""
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="fine response", rejected="a real badword")

    rows = consented_rows(settings, blocklist=_blocklist("badword"))

    assert rows == []


def test_training_time_filtering_only_drops_matching_rows(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="a real badword")
    _store_row(settings, ALICE, prompt="clean prompt", chosen="a fine response")

    rows = consented_rows(settings, blocklist=_blocklist("badword"))

    assert [r.chosen for r in rows] == ["a fine response"]


# --- export time (export_hf.py) -----------------------------------------


def test_a_blocked_row_is_absent_from_export(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="a real badword")

    result = build_export(settings, blocklist=_blocklist("badword"))

    assert result.rows == 0
    assert result.dropped_blocklist == 1
    body = (result.path / DATA_FILE).read_text(encoding="utf-8")
    assert "badword" not in body


def test_export_also_catches_a_blocked_term_in_the_rejected_field(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="fine response", rejected="a real badword")

    result = build_export(settings, blocklist=_blocklist("badword"))

    assert result.rows == 0
    assert result.dropped_blocklist == 1
    body = (result.path / DATA_FILE).read_text(encoding="utf-8")
    assert "badword" not in body


def test_export_keeps_clean_rows_and_drops_only_blocked_ones(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="a real badword")
    _store_row(settings, ALICE, prompt="clean", chosen="fine response")

    result = build_export(settings, blocklist=_blocklist("badword"))

    assert result.rows == 1
    assert result.dropped_blocklist == 1


# --- rescan / purge -------------------------------------------------------


def test_rescan_purges_rows_that_now_match_an_extended_blocklist(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="a real badword")
    _store_row(settings, ALICE, prompt="clean", chosen="fine response")
    store = InteractionStore(settings.interactions_path)
    assert len(store.all()) == 2

    blocklist = _blocklist("badword")
    removed = store.purge(lambda r: blocklist.matches(r.prompt, r.chosen))

    assert removed == 1
    assert [r.chosen for r in store.all()] == ["fine response"]


def test_rescan_via_the_cli(settings, monkeypatch, capsys, tmp_path):
    from babble.cli import main

    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hi", chosen="a real badword")

    custom = tmp_path / "custom.txt"
    custom.write_text("badword\n", encoding="utf-8")
    monkeypatch.setenv("BABBLE_BLOCKLIST_PATH", str(custom))
    monkeypatch.setenv("BABBLE_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("BABBLE_CHECKPOINT_DIR", str(settings.checkpoint_dir))
    monkeypatch.setenv("BABBLE_LOG_DIR", str(settings.log_dir))
    monkeypatch.setenv("BABBLE_HASH_SALT", settings.salt)

    code = main(["rescan-blocklist"])

    assert code == 0
    assert "purged 1" in capsys.readouterr().out
    assert InteractionStore(settings.interactions_path).all() == []
