"""The export: consented rows only, pseudonyms only, same bytes every time."""

from __future__ import annotations

import json

import pytest

from babble.consent import SCOPE_CORPUS, SCOPE_CORRECTIONS, ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.export_hf import (
    CORPUS_FILE,
    DATA_FILE,
    ExportBlocked,
    assert_pseudonymous,
    build_export,
    select_rows,
)
from babble.identity import Pseudonymiser
from babble.store import APPROVAL, CORRECTION, Interaction, InteractionStore, make_row_id

ALICE = "111111111111111111"
BOB = "222222222222222222"


def _store_row(settings, author_id, helper_id=None, prompt="hi", chosen="hey", signal=CORRECTION):
    ids = Pseudonymiser.load(settings)
    asker = ids.user(author_id)
    helper = ids.user(helper_id or author_id)
    row = Interaction(
        id=make_row_id(signal, prompt, chosen, asker, helper),
        signal=signal,
        prompt=prompt,
        rejected="junk" if signal == CORRECTION else None,
        chosen=chosen,
        prompt_author=asker,
        signal_author=helper,
        weight=1.0 if signal == CORRECTION else 0.25,
        created_at="2026-01-01T00:00:00+00:00",
    )
    InteractionStore(settings.interactions_path).append(row)
    return row


def _store_corpus_row(settings, author_id, text="hi there", source=SOURCE_MENTION):
    ids = Pseudonymiser.load(settings)
    author = ids.user(author_id)
    row = CorpusRow(
        id=make_corpus_id(text, author),
        text=text,
        author=author,
        source=source,
        # Set even though the payload must not carry them, so the assertion that
        # they're missing is proving something rather than checking a default.
        guild="g_" + "1" * 16,
        channel="c_" + "1" * 16,
        created_at="2026-01-01T00:00:00+00:00",
    )
    CorpusStore(settings.corpus_path).append(row)
    return row


def _rows_in(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- consent ------------------------------------------------------------


def test_only_consented_rows_are_published(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="kept")
    _store_row(settings, BOB, prompt="dropped")  # never consented

    result = build_export(settings)

    assert result.correction_rows == 1
    assert result.excluded_no_consent == 1
    assert [r["prompt"] for r in _rows_in(result.path / DATA_FILE)] == ["kept"]


def test_withdrawal_wins_even_if_the_row_is_still_on_disk(settings):
    consent = ConsentStore(settings.consent_path)
    consent.grant(ALICE)
    _store_row(settings, ALICE)
    assert build_export(settings).correction_rows == 1

    consent.withdraw(ALICE)

    assert build_export(settings).correction_rows == 0
    assert _rows_in(settings.export_dir / DATA_FILE) == []


def test_a_row_needs_both_participants_to_consent(settings):
    ConsentStore(settings.consent_path).grant(ALICE)  # but not Bob
    _store_row(settings, ALICE, helper_id=BOB)

    assert build_export(settings).correction_rows == 0


# --- identifiers --------------------------------------------------------


def test_the_published_file_contains_no_raw_discord_ids(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE)

    body = (build_export(settings).path / DATA_FILE).read_text(encoding="utf-8")

    assert ALICE not in body
    assert "u_" in body


def test_a_message_quoting_a_raw_id_is_dropped_rather_than_published(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="fine", chosen="ok")
    _store_row(settings, ALICE, prompt="my id is " + ALICE, chosen="noted")

    result = build_export(settings)

    assert result.correction_rows == 1
    assert result.dropped_leaky == 1
    assert ALICE not in (result.path / DATA_FILE).read_text(encoding="utf-8")


def test_a_row_with_mention_markup_is_dropped(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, chosen="hi <@444444444444444444>")

    assert build_export(settings).correction_rows == 0


def test_the_guard_refuses_an_unpseudonymised_author(settings):
    leaky = Interaction(
        id="abc",
        signal=CORRECTION,
        prompt="hi",
        rejected=None,
        chosen="hey",
        prompt_author=ALICE,  # a raw id where a hash belongs
        signal_author="u_" + "a" * 16,
        weight=1.0,
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ExportBlocked, match="not a pseudonym"):
        assert_pseudonymous([leaky], [])


def test_the_guard_refuses_a_corpus_row_with_a_non_pseudonym_author(settings):
    leaky = CorpusRow(
        id="abc",
        text="hi",
        author=ALICE,  # a raw id where a hash belongs
        source=SOURCE_MENTION,
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ExportBlocked, match="not a pseudonym"):
        assert_pseudonymous([], [], corpus=[leaky])


def test_the_guard_refuses_a_corpus_row_whose_text_quotes_a_raw_id(settings):
    leaky = CorpusRow(
        id="abc",
        text=f"my id is {ALICE}",
        author="u_" + "a" * 16,
        source=SOURCE_MENTION,
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ExportBlocked, match="raw Discord id"):
        assert_pseudonymous([], [ALICE], corpus=[leaky])


# --- the corpus -----------------------------------------------------------


def test_only_consented_corpus_rows_are_published(settings):
    ConsentStore(settings.consent_path).grant(ALICE, SCOPE_CORPUS)
    _store_corpus_row(settings, ALICE, text="kept")
    _store_corpus_row(settings, BOB, text="dropped")  # never consented

    result = build_export(settings)

    assert result.corpus_rows == 1
    assert result.corpus_excluded_no_consent == 1
    assert [r["text"] for r in _rows_in(result.path / CORPUS_FILE)] == ["kept"]


def test_a_corpus_row_whose_author_has_only_the_corrections_grant_is_not_exported(settings):
    """The consent boundary the two scopes exist to enforce.

    Someone who agreed to the old corrections notice has not agreed to have
    their ordinary messages collected -- that grant is UNKNOWN until they
    answer the corpus notice, and UNKNOWN must not export.
    """
    ConsentStore(settings.consent_path).grant(ALICE, SCOPE_CORRECTIONS)  # corpus left unanswered
    _store_corpus_row(settings, ALICE)

    result = build_export(settings)

    assert result.corpus_rows == 0
    assert result.corpus_excluded_no_consent == 1


def test_a_corpus_row_quoting_a_raw_id_is_dropped_rather_than_published(settings):
    ConsentStore(settings.consent_path).grant(ALICE, SCOPE_CORPUS)
    _store_corpus_row(settings, ALICE, text="fine, nothing to see here")
    _store_corpus_row(settings, ALICE, text="my id is " + ALICE)

    result = build_export(settings)

    assert result.corpus_rows == 1
    assert result.corpus_dropped_leaky == 1
    assert ALICE not in (result.path / CORPUS_FILE).read_text(encoding="utf-8")


def test_the_published_corpus_rows_have_exactly_the_documented_fields(settings):
    ConsentStore(settings.consent_path).grant(ALICE, SCOPE_CORPUS)
    _store_corpus_row(settings, ALICE, text="hello there")

    (row,) = _rows_in(build_export(settings).path / CORPUS_FILE)

    assert set(row) == {"id", "text", "author", "source", "created_at"}
    assert row["text"] == "hello there"
    assert row["source"] == SOURCE_MENTION
    # The guild and channel a row came from are pseudonymous but still not
    # anybody's business but babble's own -- they never leave the machine.
    assert "guild" not in row
    assert "channel" not in row


def test_corpus_rows_are_ordered_deterministically(settings):
    ConsentStore(settings.consent_path).grant(ALICE, SCOPE_CORPUS)
    for text in ("zebra", "apple", "mango"):
        _store_corpus_row(settings, ALICE, text=text)

    ids_out = [r["id"] for r in _rows_in(build_export(settings).path / CORPUS_FILE)]

    assert ids_out == sorted(ids_out)  # same created_at, so id breaks the tie


def test_running_the_corpus_export_twice_produces_identical_bytes(settings):
    ConsentStore(settings.consent_path).grant(ALICE, SCOPE_CORPUS)
    _store_corpus_row(settings, ALICE, text="a")
    _store_corpus_row(settings, ALICE, text="b")

    first = (build_export(settings).path / CORPUS_FILE).read_bytes()
    second = (build_export(settings).path / CORPUS_FILE).read_bytes()

    assert first == second


# --- shape and stability -------------------------------------------------


def test_the_published_rows_have_the_documented_fields(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="hello", chosen="hey!")

    (row,) = _rows_in(build_export(settings).path / DATA_FILE)

    assert set(row) == {
        "id",
        "prompt",
        "rejected",
        "chosen",
        "signal",
        "weight",
        "prompt_author",
        "signal_author",
        "created_at",
    }
    assert row["prompt"] == "hello"
    assert row["chosen"] == "hey!"
    assert row["signal"] == CORRECTION


def test_running_the_export_twice_produces_identical_bytes(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE, prompt="a")
    _store_row(settings, ALICE, prompt="b")
    _store_row(settings, ALICE, prompt="c", signal=APPROVAL)

    first_data = (build_export(settings).path / DATA_FILE).read_bytes()
    first_card = (settings.export_dir / "README.md").read_bytes()
    second_data = (build_export(settings).path / DATA_FILE).read_bytes()
    second_card = (settings.export_dir / "README.md").read_bytes()

    assert first_data == second_data
    assert first_card == second_card


def test_rows_are_ordered_deterministically(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    for prompt in ("zebra", "apple", "mango"):
        _store_row(settings, ALICE, prompt=prompt)

    ids = [r["id"] for r in _rows_in(build_export(settings).path / DATA_FILE)]

    assert ids == sorted(ids)  # same created_at, so id breaks the tie


def test_a_dataset_card_is_written_with_valid_frontmatter(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE)
    _store_corpus_row(settings, ALICE, text="hello there")

    card = (build_export(settings).path / "README.md").read_text(encoding="utf-8")

    assert card.startswith("---\n")
    assert card.count("---\n") >= 2
    assert "consent" in card.lower()
    assert "random weights" in card
    # Both configs are declared, each pointing at its own file, and each has
    # its own row count called out separately -- one correction, one corpus row.
    assert "config_name: default" in card
    assert f"path: {CORPUS_FILE}" in card
    assert "config_name: corrections" in card
    assert f"path: {DATA_FILE}" in card
    assert "1 rows" in card


def test_an_empty_corpus_still_produces_a_valid_export(settings):
    result = build_export(settings)

    assert result.rows == 0
    assert result.correction_rows == 0
    assert result.corpus_rows == 0
    assert (result.path / DATA_FILE).exists()
    assert (result.path / CORPUS_FILE).exists()
    assert (result.path / "README.md").exists()


def test_nothing_is_uploaded_by_building_an_export(settings, monkeypatch):
    """Capture and build must never touch the network. Push is opt-in only."""
    ConsentStore(settings.consent_path).grant(ALICE)
    _store_row(settings, ALICE)

    def explode(*args, **kwargs):
        raise AssertionError("build_export must not talk to HuggingFace")

    monkeypatch.setattr("huggingface_hub.HfApi.upload_folder", explode, raising=False)

    build_export(settings)


def test_select_rows_deduplicates(settings):
    ConsentStore(settings.consent_path).grant(ALICE)
    row = _store_row(settings, ALICE)
    # Same row appended twice by hand, bypassing the store's own dedup.
    with open(settings.interactions_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_dict()) + "\n")

    rows, _, _, _ = select_rows(settings)

    assert len(rows) == 1
