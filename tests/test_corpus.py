"""The unlabelled corpus (`corpus.py`) and the migration that seeds it from the
old correction pairs (`backfill.py`)."""

from __future__ import annotations

import json

from babble.backfill import BackfillResult, backfill_corpus
from babble.blocklist import Blocklist
from babble.consent import SCOPE_CORPUS, SCOPE_CORRECTIONS, ConsentStore
from babble.corpus import (
    SOURCE_CORRECTION,
    SOURCE_MENTION,
    SOURCE_PROMPT,
    SOURCE_REPLY,
    CorpusRow,
    CorpusStore,
    approx_tokens,
    make_corpus_id,
)
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id

ALICE = "u_" + "a" * 16
BOB = "u_" + "b" * 16


def _row(text="hello there", author=ALICE, source=SOURCE_MENTION, **kw) -> CorpusRow:
    return CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=source, **kw)


def _interaction(
    prompt="hi",
    chosen="hey",
    prompt_author=ALICE,
    signal_author=BOB,
    *,
    rejected="junk the model wrote",
    created_at="2026-01-01T00:00:00+00:00",
) -> Interaction:
    return Interaction(
        id=make_row_id(CORRECTION, prompt, chosen, prompt_author, signal_author),
        signal=CORRECTION,
        prompt=prompt,
        rejected=rejected,
        chosen=chosen,
        prompt_author=prompt_author,
        signal_author=signal_author,
        weight=1.0,
        created_at=created_at,
    )


# --- make_corpus_id ---------------------------------------------------------


def test_make_corpus_id_is_the_same_for_the_same_text_and_author_regardless_of_anything_else():
    """Source, channel and timestamp never touch the id -- only (text, author)."""
    assert make_corpus_id("hello", ALICE) == make_corpus_id("hello", ALICE)


def test_make_corpus_id_differs_when_the_author_differs():
    assert make_corpus_id("hello", ALICE) != make_corpus_id("hello", BOB)


def test_make_corpus_id_differs_when_the_text_differs():
    assert make_corpus_id("hello", ALICE) != make_corpus_id("goodbye", ALICE)


def test_the_same_text_and_author_collapse_to_one_row_even_from_different_sources_and_times(settings):
    """The scenario the docstring calls out by name: a sentence typed live at
    the bot, and the same sentence flattened later out of an old correction
    row, must land on the same id so the two capture paths can never double it
    up."""
    store = CorpusStore(settings.corpus_path)
    live = _row(text="same words", author=ALICE, source=SOURCE_MENTION, channel="c_1")
    backfilled = _row(
        text="same words",
        author=ALICE,
        source=SOURCE_CORRECTION,
        channel=None,
        created_at="2020-01-01T00:00:00+00:00",
    )

    assert store.append(live) is True
    assert store.append(backfilled) is False
    assert store.count() == 1


# --- CorpusStore -------------------------------------------------------------


def test_append_returns_true_then_false_for_the_same_id_and_the_file_keeps_one_line(settings):
    store = CorpusStore(settings.corpus_path)

    assert store.append(_row()) is True
    assert store.append(_row()) is False
    assert len(settings.corpus_path.read_text(encoding="utf-8").splitlines()) == 1


def test_all_round_trips_every_field_including_a_null_guild(settings):
    row = CorpusRow(
        id=make_corpus_id("hello there", ALICE),
        text="hello there",
        author=ALICE,
        source=SOURCE_REPLY,
        guild=None,
        channel="c_" + "1" * 12,
        created_at="2026-03-04T05:06:07+00:00",
    )
    CorpusStore(settings.corpus_path).append(row)

    (loaded,) = CorpusStore(settings.corpus_path).all()

    assert loaded == row
    assert loaded.guild is None


def test_a_torn_corpus_line_is_skipped_and_the_rows_around_it_still_load(settings):
    store = CorpusStore(settings.corpus_path)
    store.append(_row(text="before", author=ALICE))
    with open(settings.corpus_path, "a", encoding="utf-8") as fh:
        fh.write('{"id": "truncated", "tex\n')  # killed mid-write
    store.append(_row(text="after", author=BOB))

    assert sorted(r.text for r in store.all()) == ["after", "before"]


def test_purge_author_removes_exactly_that_authors_rows_and_leaves_valid_jsonl(settings):
    store = CorpusStore(settings.corpus_path)
    store.append(_row(text="one", author=ALICE))
    store.append(_row(text="two", author=BOB))
    store.append(_row(text="three", author=ALICE))

    removed = store.purge_author(ALICE)

    assert removed == 2
    assert [r.author for r in store.all()] == [BOB]
    # the rewrite is atomic and complete, not a truncated in-place edit
    lines = settings.corpus_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    json.loads(lines[0])  # does not raise


def test_purge_with_a_predicate_matching_nothing_does_not_touch_the_file(settings):
    store = CorpusStore(settings.corpus_path)
    store.append(_row(text="keep me", author=ALICE))
    before = settings.corpus_path.read_bytes()

    removed = store.purge(lambda r: r.author == BOB)

    assert removed == 0
    assert settings.corpus_path.read_bytes() == before


def test_purge_with_a_predicate_removes_only_matching_rows(settings):
    store = CorpusStore(settings.corpus_path)
    store.append(_row(text="short", author=ALICE))
    store.append(_row(text="a much longer piece of writing", author=BOB))

    removed = store.purge(lambda r: len(r.text) > 10)

    assert removed == 1
    assert [r.text for r in store.all()] == ["short"]


def test_counts_by_source_tallies_each_source_independently(settings):
    store = CorpusStore(settings.corpus_path)
    store.append(_row(text="a", author=ALICE, source=SOURCE_MENTION))
    store.append(_row(text="b", author=ALICE, source=SOURCE_MENTION))
    store.append(_row(text="c", author=BOB, source=SOURCE_PROMPT))

    assert store.counts_by_source() == {SOURCE_MENTION: 2, SOURCE_PROMPT: 1}


def test_count_and_ids(settings):
    store = CorpusStore(settings.corpus_path)
    a, b = _row(text="a", author=ALICE), _row(text="b", author=BOB)
    store.append(a)
    store.append(b)

    assert store.count() == 2
    assert store.ids() == {a.id, b.id}


def test_approx_tokens_is_utf8_length_plus_two_structural_tokens_per_row():
    rows = [_row(text="hi", author=ALICE), _row(text="héllo", author=BOB)]  # é is 2 bytes utf-8

    assert approx_tokens(rows) == (len("hi".encode()) + 2) + (len("héllo".encode()) + 2)


# --- backfill_corpus ---------------------------------------------------------


def test_backfill_flattens_one_interaction_into_a_prompt_row_and_a_correction_row(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="hi", chosen="hey", prompt_author=asker, signal_author=helper, rejected="junk")
    )

    result = backfill_corpus(settings, log=log, ids=ids)

    assert result == BackfillResult(scanned=1, considered=2, added=2)
    rows = {r.source: r for r in CorpusStore(settings.corpus_path).all()}
    assert rows[SOURCE_PROMPT].text == "hi"
    assert rows[SOURCE_PROMPT].author == asker
    assert rows[SOURCE_CORRECTION].text == "hey"
    assert rows[SOURCE_CORRECTION].author == helper


def test_backfill_never_stores_the_rejected_text(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(
            prompt="hi",
            chosen="hey",
            prompt_author=asker,
            signal_author=helper,
            rejected="the model babbled this nonsense",
        )
    )

    backfill_corpus(settings, log=log, ids=ids)

    texts = [r.text for r in CorpusStore(settings.corpus_path).all()]
    assert "the model babbled this nonsense" not in texts
    assert texts == ["hi", "hey"] or sorted(texts) == ["hey", "hi"]


def test_backfill_preserves_the_interactions_created_at_on_both_corpus_rows(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(
            prompt="hi",
            chosen="hey",
            prompt_author=asker,
            signal_author=helper,
            created_at="2020-05-04T00:00:00+00:00",
        )
    )

    backfill_corpus(settings, log=log, ids=ids)

    for row in CorpusStore(settings.corpus_path).all():
        assert row.created_at == "2020-05-04T00:00:00+00:00"


def test_backfill_is_idempotent_byte_for_byte(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="hi", chosen="hey", prompt_author=asker, signal_author=helper)
    )

    first = backfill_corpus(settings, log=log, ids=ids)
    after_first = settings.corpus_path.read_bytes()
    second = backfill_corpus(settings, log=log, ids=ids)
    after_second = settings.corpus_path.read_bytes()

    assert first.added == 2
    assert second.added == 0
    assert second.skipped_duplicate == second.considered == 2
    assert after_second == after_first


def test_backfill_skips_a_piece_whose_author_has_no_corrections_grant(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")  # the asker consents; the helper never answered
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="hi", chosen="hey", prompt_author=asker, signal_author=helper)
    )

    result = backfill_corpus(settings, log=log, ids=ids)

    assert result.added == 1
    assert result.skipped_consent == 1
    texts = [r.text for r in CorpusStore(settings.corpus_path).all()]
    assert texts == ["hi"]  # only the consented author's half made it in


def test_backfill_skips_text_matching_the_blocklist_even_though_it_predates_the_term(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    # Stored back when "badword" was perfectly innocuous.
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="hi", chosen="a badword here", prompt_author=asker, signal_author=helper)
    )
    blocklist = Blocklist(terms=frozenset({"badword"}))

    result = backfill_corpus(settings, log=log, ids=ids, blocklist=blocklist)

    assert result.added == 1
    assert result.skipped_blocklist == 1
    texts = [r.text for r in CorpusStore(settings.corpus_path).all()]
    assert "a badword here" not in texts
    assert texts == ["hi"]


def test_backfill_skips_empty_text(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="   ", chosen="hey", prompt_author=asker, signal_author=helper)
    )

    result = backfill_corpus(settings, log=log, ids=ids)

    assert result.added == 1
    assert result.skipped_empty == 1
    texts = [r.text for r in CorpusStore(settings.corpus_path).all()]
    assert texts == ["hey"]


def test_backfill_skips_someone_who_explicitly_declined_the_corpus_scope(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw", SCOPE_CORRECTIONS)
    consent.decline("alice-raw", SCOPE_CORPUS)  # granted corrections, said no to corpus
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="hi", chosen="hey", prompt_author=asker, signal_author=helper)
    )

    result = backfill_corpus(settings, log=log, ids=ids)

    assert result.added == 1
    assert result.skipped_consent == 1
    texts = [r.text for r in CorpusStore(settings.corpus_path).all()]
    assert texts == ["hey"]


def test_backfill_does_not_duplicate_a_row_the_live_capture_path_already_stored(settings, ids, log):
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-raw")
    consent.grant("bob-raw")
    asker, helper = ids.user("alice-raw"), ids.user("bob-raw")
    corpus = CorpusStore(settings.corpus_path)
    # As if the asker had already pinged the bot with "hi" and it got captured live.
    corpus.append(_row(text="hi", author=asker, source=SOURCE_MENTION))
    InteractionStore(settings.interactions_path).append(
        _interaction(prompt="hi", chosen="hey", prompt_author=asker, signal_author=helper)
    )

    result = backfill_corpus(settings, log=log, ids=ids)

    assert result.added == 1  # only the correction half is new
    assert result.skipped_duplicate == 1
    rows = corpus.all()
    assert sorted(r.text for r in rows) == ["hey", "hi"]
    assert len([r for r in rows if r.text == "hi"]) == 1  # never doubled up
