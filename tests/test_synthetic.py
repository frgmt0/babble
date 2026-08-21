"""The synthetic-correction generator: which corpus rows read as a reply,
what prompt gets postulated for them, that the pairs are stored separately
from human corrections and are re-runnable, and that post-train only touches
them when explicitly told to."""

from __future__ import annotations

import json

import pytest

from babble.blocklist import Blocklist
from babble.consent import ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.synthetic import (
    SyntheticPair,
    SyntheticPairStore,
    generate_synthetic_pairs,
    is_reactive,
    make_synthetic_id,
    reactivity_score,
    synthesize_prompt,
    synthetic_pair_count,
    trainable_synthetic_pairs,
)


def _seed_corpus_row(settings, text: str, author: str = "author-raw", *, source: str = SOURCE_MENTION) -> CorpusRow:
    row = CorpusRow(
        id=make_corpus_id(text, author),
        text=text,
        author=author,
        source=source,
        created_at="2026-01-01T00:00:00+00:00",
    )
    CorpusStore(settings.corpus_path).append(row)
    return row


# --- reactivity heuristic ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "well the visual shells were also just *fine* again i wasnt drawn to *any* of them",
        "yeah that makes sense to me",
        "no i dont think thats right",
        "same here honestly",
    ],
)
def test_reply_shaped_rows_score_as_reactive(text):
    assert is_reactive(text)


@pytest.mark.parametrize(
    "text",
    [
        "hola",
        "drat",
        "astute observations",
        "https://tenor.com/view/whatever",
        "",
        "   ",
    ],
)
def test_standalone_or_empty_rows_do_not_score_as_reactive(text):
    assert not is_reactive(text)


def test_reactivity_score_is_monotonic_with_is_reactive():
    # is_reactive is just a threshold on reactivity_score -- pin that contract.
    assert reactivity_score("hola") < reactivity_score("yeah totally agree")


# --- prompt postulation ------------------------------------------------------


def test_synthesize_prompt_never_touches_the_response_text():
    """The whole point: the response side must stay verbatim corpus text. This
    module never rewrites it -- `synthesize_prompt` only ever returns a new
    prompt, never a modified response."""
    text = "well the visual shells were also just *fine* again i wasnt drawn to *any* of them"
    prompt, method = synthesize_prompt(text)
    assert isinstance(prompt, str) and prompt
    assert isinstance(method, str) and method
    # No accidental leakage of the exact response sentence into the prompt.
    assert prompt != text


def test_synthesize_prompt_is_deterministic():
    text = "yeah that tracks"
    assert synthesize_prompt(text) == synthesize_prompt(text)


def test_synthesize_prompt_picks_a_method_per_cue():
    assert synthesize_prompt("is this thing on?")[1] == "reacts_to_question"
    assert synthesize_prompt("yeah for sure")[1] == "agreement"
    assert synthesize_prompt("nah not really")[1] == "disagreement"
    assert synthesize_prompt("it was also broken again")[1] == "continuation"


# --- SyntheticPairStore -------------------------------------------------


def test_synthetic_pair_store_dedupes_by_id(settings):
    store = SyntheticPairStore(settings.synthetic_pairs_path)
    pair = SyntheticPair(
        id=make_synthetic_id("row1", "what happened", "generic"),
        prompt="what happened",
        response="it broke again",
        source_row_id="row1",
        method="generic",
    )
    assert store.append(pair) is True
    assert store.append(pair) is False  # identical id -> no-op
    assert store.count() == 1


def test_synthetic_pair_store_is_a_separate_file_from_interactions(settings):
    """Structural guarantee that synthetic pairs can never be mistaken for
    human corrections: they live at a different path entirely."""
    assert settings.synthetic_pairs_path != settings.interactions_path
    assert settings.synthetic_pairs_path.name == "synthetic_pairs.jsonl"


def test_synthetic_pair_on_disk_is_labelled_synthetic(settings):
    store = SyntheticPairStore(settings.synthetic_pairs_path)
    store.append(
        SyntheticPair(
            id="abc", prompt="p", response="r", source_row_id="row1", method="generic"
        )
    )
    line = settings.synthetic_pairs_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["synthetic"] is True


# --- generate_synthetic_pairs ------------------------------------------------


def test_generate_synthetic_pairs_only_uses_reactive_consented_rows(settings, ids):
    ConsentStore(settings.consent_path).grant("author-raw")
    author = ids.user("author-raw")
    _seed_corpus_row(settings, "well that was also *fine* again i guess", author)
    _seed_corpus_row(settings, "a completely standalone statement about the weather", author)

    result = generate_synthetic_pairs(
        settings, ids=ids, blocklist=Blocklist.load(), continuations=False
    )

    assert result.scanned == 2
    assert result.reactive == 1
    assert result.generated == 1
    assert synthetic_pair_count(settings) == 1

    (pair,) = SyntheticPairStore(settings.synthetic_pairs_path).all()
    assert pair.response == "well that was also *fine* again i guess"  # verbatim, untouched
    assert pair.prompt  # something was postulated


def test_generate_synthetic_pairs_skips_unconsented_rows(settings, ids):
    # Never granted -- the row must never be read into a synthetic pair.
    _seed_corpus_row(settings, "yeah that also happened again", ids.user("nope-raw"))

    result = generate_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())

    assert result.scanned == 0
    assert synthetic_pair_count(settings) == 0


def test_generate_synthetic_pairs_is_idempotent_on_rerun(settings, ids):
    ConsentStore(settings.consent_path).grant("author-raw")
    author = ids.user("author-raw")
    _seed_corpus_row(settings, "well that was also *fine* again i guess", author)

    first = generate_synthetic_pairs(
        settings, ids=ids, blocklist=Blocklist.load(), continuations=False
    )
    assert first.generated == 1

    second = generate_synthetic_pairs(
        settings, ids=ids, blocklist=Blocklist.load(), continuations=False
    )
    assert second.generated == 0
    assert second.skipped_duplicate == 1
    assert synthetic_pair_count(settings) == 1  # not doubled


def test_generate_synthetic_pairs_adds_only_new_rows_as_corpus_grows(settings, ids):
    """Re-running after the corpus grows must add pairs for the new rows
    without touching or duplicating what was already generated."""
    ConsentStore(settings.consent_path).grant("author-raw")
    author = ids.user("author-raw")
    _seed_corpus_row(settings, "well that was also *fine* again i guess", author)
    generate_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load(), continuations=False)
    assert synthetic_pair_count(settings) == 1

    _seed_corpus_row(settings, "yeah honestly same here again", author)
    result = generate_synthetic_pairs(
        settings, ids=ids, blocklist=Blocklist.load(), continuations=False
    )

    assert result.generated == 1
    assert synthetic_pair_count(settings) == 2


# --- trainable_synthetic_pairs: consent re-checked at train time ------------


def test_trainable_synthetic_pairs_drops_pairs_whose_author_withdrew(settings, ids):
    consent = ConsentStore(settings.consent_path)
    consent.grant("author-raw")
    author = ids.user("author-raw")
    _seed_corpus_row(settings, "well that was also *fine* again i guess", author)
    generate_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load(), continuations=False)
    assert len(trainable_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())) == 1

    # A withdrawal purges the corpus row (mirroring `!babble forget`); even if
    # the stored synthetic pair is left untouched on disk, it must stop being
    # trainable once its source row no longer consents.
    consent.withdraw("author-raw")

    assert trainable_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load()) == []


def test_trainable_synthetic_pairs_sorted_deterministically(settings, ids):
    ConsentStore(settings.consent_path).grant("author-raw")
    author = ids.user("author-raw")
    for i in range(5):
        _seed_corpus_row(settings, f"yeah that also happened again number {i}", author)
    generate_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())

    first = trainable_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())
    second = trainable_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())
    assert [p.id for p in first] == [p.id for p in second]
    assert [p.id for p in first] == sorted(p.id for p in first)


# --- CLI wiring --------------------------------------------------------------


@pytest.mark.parametrize("command", ["synth-generate", "synth-status", "synth-corpus"])
def test_synth_subcommands_are_registered(command):
    from babble.cli import build_parser

    args = build_parser().parse_args([command])
    assert args.command == command


def test_post_train_subcommand_has_include_synthetic_flag():
    from babble.cli import build_parser

    args = build_parser().parse_args(["post-train", "--include-synthetic"])
    assert args.include_synthetic is True

    args = build_parser().parse_args(["post-train"])
    assert args.include_synthetic is False


# --- the combined default: postulated + continuation pairs ------------------


def test_default_generation_includes_continuation_cut_pairs(settings, ids):
    ConsentStore(settings.consent_path).grant("author-raw")
    author = ids.user("author-raw")
    _seed_corpus_row(settings, "well that was also *fine* again i guess", author)

    result = generate_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())

    # One postulated pair for the reactive row, plus its continuation cuts --
    # counted separately so a null result on one method can be traced.
    assert result.generated_postulated == 1
    assert result.generated_continuation >= 1
    assert result.generated == result.generated_postulated + result.generated_continuation

    pairs = SyntheticPairStore(settings.synthetic_pairs_path).all()
    continuation = [p for p in pairs if p.method == "continuation_cut"]
    assert continuation
    for pair in continuation:
        # BOTH halves verbatim slices of the source row, nothing invented.
        original = "well that was also *fine* again i guess"
        assert original.startswith(pair.prompt)
        assert original.endswith(pair.response)


def test_continuation_pairs_die_with_their_source_rows_consent(settings, ids):
    consent = ConsentStore(settings.consent_path)
    consent.grant("author-raw")
    author = ids.user("author-raw")
    _seed_corpus_row(settings, "well that was also *fine* again i guess", author)
    generate_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())
    assert trainable_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load())

    consent.withdraw("author-raw")

    assert trainable_synthetic_pairs(settings, ids=ids, blocklist=Blocklist.load()) == []
