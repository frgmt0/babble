"""The external base-stage corpus loaders: words + stories, and the fail-loud
contract. None of this touches consent -- it is nobody's Discord message."""

from __future__ import annotations

import json

import pytest

from babble.external import (
    STORY_SEPARATOR,
    EmptyCorpusError,
    load_stories,
    load_words,
    prepare_base_corpus,
    read_base_rows,
)


def _wordlist(tmp_path, lines):
    p = tmp_path / "words.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _stories(tmp_path, stories):
    p = tmp_path / "stories.txt"
    p.write_text(STORY_SEPARATOR.join(stories), encoding="utf-8")
    return p


# --- word list -----------------------------------------------------------


def test_load_words_keeps_real_words_and_drops_the_junk(tmp_path):
    src = _wordlist(tmp_path, ["Apple", "007bond", "banana", "b4nter", "cat's", "12345", "dog"])
    words = load_words(src)
    # lowercased, apostrophes allowed, digit-bearing cracklib junk dropped.
    assert words == ["apple", "banana", "cat's", "dog"]


def test_load_words_dedupes_and_respects_the_limit(tmp_path):
    src = _wordlist(tmp_path, ["cat", "CAT", "dog", "bird", "cat"])
    assert load_words(src) == ["cat", "dog", "bird"]
    assert load_words(src, limit=2) == ["cat", "dog"]


def test_load_words_on_a_missing_file_fails_loud(tmp_path):
    with pytest.raises(EmptyCorpusError):
        load_words(tmp_path / "nope.txt")


# --- stories -------------------------------------------------------------


def test_load_stories_splits_on_the_separator(tmp_path):
    src = _stories(tmp_path, ["A cat sat.", "A dog ran.", "  "])
    assert load_stories(src) == ["A cat sat.", "A dog ran."]


def test_load_stories_char_budget_cuts_on_a_story_boundary(tmp_path):
    src = _stories(tmp_path, ["one one one", "two two two", "three three three"])
    # A budget landing inside the second story keeps only the first whole story,
    # never a fragment.
    kept = load_stories(src, char_budget=len("one one one") + 5)
    assert kept == ["one one one"]


# --- the combined corpus + fail-loud -------------------------------------


def test_prepare_writes_jsonl_and_read_base_rows_round_trips(settings, tmp_path):
    words = _wordlist(tmp_path, ["apple", "banana"])
    stories = _stories(tmp_path, ["A cat sat on a mat.", "A dog ran home."])

    result = prepare_base_corpus(settings, wordlist_path=words, stories_path=stories)

    assert result.words == 2
    assert result.stories == 2
    assert result.total_rows == 4
    assert settings.base_corpus_path.exists()

    # The file is valid JSONL, one {"text": ...} per line.
    lines = settings.base_corpus_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all("text" in json.loads(line) for line in lines)

    rows = read_base_rows(settings)
    assert rows == ["apple", "banana", "A cat sat on a mat.", "A dog ran home."]


def test_prepare_never_touches_the_human_corpus(settings, tmp_path):
    # The external corpus lives at base_corpus_path, a different file from the
    # consented human corpus_path. Preparing it must not create or alter that.
    words = _wordlist(tmp_path, ["apple"])
    stories = _stories(tmp_path, ["A cat sat."])
    prepare_base_corpus(settings, wordlist_path=words, stories_path=stories)
    assert settings.base_corpus_path != settings.corpus_path
    assert not settings.corpus_path.exists()


def test_prepare_with_no_usable_text_fails_loud(settings, tmp_path):
    words = _wordlist(tmp_path, ["123", "45"])  # nothing usable
    stories = _stories(tmp_path, ["   "])  # nothing after strip
    with pytest.raises(EmptyCorpusError):
        prepare_base_corpus(settings, wordlist_path=words, stories_path=stories)


def test_read_base_rows_before_prepare_fails_loud(settings):
    with pytest.raises(EmptyCorpusError):
        read_base_rows(settings)
