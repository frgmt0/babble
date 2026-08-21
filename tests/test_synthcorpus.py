"""Synthetic corpus rows: recombined strictly from corpus material, labelled,
stored apart from the human corpus, and only ever trained on by explicit flag."""

from __future__ import annotations

import json

import pytest

from babble.blocklist import Blocklist
from babble.consent import SCOPE_CORPUS, ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.synthcorpus import (
    MarkovChain,
    SyntheticCorpusStore,
    generate_synthetic_corpus,
    make_synthetic_row_id,
    synthetic_row_count,
    trainable_synthetic_rows,
)
from babble.synthetic import continuation_cuts


def _seed_corpus(settings, ids, texts, *, author_raw="alice-raw"):
    ConsentStore(settings.consent_path).grant(author_raw, SCOPE_CORPUS)
    store = CorpusStore(settings.corpus_path)
    author = ids.user(author_raw)
    for text in texts:
        store.append(
            CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=SOURCE_MENTION)
        )


TEXTS = [
    "the cat sat on the mat and looked around",
    "the cat ran up the tree after a squirrel",
    "honestly the tree was way too tall for that",
    "no way the squirrel made it look easy",
    "looked around and saw nothing but the mat",
    "way too tall and way too fast honestly",
]


# --- the chain's corpus-internal guarantee ---------------------------------


def test_every_generated_word_and_transition_comes_from_the_corpus(settings, ids):
    chain = MarkovChain(TEXTS)
    vocab = {w for t in TEXTS for w in t.split()}
    bigrams = {(a, b) for t in TEXTS for a, b in zip(t.split(), t.split()[1:])}

    import random

    rng = random.Random(7)
    for _ in range(200):
        row = chain.sample_row(rng)
        words = row.split()
        assert words, "the chain never emits an empty row from a non-empty corpus"
        for w in words:
            assert w in vocab
        for pair in zip(words, words[1:]):
            assert pair in bigrams


def test_generation_is_deterministic_for_a_seed(settings, ids):
    _seed_corpus(settings, ids, TEXTS)
    generate_synthetic_corpus(settings, count=10, seed=3, ids=ids)
    first = [r.text for r in SyntheticCorpusStore(settings.synthetic_corpus_path).all()]

    settings.synthetic_corpus_path.unlink()
    generate_synthetic_corpus(settings, count=10, seed=3, ids=ids)
    second = [r.text for r in SyntheticCorpusStore(settings.synthetic_corpus_path).all()]

    assert first == second


def test_generated_rows_never_replay_a_real_row_verbatim(settings, ids):
    _seed_corpus(settings, ids, TEXTS)
    generate_synthetic_corpus(settings, count=50, seed=0, ids=ids)
    real = set(TEXTS)
    for row in SyntheticCorpusStore(settings.synthetic_corpus_path).all():
        assert row.text not in real


# --- storage discipline ----------------------------------------------------


def test_rows_live_in_their_own_file_and_are_labelled(settings, ids):
    _seed_corpus(settings, ids, TEXTS)
    result = generate_synthetic_corpus(settings, count=5, seed=0, ids=ids)

    assert result.generated > 0
    assert settings.synthetic_corpus_path != settings.corpus_path
    # The human corpus did not grow by a single row.
    assert CorpusStore(settings.corpus_path).count() == len(TEXTS)
    for line in settings.synthetic_corpus_path.read_text().splitlines():
        raw = json.loads(line)
        assert raw["synthetic"] is True
        assert raw["method"] == "markov_order2"


def test_rebuild_replaces_the_file_instead_of_appending(settings, ids):
    _seed_corpus(settings, ids, TEXTS)
    generate_synthetic_corpus(settings, count=8, seed=0, ids=ids)
    generate_synthetic_corpus(settings, count=8, seed=1, ids=ids)
    grown = synthetic_row_count(settings)

    result = generate_synthetic_corpus(settings, count=8, seed=2, rebuild=True, ids=ids)

    assert grown > result.stored_total or result.stored_total <= 8
    assert synthetic_row_count(settings) == result.stored_total <= 8


def test_trainable_rows_respect_the_blocklist(settings, ids, monkeypatch):
    _seed_corpus(settings, ids, TEXTS)
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)
    rows = trainable_synthetic_rows(settings)
    assert rows

    # A blocklist that matches everything: nothing is trainable any more.
    class _BlockAll:
        def matches(self, *texts) -> bool:
            return True

    assert trainable_synthetic_rows(settings, blocklist=_BlockAll()) == []


def test_ids_are_content_addressed(settings):
    assert make_synthetic_row_id("a b c", "markov_order2") == make_synthetic_row_id(
        "a b c", "markov_order2"
    )
    assert make_synthetic_row_id("a b c", "markov_order2") != make_synthetic_row_id(
        "a b d", "markov_order2"
    )


# --- the trainer only touches synthetic rows by explicit flag ---------------


def test_train_ignores_synthetic_rows_unless_flagged(settings, ids, monkeypatch):
    from babble.trainer import train

    _seed_corpus(settings, ids, [f"row number {i} with more text to chew on" for i in range(20)])
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)

    calls = []
    import babble.synthcorpus as synthcorpus

    real = synthcorpus.trainable_synthetic_rows
    monkeypatch.setattr(
        synthcorpus, "trainable_synthetic_rows", lambda *a, **k: calls.append(1) or real(*a, **k)
    )

    settings.train_synthetic = False
    result = train(settings, force=True, steps=2, seed=1, echo=False, ids=ids)
    assert result.ran
    assert calls == []  # the synthetic store was never even read

    settings.train_synthetic = True
    result = train(settings, force=True, steps=2, seed=1, echo=False, ids=ids)
    assert result.ran
    assert calls  # flag on -> the synthetic rows joined the batch pool


# --- continuation cuts (synthetic.py) --------------------------------------


def test_continuation_cuts_are_verbatim_slices():
    text = "well the visual shells were also just fine again honestly"
    for prompt, response in continuation_cuts(text, cuts=2):
        assert text.startswith(prompt)
        assert text.endswith(response)
        # Nothing invented: prompt + separator + response is the original.
        assert (prompt + " " + response) == text or (prompt + response) == text


def test_continuation_cuts_skip_short_rows():
    assert continuation_cuts("hi", cuts=2) == []
    assert continuation_cuts("no fuck you", cuts=2) == []  # 3 words < the 4-word floor


def test_continuation_cuts_deterministic_and_bounded():
    text = "one two three four five six seven eight nine"
    first = continuation_cuts(text, cuts=2)
    assert first == continuation_cuts(text, cuts=2)
    assert 1 <= len(first) <= 2
    assert all(p and r for p, r in first)


# --- val-side rows never feed the chain -------------------------------------


def _uniq_texts(n: int) -> list[str]:
    # One marker word per row, so a marker in a generated row proves exactly
    # which source rows fed the chain.
    return [f"uniq{i:02d} the cat sat on the mat tonight" for i in range(n)]


def test_exclude_val_keeps_heldout_phrasing_out_of_the_chain(settings, ids):
    from babble.valsplit import val_holdout_size, val_id_set

    texts = _uniq_texts(24)  # past val_min_rows, so the split is live
    _seed_corpus(settings, ids, texts)
    author = ids.user("alice-raw")
    row_ids = [make_corpus_id(t, author) for t in texts]
    held = val_id_set(
        row_ids, val_fraction=settings.val_fraction, val_min_rows=settings.val_min_rows
    )
    val_markers = {t.split()[0] for t, rid in zip(texts, row_ids) if rid in held}
    assert val_markers  # the split actually held something out

    result = generate_synthetic_corpus(settings, count=50, seed=0, ids=ids)

    assert result.excluded_val_rows == len(held) == val_holdout_size(24, settings.val_fraction)
    assert result.source_rows == 24 - len(held)
    for row in SyntheticCorpusStore(settings.synthetic_corpus_path).all():
        assert not val_markers & set(row.text.split())


def test_include_val_sources_restores_the_old_whole_corpus_chain(settings, ids):
    _seed_corpus(settings, ids, _uniq_texts(24))

    result = generate_synthetic_corpus(settings, count=5, seed=0, exclude_val=False, ids=ids)

    assert result.source_rows == 24
    assert result.excluded_val_rows == 0


def test_exclude_val_is_a_noop_below_the_split_floor(settings, ids):
    _seed_corpus(settings, ids, TEXTS)  # 6 rows < val_min_rows: nothing held out

    result = generate_synthetic_corpus(settings, count=5, seed=0, ids=ids)

    assert result.excluded_val_rows == 0
    assert result.source_rows == len(TEXTS)


# --- staleness: corpus growth rebuilds the file -----------------------------
#
# The val holdout is a slice of the whole id population, so appending corpus
# rows migrates some existing rows train -> val. A file generated before that
# migration can contain splices of now-held-out phrasing; the trainer must
# rebuild it before mixing, or the leak `exclude_val` closed comes back.


def _marked_texts(prefix: str, n: int) -> list[str]:
    # One marker word per row (proves which source rows fed the chain, as in
    # `_uniq_texts`) but over VARIED bodies -- identical bodies would make
    # every sample a verbatim replay, which the generator skips, and an empty
    # file cannot exercise staleness at all.
    return [f"{prefix}{i:02d} {TEXTS[i % len(TEXTS)]}" for i in range(n)]


def _grow_corpus(settings, ids, n: int) -> None:
    _seed_corpus(settings, ids, _marked_texts("fresh", n))


def test_refresh_is_a_noop_when_absent_or_fresh(settings, ids):
    from babble.synthcorpus import refresh_synthetic_corpus_if_stale

    assert refresh_synthetic_corpus_if_stale(settings, ids=ids) is None  # no file at all

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)

    assert refresh_synthetic_corpus_if_stale(settings, ids=ids) is None  # same corpus: fresh


def test_corpus_growth_marks_the_file_stale_and_one_rebuild_clears_it(settings, ids):
    from babble.synthcorpus import refresh_synthetic_corpus_if_stale

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)
    _grow_corpus(settings, ids, 8)

    result = refresh_synthetic_corpus_if_stale(settings, ids=ids)

    assert result is not None and result.generated > 0
    assert refresh_synthetic_corpus_if_stale(settings, ids=ids) is None  # rebuilt: fresh again


def test_rebuilt_file_excludes_the_holdout_of_the_grown_corpus(settings, ids):
    from babble.corpus import CorpusStore
    from babble.synthcorpus import refresh_synthetic_corpus_if_stale
    from babble.valsplit import val_id_set

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=20, seed=0, ids=ids)
    _grow_corpus(settings, ids, 8)

    assert refresh_synthetic_corpus_if_stale(settings, ids=ids) is not None

    rows = CorpusStore(settings.corpus_path).all()
    held = val_id_set(
        [r.id for r in rows], val_fraction=settings.val_fraction, val_min_rows=settings.val_min_rows
    )
    val_markers = {r.text.split()[0] for r in rows if r.id in held}
    assert val_markers  # the grown split holds something out
    for row in SyntheticCorpusStore(settings.synthetic_corpus_path).all():
        assert not val_markers & set(row.text.split())


def test_file_without_meta_counts_as_stale(settings, ids):
    from babble.synthcorpus import refresh_synthetic_corpus_if_stale

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)
    settings.synthetic_corpus_path.with_suffix(".meta.json").unlink()

    assert refresh_synthetic_corpus_if_stale(settings, ids=ids) is not None  # unknown provenance


def test_include_val_sources_file_counts_as_stale(settings, ids):
    from babble.synthcorpus import refresh_synthetic_corpus_if_stale

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=10, seed=0, exclude_val=False, ids=ids)

    result = refresh_synthetic_corpus_if_stale(settings, ids=ids)

    assert result is not None  # whole-corpus chain != train-side chain
    assert result.excluded_val_rows > 0  # and the rebuild is holdout-clean


def test_append_after_growth_keeps_the_file_stale(settings, ids):
    from babble.synthcorpus import refresh_synthetic_corpus_if_stale

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)
    _grow_corpus(settings, ids, 8)
    # An append against the grown corpus cannot vouch for the rows already
    # stored from the old one -- the file must still read as stale.
    generate_synthetic_corpus(settings, count=5, seed=1, ids=ids)

    assert refresh_synthetic_corpus_if_stale(settings, ids=ids) is not None


def test_train_rebuilds_a_stale_file_before_mixing(settings, ids, monkeypatch):
    from babble.trainer import train

    _seed_corpus(settings, ids, _marked_texts("uniq", 24))
    generate_synthetic_corpus(settings, count=10, seed=0, ids=ids)
    _grow_corpus(settings, ids, 8)

    calls = []
    import babble.synthcorpus as synthcorpus

    real = synthcorpus.refresh_synthetic_corpus_if_stale
    monkeypatch.setattr(
        synthcorpus,
        "refresh_synthetic_corpus_if_stale",
        lambda *a, **k: calls.append(real(*a, **k)) or calls[-1],
    )

    settings.train_synthetic = True
    result = train(settings, force=True, steps=2, seed=1, echo=False, ids=ids)

    assert result.ran
    assert len(calls) == 1 and calls[0] is not None  # stale file was rebuilt in-path
    assert real(settings, ids=ids) is None  # and left fresh for the next run
