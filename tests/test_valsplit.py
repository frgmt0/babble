"""valsplit is the split's single definition: the torch-free answer to "would
this row land in val?" must agree exactly with what the trainer materialises."""

from __future__ import annotations

from babble.corpus import SOURCE_MENTION, CorpusRow, make_corpus_id
from babble.trainer import split_rows
from babble.valsplit import val_holdout_size, val_id_set


def _rows(n: int) -> list[CorpusRow]:
    return [
        CorpusRow(
            id=make_corpus_id(f"text number {i}", "author"),
            text=f"text number {i}",
            author="author",
            source=SOURCE_MENTION,
        )
        for i in range(n)
    ]


def test_val_id_set_matches_the_trainer_split_exactly(settings):
    rows = _rows(37)
    split = split_rows(rows, settings)
    held = val_id_set(
        [r.id for r in rows],
        val_fraction=settings.val_fraction,
        val_min_rows=settings.val_min_rows,
    )
    assert split.enabled
    assert {r.id for r in split.val} == held
    assert len(held) == val_holdout_size(37, settings.val_fraction)


def test_val_id_set_holds_nothing_out_below_the_floor(settings):
    rows = _rows(5)
    split = split_rows(rows, settings)
    held = val_id_set(
        [r.id for r in rows],
        val_fraction=settings.val_fraction,
        val_min_rows=settings.val_min_rows,
    )
    assert not split.enabled and split.val == []
    assert held == set()
