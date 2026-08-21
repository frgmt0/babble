"""Train/val split for correction pairs -- the shared definition
`pairaugment.py` and `posttrain.py` both call so "train-side" can never mean
two different things to the generator and the trainer."""

from __future__ import annotations

from dataclasses import dataclass

from babble.pairsplit import PAIR_VAL_MIN_PAIRS, pair_split, pair_val_ids


@dataclass(frozen=True)
class _Item:
    id: str


def _items(n: int) -> list[_Item]:
    return [_Item(id=f"pair-{i:03d}") for i in range(n)]


def test_below_the_floor_nothing_is_held_out():
    ids = [f"pair-{i}" for i in range(PAIR_VAL_MIN_PAIRS - 1)]
    assert pair_val_ids(ids) == set()


def test_at_and_above_the_floor_something_is_held_out():
    ids = [f"pair-{i}" for i in range(50)]
    held = pair_val_ids(ids)
    assert held
    assert held.issubset(set(ids))
    # Roughly the configured fraction, not exact -- see valsplit.val_holdout_size.
    assert 5 <= len(held) <= 15


def test_deterministic_for_the_same_ids():
    ids = [f"pair-{i}" for i in range(50)]
    assert pair_val_ids(ids) == pair_val_ids(list(ids))


def test_split_partitions_every_item_exactly_once():
    items = _items(50)
    train, val = pair_split(items)
    assert set(i.id for i in train) | set(i.id for i in val) == {i.id for i in items}
    assert set(i.id for i in train) & set(i.id for i in val) == set()
    assert len(train) + len(val) == len(items)


def test_split_is_a_function_of_id_not_of_list_order():
    items = _items(50)
    train_a, val_a = pair_split(items)
    train_b, val_b = pair_split(list(reversed(items)))
    assert {i.id for i in train_a} == {i.id for i in train_b}
    assert {i.id for i in val_a} == {i.id for i in val_b}


def test_below_floor_split_puts_everything_in_train():
    items = _items(PAIR_VAL_MIN_PAIRS - 1)
    train, val = pair_split(items)
    assert len(train) == len(items)
    assert val == []
