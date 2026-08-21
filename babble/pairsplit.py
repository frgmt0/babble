"""Train/val split for correction pairs (`Interaction` rows) -- torch-free,
the pair analogue of `valsplit.py`'s corpus-row split.

Why this needs to be its own small function rather than something each
caller re-derives: post-train's pair validation set used to be whatever the
last 20% of a sorted example list happened to be (`posttrain._split_val`,
now gone) -- harmless while the only things in that list were real pairs and
the old postulated-prompt synthetic pairs, both essentially arbitrary with
respect to which ones landed in the tail. It stops being harmless the moment
correction-pair *augmentation* exists, because an augmented pair's whole
safety property is "derived only from a train-side pair" (see
`pairaugment.py`), which requires "train-side" to mean the same thing to
whoever generates the variants and to whoever trains on them. If generation
and training ever used two different splits, a pair that generation
correctly treated as train-side could still land in val at training time (or
vice versa), and the one number the whole feature is judged by -- held-out
pair val loss -- would quietly stop being held-out. `pairaugment.py`
(deciding which real pairs are eligible to paraphrase) and `posttrain.py`
(deciding which real pairs are held out for val) both call `pair_split`, so
the two can never drift apart -- the same discipline `valsplit.py`'s
docstring describes for corpus rows.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from .valsplit import val_id_set

#: Same fraction post-train's old positional split used.
PAIR_VAL_FRACTION = 0.2

#: Below this many real pairs, nothing is held out -- there is too little to
#: spare any (mirrors the old `_split_val`'s `len(examples) < 4` floor).
PAIR_VAL_MIN_PAIRS = 4

__all__ = ["PAIR_VAL_FRACTION", "PAIR_VAL_MIN_PAIRS", "pair_val_ids", "pair_split"]


class _HasId(Protocol):
    id: str


T = TypeVar("T", bound=_HasId)


def pair_val_ids(
    pair_ids: list[str],
    *,
    val_fraction: float = PAIR_VAL_FRACTION,
    val_min_pairs: int = PAIR_VAL_MIN_PAIRS,
) -> set[str]:
    """The ids among `pair_ids` that land on the val side -- the same
    hash-bucket ranking `valsplit.val_id_set` gives corpus rows, applied to
    correction-pair ids instead. Below `val_min_pairs` nothing is held out."""
    return val_id_set(pair_ids, val_fraction=val_fraction, val_min_rows=val_min_pairs)


def pair_split(pairs: list[T]) -> tuple[list[T], list[T]]:
    """Split any id-bearing list into `(train, val)` by `pair_val_ids`.

    Generic over `Interaction` (and anything else with a stable `.id`) so
    `pairaugment.py` and `posttrain.py` both call this instead of each
    re-deriving the id list and the split.
    """
    val_ids = pair_val_ids([p.id for p in pairs])
    train = [p for p in pairs if p.id not in val_ids]
    val = [p for p in pairs if p.id in val_ids]
    return train, val
