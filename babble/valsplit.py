"""Which rows are held out for validation, decided by row id. Torch-free.

This is the identity of the val split, factored out of `trainer.py` so that
code which must not import torch (`synthcorpus.py`, which generates synthetic
rows on the bot box) can still ask "would this row land in val?". The trainer
imports these same functions, so there is exactly one definition of the split
and the two callers cannot drift.

Why the split works this way -- ranked hash buckets rather than per-row
thresholding -- is documented on `trainer.split_rows`, which remains the
place that materialises an actual `Split` of corpus rows.
"""

from __future__ import annotations

import hashlib

__all__ = ["val_bucket", "val_holdout_size", "val_id_set"]

_VAL_SALT = "babble-val-split"


def val_bucket(row_id: str) -> float:
    """A stable float in [0, 1) derived from the row id.

    Hashing the id -- not the row's position in the file, not a shuffle -- is
    what makes the same row land on the same side of the split every time the
    trainer restarts and as more rows are appended around it.
    """
    digest = hashlib.sha256(f"{_VAL_SALT}\x1f{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def val_holdout_size(total: int, fraction: float) -> int:
    """How many rows to hold out of `total`, never all of them and never none.

    Both clamps matter at this corpus size: a 20% split of 21 rows is 4, and
    rounding or a mis-set fraction must not be allowed to leave zero rows on
    either side of the split.
    """
    holdout = round(max(0.0, min(1.0, fraction)) * total)
    return max(1, min(holdout, total - 1))


def val_id_set(row_ids: list[str], *, val_fraction: float, val_min_rows: int) -> set[str]:
    """The ids among `row_ids` that the trainer would hold out for validation.

    Mirrors `trainer.split_rows` exactly: below `val_min_rows` nothing is held
    out, otherwise the `val_holdout_size` ids with the lowest hash bucket are.
    `row_ids` should be the full consented corpus in corpus order -- the split
    is a function of the whole id population, not of any single id.
    """
    if len(row_ids) < val_min_rows:
        return set()
    holdout = val_holdout_size(len(row_ids), val_fraction)
    ranked = sorted(range(len(row_ids)), key=lambda i: (val_bucket(row_ids[i]), row_ids[i], i))
    return {row_ids[i] for i in ranked[:holdout]}
