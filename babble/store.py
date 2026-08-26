"""The corpus: every triple the bot has been taught, one JSON object per line.

A row is `(prompt, rejected, chosen)` -- what someone said, what the bot got
wrong, and what it should have said. Approvals reuse the same shape with
`rejected = None` and `chosen` being the bot's own answer, because a 👍 means
"that one was already right". Rejections (a 👎) are the mirror image: the bot's
answer on the `rejected` side and `chosen` empty -- "that one was wrong, and
nobody has said yet what would have been right". They are never trained on as
a target (`post_state.trainable_pairs` excludes them); they exist to tell us,
and the published dataset, which answers people flagged.

Authors appear only as salted hashes. There is no code path that writes a raw
Discord id into this file, which is what makes the export guard cheap to trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .util import atomic_write_text, utcnow_iso

CORRECTION = "correction"
APPROVAL = "approval"
REJECTION = "rejection"


@dataclass(frozen=True)
class Interaction:
    id: str
    signal: str  # CORRECTION | APPROVAL | REJECTION
    prompt: str
    rejected: str | None
    chosen: str
    prompt_author: str  # pseudonym, never a Discord id
    signal_author: str  # pseudonym of whoever corrected or reacted
    weight: float
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "signal": self.signal,
            "prompt": self.prompt,
            "rejected": self.rejected,
            "chosen": self.chosen,
            "prompt_author": self.prompt_author,
            "signal_author": self.signal_author,
            "weight": self.weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Interaction":
        return cls(
            id=raw["id"],
            signal=raw["signal"],
            prompt=raw.get("prompt", ""),
            rejected=raw.get("rejected"),
            chosen=raw.get("chosen", ""),
            prompt_author=raw.get("prompt_author", ""),
            signal_author=raw.get("signal_author", ""),
            weight=float(raw.get("weight", 1.0)),
            created_at=raw.get("created_at", ""),
        )

    def involves(self, author: str) -> bool:
        return author in (self.prompt_author, self.signal_author)


def make_row_id(signal: str, prompt: str, chosen: str, prompt_author: str, signal_author: str) -> str:
    """Content-addressed id.

    Timestamps are deliberately excluded: the same person thumbs-upping the same
    answer twice is one fact, not two, so it collapses to one row and the export
    stays idempotent.
    """
    payload = "\x1f".join([signal, prompt, chosen, prompt_author, signal_author])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class InteractionStore:
    """Append-only JSONL, rewritten in place only to honour a purge."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: Interaction) -> bool:
        """Add a row. Returns False if an identical one is already present."""
        if any(existing.id == row.id for existing in self.all()):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return True

    def all(self) -> list[Interaction]:
        if not self.path.exists():
            return []
        rows: list[Interaction] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(Interaction.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # a torn line never takes the dataset down with it
        return rows

    def purge_author(self, author: str) -> int:
        """Delete every row the pseudonym appears in, on either side.

        Used by `!babble forget`.
        """
        return self.purge(lambda r: r.involves(author))

    def purge(self, doomed: Callable[[Interaction], bool]) -> int:
        """Delete every row `doomed` says to. Rewrites the file atomically so an
        interrupted purge cannot leave a partially-deleted corpus.
        """
        rows = self.all()
        keep = [r for r in rows if not doomed(r)]
        removed = len(rows) - len(keep)
        if removed:
            body = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in keep)
            atomic_write_text(self.path, body)
        return removed

    def count(self) -> int:
        return len(self.all())

    def counts_by_signal(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for row in self.all():
            tally[row.signal] = tally.get(row.signal, 0) + 1
        return tally
