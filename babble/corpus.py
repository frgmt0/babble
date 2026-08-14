"""The pretraining corpus: unlabelled human text, one JSON object per line.

A row is one piece of writing somebody sent the bot. There is no "right answer"
attached to it and nothing is paired with anything -- that is the whole point.
`store.py` still holds correction *pairs*, because corrections are a real
artifact worth keeping, but this file is what the model is actually trained on.

Authors appear only as salted hashes, and so do the guild and channel a row came
from. There is no code path that writes a raw Discord id into this file, which is
what makes the export guard cheap to trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .util import atomic_write_text, utcnow_iso

# Where a piece of text came from. Provenance is kept so a row can be explained
# later ("why is this in the dataset?") without keeping anything identifying.
SOURCE_MENTION = "mention"  # someone @mentioned the bot
SOURCE_REPLY = "reply"  # someone replied to something the bot said
SOURCE_DM = "dm"  # someone messaged the bot directly
SOURCE_AMBIENT = "ambient"  # whole-channel capture, off by default
SOURCE_CORRECTION = "correction"  # the text of a correction, as its own writing
SOURCE_PROMPT = "prompt"  # what was said to the bot, taken off a correction row

SOURCES = frozenset(
    {
        SOURCE_MENTION,
        SOURCE_REPLY,
        SOURCE_DM,
        SOURCE_AMBIENT,
        SOURCE_CORRECTION,
        SOURCE_PROMPT,
    }
)


@dataclass(frozen=True)
class CorpusRow:
    id: str
    text: str
    author: str  # pseudonym, never a Discord id
    source: str
    guild: str | None = None  # pseudonym (`g_…`) or None for a DM
    channel: str | None = None  # pseudonym (`c_…`)
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "source": self.source,
            "guild": self.guild,
            "channel": self.channel,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "CorpusRow":
        return cls(
            id=raw["id"],
            text=raw.get("text", ""),
            author=raw.get("author", ""),
            source=raw.get("source", ""),
            guild=raw.get("guild"),
            channel=raw.get("channel"),
            created_at=raw.get("created_at", ""),
        )

    def involves(self, author: str) -> bool:
        return author == self.author


def make_corpus_id(text: str, author: str) -> str:
    """Content-addressed id over the text and its author, and nothing else.

    Deliberately excluded: the timestamp, the source and the channel. Somebody
    saying the same thing twice is one piece of writing, not two, so it collapses
    to one row -- and the same sentence arriving by two different routes (typed
    at the bot live, then flattened out of an old correction row by the backfill)
    collapses too. That is what makes the backfill idempotent against a corpus
    that is being appended to at the same time.
    """
    payload = "\x1f".join([text, author])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CorpusStore:
    """Append-only JSONL, rewritten in place only to honour a purge.

    Same contract as `InteractionStore`: appends are atomic-ish line writes,
    duplicate ids are dropped, and a purge rewrites the whole file atomically so
    an interrupted deletion cannot leave a half-deleted corpus behind.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: CorpusRow) -> bool:
        """Add a row. Returns False if an identical one is already present."""
        if row.id in self.ids():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return True

    def all(self) -> list[CorpusRow]:
        if not self.path.exists():
            return []
        rows: list[CorpusRow] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(CorpusRow.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # a torn line never takes the corpus down with it
        return rows

    def ids(self) -> set[str]:
        """Every stored id. Split out so a dedupe check reads as one."""
        return {row.id for row in self.all()}

    def purge_author(self, author: str) -> int:
        """Delete every row by this pseudonym. Used by `!babble forget`."""
        return self.purge(lambda r: r.involves(author))

    def purge(self, doomed: Callable[[CorpusRow], bool]) -> int:
        """Delete every row `doomed` says to, rewriting the file atomically."""
        rows = self.all()
        keep = [r for r in rows if not doomed(r)]
        removed = len(rows) - len(keep)
        if removed:
            body = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in keep)
            atomic_write_text(self.path, body)
        return removed

    def count(self) -> int:
        return len(self.all())

    def counts_by_source(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for row in self.all():
            tally[row.source] = tally.get(row.source, 0) + 1
        return tally


def approx_tokens(rows: list[CorpusRow]) -> int:
    """Roughly how many tokens the corpus is worth.

    The tokenizer is byte-level, so a row costs its UTF-8 length plus the two
    structural tokens every example is wrapped in. "Approximate" because
    chunking a row longer than the block size adds a pair per chunk; at this
    corpus size that is a rounding error and not worth loading the tokenizer for.
    """
    return sum(len(row.text.encode("utf-8")) + 2 for row in rows)
