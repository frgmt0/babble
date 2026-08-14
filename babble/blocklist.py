"""A content filter that bounces a whole row rather than editing it.

Be honest about what this is: **a speed bump, not a guarantee.** A word list
with some normalisation catches the obvious and lazy cases -- copy-pasted
slurs, dots or spaces jammed between the letters, a leetspeak substitution or
two. It does not catch a motivated adult determined to get past it. Treat it
as one layer, not the whole plan.

Two design choices matter more than the word list itself:

* **The blocklist is data, not code.** `blocklist.txt` ships a small, starter
  set across a few categories -- not an attempt at an exhaustive list, which
  does not exist. Extend it by editing the file, or point
  `BABBLE_BLOCKLIST_PATH` at your own.
* **Matching is on a normalised form.** Lowercased, accents stripped, common
  leetspeak folded back to letters (0->o, 1->i, 3->e, 4->a, 5->s, @->a, $->s),
  repeated characters collapsed, and separator punctuation glued back
  together -- so "s.l.u.r", "sluuur" and "SLÜR" all fold to the same form a
  plainly-typed term would. Letters spaced out one at a time ("s l u r") are
  folded the same way; anything longer than a couple of characters is left as
  its own word, which is what keeps ordinary words that merely *contain* a
  blocked substring (like "class" containing "ass") from tripping the filter
  -- matching requires the blocked term as a whole word, not a substring.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

BLOCKLIST_ENV = "BABBLE_BLOCKLIST_PATH"
DEFAULT_PATH = Path(__file__).resolve().parent / "blocklist.txt"

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"})
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_GLUE = re.compile(r"[.\-_]")  # punctuation used to space out a word; deleted, not a boundary
_REPEATS = re.compile(r"(.)\1+")


def _fold_word(token: str) -> str:
    """Normalise one whitespace-delimited token to its matchable form."""
    token = unicodedata.normalize("NFKD", token)
    token = "".join(ch for ch in token if not unicodedata.combining(ch))
    token = token.lower()
    token = _ZERO_WIDTH.sub("", token)
    token = token.translate(_LEET)
    token = _GLUE.sub("", token)
    return _REPEATS.sub(r"\1", token)


def normalise(text: str) -> str:
    """Fold `text` down to the space-joined form the blocklist matches against.

    Runs of single-letter tokens (real whitespace between them, the classic
    "s l u r" evasion) are glued into one word; anything longer is left as its
    own word so genuine word boundaries -- and the false-positive protection
    that comes with them -- survive.
    """
    folded = [_fold_word(tok) for tok in text.split()]
    merged: list[str] = []
    buffer = ""
    for tok in folded:
        if len(tok) == 1 and tok.isalpha():
            buffer += tok
            continue
        if buffer:
            merged.append(_REPEATS.sub(r"\1", buffer))
            buffer = ""
        if tok:
            merged.append(tok)
    if buffer:
        merged.append(_REPEATS.sub(r"\1", buffer))
    return " ".join(merged)


def row_fingerprint(*parts: str) -> str:
    """A stable hash for logging a rejection without ever logging the text."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Blocklist:
    """Loaded terms plus their compiled word-boundary patterns."""

    terms: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def load(cls, path: Path | None = None) -> "Blocklist":
        if path is None:
            override = os.environ.get(BLOCKLIST_ENV)
            path = Path(override) if override else DEFAULT_PATH
        if not path.exists():
            return cls(frozenset())
        terms = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            normalised = normalise(term)
            if normalised:
                terms.add(normalised)
        return cls(frozenset(terms))

    @property
    def enabled(self) -> bool:
        return bool(self.terms)

    def hit(self, text: str | None) -> str | None:
        """The first blocked term found in `text`, or None."""
        if not self.enabled or not text:
            return None
        normalised = normalise(text)
        if not normalised:
            return None
        for term in self.terms:
            if re.search(rf"\b{re.escape(term)}\b", normalised):
                return term
        return None

    def matches(self, *texts: str | None) -> bool:
        """True if any of `texts` contains a blocked term."""
        return any(self.hit(t) is not None for t in texts)
