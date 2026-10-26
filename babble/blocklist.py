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
* **Edits apply live.** A list loaded from a file re-stats it at most once a
  second and re-reads it when it changed, so a running bot needs no restart.
  If the file goes missing, unreadable or unparsable the last good terms are
  kept (never fail open) and a later good write is picked up. A reload that
  parses to *no* terms is likewise ignored while terms are loaded (it is
  indistinguishable from an editor mid-save), so to disable the blocklist on a
  running bot, unset or repoint `BABBLE_BLOCKLIST_PATH` and restart rather than
  emptying the file. Save edits atomically (write a temp file, then rename) so
  a partially written file is never read.
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
import threading
import time
from collections.abc import Callable
from pathlib import Path

BLOCKLIST_ENV = "BABBLE_BLOCKLIST_PATH"
RELOAD_INTERVAL = 1.0  # seconds between stat() checks of the source file
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


def _parse_terms(text: str) -> frozenset[str]:
    terms = set()
    for line in text.splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        normalised = normalise(term)
        if normalised:
            terms.add(normalised)
    return frozenset(terms)


class Blocklist:
    """Loaded terms plus their compiled word-boundary patterns.

    A list loaded from a file watches that file: `terms` (and so `enabled`,
    `hit`, `matches`) re-stats the path at most once per `check_interval`
    seconds and re-parses it when its mtime or size changed. A missing,
    unreadable or broken file keeps the last good terms (fail safe, never
    open) and is retried on a later check. An in-memory list never touches
    the filesystem.
    """

    def __init__(
        self,
        terms: frozenset[str] = frozenset(),
        *,
        path: Path | None = None,
        check_interval: float = RELOAD_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        signature: tuple[int, int] | None = None,
    ) -> None:
        self._terms = frozenset(terms)
        self._path = path
        self._interval = check_interval
        self._clock = clock
        self._signature = signature
        self._last_check = clock()
        self._lock = threading.Lock()
        self.generation = 0  # bumped on every successful hot reload

    @classmethod
    def load(cls, path: Path | None = None) -> "Blocklist":
        if path is None:
            override = os.environ.get(BLOCKLIST_ENV)
            path = Path(override) if override else DEFAULT_PATH
        # The path is watched even if absent now, so a file that appears later is picked up.
        bl = cls(frozenset(), path=path)
        try:
            bl._signature, bl._terms = bl._read()
        except FileNotFoundError:
            pass
        return bl

    def _read(self) -> tuple[tuple[int, int], frozenset[str]]:
        assert self._path is not None
        st = self._path.stat()  # stat first: a write racing the read forces another reload
        text = self._path.read_text(encoding="utf-8")
        return (st.st_mtime_ns, st.st_size), _parse_terms(text)

    def _maybe_reload(self) -> None:
        if self._path is None:
            return
        with self._lock:
            now = self._clock()
            if now - self._last_check < self._interval:
                return
            self._last_check = now
            try:
                st = self._path.stat()
                if (st.st_mtime_ns, st.st_size) == self._signature:
                    return
                signature, terms = self._read()
            except Exception:
                return  # keep the last good terms; retry at the next check
            if not terms and self._terms:
                # An editor mid-save (truncate, then write) looks like an empty
                # file. Keep the old terms and leave the signature unstored so
                # the next check retries.
                return
            self._terms = terms  # fully built before the swap
            self._signature = signature
            self.generation += 1

    @property
    def terms(self) -> frozenset[str]:
        self._maybe_reload()
        return self._terms

    @property
    def enabled(self) -> bool:
        return bool(self.terms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Blocklist):
            return NotImplemented
        return self.terms == other.terms

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:  # never show the terms
        return f"Blocklist(n={len(self._terms)}, path={self._path})"

    def hit(self, text: str | None) -> str | None:
        """The first blocked term found in `text`, or None."""
        terms = self.terms
        if not terms or not text:
            return None
        normalised = normalise(text)
        if not normalised:
            return None
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", normalised):
                return term
        return None

    def matches(self, *texts: str | None) -> bool:
        """True if any of `texts` contains a blocked term."""
        return any(self.hit(t) is not None for t in texts)
