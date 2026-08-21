"""Synthetic correction pairs, postulated from the corpus itself.

ro's ask: give post-train more `(prompt, response)` pairs than the human
corrections alone provide, without inventing text nobody in Discord wrote.
The technique is his: take a corpus row that reads like a reply or an
interjection -- his example, `"well the visual shells were also just *fine*
again i wasnt drawn to *any* of them"` -- and postulate the prompt it was
plausibly answering. The **response half stays verbatim corpus text**; only
the prompt is synthesized. That is deliberate and load-bearing: it is what
makes "maintains the corpus voice to a tee" true by construction rather than
by hoping a second model matches it. See `is_reactive` / `synthesize_prompt`.

**Honesty about the method.** There is no LLM credential wired into this
repo (`babble/config.py` has no API-key setting at all, and the dependency
list is `torch` / `discord.py` / `huggingface_hub` -- no LLM client), so this
does not call out to one to "postulate" anything. What is here instead is a
small set of surface heuristics (an interjection at the front, a continuation
word like "also"/"again", an emphasis marker, a trailing "?") plus a
stopword-stripped topic phrase, templated into a plausible-sounding prompt.
It is not language understanding. It is enough to demonstrate the mechanism
ro described, and the generated prompts should be read that way -- see the
counts and samples `babble synth-generate` prints.

**Kept strictly apart from human corrections.** Every pair here is
content-addressed against its *source corpus row*, lives in its own file
(`data/synthetic_pairs.jsonl`, `Settings.synthetic_pairs_path`), and is never
appended to `interactions.jsonl`. `babble post-train` only touches it when
told to with `--include-synthetic`. `babble synth-status` counts it
separately from human pairs so it can be watched, and deleting the file (or
never passing `--include-synthetic`) turns it off completely.

**Re-runnable.** `generate_synthetic_pairs` scans whatever the corpus holds
right now and only appends pairs it has not already generated (same source
row + same postulated prompt = same id), so running it again after the corpus
has grown costs nothing for the rows it already covered.

**Consent, twice.** A synthetic pair is only ever built from a corpus row
that currently passes the same consent + blocklist gate the trainer applies
(`_consented_rows` below, deliberately reimplemented rather than importing
`trainer.py`, which pulls in `torch` -- this module stays as cheap to import
as `post_state.py`). `trainable_synthetic_pairs` re-checks that gate again at
train time, so a withdrawal that purges a corpus row also stops any synthetic
pair built from it from being trained on, the same belt-and-braces promise
`trainable_pairs` gives human corrections.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .blocklist import Blocklist
from .config import Settings
from .consent import ConsentStore, CorpusConsent
from .corpus import CorpusRow, CorpusStore
from .identity import Pseudonymiser
from .util import utcnow_iso

__all__ = [
    "SyntheticPair",
    "SyntheticPairStore",
    "SynthGenerateResult",
    "is_reactive",
    "reactivity_score",
    "synthesize_prompt",
    "continuation_cuts",
    "generate_synthetic_pairs",
    "trainable_synthetic_pairs",
    "synthetic_pair_count",
]

#: Method tag for pairs cut out of a single real row: prompt is the row's
#: opening, response is the rest, both verbatim. See `continuation_cuts`.
METHOD_CONTINUATION = "continuation_cut"


@dataclass(frozen=True)
class SyntheticPair:
    """One postulated-prompt pair. `response` is verbatim corpus text --
    never rewritten -- so its voice is exactly the corpus's by construction.
    `method` names which heuristic rule produced `prompt`, kept so a bad rule
    can be found and dropped without guessing which pairs it wrote."""

    id: str
    prompt: str
    response: str
    source_row_id: str
    method: str
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "source_row_id": self.source_row_id,
            "method": self.method,
            "synthetic": True,  # belt-and-braces: true even for a reader that skips the filename
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SyntheticPair":
        return cls(
            id=raw["id"],
            prompt=raw.get("prompt", ""),
            response=raw.get("response", ""),
            source_row_id=raw.get("source_row_id", ""),
            method=raw.get("method", ""),
            created_at=raw.get("created_at", ""),
        )


def make_synthetic_id(source_row_id: str, prompt: str, method: str) -> str:
    """Content-addressed over the source row, the postulated prompt and the
    rule that made it -- so a rerun after the heuristic changes produces a
    fresh id (and a fresh pair) rather than silently reusing a stale one, and
    two different rules that happen to postulate the same prompt for the same
    row are still two distinct, traceable pairs."""
    payload = "\x1f".join(["synthetic", source_row_id, prompt, method])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class SyntheticPairStore:
    """Append-only JSONL at `Settings.synthetic_pairs_path`. Same
    append/dedupe contract as `InteractionStore`, deliberately not shared code
    with it: keeping this a separate, small class is part of what makes "never
    silently mixed into the human corrections" true rather than a convention
    someone has to remember."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: SyntheticPair) -> bool:
        return self.extend([row]) == 1

    def extend(self, rows: Iterable[SyntheticPair]) -> int:
        seen = self.ids()
        fresh = []
        for row in rows:
            if row.id in seen:
                continue
            seen.add(row.id)
            fresh.append(row)
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for row in fresh:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return len(fresh)

    def all(self) -> list[SyntheticPair]:
        if not self.path.exists():
            return []
        rows: list[SyntheticPair] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(SyntheticPair.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # a torn line never takes the store down with it
        return rows

    def ids(self) -> set[str]:
        return {r.id for r in self.all()}

    def count(self) -> int:
        return len(self.all())


# --- picking which corpus rows read as a reply or an interjection ---------

# A word at the front of the row that reads as a reaction rather than the
# opening of a standalone statement.
_INTERJECTIONS = frozenset(
    {
        "yeah", "yea", "yeh", "yep", "yes", "no", "nah", "nope", "true", "fr",
        "lol", "lmao", "lmaooo", "same", "honestly", "literally", "wait",
        "oh", "ok", "okay", "well", "also", "still", "again", "tbh", "ngl",
        "fair", "real", "based", "damn", "bruh", "haha", "lowkey", "highkey",
        "genuinely", "actually", "exactly", "true true", "facts",
    }
)

# Words that only make sense continuing something already said.
_CONTINUATION = re.compile(r"\b(also|too|either|still|again|another|same)\b", re.IGNORECASE)
_ANAPHORA = re.compile(r"\b(it|that|this|them|those|they|there)\b", re.IGNORECASE)
_EMPHASIS = re.compile(r"\*[^*]+\*")

REACTIVE_THRESHOLD = 1.5


def reactivity_score(text: str) -> float:
    """How much `text` reads like a reply to something rather than a
    standalone statement -- see the module docstring for why this is surface
    heuristics and not semantic understanding.
    """
    stripped = text.strip()
    if not stripped or stripped.lower().startswith(("http://", "https://", "!babble")):
        return 0.0
    lower = stripped.lower()
    first_word = lower.split(" ", 1)[0].strip(".,!?~;:")

    score = 0.0
    if first_word in _INTERJECTIONS:
        score += 2.0
    if _CONTINUATION.search(lower):
        score += 1.5
    if _ANAPHORA.search(lower):
        score += 1.0
    if _EMPHASIS.search(stripped):
        score += 1.0
    if len(stripped) <= 80:
        score += 0.5
    if stripped.endswith("?"):
        score += 0.5
    return score


def is_reactive(text: str) -> bool:
    """Does this corpus row read as a reply or an interjection -- the kind of
    line ro's example is -- rather than a self-contained statement?"""
    return reactivity_score(text) >= REACTIVE_THRESHOLD


# --- postulating the prompt a reactive row was plausibly answering --------

_TOPIC_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "so", "of", "to", "in", "on", "at", "for", "with",
        "just", "that", "this", "it", "its", "i", "im", "you", "your", "we",
        "they", "he", "she", "not", "dont", "didnt", "wasnt", "isnt", "cant",
        "also", "again", "still", "too", "either", "another", "same", "very",
        "really", "well", "yeah", "yea", "yes", "no", "nah", "nope", "true",
        "fr", "lol", "lmao", "same", "honestly", "literally", "wait", "oh",
        "ok", "okay", "tbh", "ngl", "fair", "real", "based", "damn", "bruh",
        "haha", "actually", "genuinely", "exactly", "facts", "there", "here",
        "them", "those", "what", "who", "did", "do", "does", "my", "me",
    }
)


def _topic_words(text: str, limit: int = 5) -> str:
    """A rough topic phrase pulled from `text`'s content words -- not NLP,
    just stopword-stripped enough to fill a template so the postulated prompt
    reads like it's about the same thing as the response. Falls back to
    "that" for a row that is nothing but stopwords/punctuation."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    content = [w for w in words if w not in _TOPIC_STOPWORDS and len(w) > 2]
    if not content:
        return "that"
    return " ".join(content[:limit])


def synthesize_prompt(text: str) -> tuple[str, str]:
    """Postulate the plausible prompt `text` was replying to. Returns
    `(prompt, method)`; `method` names the rule so it can be audited later."""
    stripped = text.strip()
    lower = stripped.lower()
    first_word = lower.split(" ", 1)[0].strip(".,!?~;:")
    topic = _topic_words(stripped)

    if lower.endswith("?"):
        return f"so what's the deal with {topic}", "reacts_to_question"
    if first_word in {"yeah", "yea", "yeh", "yep", "yes", "true", "fr", "same", "fair", "based", "real", "facts"}:
        return f"does anyone else think {topic}", "agreement"
    if first_word in {"no", "nah", "nope"}:
        return f"i think {topic}, right?", "disagreement"
    if _CONTINUATION.search(lower):
        return f"what did you think of {topic}", "continuation"
    if _EMPHASIS.search(stripped):
        return f"how was {topic}", "emphasis"
    if _ANAPHORA.search(lower):
        return f"what happened with {topic}", "anaphora"
    return f"what about {topic}", "generic"


# --- continuation cuts: pairs where BOTH sides are verbatim corpus text ----

#: Rows shorter than this many words are not cut: a two-word row yields a
#: one-word prompt and a one-word response, which teaches nothing.
_MIN_CUT_WORDS = 4


def continuation_cuts(text: str, cuts: int = 2) -> list[tuple[str, str]]:
    """Cut `text` at up to `cuts` word boundaries into (prefix, rest) pairs.

    Both halves are byte-slices of the original row -- nothing is rejoined or
    re-spaced, so each half is text somebody actually typed, exactly as they
    typed it. The postulated-prompt pairs above only ever synthesize the
    prompt side; these grow the *response* pool too, which is the half the
    loss is actually computed over (`build_example` masks the prompt out).

    Cut points are evenly spaced through the row's words, deterministically,
    so a rerun on the same corpus regenerates the same pairs. A row shorter
    than `_MIN_CUT_WORDS` words yields nothing.
    """
    stripped = text.strip()
    words = list(re.finditer(r"\S+", stripped))
    if len(words) < _MIN_CUT_WORDS:
        return []
    cuts = max(1, cuts)
    # Word indices to cut before, spread across the row's interior. With
    # cuts=2 on a 9-word row that is before words 3 and 6.
    positions = sorted(
        {max(1, min(len(words) - 1, round(len(words) * (i + 1) / (cuts + 1)))) for i in range(cuts)}
    )
    out: list[tuple[str, str]] = []
    for pos in positions:
        boundary = words[pos].start()
        prompt = stripped[:boundary].rstrip()
        response = stripped[boundary:]
        if prompt and response:
            out.append((prompt, response))
    return out


# --- consent-gated corpus reads, kept torch-free ---------------------------


def _consented_rows(settings: Settings, ids: Pseudonymiser, blocklist: Blocklist) -> list[CorpusRow]:
    """Corpus rows whose author currently consents, blocklist-clean -- the
    same filter `trainer.corpus_rows` applies, reimplemented here rather than
    imported so this module never pulls in `torch` just to read a JSONL file
    (see the module docstring)."""
    gate = CorpusConsent(ConsentStore(settings.consent_path), ids)
    rows = CorpusStore(settings.corpus_path).all()
    return [r for r in rows if gate.allows(r) and not blocklist.matches(r.text)]


@dataclass
class SynthGenerateResult:
    scanned: int = 0
    reactive: int = 0
    generated: int = 0
    generated_postulated: int = 0
    generated_continuation: int = 0
    skipped_duplicate: int = 0
    skipped_blocklist: int = 0


def generate_synthetic_pairs(
    settings: Settings,
    *,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
    postulate: bool = True,
    continuations: bool = True,
    cuts: int = 2,
) -> SynthGenerateResult:
    """Scan the current consented corpus and append any pair not already
    stored to `settings.synthetic_pairs_path`. Two generators, both corpus-
    internal: `postulate` invents a plausible prompt for a reply-shaped row
    (response verbatim); `continuations` cuts a row into (prefix, rest) with
    *both* halves verbatim -- see `continuation_cuts` for why that is the half
    that actually grows the training signal. Safe to rerun as the corpus
    grows -- see the module docstring.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    rows = _consented_rows(settings, ids, blocklist)

    store = SyntheticPairStore(settings.synthetic_pairs_path)
    existing = store.ids()

    result = SynthGenerateResult(scanned=len(rows))
    fresh: list[SyntheticPair] = []

    def add(row: CorpusRow, prompt: str, response: str, method: str, kind: str) -> None:
        if blocklist.matches(prompt, response):
            result.skipped_blocklist += 1
            return
        pair_id = make_synthetic_id(row.id, prompt, method)
        if pair_id in existing:
            result.skipped_duplicate += 1
            return
        existing.add(pair_id)
        fresh.append(
            SyntheticPair(
                id=pair_id, prompt=prompt, response=response, source_row_id=row.id, method=method
            )
        )
        if kind == "postulated":
            result.generated_postulated += 1
        else:
            result.generated_continuation += 1

    for row in rows:
        if postulate and is_reactive(row.text):
            result.reactive += 1
            prompt, method = synthesize_prompt(row.text)
            add(row, prompt, row.text, method, "postulated")
        if continuations:
            for prompt, response in continuation_cuts(row.text, cuts=cuts):
                add(row, prompt, response, METHOD_CONTINUATION, "continuation")

    result.generated = store.extend(fresh)
    return result


def trainable_synthetic_pairs(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> list[SyntheticPair]:
    """Stored synthetic pairs whose source corpus row still consents and still
    clears the blocklist, checked right now -- withdrawal purges the source
    row, and this is what stops a synthetic pair built from it outliving that
    purge. Sorted by id (content-addressed) for a deterministic order.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    allowed_row_ids = {row.id for row in _consented_rows(settings, ids, blocklist)}
    pairs = SyntheticPairStore(settings.synthetic_pairs_path).all()
    trainable = [
        p
        for p in pairs
        if p.source_row_id in allowed_row_ids and not blocklist.matches(p.prompt, p.response)
    ]
    return sorted(trainable, key=lambda p: p.id)


def synthetic_pair_count(settings: Settings) -> int:
    """Total stored synthetic pairs -- the raw count, not the consent-filtered
    one, mirroring `post_state.pair_count`."""
    return SyntheticPairStore(settings.synthetic_pairs_path).count()
