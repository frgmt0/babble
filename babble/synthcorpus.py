"""Synthetic corpus rows, recombined from the corpus itself. Torch-free.

ro's ask is synthetic data that "mimics yet expands" what the model trains on,
using only text the corpus already contains. `synthetic.py` answered the pair
half of that by postulating a prompt for a real row -- but it never grew the
pool of *target* text, which is why it measured as a null result. This module
grows the target pool: it emits new rows sampled from a word-level Markov
chain built over the consented corpus.

Why this counts as corpus-internal and voice-preserving *by construction*:

- Every emitted word is a word somebody actually typed, spelled exactly as
  they typed it (tokens are whitespace-split, punctuation left attached).
- Every consecutive word pair follows an order-2 transition observed in the
  corpus: the chain only moves `(w1, w2) -> w3` where "w1 w2 w3" appears
  verbatim in some corpus row (with order-1 backoff only from contexts that
  have no order-2 continuation, and even then to an observed bigram).
- Row starts are sampled from real row openings; a row ends where a real row
  ended, or at a length cap drawn from the real row-length distribution.

So a synthetic row is a splice of real phrasing -- lowercase Discord cadence,
slang, typos and all. What it adds is *recombination*: the model sees corpus
vocabulary in new but in-distribution orders, which is more next-token signal
per real byte than replaying the same 400 rows again.

Kept strictly apart from the human corpus, same discipline as
`synthetic.py`: its own file (`data/synthetic_corpus.jsonl`,
`Settings.synthetic_corpus_path`), every row carries `"synthetic": true` and
the method that made it, nothing is ever appended to `corpus.jsonl`, and the
trainer mixes these rows in only while `Settings.train_synthetic` allows it
(`BABBLE_TRAIN_SYNTHETIC`, on by default) -- and even then only into the
*train* side of the split, never into validation, so val loss always
measures real held-out human text. Generation itself also excludes val-side
rows by default (`exclude_val`): the chain never even sees held-out text, so
a synthetic row cannot smuggle val phrasing into training. Deleting the file
(or setting `BABBLE_TRAIN_SYNTHETIC=0`) turns it off completely.

The exclude-val guarantee only holds for the corpus the file was generated
from. The val holdout is the lowest-hash-bucket slice of the *whole* id
population (`valsplit.val_id_set`), so appending corpus rows migrates some
existing rows train -> val -- and a file generated earlier may then contain
splices of now-held-out phrasing. A sidecar meta file records the exact
source-id set each generation used (`source_fingerprint`), and the trainer
calls `refresh_synthetic_corpus_if_stale` before every mix: if the current
train-side id set no longer matches the recorded one, the file is rebuilt
from the current corpus before anything is trained on.

Consent: rows are generated only from text that currently passes the same
consent + blocklist gate the trainer applies. A generated row is a splice of
many source rows, so per-source-row withdrawal tracking is not possible the
way it is for `synthetic.py` pairs; instead the file is cheap to regenerate
and `babble synth-corpus --rebuild` rewrites it from the *current* consented
corpus, which is the supported way to honour a withdrawal. The trainer's mix
step re-checks the blocklist on every synthetic row at training time.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .blocklist import Blocklist
from .config import Settings
from .identity import Pseudonymiser
from .synthetic import _consented_rows
from .util import atomic_write_text, utcnow_iso
from .valsplit import val_id_set

__all__ = [
    "SyntheticRow",
    "SyntheticCorpusStore",
    "MarkovChain",
    "SynthCorpusResult",
    "generate_synthetic_corpus",
    "refresh_synthetic_corpus_if_stale",
    "source_fingerprint",
    "trainable_synthetic_rows",
    "synthetic_row_count",
]

#: Marks the end of a source row inside the chain, so generated rows learn to
#: stop where real rows stopped instead of rambling to the length cap.
_END = "\x00end"

METHOD_MARKOV2 = "markov_order2"


@dataclass(frozen=True)
class SyntheticRow:
    """One recombined corpus-style row. `text` is a splice of real corpus
    phrasing (see module docstring); `method` names the generator rule."""

    id: str
    text: str
    method: str
    seed: int
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "method": self.method,
            "seed": self.seed,
            "synthetic": True,  # belt-and-braces, same as SyntheticPair
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SyntheticRow":
        return cls(
            id=raw["id"],
            text=raw.get("text", ""),
            method=raw.get("method", ""),
            seed=int(raw.get("seed", 0)),
            created_at=raw.get("created_at", ""),
        )


def make_synthetic_row_id(text: str, method: str) -> str:
    payload = "\x1f".join(["synthcorpus", method, text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def source_fingerprint(row_ids: Iterable[str]) -> str:
    """Content address of the exact source-row id set a chain was built from.

    Order-independent (the set is sorted first) so it only changes when the
    membership changes -- which is precisely when a previously generated file
    can start leaking now-held-out phrasing and must be rebuilt."""
    payload = "\x1f".join(["synthcorpus-sources", *sorted(row_ids)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _read_meta(path: Path) -> dict:
    try:
        raw = json.loads(_meta_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


class SyntheticCorpusStore:
    """JSONL at `Settings.synthetic_corpus_path`. Append-dedupe like the other
    stores, plus `rewrite` -- regeneration replaces the whole file, which is
    how a consent withdrawal is honoured for spliced rows."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def all(self) -> list[SyntheticRow]:
        if not self.path.exists():
            return []
        rows: list[SyntheticRow] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(SyntheticRow.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # a torn line never takes the store down with it
        return rows

    def ids(self) -> set[str]:
        return {r.id for r in self.all()}

    def count(self) -> int:
        return len(self.all())

    def extend(self, rows: Iterable[SyntheticRow]) -> int:
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

    def rewrite(self, rows: Iterable[SyntheticRow]) -> int:
        """Replace the whole file with `rows`, atomically."""
        deduped: list[SyntheticRow] = []
        seen: set[str] = set()
        for row in rows:
            if row.id in seen:
                continue
            seen.add(row.id)
            deduped.append(row)
        body = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in deduped)
        atomic_write_text(self.path, body)
        return len(deduped)


class MarkovChain:
    """Order-2 word-level chain over the corpus, with order-1 backoff.

    Words are whitespace tokens with punctuation attached -- "*fine*" and
    "wasnt" stay exactly as typed. Newlines inside a row count as spaces; the
    generated row is single-line, which matches almost every real row (the
    corpus median is 25 bytes).
    """

    def __init__(self, texts: list[str]) -> None:
        self.starts: list[tuple[str, ...]] = []  # first two words of each row
        self.order2: dict[tuple[str, str], list[str]] = {}
        self.order1: dict[str, list[str]] = {}
        self.lengths: list[int] = []  # word counts of real rows
        self.real_texts = {t.strip() for t in texts}
        for text in texts:
            words = text.split()
            if not words:
                continue
            self.lengths.append(len(words))
            padded = words + [_END]
            self.starts.append(tuple(padded[:2]))
            for w1, w2 in zip(padded, padded[1:]):
                self.order1.setdefault(w1, []).append(w2)
            for w1, w2, w3 in zip(padded, padded[1:], padded[2:]):
                self.order2.setdefault((w1, w2), []).append(w3)

    def sample_row(self, rng: random.Random, max_words: int | None = None) -> str:
        if not self.starts:
            return ""
        cap = max_words if max_words is not None else max(2, rng.choice(self.lengths))
        start = rng.choice(self.starts)
        words = [w for w in start if w != _END]
        while len(words) < cap:
            nxt = None
            if len(words) >= 2:
                nxt = self._step2(rng, words[-2], words[-1])
            if nxt is None:
                nxt = self._step1(rng, words[-1])
            if nxt is None or nxt == _END:
                break
            words.append(nxt)
        return " ".join(words)

    def _step2(self, rng: random.Random, w1: str, w2: str) -> str | None:
        options = self.order2.get((w1, w2))
        return rng.choice(options) if options else None

    def _step1(self, rng: random.Random, w1: str) -> str | None:
        options = self.order1.get(w1)
        return rng.choice(options) if options else None


@dataclass
class SynthCorpusResult:
    source_rows: int = 0
    excluded_val_rows: int = 0
    requested: int = 0
    generated: int = 0
    skipped_real_duplicate: int = 0
    skipped_blocklist: int = 0
    skipped_short: int = 0
    stored_total: int = 0


def generate_synthetic_corpus(
    settings: Settings,
    *,
    count: int = 400,
    seed: int = 0,
    rebuild: bool = False,
    exclude_val: bool = True,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> SynthCorpusResult:
    """Sample `count` recombined rows from the current consented corpus into
    `settings.synthetic_corpus_path`. Deterministic for a given corpus and
    seed. `rebuild=True` replaces the file instead of appending -- the
    supported way to regenerate after the corpus grew or a consent changed.

    `exclude_val` (the default) builds the chain only from rows the trainer
    puts on the *train* side of its deterministic split. Synthetic rows are
    only ever mixed into training, but a chain built over the whole corpus
    can splice held-out phrasing into them, and then val loss is partly
    scoring text the model saw in recombined form. Excluding val-side rows at
    generation time keeps the holdout clean; pass `exclude_val=False` only
    for experiments that need the old behaviour.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    rows = _consented_rows(settings, ids, blocklist)
    excluded = 0
    if exclude_val:
        held_out = val_id_set(
            [r.id for r in rows],
            val_fraction=settings.val_fraction,
            val_min_rows=settings.val_min_rows,
        )
        excluded = sum(1 for r in rows if r.id in held_out)
        rows = [r for r in rows if r.id not in held_out]
    fingerprint = source_fingerprint(r.id for r in rows)
    chain = MarkovChain([r.text for r in rows])
    rng = random.Random(seed)

    result = SynthCorpusResult(
        source_rows=len(rows), excluded_val_rows=excluded, requested=count
    )
    fresh: list[SyntheticRow] = []
    seen_texts: set[str] = set()
    attempts = 0
    # Draw until `count` rows landed or the attempt budget runs out -- a tiny
    # corpus can only support so many distinct recombinations, and looping
    # forever chasing duplicates would hang the command.
    while len(fresh) < count and attempts < count * 20:
        attempts += 1
        text = chain.sample_row(rng).strip()
        if len(text.split()) < 2:
            result.skipped_short += 1
            continue
        if text in chain.real_texts or text in seen_texts:
            # A verbatim replay of a real row is not an expansion, and it
            # would double-count that row's text under a synthetic label.
            result.skipped_real_duplicate += 1
            continue
        if blocklist.matches(text):
            result.skipped_blocklist += 1
            continue
        seen_texts.add(text)
        fresh.append(
            SyntheticRow(
                id=make_synthetic_row_id(text, METHOD_MARKOV2),
                text=text,
                method=METHOD_MARKOV2,
                seed=seed,
            )
        )

    store = SyntheticCorpusStore(settings.synthetic_corpus_path)
    prior_meta = _read_meta(store.path)
    prior_count = store.count()
    if rebuild:
        result.generated = store.rewrite(fresh)
        file_fingerprint: str | None = fingerprint
    else:
        result.generated = store.extend(fresh)
        # The meta fingerprint vouches for the WHOLE file, so an append may
        # only claim it when everything already stored came from this same
        # source set. Otherwise the older rows keep the file stale (or of
        # unknown provenance) and the trainer will rebuild before mixing.
        if prior_count == 0 or prior_meta.get("fingerprint") == fingerprint:
            file_fingerprint = fingerprint
        else:
            file_fingerprint = prior_meta.get("fingerprint") or None
    result.stored_total = store.count()
    atomic_write_text(
        _meta_path(store.path),
        json.dumps(
            {
                "fingerprint": file_fingerprint,
                "requested": count,
                "seed": seed,
                "exclude_val": exclude_val,
                "source_rows": len(rows),
                "generated_at": utcnow_iso(),
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    return result


def refresh_synthetic_corpus_if_stale(
    settings: Settings,
    *,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> SynthCorpusResult | None:
    """Rebuild `data/synthetic_corpus.jsonl` if the corpus has moved under it.

    The val holdout is a function of the whole corpus id population, so
    appending rows migrates some existing rows train -> val; a file generated
    before that migration can contain splices of now-held-out phrasing, and
    training on it would quietly re-open the val leak `exclude_val` closed.
    The trainer calls this before every synthetic mix. Returns the rebuild
    result, or None when the file is absent/empty or provably fresh (its
    recorded source fingerprint matches the current train-side id set).

    A file with no meta sidecar (generated before fingerprints existed, or
    with `--include-val-sources`) counts as stale: unknown provenance is
    treated exactly like known-bad provenance."""
    store = SyntheticCorpusStore(settings.synthetic_corpus_path)
    stored = store.count()
    if stored == 0:
        return None
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    rows = _consented_rows(settings, ids, blocklist)
    held_out = val_id_set(
        [r.id for r in rows],
        val_fraction=settings.val_fraction,
        val_min_rows=settings.val_min_rows,
    )
    expected = source_fingerprint(r.id for r in rows if r.id not in held_out)
    meta = _read_meta(store.path)
    if meta.get("fingerprint") == expected:
        return None
    requested = meta.get("requested")
    count = requested if isinstance(requested, int) and requested > 0 else stored
    seed = meta.get("seed")
    return generate_synthetic_corpus(
        settings,
        count=count,
        seed=seed if isinstance(seed, int) else 0,
        rebuild=True,
        exclude_val=True,
        ids=ids,
        blocklist=blocklist,
    )


def trainable_synthetic_rows(
    settings: Settings, blocklist: Blocklist | None = None
) -> list[SyntheticRow]:
    """Stored synthetic rows that clear the blocklist right now, sorted by id.

    There is no per-row consent check here because a spliced row has no single
    source row to check against -- regeneration (`--rebuild`) from the current
    consented corpus is how consent is honoured; see the module docstring.
    """
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    rows = SyntheticCorpusStore(settings.synthetic_corpus_path).all()
    return sorted(
        (r for r in rows if r.text and not blocklist.matches(r.text)),
        key=lambda r: r.id,
    )


def synthetic_row_count(settings: Settings) -> int:
    return SyntheticCorpusStore(settings.synthetic_corpus_path).count()
