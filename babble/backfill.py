"""Flattening the old correction pairs into the corpus.

Before the corpus existed, the only thing babble kept was `(prompt, rejected,
chosen)` triples. The prompt and the chosen answer in each of those are real
human writing -- somebody typed them -- and they are the entire corpus the
project has any history of. This walks the interaction store and files each side
of each pair as its own unlabelled corpus row.

**Which consent applies.** This is deliberately gated on the *corrections* grant,
not the corpus one. That text was collected under the corrections notice and has
already been published under it; re-filing it is not a new collection, it is the
same words in a second index. The corpus grant governs what gets collected
*from now on* -- somebody's ordinary messages -- and nothing here touches that.
A person who has explicitly said no to the corpus is skipped anyway, because an
explicit no should not need a subtlety to be honoured.

Running it twice does nothing the second time: corpus ids are content-addressed
over `(text, author)`, so every row it would write is already there.
"""

from __future__ import annotations

from dataclasses import dataclass

from .blocklist import Blocklist
from .config import Settings
from .consent import CAPTURE_OK, NEGATIVE, SCOPE_CORPUS, SCOPE_CORRECTIONS, ConsentStore
from .corpus import SOURCE_CORRECTION, SOURCE_PROMPT, CorpusRow, CorpusStore, make_corpus_id
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .store import CORRECTION, Interaction, InteractionStore


@dataclass
class BackfillResult:
    """What the migration did, in enough detail to explain any missing row."""

    scanned: int = 0  # interaction rows walked
    considered: int = 0  # pieces of text those rows offered up
    added: int = 0
    skipped_duplicate: int = 0
    skipped_consent: int = 0
    skipped_blocklist: int = 0
    skipped_empty: int = 0


def _pieces(row: Interaction) -> list[tuple[str, str, str]]:
    """The human-written halves of one row: (text, author, source).

    The prompt is always somebody's writing. `chosen` only is on a **correction**,
    where a person typed what the bot should have said. On an approval `chosen`
    is `exchange.response` -- the bot's own answer, which somebody merely agreed
    with by reacting 👍 -- and filing that as their writing would put randomly
    initialised model output into the corpus under a human's pseudonym, train on
    it, and publish it as a thing a person said.

    `rejected` is absent for exactly the same reason, on every row.
    """
    pieces = [(row.prompt, row.prompt_author, SOURCE_PROMPT)]
    if row.signal == CORRECTION:
        pieces.append((row.chosen, row.signal_author, SOURCE_CORRECTION))
    return pieces


def backfill_corpus(
    settings: Settings,
    log: EventLog | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> BackfillResult:
    """Flatten every consented correction pair into the corpus. Idempotent."""
    settings.ensure_dirs()
    log = log or NullLog()
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)

    # Pseudonyms whose owner still consents to the corrections collection, and
    # has not separately said no to the corpus.
    allowed = {
        ids.user(uid)
        for uid in consent.known_ids()
        if consent.decision(uid, SCOPE_CORRECTIONS) in CAPTURE_OK
        and consent.decision(uid, SCOPE_CORPUS) not in NEGATIVE
    }

    corpus = CorpusStore(settings.corpus_path)
    seen = corpus.ids()
    result = BackfillResult()
    fresh: list[CorpusRow] = []

    for row in InteractionStore(settings.interactions_path).all():
        result.scanned += 1
        for text, author, source in _pieces(row):
            result.considered += 1
            if not text.strip():
                result.skipped_empty += 1
                continue
            if author not in allowed:
                result.skipped_consent += 1
                continue
            # Re-checked here, not trusted from capture time: a term added to the
            # blocklist since must keep the row out of the corpus too.
            if blocklist.matches(text):
                result.skipped_blocklist += 1
                continue
            row_id = make_corpus_id(text, author)
            if row_id in seen:
                result.skipped_duplicate += 1
                continue
            seen.add(row_id)
            fresh.append(
                CorpusRow(
                    id=row_id,
                    text=text,
                    author=author,
                    source=source,
                    created_at=row.created_at,  # keep the original capture time
                )
            )

    for new_row in fresh:
        result.added += int(corpus.append(new_row))

    log.event(
        "corpus.backfill",
        scanned=result.scanned,
        considered=result.considered,
        added=result.added,
        skipped_duplicate=result.skipped_duplicate,
        skipped_consent=result.skipped_consent,
        skipped_blocklist=result.skipped_blocklist,
        skipped_empty=result.skipped_empty,
        total=corpus.count(),
    )
    return result
