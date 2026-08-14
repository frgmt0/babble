"""Publishing to HuggingFace: the corpus, and the corrections beside it.

The dataset has two configs, because babble now collects two different things:

* **`default`** — the unlabelled corpus. One row of human writing per line, no
  answer attached to any of it. This is what the model trains on.
* **`corrections`** — the `(prompt, rejected, chosen)` triples. Not the training
  objective any more, still a real artifact, still published.

Two rules govern the whole file:

* **Only consented rows leave the machine.** Consent is re-checked here against
  the live consent store, not trusted from capture time, so a withdrawal always
  wins even if the purge failed for some reason. Each config is checked against
  the grant that actually covers it: the corpus against `corpus`, the
  corrections against `corrections`.
* **No raw identifier is ever written.** Author fields must look like
  pseudonyms or the export aborts, and any row whose *text* contains a known raw
  Discord id or mention markup is dropped rather than published.

Nothing here runs automatically from capture; you have to type
`babble export --push`, or let the trainer's scheduled publish do it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .blocklist import Blocklist
from .config import Settings
from .consent import SCOPE_CORRECTIONS, ConsentStore, CorpusConsent
from .corpus import CorpusRow, CorpusStore
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .store import APPROVAL, CORRECTION, Interaction, InteractionStore

PSEUDONYM = re.compile(r"^u_[0-9a-f]{16}$")
MENTION = re.compile(r"<@[!&]?\d+>")

# Discord's own CDN puts the raw channel id and message id in the path of every
# attachment url: `.../attachments/<channel_id>/<message_id>/name.png`. A
# message that is nothing but an uploaded image therefore carries two raw
# snowflakes in its text, and nothing else in the leak check would notice --
# they are not user ids and they are not mention markup. Correcting the bot with
# an upload still works and is still stored; that row just does not get
# published, the same way any other leaky row does not.
ATTACHMENT = re.compile(
    r"https?://(?:cdn\.discordapp\.com|media\.discordapp\.net)/attachments/\d{15,}/\d{15,}",
    re.IGNORECASE,
)

#: The corpus, and the `default` config: what anyone loading the dataset gets.
CORPUS_FILE = "data/corpus.jsonl"
#: The corrections, under their own config. Same path it has always had, so an
#: existing download script pointed straight at the file keeps working.
DATA_FILE = "data/train.jsonl"


class ExportBlocked(RuntimeError):
    """Raised when the output would contain something it must never contain."""


@dataclass
class ExportResult:
    path: Path
    rows: int  # everything published, both configs
    corpus_rows: int
    corpus_excluded_no_consent: int
    corpus_dropped_leaky: int
    corpus_dropped_blocklist: int
    corpus_chars: int
    correction_rows: int
    excluded_no_consent: int
    dropped_leaky: int
    dropped_blocklist: int
    corrections: int
    approvals: int
    contributors: int


def _text_fields(row: Interaction) -> list[str]:
    return [row.prompt or "", row.rejected or "", row.chosen or ""]


def _leak_candidates(known_ids: list[str]) -> list[str]:
    """Ids long enough to actually be a Discord snowflake.

    Real snowflakes are 17-19 digits. Filtering by length keeps every genuine id
    in scope while stopping a short synthetic test id from matching ordinary
    prose and silently dropping innocent rows.
    """
    return [uid for uid in known_ids if len(uid) >= 15]


def _leaks(blob: str, raw_ids: list[str]) -> bool:
    """Does this text carry an identifier out with it?"""
    if MENTION.search(blob) or ATTACHMENT.search(blob):
        return True
    return any(uid and uid in blob for uid in raw_ids)


def select_rows(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> tuple[list[Interaction], int, int, int]:
    """(publishable corrections, excluded for consent, dropped for leaks, dropped for the blocklist)."""
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)
    raw_ids = _leak_candidates(consent.known_ids())
    allowed = {ids.user(uid) for uid in consent.granted_ids(SCOPE_CORRECTIONS)}

    rows = InteractionStore(settings.interactions_path).all()
    consented = [r for r in rows if r.prompt_author in allowed and r.signal_author in allowed]
    excluded = len(rows) - len(consented)

    # Someone typed a raw id into their message; do not publish it.
    leak_free = [r for r in consented if not _leaks("\n".join(_text_fields(r)), raw_ids)]
    dropped_leaky = len(consented) - len(leak_free)

    # Same three-place enforcement as consent: a row stored before a blocked
    # term was added must not survive to be published once it is.
    clean = [r for r in leak_free if not blocklist.matches(r.prompt, r.chosen, r.rejected)]
    dropped_blocklist = len(leak_free) - len(clean)

    # Deterministic order, and one row per fact, so re-running is a no-op.
    unique = {row.id: row for row in clean}
    ordered = sorted(unique.values(), key=lambda r: (r.created_at, r.id))
    return ordered, excluded, dropped_leaky, dropped_blocklist


def select_corpus_rows(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> tuple[list[CorpusRow], int, int, int]:
    """(publishable corpus rows, excluded for consent, dropped for leaks, dropped for the blocklist).

    Same shape and the same three filters as `select_rows`, against whichever
    grant governs each row's `source`. A person who consented to corrections and
    has not answered the corpus notice publishes the correction text they already
    agreed to publish, and none of the ordinary messages they never did -- which
    is the entire point of there being two grants.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)
    raw_ids = _leak_candidates(consent.known_ids())
    gate = CorpusConsent(consent, ids)

    rows = CorpusStore(settings.corpus_path).all()
    consented = gate.keep(rows)
    excluded = len(rows) - len(consented)

    leak_free = [r for r in consented if not _leaks(r.text, raw_ids)]
    dropped_leaky = len(consented) - len(leak_free)

    clean = [r for r in leak_free if not blocklist.matches(r.text)]
    dropped_blocklist = len(leak_free) - len(clean)

    unique = {row.id: row for row in clean}
    ordered = sorted(unique.values(), key=lambda r: (r.created_at, r.id))
    return ordered, excluded, dropped_leaky, dropped_blocklist


def assert_pseudonymous(
    rows: list[Interaction],
    known_raw_ids: list[str],
    corpus: list[CorpusRow] | None = None,
) -> None:
    """Last line of defence, run against the exact rows about to be written."""
    raw_ids = _leak_candidates(known_raw_ids)
    for row in rows:
        for field_name in ("prompt_author", "signal_author"):
            value = getattr(row, field_name)
            if not PSEUDONYM.match(value or ""):
                raise ExportBlocked(
                    f"row {row.id}: {field_name}={value!r} is not a pseudonym — refusing to export"
                )
        blob = "\n".join(_text_fields(row))
        if ATTACHMENT.search(blob):
            raise ExportBlocked(
                f"row {row.id}: contains a raw Discord id — refusing to export"
            )
        for raw in raw_ids:
            if raw in blob:
                raise ExportBlocked(f"row {row.id}: contains a raw Discord id — refusing to export")

    for corpus_row in corpus or []:
        if not PSEUDONYM.match(corpus_row.author or ""):
            raise ExportBlocked(
                f"corpus row {corpus_row.id}: author={corpus_row.author!r} is not a pseudonym "
                "— refusing to export"
            )
        if ATTACHMENT.search(corpus_row.text):
            raise ExportBlocked(
                f"corpus row {corpus_row.id}: contains a raw Discord id — refusing to export"
            )
        for raw in raw_ids:
            if raw in corpus_row.text:
                raise ExportBlocked(
                    f"corpus row {corpus_row.id}: contains a raw Discord id — refusing to export"
                )


def row_payload(row: Interaction) -> dict:
    """The published shape of a correction. Field names chosen to be obvious on the Hub."""
    return {
        "id": row.id,
        "prompt": row.prompt,
        "rejected": row.rejected,
        "chosen": row.chosen,
        "signal": row.signal,
        "weight": row.weight,
        "prompt_author": row.prompt_author,
        "signal_author": row.signal_author,
        "created_at": row.created_at,
    }


def corpus_payload(row: CorpusRow) -> dict:
    """The published shape of a corpus row.

    The guild and channel a row came from are stored locally but not published.
    They are pseudonymous, so publishing them would not identify anybody -- they
    are simply not needed by anyone downloading a text corpus, and the smallest
    thing that answers the question is the right thing to publish.
    """
    return {
        "id": row.id,
        "text": row.text,
        "author": row.author,
        "source": row.source,
        "created_at": row.created_at,
    }


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in payloads)
    path.write_text(body, encoding="utf-8")


def build_export(
    settings: Settings,
    out_dir: Path | None = None,
    log: EventLog | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> ExportResult:
    log = log or NullLog()
    ids = ids or Pseudonymiser.load(settings)
    out_dir = out_dir or settings.export_dir

    rows, excluded, dropped_leaky, dropped_blocklist = select_rows(settings, ids, blocklist)
    corpus, corpus_excluded, corpus_leaky, corpus_blocked = select_corpus_rows(
        settings, ids, blocklist
    )
    assert_pseudonymous(rows, ConsentStore(settings.consent_path).known_ids(), corpus)

    corrections = sum(1 for r in rows if r.signal == CORRECTION)
    approvals = sum(1 for r in rows if r.signal == APPROVAL)
    contributors = len(
        {r.signal_author for r in rows} | {r.prompt_author for r in rows} | {c.author for c in corpus}
    )

    _write_jsonl(out_dir / CORPUS_FILE, [corpus_payload(c) for c in corpus])
    _write_jsonl(out_dir / DATA_FILE, [row_payload(r) for r in rows])

    result = ExportResult(
        path=out_dir,
        rows=len(corpus) + len(rows),
        corpus_rows=len(corpus),
        corpus_excluded_no_consent=corpus_excluded,
        corpus_dropped_leaky=corpus_leaky,
        corpus_dropped_blocklist=corpus_blocked,
        corpus_chars=sum(len(c.text) for c in corpus),
        correction_rows=len(rows),
        excluded_no_consent=excluded,
        dropped_leaky=dropped_leaky,
        dropped_blocklist=dropped_blocklist,
        corrections=corrections,
        approvals=approvals,
        contributors=contributors,
    )
    (out_dir / "README.md").write_text(dataset_card(result), encoding="utf-8")

    log.event(
        "export.run",
        rows=result.rows,
        corpus_rows=result.corpus_rows,
        corpus_chars=result.corpus_chars,
        corpus_excluded_no_consent=corpus_excluded,
        corpus_dropped_leaky=corpus_leaky,
        corpus_dropped_blocklist=corpus_blocked,
        correction_rows=result.correction_rows,
        corrections=corrections,
        approvals=approvals,
        contributors=contributors,
        excluded_no_consent=excluded,
        dropped_leaky=dropped_leaky,
        dropped_blocklist=dropped_blocklist,
        out=str(out_dir),
    )
    return result


def push(settings: Settings, repo_id: str, out_dir: Path, log: EventLog | None = None, private: bool = False) -> str:
    """Upload an already-built export. Explicit, never automatic."""
    import os

    from huggingface_hub import HfApi  # imported here so export works without it

    log = log or NullLog()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("set HF_TOKEN to push (nothing was uploaded)")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="babble: sync corpus",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    log.event("export.push", repo=repo_id, private=private, url=url)
    return url


def dataset_card(result: ExportResult) -> str:
    """The README.md that ships with the dataset.

    Deterministic on purpose -- no build timestamp -- so an unchanged corpus
    produces a byte-identical export and pushing twice is a no-op.
    """
    return f"""---
license: mit
language:
- en
task_categories:
- text-generation
tags:
- discord
- pretraining
- corpus
- human-feedback
- from-scratch
pretty_name: babble
configs:
- config_name: default
  data_files:
  - split: train
    path: {CORPUS_FILE}
- config_name: corrections
  data_files:
  - split: train
    path: {DATA_FILE}
---

# babble

Training data for [babble](https://github.com/kowo-co/babble): a ~3M parameter
byte-level transformer that started from **random weights** and has only ever
learned from what people typed at it in Discord.

There is no scraped chat history and no borrowed pretraining corpus. Every row
here is somebody talking to a small confused model on purpose, knowing it was
being collected.

## The two configs

### `default` — the corpus ({result.corpus_rows} rows, {result.corpus_chars:,} characters)

Unlabelled text. One row is one thing a person sent the bot. Nothing is paired
with anything, there is no right answer attached, and this is the whole of the
model's training data — plain next-token prediction over these rows.

| field | meaning |
| --- | --- |
| `id` | content hash of `(text, author)`; stable across exports |
| `text` | what the person wrote |
| `author` | salted hash of who wrote it |
| `source` | how it reached the bot — `mention`, `reply`, `dm`, `ambient`, or `prompt`/`correction` for text lifted off an older correction pair |
| `created_at` | UTC timestamp of capture |

### `corrections` — the pairs ({result.correction_rows} rows)

The older artifact: {result.corrections} corrections and {result.approvals}
approvals. Someone pinged the bot, the bot answered badly, and a human replied
with what it should have said. These are **no longer what the model trains on**
— they are kept and published because they are a real record of people teaching
a model, and because the text in them is also in the corpus above.

| field | meaning |
| --- | --- |
| `id` | content hash of the row; stable across exports |
| `prompt` | what was said to the bot |
| `rejected` | what the bot answered and got corrected on (`null` for 👍 rows) |
| `chosen` | what it should have said — text, an emoji, a gif url, anything |
| `signal` | `correction` or `approval` |
| `weight` | how much the old paired objective leaned on the row; nothing reads it now |
| `prompt_author` | salted hash of the asker |
| `signal_author` | salted hash of whoever corrected or reacted |
| `created_at` | UTC timestamp of capture |

Between them, **{result.rows} rows** from {result.contributors} pseudonymous
participants.

## Consent

Every participant saw an explicit notice and opted in before anything of theirs
was kept. The notice says exactly what this file is: the messages you send the
bot, stored, trained on, and published here for anyone to download.

Consent is tracked in two parts, and the split is deliberate. People who opted
in when babble only kept **corrections** agreed to something narrower than the
corpus, so that older agreement does not carry over — they are asked again, and
until they answer, none of their ordinary messages are collected or published.

By default only messages **addressed to the bot** are collected: an @mention, a
reply to it, or a DM. Anything else you say in a channel is yours and is never
seen. One person can widen that for themselves in one channel with a command, and
narrow it again the same way; it never widens anything for anybody else.

People who declined, ignored the notice, or later withdrew are **not** in this
file, and withdrawal deletes their rows from both configs locally as well.

Discord ids and usernames are never stored in the dataset at all: authors appear
only as salted hashes, and the salt is not published. Messages containing raw
Discord ids or mention markup are dropped rather than published, and so is any
row that matches babble's content blocklist — a speed bump against slurs and
hate terms, not a guarantee.

Because withdrawal is retroactive, **rows can disappear between exports**. That
is the consent model working, not corruption.

## Caveats

The corpus is tiny, the model is tiny, and what it produces is mostly wrong. That
is the entire point — this is a record of something learning to talk from zero
in public.
"""
