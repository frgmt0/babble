"""Publishing the corrections to HuggingFace.

Two rules govern this file:

* **Only consented rows leave the machine.** Consent is re-checked here against
  the live consent store, not trusted from capture time, so a withdrawal always
  wins even if the purge failed for some reason.
* **No raw identifier is ever written.** Author fields must look like
  pseudonyms or the export aborts, and any row whose *text* contains a known raw
  Discord id or mention markup is dropped rather than published.

Nothing here runs automatically. Capture never pushes; you have to type
`babble export --push`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .blocklist import Blocklist
from .config import Settings
from .consent import ConsentStore
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .store import APPROVAL, CORRECTION, Interaction, InteractionStore

PSEUDONYM = re.compile(r"^u_[0-9a-f]{16}$")
MENTION = re.compile(r"<@[!&]?\d+>")
DATA_FILE = "data/train.jsonl"


class ExportBlocked(RuntimeError):
    """Raised when the output would contain something it must never contain."""


@dataclass
class ExportResult:
    path: Path
    rows: int
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


def select_rows(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> tuple[list[Interaction], int, int, int]:
    """(publishable rows, excluded for consent, dropped for leaks, dropped for the blocklist)."""
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)
    raw_ids = _leak_candidates(consent.known_ids())
    allowed = {ids.user(uid) for uid in consent.granted_ids()}

    rows = InteractionStore(settings.interactions_path).all()
    consented = [r for r in rows if r.prompt_author in allowed and r.signal_author in allowed]
    excluded = len(rows) - len(consented)

    leak_free = []
    for row in consented:
        blob = "\n".join(_text_fields(row))
        if MENTION.search(blob) or any(uid and uid in blob for uid in raw_ids):
            continue  # someone typed a raw id into their message; do not publish it
        leak_free.append(row)
    dropped_leaky = len(consented) - len(leak_free)

    # Same three-place enforcement as consent: a row stored before a blocked
    # term was added must not survive to be published once it is.
    clean = [r for r in leak_free if not blocklist.matches(r.prompt, r.chosen, r.rejected)]
    dropped_blocklist = len(leak_free) - len(clean)

    # Deterministic order, and one row per fact, so re-running is a no-op.
    unique = {row.id: row for row in clean}
    ordered = sorted(unique.values(), key=lambda r: (r.created_at, r.id))
    return ordered, excluded, dropped_leaky, dropped_blocklist


def assert_pseudonymous(rows: list[Interaction], known_raw_ids: list[str]) -> None:
    """Last line of defence, run against the exact rows about to be written."""
    for row in rows:
        for field_name in ("prompt_author", "signal_author"):
            value = getattr(row, field_name)
            if not PSEUDONYM.match(value or ""):
                raise ExportBlocked(
                    f"row {row.id}: {field_name}={value!r} is not a pseudonym — refusing to export"
                )
        blob = "\n".join(_text_fields(row))
        for raw in _leak_candidates(known_raw_ids):
            if raw in blob:
                raise ExportBlocked(f"row {row.id}: contains a raw Discord id — refusing to export")


def row_payload(row: Interaction) -> dict:
    """The published shape. Field names chosen to be obvious on the Hub."""
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
    assert_pseudonymous(rows, ConsentStore(settings.consent_path).known_ids())

    corrections = sum(1 for r in rows if r.signal == CORRECTION)
    approvals = sum(1 for r in rows if r.signal == APPROVAL)
    contributors = len({r.signal_author for r in rows} | {r.prompt_author for r in rows})

    data_path = out_dir / DATA_FILE
    data_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row_payload(r), ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    data_path.write_text(body, encoding="utf-8")

    result = ExportResult(
        path=out_dir,
        rows=len(rows),
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
        commit_message="babble: sync corrections",
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
- human-feedback
- corrections
- from-scratch
pretty_name: babble corrections
configs:
- config_name: default
  data_files:
  - split: train
    path: {DATA_FILE}
---

# babble — corrections

Training data for [babble](https://github.com/kowo-co/babble): a ~3M parameter
byte-level transformer that started from **random weights** and has only ever
learned from people correcting it in Discord.

There is no pretraining corpus. There is no scraped chat history. Every row here
is somebody deliberately teaching a small confused model to talk.

## How a row happens

1. Someone @mentions the bot.
2. The bot replies with whatever its current weights produce. Early on this is
   noise, and it is supposed to be.
3. The human either reacts 👍 (a weak "that was fine") or **replies with what it
   should have said** (the strong signal, and the one that matters).

## Fields

| field | meaning |
| --- | --- |
| `id` | content hash of the row; stable across exports |
| `prompt` | what was said to the bot |
| `rejected` | what the bot answered and got corrected on (`null` for 👍 rows) |
| `chosen` | what it should have said — text, an emoji, a gif url, anything |
| `signal` | `correction` or `approval` |
| `weight` | how much the trainer leans on the row (corrections count for more) |
| `prompt_author` | salted hash of the asker |
| `signal_author` | salted hash of whoever corrected or reacted |
| `created_at` | UTC timestamp of capture |

Currently **{result.rows} rows** — {result.corrections} corrections,
{result.approvals} approvals, from {result.contributors} pseudonymous participants.

## Consent

Every participant saw an explicit notice the first time they pinged the bot and
opted in before anything of theirs was kept. People who declined, ignored the
notice, or later withdrew are **not** in this file, and withdrawal deletes their
rows locally as well.

Discord ids and usernames are never stored in the dataset at all: authors appear
only as salted hashes, and the salt is not published. Messages containing raw
Discord ids or mention markup are dropped rather than published, and so is any
row where either side matches babble's content blocklist — a speed bump
against slurs and hate terms, not a guarantee.

Because withdrawal is retroactive, **rows can disappear between exports**. That
is the consent model working, not corruption.

## Caveats

The corpus is tiny, the model is tiny, and the responses are mostly wrong. That
is the entire point — this is a record of something learning to talk from zero
in public.
"""
