"""Auto-publish keyed to the *data*, not to training.

The old cadence pushed the dataset every N training checkpoints. That was fine
while a trainer ran continuously, but babble is in a collection phase now: no
trainer, no checkpoints, so a checkpoint-keyed publish would never fire and the
public dataset would quietly freeze. This keys the same publish to corpus
growth instead -- once enough new rows or characters have accumulated since the
last publish, it exports and pushes through the exact same consent/blocklist
gate as a manual `babble export --push`.

The trainer's own checkpoint-keyed publish (`trainer._maybe_auto_publish`) is
left intact for when someone runs `babble train` by hand; this is the path that
keeps the dataset live while only the bot is running.

Two rules, the same ones the training-feed publish already obeys:

* **Never disturb collection.** Every failure -- export blocked, bad token, no
  network, HF down -- is caught, logged, reported in the feed, and swallowed.
  `maybe_publish()` never raises.
* **Consent and the blocklist are re-checked at export**, not trusted from
  capture time, because that check lives in `build_export`/`select_corpus_rows`
  and this calls straight into it rather than reimplementing a looser one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings
from .corpus import CorpusStore
from .discord_feed import CollectionFeed
from .export_hf import CORPUS_FILE, DATA_FILE, ExportBlocked, ExportResult, build_export
from .export_hf import push as push_export
from .logs import EventLog, NullLog

#: Publish once the corpus has grown by either of these since the last publish.
#: Deliberately small for a corpus measured in tens of rows -- a handful of new
#: rows, or a paragraph's worth of new text, is a meaningful update at this size.
DEFAULT_EVERY_ROWS = 10
DEFAULT_EVERY_CHARS = 2_000


@dataclass
class _Baseline:
    rows: int = 0
    chars: int = 0
    content_hash: str | None = None


class GrowthPublisher:
    """Publishes the dataset when the corpus has grown by a meaningful amount.

    The baseline (corpus size at the last publish *attempt*) is persisted, so a
    bot restart neither loses it nor re-publishes on boot, and so a failed push
    does not hammer HuggingFace on every subsequent message: an attempt advances
    the baseline whatever its outcome, and the next attempt waits for another
    threshold of growth. No data is lost by that -- `build_export` always writes
    the whole corpus, so the next successful push carries everything.
    """

    def __init__(
        self,
        settings: Settings,
        log: EventLog | None = None,
        *,
        feed: CollectionFeed | None = None,
        every_rows: int = DEFAULT_EVERY_ROWS,
        every_chars: int = DEFAULT_EVERY_CHARS,
        exporter: Callable[..., ExportResult] = build_export,
        pusher: Callable[..., str] = push_export,
    ) -> None:
        self.settings = settings
        self.log = log or NullLog()
        self.feed = feed
        self.every_rows = every_rows
        self.every_chars = every_chars
        self._exporter = exporter
        self._pusher = pusher
        self._baseline = self._load_baseline()

    @property
    def state_path(self) -> Path:
        return self.settings.data_dir / "publish_state.json"

    def _load_baseline(self) -> _Baseline:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _Baseline()
        return _Baseline(
            rows=int(raw.get("rows", 0)),
            chars=int(raw.get("chars", 0)),
            content_hash=raw.get("content_hash"),
        )

    def _save_baseline(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "rows": self._baseline.rows,
                    "chars": self._baseline.chars,
                    "content_hash": self._baseline.content_hash,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _grown_enough(self, rows: int, chars: int) -> bool:
        return (
            rows - self._baseline.rows >= self.every_rows
            or chars - self._baseline.chars >= self.every_chars
        )

    def maybe_publish(self) -> None:
        """Publish if the corpus has grown past the threshold. Never raises."""
        if not (self.every_rows or self.every_chars):
            return
        totals = CorpusStore(self.settings.corpus_path).totals()
        if not self._grown_enough(totals.rows, totals.chars):
            return

        grew_rows = totals.rows - self._baseline.rows
        grew_chars = totals.chars - self._baseline.chars
        # An attempt advances the baseline whatever happens next, so we do not
        # re-attempt until another threshold of growth. Data is never lost: the
        # export below always writes the entire corpus.
        self._baseline.rows = totals.rows
        self._baseline.chars = totals.chars
        self._save_baseline()

        try:
            result = self._exporter(self.settings, log=self.log)
        except ExportBlocked as exc:
            self.log.event("publish.blocked", error=str(exc))
            if self.feed is not None:
                self.feed.publish_failed(f"export blocked: {exc}")
            return

        content_hash = _content_hash(result.path)
        if content_hash == self._baseline.content_hash:
            # The corpus grew but the *publishable* bytes did not -- every new
            # row was unconsented or blocklisted and dropped at export. Nothing
            # to push; the baseline already advanced so we won't spin on it.
            self.log.event("publish.skipped", reason="unchanged", rows=result.rows)
            return

        try:
            url = self._pusher(self.settings, self.settings.hf_repo, result.path, log=self.log)
        except Exception as exc:  # network, rate limit, bad token, HF down -- never fatal
            self.log.event("publish.failed", error=f"{type(exc).__name__}: {exc}")
            if self.feed is not None:
                self.feed.publish_failed(f"{type(exc).__name__}: {exc}")
            return

        self._baseline.content_hash = content_hash
        self._save_baseline()
        self.log.event("publish.ok", rows=result.rows, url=url, grew_rows=grew_rows)
        if self.feed is not None:
            self.feed.published(rows=result.rows, url=url, grew_rows=grew_rows, grew_chars=grew_chars)


def _content_hash(export_dir: Path) -> str:
    """Hash of both published files, so a change to either counts as changed."""
    digest = hashlib.sha256()
    for name in (CORPUS_FILE, DATA_FILE):
        path = export_dir / name
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()
