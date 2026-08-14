"""A best-effort Discord feed for training progress, decoupled from the gateway.

`babble train --loop` runs as its own process with no Discord login -- that is
deliberate, the trainer must work standalone. So this posts over a plain HTTPS
webhook rather than teaching the trainer to speak the gateway protocol or
merging it with the bot: a webhook is one POST with a URL, no login, no
heartbeat, nothing to keep alive, and it degrades to "unset env var" instead
of "half-started process" when nobody wants it.

Two rules govern every call in here:

* **Never disturb training.** Every failure -- bad URL, no network, Discord
  down, rate limited -- is caught, logged, and swallowed. `checkpoint()` never
  raises.
* **Unconfigured is silent.** No `BABBLE_LOG_WEBHOOK_URL` means every method
  is a no-op: no request attempted, no log noise, no behaviour change.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from . import __version__
from .blocklist import Blocklist
from .corpus import (
    SOURCE_AMBIENT,
    SOURCE_CORRECTION,
    SOURCE_DM,
    SOURCE_MENTION,
    SOURCE_PROMPT,
    SOURCE_REPLY,
)
from .logs import EventLog, NullLog

WEBHOOK_ENV = "BABBLE_LOG_WEBHOOK_URL"
EVERY_ENV = "BABBLE_LOG_EVERY_N"

SAMPLE_LIMIT = 200
ERROR_BODY_LIMIT = 300
POST_LIMIT = 1990  # a hair under Discord's 2000, so a coalesced burst never 400s

# Discord's edge rejects urllib's default "Python-urllib/3.x" User-Agent with a
# 403 before the request ever reaches the webhook. This is the documented
# bot-UA form: https://discord.com/developers/docs/reference#user-agent
USER_AGENT = f"DiscordBot (https://github.com/kowo-co/babble, {__version__})"

# Mentions the model could emit and that must never resolve, belt and suspenders
# alongside `allowed_mentions: {"parse": []}` on the outgoing payload.
_MENTION = re.compile(r"@(everyone|here)\b|<@[!&]?\d+>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def neuter_sample(text: str) -> str:
    """Make raw model output safe to drop into a Discord message.

    Strips control bytes, breaks mention syntax with a zero-width space so it
    can never resolve to a ping even if `allowed_mentions` were ever dropped,
    flattens backticks so the sample can't escape its code span, collapses
    newlines so the post stays a couple of lines, and truncates.
    """
    text = _CONTROL.sub("", text)
    text = text.replace("`", "'").replace("\n", "⏎")
    text = _MENTION.sub(lambda m: "@​" + m.group(0)[1:], text)
    if len(text) > SAMPLE_LIMIT:
        text = text[: SAMPLE_LIMIT - 1] + "…"
    return text


def post_webhook(url: str, content: str, *, timeout: float = 5.0) -> None:
    """One POST to a Discord webhook. Raises on any failure; callers decide what that means."""
    payload = json.dumps({"content": content, "allowed_mentions": {"parse": []}}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def _error_body(exc: Exception) -> str | None:
    """Pull the response body off an HTTPError, if there is one, truncated for logging."""
    if not isinstance(exc, urllib.error.HTTPError):
        return None
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    if not body:
        return None
    if len(body) > ERROR_BODY_LIMIT:
        body = body[:ERROR_BODY_LIMIT] + "…"
    return body


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass
class TrainingFeed:
    """Posts trainer lifecycle events to a Discord webhook. A no-op unconfigured."""

    webhook_url: str | None
    every_n: int = 1
    log: EventLog = field(default_factory=NullLog)
    sender: Callable[[str, str], None] = post_webhook
    _checkpoints_seen: int = field(default=0, init=False)
    _idle_posted: bool = field(default=False, init=False)

    @classmethod
    def from_env(cls, log: EventLog | None = None) -> "TrainingFeed":
        url = (os.environ.get(WEBHOOK_ENV) or "").strip() or None
        every = max(1, _env_int(EVERY_ENV, 1))
        return cls(webhook_url=url, every_n=every, log=log or NullLog())

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    # --- lifecycle events -------------------------------------------------

    def start(self, *, resumed: bool, step: int) -> None:
        if not self.enabled:
            return
        if resumed:
            self._post(f"🔄 **babble** resumed training at step **{step:,}**")
        else:
            self._post("▶️ **babble** trainer started")

    def idle(self) -> None:
        """Report going idle once; call `active()` to re-arm for the next stretch."""
        if not self.enabled or self._idle_posted:
            return
        self._idle_posted = True
        self._post("💤 **babble** is idle — no consented rows to train on yet")

    def active(self) -> None:
        """A cycle actually ran: the next idle stretch is worth announcing again."""
        self._idle_posted = False

    def cycle_start(
        self,
        *,
        cycle: int,
        stored: int,
        trained: int,
        dropped_consent: int,
        dropped_blocklist: int,
        examples: int,
        batch_size: int,
        lr: float,
        tokens: int = 0,
        train_rows: int = 0,
        val_rows: int = 0,
    ) -> None:
        """One line per cycle for the numbers that barely change: the shape of
        the corpus and the hyperparameters, so the per-checkpoint line doesn't
        have to repeat them.

        `stored`/`trained`/`dropped` count corpus **rows**; `examples` counts what
        those rows tokenise into, which is larger whenever a row was long enough
        to need more than one block.
        """
        if not self.enabled:
            return
        dropped = dropped_consent + dropped_blocklist
        note = ""
        if dropped:
            reasons = []
            if dropped_consent:
                reasons.append(f"{dropped_consent} no consent")
            if dropped_blocklist:
                reasons.append(f"{dropped_blocklist} blocklist")
            note = f" ({', '.join(reasons)})"
        split = f" · {train_rows} train / {val_rows} val rows" if train_rows or val_rows else ""
        token_part = f" ≈{tokens:,} tokens" if tokens else ""
        self._post(
            f"🚀 cycle **{cycle}** starting · {stored} stored → {trained} training, "
            f"{dropped} dropped{note}{split} · {examples} examples{token_part} · "
            f"batch {batch_size} @ lr {lr:g}"
        )

    def cycle_end(self, *, cycle: int, steps: int, seconds: float) -> None:
        if not self.enabled:
            return
        self._post(f"✅ cycle **{cycle}** done · {steps} steps in {seconds:.1f}s")

    def checkpoint(
        self,
        *,
        cycle: int,
        step: int,
        loss: float,
        prev_loss: float | None,
        rows: int,
        prefix: str,
        sample: str,
        probe_side: str = "",
        val_loss: float | None = None,
        prev_val_loss: float | None = None,
        val_rows: int = 0,
        val_enabled: bool | None = None,
        val_disabled_reason: str | None = None,
        overfit_signal: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self._checkpoints_seen += 1
        if self._checkpoints_seen % self.every_n != 0:
            return
        delta = "" if prev_loss is None else f" ({loss - prev_loss:+.3f})"
        lines = [
            f"🔁 cycle **{cycle}** · step **{step:,}** · loss **{loss:.4f}**{delta} · {rows} rows"
        ]
        # `val_enabled=None` means the caller didn't pass any validation state at
        # all -- leave the post exactly as it was before validation existed.
        if val_enabled is True:
            if val_loss is None:
                lines.append("val: no held-out rows this checkpoint")
            else:
                val_delta = "" if prev_val_loss is None else f" ({val_loss - prev_val_loss:+.3f})"
                warning = "  ⚠️ val rising while train falls" if overfit_signal else ""
                lines.append(
                    f"val **{val_loss:.4f}**{val_delta} · {val_rows} held out{warning}"
                )
        elif val_enabled is False:
            reason = f" — {val_disabled_reason}" if val_disabled_reason else ""
            lines.append(f"val: disabled{reason}")
        # Prefix and sample both come out of the corpus / the model, so both are
        # neutered before they hit Discord. It is labelled a continuation because
        # that is what it is: the model was seeded with the opening of a real row
        # and asked to carry on. There is no expected answer to print alongside
        # it -- the corpus has no answers in it -- and inventing a field to hold
        # one would be the single most misleading thing this line could do.
        # `probe_side` says whether the prefix came from a row it trained on:
        # without it, nonsense here is unreadable.
        side = f" _({probe_side})_" if probe_side else ""
        lines.append(
            f"> continuation{side}: `{neuter_sample(prefix)}` → `{neuter_sample(sample)}`"
        )
        self._post("\n".join(lines))

    def publish(self, *, rows: int, url: str) -> None:
        if not self.enabled:
            return
        self._post(f"📤 **babble** auto-published **{rows}** row(s) to {url}")

    def publish_failed(self, error: str) -> None:
        if not self.enabled:
            return
        self._post(f"⚠️ **babble** auto-publish to HuggingFace failed — {error}")

    # --- plumbing -----------------------------------------------------

    def _post(self, content: str) -> None:
        if not self.webhook_url:
            return
        try:
            self.sender(self.webhook_url, content)
        except Exception as exc:  # network, DNS, HTTP, rate limit -- never fatal to training
            error = f"{type(exc).__name__}: {exc}"
            body = _error_body(exc)
            if body:
                error = f"{error} body={body!r}"
            self.log.event("feed.post_failed", error=error)


# --- collection feed ------------------------------------------------------

#: How a captured row reached the bot, in words, for the feed. The corpus stores
#: a terse `source`; this is the human-readable version that says which surface
#: it came from -- a ping, a DM, or a widened `!babble all` channel grant.
SOURCE_LABELS = {
    SOURCE_MENTION: "a ping",
    SOURCE_REPLY: "a reply",
    SOURCE_DM: "a DM",
    SOURCE_AMBIENT: "a widened channel (`!babble all`)",
    SOURCE_CORRECTION: "a correction",
    SOURCE_PROMPT: "a prompt (from a correction)",
}

#: Text withheld because it matched the blocklist. It should never get this far
#: -- `core` refuses to store a blocked row in the first place -- but the feed
#: re-checks anyway so "collected text on Discord is blocklist-filtered" is true
#: at the one place text actually reaches Discord, not only upstream of it.
WITHHELD = "⟨withheld by the content filter⟩"

#: Milestone intervals, scaled by size: fire often while the corpus is tiny,
#: rarely once it is large. Each entry is (below_this_size, use_this_interval);
#: the last interval applies to everything at or above the final threshold. At
#: 54 rows the row interval is 25 (last crossed 50, next 75); at 10k it is 500.
ROW_MILESTONES = ((100, 25), (1_000, 100), (10_000, 500), (float("inf"), 2_500))
CHAR_MILESTONES = ((10_000, 2_000), (100_000, 20_000), (float("inf"), 100_000))


def milestone_interval(value: int, table: tuple) -> int:
    """The interval that applies at `value`, from a size-scaled `table`."""
    for threshold, interval in table:
        if value < threshold:
            return interval
    return table[-1][1]


def _reached_milestone(value: int, table: tuple) -> int:
    """The highest milestone at or below `value`, at `value`'s own scale."""
    interval = milestone_interval(value, table)
    return (value // interval) * interval


def _crossed_milestone(value: int, last: int, table: tuple) -> int | None:
    """The highest milestone `value` has reached past `last`, or None.

    Uses the interval that applies at `value`, so a corpus that jumps across a
    scale boundary (99 -> 101 rows) reports the new-scale milestone (100) rather
    than silently skipping it.
    """
    interval = milestone_interval(value, table)
    reached = (value // interval) * interval
    return reached if reached >= interval and reached > last else None


@dataclass
class _PendingRow:
    text: str
    source: str
    author: str


@dataclass
class CollectionFeed:
    """Posts *collection* events -- rows arriving, consent changing, growth
    milestones, dataset publishes -- to the same Discord webhook the training
    feed uses. It is what that channel shows while no trainer is running.

    Same two rules as `TrainingFeed`: every send failure is caught and logged,
    never raised, so a Discord outage cannot break capture; and an unset webhook
    makes every method a no-op.

    Rows are coalesced: one arriving inside `window_seconds` of the last is held
    and listed together, so someone who ran `!babble all` and is typing does not
    produce one post per message. The buffer is flushed by `flush_due()` on a
    timer (production) and by `flush()` outright (shutdown, tests).
    """

    webhook_url: str | None
    log: EventLog = field(default_factory=NullLog)
    sender: Callable[[str, str], None] = post_webhook
    blocklist: Blocklist | None = None
    window_seconds: float = 3.0
    max_coalesce: int = 8  # a burst larger than this flushes at once, never grows unbounded
    clock: Callable[[], float] = time.monotonic
    _pending: list = field(default_factory=list, init=False)
    _pending_since: float = field(default=0.0, init=False)
    _pending_totals: tuple[int, int, int] | None = field(default=None, init=False)
    _last_row_milestone: int = field(default=0, init=False)
    _last_char_milestone: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @classmethod
    def from_env(cls, log: EventLog | None = None, blocklist: Blocklist | None = None) -> "CollectionFeed":
        url = (os.environ.get(WEBHOOK_ENV) or "").strip() or None
        return cls(
            webhook_url=url,
            log=log or NullLog(),
            blocklist=blocklist if blocklist is not None else Blocklist.load(),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def prime(self, *, rows: int, chars: int) -> None:
        """Seed the milestone markers from the corpus's current size.

        The markers live in memory, so without this a restart would re-announce
        the last milestone on the very next captured row. Called once at startup
        with the size already on disk, so only genuinely new milestones fire.
        """
        self._last_row_milestone = _reached_milestone(rows, ROW_MILESTONES)
        self._last_char_milestone = _reached_milestone(chars, CHAR_MILESTONES)

    # --- row capture ----------------------------------------------------

    def row(self, *, text: str, source: str, author: str, rows: int, chars: int, contributors: int) -> None:
        """One freshly-stored corpus row. Buffered and coalesced, not posted yet."""
        if not self.enabled:
            return
        with self._lock:
            self._flush_if_due_locked()
            self._pending.append(_PendingRow(text=text, source=source, author=author))
            if len(self._pending) == 1:
                self._pending_since = self.clock()
            self._pending_totals = (rows, chars, contributors)
            if len(self._pending) >= self.max_coalesce:
                self._flush_locked()

    def flush_due(self) -> None:
        """Post the buffer if its oldest row has waited out the window. On a timer."""
        if not self.enabled:
            return
        with self._lock:
            self._flush_if_due_locked()

    def flush(self) -> None:
        """Post whatever is buffered, window or no window."""
        if not self.enabled:
            return
        with self._lock:
            self._flush_locked()

    def _flush_if_due_locked(self) -> None:
        if self._pending and self.clock() - self._pending_since >= self.window_seconds:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        pending, totals = self._pending, self._pending_totals
        self._pending = []
        self._pending_totals = None
        self._post(self._render_rows(pending, totals))
        if totals is not None:
            self._announce_milestones(totals[0], totals[1])

    def _render_rows(self, pending: list, totals: tuple[int, int, int] | None) -> str:
        header = f"🌱 corpus **+{len(pending)}**" if len(pending) > 1 else "🌱 corpus **+1**"
        lines = [header]
        for row in pending:
            body = WITHHELD if self._is_blocked(row.text) else f"`{neuter_sample(row.text)}`"
            label = SOURCE_LABELS.get(row.source, row.source)
            lines.append(f"> {body} — {label} · {row.author}")
        if totals is not None:
            rows, chars, contributors = totals
            lines.append(
                f"now **{rows:,}** rows · **{chars:,}** chars · "
                f"**{contributors:,}** {'contributor' if contributors == 1 else 'contributors'}"
            )
        content = "\n".join(lines)
        return content if len(content) <= POST_LIMIT else content[: POST_LIMIT - 1] + "…"

    def _announce_milestones(self, rows: int, chars: int) -> None:
        row_hit = _crossed_milestone(rows, self._last_row_milestone, ROW_MILESTONES)
        if row_hit is not None:
            self._last_row_milestone = row_hit
            self._post(f"📈 milestone — **{row_hit:,}** corpus rows collected")
        char_hit = _crossed_milestone(chars, self._last_char_milestone, CHAR_MILESTONES)
        if char_hit is not None:
            self._last_char_milestone = char_hit
            self._post(f"📈 milestone — **{char_hit:,}** characters collected")

    def _is_blocked(self, text: str) -> bool:
        return self.blocklist is not None and self.blocklist.matches(text)

    # --- consent events -------------------------------------------------

    def consent_granted(self, *, author: str) -> None:
        self._post(f"✅ **{author}** opted in — their messages now go into the corpus")

    def consent_declined(self, *, author: str) -> None:
        self._post(f"🚫 **{author}** opted out — nothing of theirs is collected")

    def channel_widened(self, *, author: str, channel: str) -> None:
        self._post(
            f"📡 **{author}** opened channel {channel} with `!babble all` — "
            "everything they say there is now collected"
        )

    def channel_narrowed(self, *, author: str, channel: str) -> None:
        self._post(
            f"🔕 **{author}** turned `!babble all` back off in channel {channel} — "
            "only what they send the bot there is collected now"
        )

    def consent_withdrawn(self, *, author: str, corpus_purged: int, correction_purged: int) -> None:
        corpus = f"{corpus_purged} corpus" if corpus_purged != 1 else "1 corpus"
        corr = f"{correction_purged} correction" if correction_purged != 1 else "1 correction"
        self._post(
            f"🗑️ **{author}** withdrew — purged **{corpus_purged + correction_purged}** "
            f"stored row(s) ({corpus} · {corr})"
        )

    # --- publish --------------------------------------------------------

    def published(self, *, rows: int, url: str, grew_rows: int, grew_chars: int) -> None:
        self._post(
            f"📤 published **{rows:,}** row(s) to {url} — corpus grew "
            f"**+{grew_rows:,}** rows / **+{grew_chars:,}** chars since the last publish"
        )

    def publish_failed(self, error: str) -> None:
        self._post(f"⚠️ dataset publish failed — {error}")

    # --- plumbing -------------------------------------------------------

    def _post(self, content: str) -> None:
        if not self.webhook_url:
            return
        try:
            self.sender(self.webhook_url, content)
        except Exception as exc:  # network, DNS, HTTP, rate limit -- never fatal to collection
            error = f"{type(exc).__name__}: {exc}"
            body = _error_body(exc)
            if body:
                error = f"{error} body={body!r}"
            self.log.event("feed.post_failed", error=error)
