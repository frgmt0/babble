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
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from . import __version__
from .logs import EventLog, NullLog

WEBHOOK_ENV = "BABBLE_LOG_WEBHOOK_URL"
EVERY_ENV = "BABBLE_LOG_EVERY_N"

SAMPLE_LIMIT = 200
ERROR_BODY_LIMIT = 300

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

    def checkpoint(
        self,
        *,
        cycle: int,
        step: int,
        loss: float,
        prev_loss: float | None,
        rows: int,
        prompt: str,
        sample: str,
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
        lines.append(f"> {prompt!r} → `{neuter_sample(sample)}`")
        self._post("\n".join(lines))

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
