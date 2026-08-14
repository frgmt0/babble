"""Append-only event log, written twice: machine-readable and human-readable.

Design rules, all of them load-bearing:

* **Append only.** Files open in "a" mode and are never truncated, not on
  restart and above all not on read. Following the log cannot perturb the run.
* **Rotation by size, never by read.** When a file passes `log_max_bytes` it is
  renamed to `.1`, `.2`, ... and a fresh one starts.
* **Pseudonymous.** Ids go through the same salted hash the exported dataset
  uses, so a log is never more revealing than the public data.
* **Content only from consented users.** Everything else logs the shape of the
  data (how many characters) and the reason it was skipped, never the text.
* **Cheap.** Buffered writes with a time-based flush, so a tail is live within a
  second without paying a syscall per field.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, TextIO

from .config import Settings
from .identity import Pseudonymiser
from .util import utcnow_iso

# Events worth a flush the moment they happen -- the ones a human is likely to be
# waiting on, or that tend to precede a crash.
URGENT = frozenset(
    {
        "bot.start",
        "bot.ready",
        "bot.stop",
        "bot.error",
        "consent.prompt",
        "consent.accept",
        "consent.decline",
        "consent.withdraw",
        "capture.correction",
        "capture.approval",
        "capture.skipped",
        "capture.blocked",
        "train.start",
        "train.resume",
        "train.checkpoint",
        "train.stop",
        "train.interrupt",
        "export.run",
        "export.push",
        "export.blocked",
    }
)


def _size_of(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    text = str(value)
    if text == "":
        return '""'
    if any(c in text for c in " \t\n\"'"):
        return json.dumps(text, ensure_ascii=False)
    return text


class EventLog:
    """Writes `logs/babble.jsonl` and `logs/babble.log` side by side."""

    def __init__(
        self,
        settings: Settings,
        pseudonymiser: Pseudonymiser | None = None,
        *,
        component: str = "babble",
        echo: bool = False,
    ) -> None:
        self.settings = settings
        self.component = component
        self.echo = echo
        self._ids = pseudonymiser
        self._closed = False

        settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = settings.log_dir / "babble.jsonl"
        self.text_path = settings.log_dir / "babble.log"
        self._jsonl = self._open(self.jsonl_path)
        self._text = self._open(self.text_path)
        # Sizes are tracked by hand rather than with tell(), because tell() on a
        # buffered text stream forces a flush -- which would undo the buffering
        # this class exists to provide.
        self._jsonl_size = _size_of(self.jsonl_path)
        self._text_size = _size_of(self.text_path)
        self._last_flush = time.monotonic()
        atexit.register(self.flush)

    @staticmethod
    def _open(path: Path) -> TextIO:
        # "a" is the whole privacy story for readers: never truncates, and every
        # write lands at the end even if something else is tailing the file.
        return open(path, "a", encoding="utf-8", buffering=1 << 16)

    # --- writing -------------------------------------------------------

    def event(self, name: str, **fields: Any) -> dict:
        """Record one event. Returns the record, which is handy in tests."""
        record = {"ts": utcnow_iso(), "component": self.component, "event": name}
        record.update({k: v for k, v in fields.items() if v is not None})

        if self._closed:
            return record

        self._rotate_if_needed()
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self._jsonl.write(line)
        self._jsonl_size += len(line.encode("utf-8"))

        rendered = " ".join(
            f"{k}={_fmt(v)}" for k, v in record.items() if k not in ("ts", "component", "event")
        )
        human = f"{record['ts']}  {name:<22} {rendered}".rstrip() + "\n"
        self._text.write(human)
        self._text_size += len(human.encode("utf-8"))
        if self.echo:
            print(human, end="", flush=True)

        now = time.monotonic()
        if name in URGENT or now - self._last_flush >= self.settings.log_flush_seconds:
            self.flush()
            self._last_flush = now
        return record

    # --- identity helpers ----------------------------------------------

    def user(self, user_id: object) -> str:
        """Pseudonym for a raw id, so callers never have to remember to hash."""
        return self._ids.user(user_id) if self._ids else f"u_raw:{user_id}"

    def channel(self, channel_id: object) -> str:
        return self._ids.channel(channel_id) if self._ids else f"c_raw:{channel_id}"

    def guild(self, guild_id: object) -> str | None:
        """Pseudonym for a guild id, or None (dropped from the record) if there isn't one."""
        if guild_id is None:
            return None
        return self._ids.guild(guild_id) if self._ids else f"g_raw:{guild_id}"

    def preview(self, text: str | None, *, allowed: bool) -> dict:
        """Describe a piece of user text for the log.

        `allowed` is the consent decision. When it is false we record the size and
        nothing else -- a log line must never become a side channel around the
        consent gate.
        """
        if text is None:
            return {"chars": 0}
        if not allowed:
            return {"chars": len(text), "text": "<withheld: no consent>"}
        return {"chars": len(text), "text": _clip(text, self.settings.log_preview_chars)}

    # --- housekeeping --------------------------------------------------

    def _rotate_if_needed(self) -> None:
        limit = self.settings.log_max_bytes
        if self._jsonl_size >= limit:
            self._jsonl = self._roll(self._jsonl, self.jsonl_path)
            self._jsonl_size = 0
        if self._text_size >= limit:
            self._text = self._roll(self._text, self.text_path)
            self._text_size = 0

    def _roll(self, handle: TextIO, path: Path) -> TextIO:
        handle.flush()
        handle.close()
        self._shift(path)
        return self._open(path)

    def _shift(self, path: Path) -> None:
        keep = self.settings.log_backups
        oldest = path.with_suffix(path.suffix + f".{keep}")
        oldest.unlink(missing_ok=True)
        for index in range(keep - 1, 0, -1):
            src = path.with_suffix(path.suffix + f".{index}")
            if src.exists():
                os.replace(src, path.with_suffix(path.suffix + f".{index + 1}"))
        if path.exists():
            os.replace(path, path.with_suffix(path.suffix + ".1"))

    def flush(self) -> None:
        if self._closed:
            return
        for handle in (self._jsonl, self._text):
            try:
                handle.flush()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        self.flush()
        self._closed = True
        for handle in (self._jsonl, self._text):
            try:
                handle.close()
            except (OSError, ValueError):
                pass


class NullLog(EventLog):
    """Drops everything. For code paths that run before settings exist.

    Every method that the rest of the code calls is overridden, because this
    deliberately never runs `EventLog.__init__` and so has no settings or files.
    """

    def __init__(self) -> None:  # noqa: D107 - deliberately does not call super
        self.component = "null"
        self._closed = True
        self._ids = None

    def event(self, name: str, **fields: Any) -> dict:
        return {"event": name, **fields}

    def user(self, user_id: object) -> str:
        return "u_null"

    def channel(self, channel_id: object) -> str:
        return "c_null"

    def guild(self, guild_id: object) -> str | None:
        return None if guild_id is None else "g_null"

    def preview(self, text: str | None, *, allowed: bool) -> dict:
        return {"chars": len(text or "")}

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def _clip(text: str, limit: int) -> str:
    collapsed = text.replace("\n", "\\n")
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


# --- read-only viewing ---------------------------------------------------
# Nothing below opens a file for writing. Watching the bot must never change it.


def tail(path: Path, lines: int = 40) -> list[str]:
    """Last N lines, without reading the whole file."""
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        block = 4096
        data = b""
        while end > 0 and data.count(b"\n") <= lines:
            step = min(block, end)
            end -= step
            fh.seek(end)
            data = fh.read(step) + data
    return data.decode("utf-8", "replace").splitlines()[-lines:]


def follow(path: Path, *, from_start: bool = False, poll: float = 0.4) -> Iterator[str]:
    """Yield lines as they are appended, like `tail -f`. Opens read-only."""
    while not path.exists():
        time.sleep(poll)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        if not from_start:
            fh.seek(0, os.SEEK_END)
        pending = ""
        while True:
            chunk = fh.readline()
            if not chunk:
                time.sleep(poll)
                continue
            pending += chunk
            if pending.endswith("\n"):
                yield pending.rstrip("\n")
                pending = ""
