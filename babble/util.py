"""Small shared helpers. Nothing clever lives here."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utcnow_iso() -> str:
    """Timestamp used in every stored row and log line."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utcnow_stamp() -> str:
    """Filesystem-safe timestamp -- same moment as `utcnow_iso()`, without the
    colons that break using it directly in a filename."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file such that a kill -9 mid-write can never leave a half file.

    The trainer and the consent store both depend on this: we write a temp file
    beside the target and rename it, which is atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def truncate(text: str, limit: int) -> str:
    """Shorten for display/logging, marking that we did."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"
