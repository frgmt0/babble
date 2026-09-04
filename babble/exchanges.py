"""What the bot said, and to whom, so a later reply can be scored against it.

A correction arrives minutes after the answer it corrects, and possibly after a
restart, so this cannot live in memory. It maps a bot message id to the exchange
that produced it.

Only exchanges from users who have already granted consent are recorded. That is
a retention decision, not an optimisation: if someone has not opted in, their
message should not sit in a cache on disk waiting for a grade that can never be
captured anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .conversation import ConversationTurn
from .util import atomic_write_text, utcnow_iso


@dataclass(frozen=True)
class Exchange:
    prompt: str
    response: str
    prompt_author_id: str  # raw id: needed to re-check consent at capture time
    created_at: str = ""
    step: int = 0
    # Context used to produce this response. The current prompt/response remain
    # separate so correction and corpus paths never mistake model history for
    # a new piece of human writing.
    history: tuple[ConversationTurn, ...] = ()
    channel_id: str = ""
    guild_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "prompt_author_id": self.prompt_author_id,
            "created_at": self.created_at or utcnow_iso(),
            "step": self.step,
            "history": [turn.to_dict() for turn in self.history],
            "channel_id": self.channel_id,
            "guild_id": self.guild_id,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Exchange":
        raw_history = raw.get("history", ())
        if not isinstance(raw_history, (list, tuple)):
            raw_history = ()
        history = tuple(
            turn
            for item in raw_history
            if (turn := ConversationTurn.from_dict(item)) is not None
        )
        return cls(
            prompt=raw.get("prompt", ""),
            response=raw.get("response", ""),
            prompt_author_id=str(raw.get("prompt_author_id", "")),
            created_at=raw.get("created_at", ""),
            step=int(raw.get("step", 0)),
            history=history,
            channel_id=str(raw.get("channel_id", "")),
            guild_id=(str(raw["guild_id"]) if raw.get("guild_id") is not None else None),
        )


class ExchangeLog:
    def __init__(self, path: Path, max_entries: int = 1000) -> None:
        self.path = path
        self.max_entries = max_entries
        self._entries: dict[str, Exchange] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for message_id, entry in (raw or {}).items():
            if isinstance(entry, dict):
                self._entries[str(message_id)] = Exchange.from_dict(entry)

    def _save(self) -> None:
        payload = {mid: ex.to_dict() for mid, ex in self._entries.items()}
        atomic_write_text(self.path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def record(self, bot_message_id: object, exchange: Exchange) -> None:
        self._entries[str(bot_message_id)] = exchange
        # Insertion-ordered, so trimming the front drops the oldest.
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._save()

    def get(self, bot_message_id: object) -> Exchange | None:
        return self._entries.get(str(bot_message_id))

    def forget_author(self, user_id: object) -> int:
        """Drop every pending exchange belonging to a user who opted out."""
        target = str(user_id)
        doomed = [mid for mid, ex in self._entries.items() if ex.prompt_author_id == target]
        for mid in doomed:
            del self._entries[mid]
        if doomed:
            self._save()
        return len(doomed)

    def __len__(self) -> int:
        return len(self._entries)
