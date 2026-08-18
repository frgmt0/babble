"""Shared fixtures, including the fake Discord.

`FakeDiscord` plays the part `bot.py` plays in production: it invents message
ids, hands events to the brain, posts whatever comes back, and tells the brain
what id each posted message got. Tests then read like transcripts.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from babble.config import CORRECTION_MARKER, Settings
from babble.core import Babble, Generation, IncomingMessage, ReactionEvent
from babble.identity import Pseudonymiser
from babble.logs import EventLog

TEST_SALT = "salt-for-tests-only"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A whole babble installation inside tmp_path, with a tiny model."""
    s = Settings.for_root(tmp_path)
    s.salt = TEST_SALT
    # Small enough that a training test is a second, not a minute.
    s.n_layer, s.n_head, s.n_embd, s.block_size = 2, 2, 32, 64
    s.batch_size = 2
    s.checkpoint_every = 2
    s.max_new_tokens = 16
    s.ensure_dirs()
    return s


@pytest.fixture
def ids(settings: Settings) -> Pseudonymiser:
    return Pseudonymiser.load(settings)


@pytest.fixture
def log(settings: Settings, ids: Pseudonymiser) -> EventLog:
    return EventLog(settings, ids, component="test")


class FakeGenerator:
    """Stands in for the model, so bot tests never import torch."""

    def __init__(self, text: str = "wug wug blorp") -> None:
        self.text = text
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Generation:
        self.prompts.append(prompt)
        return Generation(text=self.text, step=42, temperature=1.0, top_k=40, ms=1.2)


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def brain(settings: Settings, generator: FakeGenerator, log: EventLog) -> Babble:
    return Babble(settings, generator=generator, log=log, bot_user_id="bot-9999")


@dataclass
class SentMessage:
    id: str
    content: str
    kind: str


class FakeDiscord:
    """A pretend gateway. Same contract as `bot.py`, none of the network."""

    def __init__(self, brain: Babble) -> None:
        self.brain = brain
        self.sent: list[SentMessage] = []
        self._ids = itertools.count(1000)

    def _next_id(self) -> str:
        return str(next(self._ids))

    def ping(
        self,
        user: str,
        text: str = "hello",
        *,
        reply_to: str | None = None,
        attachments: tuple[str, ...] = (),
        channel: str = "chan-1",
    ) -> list[SentMessage]:
        """@mention the bot, or reply to one of its messages."""
        return self._dispatch(
            self.brain.handle_message,
            IncomingMessage(
                message_id=self._next_id(),
                author_id=user,
                content=text,
                channel_id=channel,
                mentions_bot=reply_to is None,
                reply_to_message_id=reply_to,
                reply_to_is_bot=reply_to is not None,
                attachment_urls=attachments,
            ),
        )

    def dm(self, user: str, text: str = "hello", *, channel: str = "dm-1") -> list[SentMessage]:
        """A direct message: addressed to the bot without mentioning anybody."""
        return self._dispatch(
            self.brain.handle_message,
            IncomingMessage(
                message_id=self._next_id(),
                author_id=user,
                content=text,
                channel_id=channel,
                is_dm=True,
            ),
        )

    def correct(
        self, user: str, text: str, *, reply_to: str, attachments: tuple[str, ...] = ()
    ) -> list[SentMessage]:
        """Reply with the correction marker, the way a person teaching it would.

        `ping(..., reply_to=...)` is deliberately left as the *unmarked* reply,
        so a test that wants "someone replied but wasn't teaching" still reads
        like one.
        """
        return self.ping(
            user, f"{CORRECTION_MARKER} {text}", reply_to=reply_to, attachments=attachments
        )

    def say(self, user: str, text: str, *, channel: str = "chan-1") -> list[SentMessage]:
        """An ordinary channel message that does not mention the bot."""
        return self._dispatch(
            self.brain.handle_message,
            IncomingMessage(
                message_id=self._next_id(), author_id=user, content=text, channel_id=channel
            ),
        )

    def react(self, user: str, message_id: str, emoji: str = "👍") -> list[SentMessage]:
        return self._dispatch(
            self.brain.handle_reaction,
            ReactionEvent(
                message_id=message_id, emoji=emoji, user_id=user, channel_id="chan-1"
            ),
        )

    def accept(self, user: str) -> list[SentMessage]:
        return self.say(user, "!babble accept")

    def decline(self, user: str) -> list[SentMessage]:
        return self.say(user, "!babble decline")

    def collect_all(self, user: str, *, channel: str = "chan-1") -> list[SentMessage]:
        """`!babble all` — widen collection to everything this person says here."""
        return self.say(user, "!babble all", channel=channel)

    def only_pings(self, user: str, *, channel: str = "chan-1") -> list[SentMessage]:
        """`!babble pings` — the opposite command, undoing the widening."""
        return self.say(user, "!babble pings", channel=channel)

    def onboard(self, user: str) -> None:
        """Get a user past the consent gate the way a real one would."""
        self.ping(user)  # triggers the notice
        self.accept(user)

    def _dispatch(self, handler, payload) -> list[SentMessage]:
        posted = []
        for reply in handler(payload):
            message = SentMessage(self._next_id(), reply.content, reply.kind)
            self.brain.remember(message.id, reply)  # exactly what bot.py does
            self.sent.append(message)
            posted.append(message)
        return posted

    @property
    def last(self) -> SentMessage:
        return self.sent[-1]


@pytest.fixture
def fake(brain: Babble) -> FakeDiscord:
    return FakeDiscord(brain)


@pytest.fixture
def read_log(settings: Settings, log: EventLog):
    """Read back structured log events. Flushes first, never truncates."""

    def _read(event: str | None = None) -> list[dict]:
        log.flush()
        path = settings.log_dir / "babble.jsonl"
        if not path.exists():
            return []
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [e for e in entries if event is None or e.get("event") == event]

    return _read
