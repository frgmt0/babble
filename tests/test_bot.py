"""Direct tests of `babble/bot.py`, the discord.py adapter itself.

These stay narrow on purpose: the pure helper functions (`_mentions_bot`,
`_guild_visibility`), exercised with small duck-typed stand-ins for discord.py
objects. No gateway, no real `client.run()`. Everything else the bot does is
already covered end-to-end through `babble/core.py` via the `FakeDiscord` in
conftest.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord

from babble.bot import BabbleClient, _guild_visibility, _mentions_bot
from babble.core import IncomingMessage, Reply


class FakeUser:
    def __init__(self, id_: int) -> None:
        self.id = id_

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeUser) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)


class FakeRole:
    def __init__(self, id_: int, *, default: bool = False) -> None:
        self.id = id_
        self._default = default

    def is_default(self) -> bool:
        return self._default


class FakeMember:
    def __init__(self, roles) -> None:
        self.roles = list(roles)


class FakeMessage:
    def __init__(self, *, mentions=(), role_mentions=(), guild=None) -> None:
        self.mentions = list(mentions)
        self.role_mentions = list(role_mentions)
        self.guild = guild


BOT = FakeUser(999)
OTHER = FakeUser(111)


# --- _mentions_bot ---------------------------------------------------------


def test_a_direct_mention_still_triggers_a_response():
    message = FakeMessage(mentions=[BOT])
    assert _mentions_bot(BOT, message) is True


def test_someone_elses_mention_does_not_trigger_a_response():
    message = FakeMessage(mentions=[OTHER])
    assert _mentions_bot(BOT, message) is False


def test_a_mention_of_a_role_the_bot_holds_triggers_a_response():
    bot_role = FakeRole(42)
    everyone_role = FakeRole(1, default=True)
    guild = SimpleNamespace(me=FakeMember([everyone_role, bot_role]))
    message = FakeMessage(role_mentions=[bot_role], guild=guild)

    assert _mentions_bot(BOT, message) is True


def test_a_role_mention_for_a_role_the_bot_does_not_have_is_ignored():
    other_role = FakeRole(7)
    bot_role = FakeRole(42)
    guild = SimpleNamespace(me=FakeMember([bot_role]))
    message = FakeMessage(role_mentions=[other_role], guild=guild)

    assert _mentions_bot(BOT, message) is False


def test_everyone_and_here_never_trigger_a_response_even_via_the_default_role():
    # @everyone/@here never surface in discord.py's `role_mentions` in the
    # first place; this pins the defence (excluding the default role) that
    # holds even if that ever changed.
    bot_role = FakeRole(42)
    everyone_role = FakeRole(1, default=True)
    guild = SimpleNamespace(me=FakeMember([everyone_role, bot_role]))
    message = FakeMessage(role_mentions=[everyone_role], guild=guild)

    assert _mentions_bot(BOT, message) is False


def test_role_mention_with_no_guild_is_safe():
    message = FakeMessage(role_mentions=[FakeRole(1)], guild=None)
    assert _mentions_bot(BOT, message) is False


def test_role_mention_with_no_cached_member_is_safe():
    guild = SimpleNamespace(me=None)
    message = FakeMessage(role_mentions=[FakeRole(1)], guild=guild)
    assert _mentions_bot(BOT, message) is False


def test_no_bot_user_yet_is_safe():
    message = FakeMessage(mentions=[OTHER])
    assert _mentions_bot(None, message) is False


# --- _guild_visibility -------------------------------------------------


class FakeChannel:
    def __init__(self, *, view: bool, send: bool) -> None:
        self._perms = discord.Permissions(view_channel=view, send_messages=send)

    def permissions_for(self, member) -> discord.Permissions:
        return self._perms


def test_guild_visibility_counts_seeable_and_sendable_channels():
    guild = SimpleNamespace(
        me=object(),
        text_channels=[
            FakeChannel(view=True, send=True),
            FakeChannel(view=True, send=False),
            FakeChannel(view=False, send=False),
        ],
    )

    assert _guild_visibility(guild) == (3, 2, 1)


def test_guild_visibility_with_no_cached_member_reports_nothing_visible():
    guild = SimpleNamespace(me=None, text_channels=[FakeChannel(view=True, send=True)])

    assert _guild_visibility(guild) == (1, 0, 0)


def test_guild_visibility_with_no_text_channels():
    guild = SimpleNamespace(me=object(), text_channels=[])

    assert _guild_visibility(guild) == (0, 0, 0)


# --- _send_reply: a failed send must say why, distinguishably ----------


class FakeSendChannel:
    id = 4242


class FakeSentMessage:
    def __init__(self, id_: int) -> None:
        self.id = id_


class FakeOutboundMessage:
    """Stands in for the `discord.Message` being replied to."""

    def __init__(self, *, reply_exc: Exception | None = None, reply_result=None) -> None:
        self.id = 1
        self.channel = FakeSendChannel()
        self._reply_exc = reply_exc
        self._reply_result = reply_result

    async def reply(self, content: str, mention_author: bool = False):
        if self._reply_exc is not None:
            raise self._reply_exc
        return self._reply_result


def _http_exception(cls, status: int, reason: str):
    return cls(SimpleNamespace(status=status, reason=reason), "denied")


def test_a_forbidden_send_is_logged_as_forbidden(settings, log, brain, read_log):
    client = BabbleClient(settings, log, brain=brain)
    outbound = FakeOutboundMessage(reply_exc=_http_exception(discord.Forbidden, 403, "Forbidden"))
    incoming = IncomingMessage(message_id="1", author_id="a", channel_id="chan-9", guild_id="guild-9")

    result = asyncio.run(client._send_reply(outbound, Reply("hi"), incoming))

    assert result is None
    (event,) = read_log("bot.error")
    assert event["reason"] == "forbidden"
    assert event["channel"].startswith("c_")
    assert event["guild"].startswith("g_")


def test_a_non_forbidden_send_failure_is_logged_distinctly(settings, log, brain, read_log):
    client = BabbleClient(settings, log, brain=brain)
    outbound = FakeOutboundMessage(
        reply_exc=_http_exception(discord.HTTPException, 500, "Server Error")
    )
    incoming = IncomingMessage(message_id="1", author_id="a", channel_id="chan-9")

    result = asyncio.run(client._send_reply(outbound, Reply("hi"), incoming))

    assert result is None
    (event,) = read_log("bot.error")
    assert event["reason"] == "http_error"


def test_a_successful_send_is_not_logged_as_an_error(settings, log, brain, read_log):
    client = BabbleClient(settings, log, brain=brain)
    sent = FakeSentMessage(999)
    outbound = FakeOutboundMessage(reply_result=sent)
    incoming = IncomingMessage(message_id="1", author_id="a", channel_id="chan-9")

    result = asyncio.run(client._send_reply(outbound, Reply("hi"), incoming))

    assert result is sent
    assert read_log("bot.error") == []
