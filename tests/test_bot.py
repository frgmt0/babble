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


def test_prewarm_runs_off_the_bot_lock_and_is_logged(settings, log, brain, read_log):
    client = BabbleClient(settings, log, brain=brain)
    seen = []

    def prewarm(reply):
        seen.append(client._lock.locked())
        return {"tokens": 3, "prefilled": 2}

    brain.prewarm = prewarm

    async def go():
        # A generation holds the bot lock; the warm must not queue behind it.
        async with client._lock:
            client._schedule_prewarm(Reply("hi", continues=True))
            await asyncio.wait_for(asyncio.gather(*client._prewarms), 5)

    asyncio.run(go())
    assert seen == [True]
    (event,) = read_log("bot.prewarm")
    assert event["tokens"] == 3 and event["prefilled"] == 2


def test_a_failing_prewarm_is_logged_not_raised(settings, log, brain, read_log):
    client = BabbleClient(settings, log, brain=brain)

    def boom(reply):
        raise RuntimeError("engine gone")

    brain.prewarm = boom

    async def go():
        client._schedule_prewarm(Reply("hi", continues=True))
        await asyncio.gather(*client._prewarms)

    asyncio.run(go())
    (event,) = read_log("bot.error")
    assert event["where"] == "prewarm"


# --- /pretend: owner-only posting as the bot ---------------------------------


class FakeResponse:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    async def send_message(self, content, ephemeral=False):
        self._sink.append(("response", content, ephemeral))

    async def defer(self, ephemeral=False, thinking=False):
        self._sink.append(("defer", ephemeral))


class FakeFollowup:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    async def send(self, content, ephemeral=False):
        self._sink.append(("followup", content, ephemeral))


class FakePosted:
    jump_url = "https://discord.com/channels/1/2/3"


class FakeTarget:
    def __init__(self, posts: list) -> None:
        self._posts = posts

    async def reply(self, mention_author=True, **kwargs):
        self._posts.append(("reply", mention_author, kwargs))
        return FakePosted()


class FakePretendChannel:
    id = 4242

    def __init__(self, *, send_exc: Exception | None = None) -> None:
        self.posts: list = []
        self.fetched: list = []
        self._send_exc = send_exc

    async def send(self, **kwargs):
        if self._send_exc is not None:
            raise self._send_exc
        self.posts.append(("send", kwargs))
        return FakePosted()

    async def fetch_message(self, id_):
        self.fetched.append(id_)
        return FakeTarget(self.posts)


class FakeAttachment:
    def __init__(self, name: str) -> None:
        self.name = name

    def is_spoiler(self) -> bool:
        return False

    async def to_file(self, spoiler=False):
        return f"file:{self.name}"


def _interaction(user_id: int, channel) -> SimpleNamespace:
    sink: list = []
    return SimpleNamespace(
        user=FakeUser(user_id),
        channel=channel,
        guild_id=77,
        response=FakeResponse(sink),
        followup=FakeFollowup(sink),
        sink=sink,
    )


def _pretend(client, interaction, **kwargs):
    return asyncio.run(client._pretend(interaction, **kwargs))


def test_pretend_is_registered_with_message_images_and_reply_to(settings, log, brain):
    client = BabbleClient(settings, log, brain=brain)
    command = client.tree.get_command("pretend")
    assert command is not None
    names = [p.name for p in command.parameters]
    assert names == ["message", "image", "image2", "image3", "reply_to"]
    assert not any(p.required for p in command.parameters)
    message = command.get_parameter("message")
    assert message.max_value == 2000


def test_pretend_refuses_a_non_owner_and_posts_nothing(settings, log, brain, read_log, monkeypatch):
    monkeypatch.setenv("BABBLE_OWNER_IDS", "1, 2")
    client = BabbleClient(settings, log, brain=brain)
    channel = FakePretendChannel()
    interaction = _interaction(3, channel)

    _pretend(client, interaction, message="hi")

    assert channel.posts == []
    assert interaction.sink == [("response", "Only the bot owner can use this.", True)]
    (event,) = read_log("bot.pretend")
    assert event["status"] == "denied"


def test_pretend_falls_back_to_the_application_owner(settings, log, brain, monkeypatch):
    monkeypatch.delenv("BABBLE_OWNER_IDS", raising=False)
    client = BabbleClient(settings, log, brain=brain)

    async def app_info():
        return SimpleNamespace(team=None, owner=FakeUser(5))

    client.application_info = app_info
    assert asyncio.run(client._is_owner(FakeUser(5))) is True
    assert asyncio.run(client._is_owner(FakeUser(6))) is False


def test_pretend_posts_text_and_images_for_the_owner(settings, log, brain, read_log, monkeypatch):
    monkeypatch.setenv("BABBLE_OWNER_IDS", "1")
    client = BabbleClient(settings, log, brain=brain)
    channel = FakePretendChannel()
    interaction = _interaction(1, channel)

    _pretend(client, interaction, message="boop", image=FakeAttachment("a.png"), image3=FakeAttachment("b.gif"))

    assert channel.posts == [("send", {"content": "boop", "files": ["file:a.png", "file:b.gif"]})]
    assert interaction.sink[-1] == ("followup", f"Posted: {FakePosted.jump_url}", True)
    (event,) = read_log("bot.pretend")
    assert event["status"] == "sent" and event["chars"] == 4 and event["files"] == 2
    assert "boop" not in str(event)
    # Never remembered as an exchange: nobody can "correct" a pretend post.
    assert brain.exchanges.get("3") is None


def test_pretend_can_reply_to_a_message_link(settings, log, brain, monkeypatch):
    monkeypatch.setenv("BABBLE_OWNER_IDS", "1")
    client = BabbleClient(settings, log, brain=brain)
    channel = FakePretendChannel()
    interaction = _interaction(1, channel)

    _pretend(
        client, interaction, message="yes",
        reply_to="https://discord.com/channels/111/222/1234567890123456789",
    )

    assert channel.fetched == [1234567890123456789]
    assert channel.posts == [("reply", False, {"content": "yes", "files": []})]


def test_pretend_needs_something_to_say(settings, log, brain, monkeypatch):
    monkeypatch.setenv("BABBLE_OWNER_IDS", "1")
    client = BabbleClient(settings, log, brain=brain)
    channel = FakePretendChannel()
    interaction = _interaction(1, channel)

    _pretend(client, interaction)

    assert channel.posts == []
    assert interaction.sink[0][0] == "response"


def test_pretend_reports_a_forbidden_channel(settings, log, brain, read_log, monkeypatch):
    monkeypatch.setenv("BABBLE_OWNER_IDS", "1")
    client = BabbleClient(settings, log, brain=brain)
    channel = FakePretendChannel(send_exc=_http_exception(discord.Forbidden, 403, "Forbidden"))
    interaction = _interaction(1, channel)

    _pretend(client, interaction, message="hi")

    assert interaction.sink[-1] == ("followup", "Discord refused that post (forbidden).", True)
    (event,) = read_log("bot.error")
    assert event["where"] == "pretend" and event["reason"] == "forbidden"
