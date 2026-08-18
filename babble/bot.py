"""The discord.py adapter -- the only file that imports discord.

It does three things and nothing else: turn gateway events into the plain
dataclasses `core` understands, post whatever `core` hands back, and tell `core`
the id of each message it posted so corrections can find their target.

All of the actual behaviour lives in `core.py`, which is why the bot's logic is
tested without a token.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import discord

from .backfill import backfill_corpus
from .config import TOKEN_ENV, Settings, discord_token
from .core import Babble, IncomingMessage, ReactionEvent
from .discord_feed import CollectionFeed
from .generate import CheckpointGenerator
from .identity import Pseudonymiser
from .logs import EventLog
from .publish import GrowthPublisher
from .trainer import AutoTrainTrigger

#: How often the background task drains the collection feed's coalescing buffer.
#: Shorter than the feed's window, so the last row of a trickle still posts a
#: second or two after it lands rather than waiting for the next message.
FEED_FLUSH_SECONDS = 1.0


class BabbleClient(discord.Client):
    def __init__(self, settings: Settings, log: EventLog, brain: Babble | None = None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # privileged: enable it in the dev portal
        super().__init__(
            intents=intents,
            # The model emits random bytes. It must never be able to ping anyone.
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.log = log
        # The collection feed reports every corpus change to the same channel the
        # training feed uses; the publisher pushes the dataset when the corpus has
        # grown. Both are wired into the brain here, unless a brain was handed in
        # (tests), so the real bot always reports collection while no trainer runs.
        self.feed: CollectionFeed | None = None
        if brain is None:
            self.feed = CollectionFeed.from_env(log)
            publisher = GrowthPublisher(
                settings,
                log,
                feed=self.feed,
                every_rows=settings.hf_publish_every_rows,
                every_chars=settings.hf_publish_every_chars,
            )
            # The training trigger: after every N new corpus rows, a fresh
            # pretrain run fires and writes a new latest.pt, which the
            # CheckpointGenerator hot-reloads.
            train_trigger = AutoTrainTrigger(settings, log)
            brain = Babble(
                settings,
                generator=CheckpointGenerator(settings, log),
                log=log,
                feed=self.feed,
                publisher=publisher,
                train_trigger=train_trigger,
            )
            # Seed milestone markers from the corpus already on disk, so a
            # restart does not re-announce the last milestone on the next row.
            if self.feed.enabled:
                totals = brain.corpus.totals()
                self.feed.prime(rows=totals.rows, chars=totals.chars)
        self.brain = brain
        # Generation and file writes are serialised: one brain, one thread at a time.
        self._lock = asyncio.Lock()

    async def setup_hook(self) -> None:
        # Drain the feed's coalescing buffer on a timer, so a lone row at the end
        # of a trickle still posts rather than waiting for the next capture. The
        # flush is best-effort and off the event loop, exactly like every post.
        if self.feed is not None and self.feed.enabled:
            self.loop.create_task(self._flush_feed_loop())

    async def _flush_feed_loop(self) -> None:
        while True:
            await asyncio.sleep(FEED_FLUSH_SECONDS)
            try:
                await asyncio.to_thread(self.feed.flush_due)
            except Exception as exc:  # a flusher hiccup must never take the bot down
                self.log.event("bot.error", where="feed_flush", error=f"{type(exc).__name__}: {exc}")

    async def on_ready(self) -> None:
        self.brain.bot_user_id = str(self.user.id) if self.user else None
        self.log.event(
            "bot.ready",
            bot=str(self.user),
            guilds=len(self.guilds),
            latency_ms=round(self.latency * 1000, 1),
            step=getattr(self.brain.generator, "step", 0),
        )
        # A shared/multi-guild deployment can have a channel it never sees a
        # single event from because it lacks View Channel there. That gap is
        # invisible from the message log alone, so spell it out here instead
        # of requiring a manual permissions poke.
        for guild in self.guilds:
            total, visible, sendable = _guild_visibility(guild)
            self.log.event(
                "bot.guild",
                guild=self.log.guild(guild.id),
                channels=total,
                visible=visible,
                sendable=sendable,
            )
        print(f"babble is online as {self.user} in {len(self.guilds)} guild(s)", flush=True)

    async def on_message(self, message: discord.Message) -> None:
        if self.user and message.author.id == self.user.id:
            return
        incoming = await self._to_incoming(message)
        replies = await self._think(self.brain.handle_message, incoming)
        for reply in replies:
            sent = await self._send_reply(message, reply, incoming)
            if sent is None:
                continue
            self.brain.remember(sent.id, reply)
            self.log.event(
                "bot.sent",
                kind=reply.kind,
                message_id=str(sent.id),
                chars=len(reply.content),
                remembered=reply.exchange is not None or None,
            )

    async def _send_reply(
        self, message: discord.Message, reply: Any, incoming: IncomingMessage
    ) -> discord.Message | None:
        """Post one reply, or log why it failed and return None instead of raising."""
        try:
            return await message.reply(reply.content, mention_author=False)
        except discord.HTTPException as exc:
            reason = "forbidden" if isinstance(exc, discord.Forbidden) else "http_error"
            self.log.event(
                "bot.error",
                where="reply",
                reason=reason,
                channel=self.log.channel(incoming.channel_id),
                guild=self.log.guild(incoming.guild_id),
                error=f"{type(exc).__name__}: {exc}",
            )
            return None

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.user and payload.user_id == self.user.id:
            return
        event = ReactionEvent(
            message_id=str(payload.message_id),
            emoji=str(payload.emoji),
            user_id=str(payload.user_id),
            channel_id=str(payload.channel_id),
            guild_id=str(payload.guild_id) if payload.guild_id else None,
            user_is_bot=bool(payload.member.bot) if payload.member else False,
        )
        await self._think(self.brain.handle_reaction, event)

    # --- plumbing -------------------------------------------------------

    async def _to_incoming(self, message: discord.Message) -> IncomingMessage:
        reply_to_id: str | None = None
        reply_to_is_bot = False
        ref = message.reference
        if ref is not None and ref.message_id:
            reply_to_id = str(ref.message_id)
            reply_to_is_bot = await self._is_ours(message, ref)

        return IncomingMessage(
            message_id=str(message.id),
            author_id=str(message.author.id),
            content=message.content or "",
            channel_id=str(getattr(message.channel, "id", 0)),
            guild_id=str(message.guild.id) if message.guild else None,
            author_is_bot=bool(message.author.bot),
            mentions_bot=_mentions_bot(self.user, message),
            reply_to_message_id=reply_to_id,
            reply_to_is_bot=reply_to_is_bot,
            attachment_urls=tuple(a.url for a in message.attachments),
            # A one-to-one DM is by definition addressed to us: there is nobody
            # else in it to address. A *group* DM is not -- it has other people
            # in it, and `message.guild is None` is true there too, which would
            # have made every member's every message collectable without a
            # mention, a reply or a widening. `DMChannel` is the 1:1 one.
            is_dm=isinstance(message.channel, discord.DMChannel),
        )

    async def _is_ours(self, message: discord.Message, ref: discord.MessageReference) -> bool:
        """Did we write the message being replied to? Cheapest check first."""
        resolved = ref.resolved
        if isinstance(resolved, discord.Message):
            return bool(self.user and resolved.author.id == self.user.id)
        # We remember every generation we posted, so this settles it without a call.
        if self.brain.exchanges.get(str(ref.message_id)) is not None:
            return True
        try:
            fetched = await message.channel.fetch_message(ref.message_id)
        except (discord.HTTPException, AttributeError):
            return False
        return bool(self.user and fetched.author.id == self.user.id)

    async def _think(self, handler: Callable[[Any], list], payload: Any) -> list:
        """Run core logic off the event loop; it does CPU work and file IO."""
        async with self._lock:
            try:
                return await asyncio.to_thread(handler, payload)
            except Exception as exc:  # a bad message must never kill the gateway
                self.log.event(
                    "bot.error",
                    where=getattr(handler, "__name__", "handler"),
                    error=f"{type(exc).__name__}: {exc}",
                )
                return []


def _mentions_bot(user: discord.abc.User | None, message: discord.Message) -> bool:
    """A direct @mention, or a mention of a role the bot holds.

    `message.mentions` never includes role pings, so a server that pings the
    bot's role instead of mentioning it directly was being silently ignored.
    The guild's default (@everyone) role is deliberately excluded from the
    role check -- @everyone and @here must never trigger a response.
    """
    if user and user in message.mentions:
        return True
    if not message.role_mentions:
        return False
    guild = message.guild
    me = guild.me if guild else None
    if me is None:
        return False
    bot_role_ids = {role.id for role in me.roles if not role.is_default()}
    return any(role.id in bot_role_ids for role in message.role_mentions)


def _guild_visibility(guild: discord.Guild) -> tuple[int, int, int]:
    """(text channels, how many we can see, how many we can also send in)."""
    me = guild.me
    total = visible = sendable = 0
    for channel in guild.text_channels:
        total += 1
        perms = channel.permissions_for(me) if me else discord.Permissions.none()
        if perms.view_channel:
            visible += 1
            if perms.send_messages:
                sendable += 1
    return total, visible, sendable


def run_bot(settings: Settings | None = None) -> int:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    log = EventLog(settings, Pseudonymiser.load(settings), component="bot")

    token = discord_token()
    if not token:
        log.event("bot.error", reason="missing_token", env=TOKEN_ENV)
        print(
            f"No Discord token. Set {TOKEN_ENV} in your environment or .env file.\n"
            "Everything except the gateway works without it:\n"
            "  babble fake-data && babble train --force --steps 50\n"
            "  babble sample --prompt hello\n"
            "  babble export",
            flush=True,
        )
        log.close()
        return 2

    # The bot shares the box with the trainer; generation gets one thread.
    import torch

    torch.set_num_threads(1)

    log.event("bot.start", data_dir=str(settings.data_dir), checkpoints=str(settings.checkpoint_dir))
    # An install that predates the corpus has all of its text sitting in
    # interactions.jsonl. Migrate it here too, not just in the trainer, so
    # `!babble status` tells the truth on a box where the bot came up first.
    backfill_corpus(settings, log=log, log_noop=False)
    client = BabbleClient(settings, log)
    try:
        client.run(token, log_handler=None)
    except discord.LoginFailure:
        log.event("bot.error", reason="login_failed")
        print("Discord rejected that token.", flush=True)
        return 2
    except KeyboardInterrupt:
        log.event("bot.stop", reason="keyboard_interrupt")
        return 0
    finally:
        log.event("bot.stop", reason="exit")
        log.close()
    return 0
