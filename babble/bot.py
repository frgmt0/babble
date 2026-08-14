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

from .config import TOKEN_ENV, Settings, discord_token
from .core import Babble, IncomingMessage, ReactionEvent
from .generate import CheckpointGenerator
from .identity import Pseudonymiser
from .logs import EventLog


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
        self.brain = brain or Babble(
            settings, generator=CheckpointGenerator(settings, log), log=log
        )
        # Generation and file writes are serialised: one brain, one thread at a time.
        self._lock = asyncio.Lock()

    async def on_ready(self) -> None:
        self.brain.bot_user_id = str(self.user.id) if self.user else None
        self.log.event(
            "bot.ready",
            bot=str(self.user),
            guilds=len(self.guilds),
            latency_ms=round(self.latency * 1000, 1),
            step=getattr(self.brain.generator, "step", 0),
        )
        print(f"babble is online as {self.user} in {len(self.guilds)} guild(s)", flush=True)

    async def on_message(self, message: discord.Message) -> None:
        if self.user and message.author.id == self.user.id:
            return
        incoming = await self._to_incoming(message)
        replies = await self._think(self.brain.handle_message, incoming)
        for reply in replies:
            try:
                sent = await message.reply(reply.content, mention_author=False)
            except discord.HTTPException as exc:
                self.log.event("bot.error", where="reply", error=f"{type(exc).__name__}: {exc}")
                continue
            self.brain.remember(sent.id, reply)
            self.log.event(
                "bot.sent",
                kind=reply.kind,
                message_id=str(sent.id),
                chars=len(reply.content),
                remembered=reply.exchange is not None or None,
            )

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.user and payload.user_id == self.user.id:
            return
        event = ReactionEvent(
            message_id=str(payload.message_id),
            emoji=str(payload.emoji),
            user_id=str(payload.user_id),
            channel_id=str(payload.channel_id),
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
            author_is_bot=bool(message.author.bot),
            mentions_bot=bool(self.user and self.user in message.mentions),
            reply_to_message_id=reply_to_id,
            reply_to_is_bot=reply_to_is_bot,
            attachment_urls=tuple(a.url for a in message.attachments),
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
            "  babble fake-data && babble train --steps 50\n"
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
