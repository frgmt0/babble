"""All of the bot's behaviour, with none of Discord in it.

`Babble` takes plain dataclasses in and returns `Reply` objects out. It never
sends anything and never awaits anything, which is why the whole feedback loop --
consent gate, correction capture, thumbs-up, purge -- is testable without a
token, a gateway connection or an event loop.

`bot.py` is the only file that knows discord.py exists; it translates events into
these dataclasses and posts the replies that come back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .blocklist import Blocklist, row_fingerprint
from .config import Settings
from .consent import (
    CAPTURE_OK,
    DECLINED,
    GRANTED,
    PENDING,
    UNKNOWN,
    WITHDRAWN,
    ConsentStore,
)
from .exchanges import Exchange, ExchangeLog
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .store import APPROVAL, CORRECTION, Interaction, InteractionStore, make_row_id
from .util import truncate, utcnow_iso

BLOCKED_OUTPUT = "*…(that one didn't clear the content filter — regenerate by pinging me again)*"

DISCORD_LIMIT = 2000
COMMAND_PREFIX = "!babble"
THUMBS_UP = "👍"

# --- copy ----------------------------------------------------------------

FOOTER = (
    "-# like this response? react with 👍 — if not, correct me with a reply. "
    "replies teach me a lot more than reactions do."
)

FOOTER_UNCONSENTED = (
    "-# you haven't opted in, so nothing from this exchange is stored or trained on. "
    "`!babble consent` for the details."
)

CONSENT_NOTICE = """**first time we've talked — here's the deal before we start.**

i'm a language model with **random weights**. i wasn't trained on the internet, on this server's \
history, or on anything at all. the only way i ever learn is from people correcting me here.

if you opt in, this gets **stored, used to train me, and published to a public HuggingFace \
dataset** that anyone can download:
· the messages you send me
· what i answered
· your corrections and 👍 reactions

what never leaves this machine: your discord id, your username, anything you say that isn't \
addressed to me. in the published data you are a salted hash like `u_9f2c…`.

**nothing of yours is stored until you say yes.**

**`!babble accept`** — opt in
**`!babble decline`** — no thanks. i'll still reply, i just won't keep anything
**`!babble forget`** — any time later: opt out *and* delete everything of yours i've kept"""

HELP_TEXT = """**babble** — a from-scratch model that only learns from you.

ping me and i'll answer. reply to my answer to correct it, or react 👍 if it was fine.
corrections are worth far more than reactions — they're the only strong signal i get.

`!babble consent` — what i store, and your current choice
`!babble accept` / `!babble decline` — opt in or out
`!babble forget` — opt out and delete everything of yours
`!babble status` — how training is going"""


@dataclass(frozen=True)
class IncomingMessage:
    """A Discord message, reduced to what the logic actually needs."""

    message_id: str
    author_id: str
    content: str = ""
    channel_id: str = "0"
    author_is_bot: bool = False
    mentions_bot: bool = False
    reply_to_message_id: str | None = None
    reply_to_is_bot: bool = False
    attachment_urls: Sequence[str] = ()


@dataclass(frozen=True)
class ReactionEvent:
    message_id: str
    emoji: str
    user_id: str
    channel_id: str = "0"
    user_is_bot: bool = False


@dataclass(frozen=True)
class Generation:
    """What a generator hands back: the text plus how it was produced."""

    text: str
    step: int = 0
    temperature: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 0
    ms: float = 0.0


@dataclass(frozen=True)
class Reply:
    """Something for the adapter to send.

    If `exchange` is set the adapter must call `Babble.remember()` with the id of
    the message it just posted, so a later correction can find its way back.
    """

    content: str
    reply_to: str | None = None
    kind: str = "message"
    exchange: Exchange | None = None


# --- text hygiene --------------------------------------------------------

_MENTION_USER = re.compile(r"<@!?\d+>")
_MENTION_ROLE = re.compile(r"<@&\d+>")
_MENTION_CHANNEL = re.compile(r"<#\d+>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def scrub_mentions(text: str) -> str:
    """Strip Discord mention markup, which embeds raw snowflake ids in content.

    This is the reason a stored prompt can never smuggle a user id into the
    public dataset through the message body.
    """
    text = _MENTION_ROLE.sub("@role", text)
    text = _MENTION_USER.sub("@user", text)
    return _MENTION_CHANNEL.sub("#channel", text)


def clean_for_discord(text: str, limit: int = DISCORD_LIMIT) -> str:
    """Make raw model output safe to post without changing what it says."""
    cleaned = _CONTROL.sub("", text).strip()
    if not cleaned:
        return "*…(nothing printable that time — that's what noise looks like)*"
    return truncate(cleaned, limit)


def compose(body: str, footer: str) -> str:
    return f"{clean_for_discord(body, DISCORD_LIMIT - len(footer) - 1)}\n{footer}"


def _as_generation(result: Generation | str) -> Generation:
    return result if isinstance(result, Generation) else Generation(text=str(result))


# --- the bot -------------------------------------------------------------

class Babble:
    def __init__(
        self,
        settings: Settings,
        *,
        generator: Callable[[str], Generation | str],
        consent: ConsentStore | None = None,
        store: InteractionStore | None = None,
        exchanges: ExchangeLog | None = None,
        ids: Pseudonymiser | None = None,
        log: EventLog | None = None,
        blocklist: Blocklist | None = None,
        bot_user_id: str | None = None,
    ) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.generator = generator
        self.ids = ids or Pseudonymiser.load(settings)
        self.consent = consent or ConsentStore(settings.consent_path)
        self.store = store or InteractionStore(settings.interactions_path)
        self.exchanges = exchanges or ExchangeLog(settings.exchanges_path)
        self.log = log or NullLog()
        self.blocklist = blocklist if blocklist is not None else Blocklist.load()
        self.bot_user_id = str(bot_user_id) if bot_user_id else None

    # --- entry points ---------------------------------------------------

    def handle_message(self, msg: IncomingMessage) -> list[Reply]:
        if msg.author_is_bot:
            return []

        command = _parse_command(msg.content)
        if command is not None:
            return self._handle_command(msg, *command)

        if not (msg.mentions_bot or msg.reply_to_is_bot):
            return []

        self.log.event(
            "bot.ping",
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
            is_reply=msg.reply_to_is_bot,
            chars=len(msg.content),
            attachments=len(msg.attachment_urls) or None,
        )

        # A reply to something we said and still remember is a correction.
        if msg.reply_to_is_bot and msg.reply_to_message_id:
            exchange = self.exchanges.get(msg.reply_to_message_id)
            if exchange is not None:
                return self._handle_correction(msg, exchange)

        return self._respond(msg)

    def handle_reaction(self, evt: ReactionEvent) -> list[Reply]:
        if evt.user_is_bot or _normalise_emoji(evt.emoji) != THUMBS_UP:
            return []

        exchange = self.exchanges.get(evt.message_id)
        if exchange is None:
            # A 👍 on some older or unremembered message. Nothing to attach it to.
            self.log.event(
                "reaction.ignored",
                user=self.log.user(evt.user_id),
                reason="no_matching_exchange",
            )
            return []

        if not self.consent.may_capture(evt.user_id, exchange.prompt_author_id):
            self._log_skip(APPROVAL, evt.user_id, exchange.prompt_author_id)
            return []

        if self.blocklist.matches(exchange.prompt, exchange.response):
            self._log_blocked("approval", exchange.prompt, exchange.response, evt.user_id)
            return []

        row = Interaction(
            id=make_row_id(
                APPROVAL,
                exchange.prompt,
                exchange.response,
                self.ids.user(exchange.prompt_author_id),
                self.ids.user(evt.user_id),
            ),
            signal=APPROVAL,
            prompt=exchange.prompt,
            rejected=None,
            chosen=exchange.response,
            prompt_author=self.ids.user(exchange.prompt_author_id),
            signal_author=self.ids.user(evt.user_id),
            weight=self.settings.approval_weight,
            created_at=utcnow_iso(),
        )
        fresh = self.store.append(row)
        self.log.event(
            "capture.approval",
            row=row.id,
            user=self.log.user(evt.user_id),
            weight=row.weight,
            duplicate=None if fresh else True,
            chars=len(row.chosen),
            total_rows=self.store.count(),
        )
        return []

    def remember(self, bot_message_id: object, reply: Reply) -> None:
        """Called by the adapter once it knows the id of the message it sent."""
        if reply.exchange is not None:
            self.exchanges.record(bot_message_id, reply.exchange)

    # --- behaviour ------------------------------------------------------

    def _respond(self, msg: IncomingMessage) -> list[Reply]:
        decision = self.consent.decision(msg.author_id)

        # First contact is the consent moment. No generation, no storage.
        if decision == UNKNOWN:
            self.consent.mark_prompted(msg.author_id)
            self.log.event(
                "consent.prompt",
                user=self.log.user(msg.author_id),
                channel=self.log.channel(msg.channel_id),
                trigger="first_ping",
            )
            return [Reply(CONSENT_NOTICE, reply_to=msg.message_id, kind="consent")]

        allowed = decision in CAPTURE_OK
        prompt = self._message_text(msg)
        generation = _as_generation(self.generator(prompt))
        body = clean_for_discord(generation.text, DISCORD_LIMIT - len(FOOTER) - 1)

        # The model can emit anything, including a blocked term. Catch it here,
        # before it is ever sent, not after -- this is the one send site every
        # generation passes through.
        blocked = self.blocklist.matches(body)
        if blocked:
            self._log_blocked("generate", prompt, body, msg.author_id)
            body = BLOCKED_OUTPUT

        preview = self.log.preview(prompt, allowed=allowed)
        self.log.event(
            "bot.generate",
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
            step=generation.step,
            temperature=generation.temperature,
            top_k=generation.top_k,
            max_new_tokens=generation.max_new_tokens,
            ms=round(generation.ms, 1),
            prompt_chars=preview["chars"],
            prompt=preview.get("text"),
            out_chars=len(body),
            output=truncate(body.replace("\n", "\\n"), self.settings.log_preview_chars),
            capturable=allowed,
        )

        # Only remember the exchange if a correction to it could ever be stored.
        # A blocked generation is never remembered either -- there is nothing
        # here worth teaching the model to reproduce or correct.
        exchange = (
            Exchange(
                prompt=prompt,
                response=body,
                prompt_author_id=str(msg.author_id),
                created_at=utcnow_iso(),
                step=generation.step,
            )
            if allowed and not blocked
            else None
        )
        footer = FOOTER if allowed else FOOTER_UNCONSENTED
        return [
            Reply(
                f"{body}\n{footer}",
                reply_to=msg.message_id,
                kind="generation",
                exchange=exchange,
            )
        ]

    def _handle_correction(self, msg: IncomingMessage, exchange: Exchange) -> list[Reply]:
        corrector = msg.author_id

        if self.consent.decision(corrector) == UNKNOWN:
            self.consent.mark_prompted(corrector)
            self.log.event(
                "consent.prompt",
                user=self.log.user(corrector),
                channel=self.log.channel(msg.channel_id),
                trigger="first_correction",
            )
            return [Reply(CONSENT_NOTICE, reply_to=msg.message_id, kind="consent")]

        if not self.consent.may_capture(corrector, exchange.prompt_author_id):
            self._log_skip(CORRECTION, corrector, exchange.prompt_author_id)
            return [
                Reply(
                    "-# noted, but not stored — someone in this thread hasn't opted in, "
                    "so i can't learn from it. `!babble consent`",
                    reply_to=msg.message_id,
                    kind="ack",
                )
            ]

        correction = self._message_text(msg)
        if not correction:
            return [
                Reply(
                    "-# that reply was empty as far as i can tell, so there's nothing to learn.",
                    reply_to=msg.message_id,
                    kind="ack",
                )
            ]

        # The old, rejected answer is checked too: the blocklist can be extended
        # after a generation was sent and remembered, and that generation is
        # about to be published in this row's `rejected` field.
        if self.blocklist.matches(exchange.prompt, exchange.response, correction):
            self._log_blocked("correction", exchange.prompt, correction, corrector)
            return [
                Reply(
                    "-# that correction wasn't accepted — it matched the content filter.",
                    reply_to=msg.message_id,
                    kind="ack",
                )
            ]

        row = Interaction(
            id=make_row_id(
                CORRECTION,
                exchange.prompt,
                correction,
                self.ids.user(exchange.prompt_author_id),
                self.ids.user(corrector),
            ),
            signal=CORRECTION,
            prompt=exchange.prompt,
            rejected=exchange.response,
            chosen=correction,
            prompt_author=self.ids.user(exchange.prompt_author_id),
            signal_author=self.ids.user(corrector),
            weight=self.settings.correction_weight,
            created_at=utcnow_iso(),
        )
        fresh = self.store.append(row)
        total = self.store.count()
        self.log.event(
            "capture.correction",
            row=row.id,
            user=self.log.user(corrector),
            prompt_author=self.log.user(exchange.prompt_author_id),
            weight=row.weight,
            duplicate=None if fresh else True,
            prompt_chars=len(row.prompt),
            rejected_chars=len(row.rejected or ""),
            chosen_chars=len(correction),
            chosen=truncate(correction.replace("\n", "\\n"), self.settings.log_preview_chars),
            step=exchange.step,
            total_rows=total,
        )
        return [
            Reply(
                f"-# got it — correction #{total} filed. i'll pick it up on the next training cycle.",
                reply_to=msg.message_id,
                kind="ack",
            )
        ]

    # --- commands -------------------------------------------------------

    def _handle_command(self, msg: IncomingMessage, verb: str, args: list[str]) -> list[Reply]:
        self.log.event(
            "command",
            verb=verb or "help",
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
        )
        reply = lambda text: [Reply(text, reply_to=msg.message_id, kind="command")]  # noqa: E731

        if verb in ("accept", "yes", "agree", "optin", "opt-in"):
            self.consent.grant(msg.author_id)
            self.log.event("consent.accept", user=self.log.user(msg.author_id))
            return reply(
                "you're in — thank you. from now on our exchanges and your corrections go into "
                "the training set and the public dataset.\n"
                "-# change your mind whenever with `!babble forget`, which also deletes what i've kept."
            )

        if verb in ("decline", "no", "nope", "optout", "opt-out"):
            self.consent.decline(msg.author_id)
            self.log.event("consent.decline", user=self.log.user(msg.author_id))
            return reply(
                "understood — nothing of yours will be stored, trained on, or published.\n"
                "-# i'll still reply if you ping me. `!babble accept` if you ever change your mind."
            )

        if verb in ("forget", "delete", "withdraw", "erase"):
            author = self.ids.user(msg.author_id)
            purged = self.store.purge_author(author)
            dropped = self.exchanges.forget_author(msg.author_id)
            self.consent.withdraw(msg.author_id)
            self.log.event(
                "consent.withdraw",
                user=author,
                rows_purged=purged,
                pending_dropped=dropped,
            )
            return reply(
                f"done. consent withdrawn and **{purged}** stored "
                f"{'row' if purged == 1 else 'rows'} of yours deleted.\n"
                "-# anything already published to HuggingFace in a previous export won't vanish "
                "from people's downloads, but it's gone from here and from every export after this."
            )

        if verb in ("consent", "privacy", "data"):
            state = self.consent.decision(msg.author_id)
            return reply(f"{CONSENT_NOTICE}\n\n-# your current setting: **{_describe(state)}**")

        if verb in ("status", "stats"):
            from .stats import render_snapshot, snapshot  # local: keeps core import cheap

            state = self.consent.decision(msg.author_id)
            return reply(
                f"{render_snapshot(snapshot(self.settings))}\n"
                f"-# you: **{_describe(state)}**"
            )

        return reply(HELP_TEXT)

    # --- helpers --------------------------------------------------------

    def _message_text(self, msg: IncomingMessage) -> str:
        """The user's words, minus our own @mention, minus any raw ids.

        Attachment urls are folded in so that correcting the bot by dropping a gif
        in works exactly like correcting it with a tenor link.
        """
        text = msg.content
        if self.bot_user_id:
            text = re.sub(rf"<@!?{re.escape(self.bot_user_id)}>", " ", text)
        text = scrub_mentions(text).strip()
        if msg.attachment_urls:
            text = " ".join(filter(None, [text, *msg.attachment_urls])).strip()
        return re.sub(r"[ \t]{2,}", " ", text)

    def _log_skip(self, signal: str, signal_author: object, prompt_author: object) -> None:
        """Record that we threw data away, and why. Never what the data was."""
        blockers = {
            "signal_author": self.consent.decision(signal_author),
            "prompt_author": self.consent.decision(prompt_author),
        }
        missing = [role for role, state in blockers.items() if state not in CAPTURE_OK]
        self.log.event(
            "capture.skipped",
            signal=signal,
            reason="no_consent",
            missing=",".join(missing),
            user=self.log.user(signal_author),
            signal_author_state=blockers["signal_author"],
            prompt_author_state=blockers["prompt_author"],
        )

    def _log_blocked(self, stage: str, prompt: str, chosen: str, user_id: object) -> None:
        """Record a blocklist rejection: the reason and a fingerprint, never the text."""
        self.log.event(
            "capture.blocked",
            stage=stage,
            user=self.log.user(user_id),
            row=row_fingerprint(prompt, chosen),
        )


def _parse_command(content: str) -> tuple[str, list[str]] | None:
    """`!babble [verb] [args]`, or None if this is an ordinary message."""
    stripped = content.strip()
    if not stripped.lower().startswith(COMMAND_PREFIX):
        return None
    rest = stripped[len(COMMAND_PREFIX) :]
    if rest and not rest[0].isspace():
        return None  # "!babbles about nothing" is a message, not a command
    parts = rest.split()
    verb = parts[0].lower() if parts else "help"
    return verb, parts[1:]


def _normalise_emoji(emoji: str) -> str:
    """Discord sends 👍 with and without the U+FE0F variation selector."""
    return emoji.replace("️", "").strip()


def _describe(state: str) -> str:
    return {
        GRANTED: "opted in",
        DECLINED: "opted out",
        WITHDRAWN: "withdrawn",
        PENDING: "not answered yet",
        UNKNOWN: "never asked",
    }.get(state, state)
