"""All of the bot's behaviour, with none of Discord in it.

`Babble` takes plain dataclasses in and returns `Reply` objects out. It never
sends anything and never awaits anything, which is why the whole feedback loop --
consent gate, corpus capture, correction capture, thumbs-up, purge -- is testable
without a token, a gateway connection or an event loop.

`bot.py` is the only file that knows discord.py exists; it translates events into
these dataclasses and posts the replies that come back.

There are two things being collected here and they are not the same thing:

* the **corpus** (`corpus.py`) -- the plain text of what people send the bot.
  This is what the model trains on. Collecting it needs a `corpus` grant.
* **corrections** (`store.py`) -- the `(prompt, rejected, chosen)` triples. No
  longer a training objective, still captured and still published, still under
  the older, narrower `corrections` grant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .blocklist import Blocklist, row_fingerprint
from .config import CORRECTION_MARKER, Settings
from .consent import (
    CAPTURE_OK,
    DECLINED,
    GRANTED,
    PENDING,
    SCOPE_CORPUS,
    SCOPE_CORRECTIONS,
    UNKNOWN,
    WITHDRAWN,
    ConsentStore,
)
from .corpus import (
    SOURCE_AMBIENT,
    SOURCE_CORRECTION,
    SOURCE_DM,
    SOURCE_MENTION,
    SOURCE_PROMPT,
    SOURCE_REPLY,
    CorpusRow,
    CorpusStore,
    make_corpus_id,
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
    f"-# what you send me goes into the corpus i learn from. react 👍 if this one was fine, or "
    f"teach me something specific by replying with `{CORRECTION_MARKER} what i should have said`."
)

FOOTER_UNCONSENTED = (
    "-# you haven't opted in, so nothing from this exchange is stored or trained on. "
    "`!babble consent` for the details."
)

# Shown to someone who opted in under the old corrections-only notice. They are
# still a full participant for corrections; their ordinary messages are simply
# not being collected until they answer the new one.
FOOTER_CORPUS_PENDING = (
    "-# i keep more than corrections now, and i'm not keeping yours until you say so. "
    "`!babble consent`"
)

PRIVACY_URL = "https://booper.frgmt.xyz/privacy"
TERMS_URL = "https://booper.frgmt.xyz/terms"
AUP_URL = "https://booper.frgmt.xyz/aup"

# The <> around each URL suppress Discord's embed previews.
CONSENT_NOTICE = f"""please take the time to read:
<{PRIVACY_URL}> · <{TERMS_URL}> · <{AUP_URL}>
as they outline what data we do/don't collect. to proceed:

**`!babble accept`** — opt in
**`!babble decline`** — no thanks (i'll still reply, nothing is stored)"""

# The re-ask. Anyone who said yes to the old notice said yes to something
# narrower than this, so their existing corrections stand and their ordinary
# messages wait here until they answer again.
CORPUS_NOTICE = f"""**heads up — what i collect has changed.**

you opted in back when i only kept corrections. i now keep the messages you send me, to train on \
and publish in a public dataset. that's more than you agreed to, so nothing new of yours is \
collected until you say yes again. your existing corrections stay as they are.

**`!babble accept`** — opt in to this
**`!babble decline`** — no thanks
**`!babble forget`** — opt out *and* delete everything of yours

-# full details: <{PRIVACY_URL}>"""

HELP_TEXT = f"""**babble** — a from-scratch model learning to talk from what you send it.

ping me and i'll carry on from whatever you said. it will be nonsense for a long time.

everything you send me goes into an unlabelled corpus, and that corpus is the whole
of my training data. there's no right answer attached to any of it.

**to teach me something specific, start your reply with `{CORRECTION_MARKER}`:**
> `{CORRECTION_MARKER} hey, what's up`

that marker is how i tell teaching apart from talking. the marker is stripped before
anything is stored. corrections are filed separately *and* go into the corpus.

`!babble consent` — what i store, and your current choice
`!babble accept` / `!babble decline` — opt in or out
`!babble all` — in this channel, collect everything you say, not just pings at me
`!babble pings` — undo that here
`!babble forget` — opt out and delete everything of yours
`!babble status` — how training is going"""


class _CollectionFeed(Protocol):
    """Just the collection-feed surface `core` drives. Kept structural so the
    brain stays testable with a fake and never has to import the webhook code."""

    def row(self, *, text: str, source: str, author: str, rows: int, chars: int, contributors: int) -> None: ...
    def consent_granted(self, *, author: str) -> None: ...
    def consent_declined(self, *, author: str) -> None: ...
    def channel_widened(self, *, author: str, channel: str) -> None: ...
    def channel_narrowed(self, *, author: str, channel: str) -> None: ...
    def consent_withdrawn(self, *, author: str, corpus_purged: int, correction_purged: int) -> None: ...


class _Publisher(Protocol):
    """The one method `core` calls after a fresh row: publish if the corpus has
    grown enough. Everything else about publishing is the publisher's business."""

    def maybe_publish(self) -> None: ...


class _TrainTrigger(Protocol):
    """The one method `core` calls after a fresh row: run a training pass if
    the corpus has grown by enough rows since the last one. A trigger, not a
    loop -- the trigger's own business, kept structural so the brain never
    imports torch."""

    def maybe_run(self) -> None: ...


class _PostTrigger(Protocol):
    """The one method `core` calls after a fresh correction pair: run a
    post-train if the pairs have grown by enough since the last one. Same
    shape as `_TrainTrigger`, kept separate because it fires off a different
    count (pairs, not corpus rows)."""

    def maybe_run(self) -> None: ...


class _AugmentTrigger(Protocol):
    """The one method `core` calls after a fresh correction pair: paraphrase
    THIS pair into extra post-train variants, out of band. Unlike
    `_PostTrigger` this is not threshold-based -- every new correction fires
    it (subject to its own on/off knob), so the pair set compounds as
    corrections keep arriving rather than waiting for a batch."""

    def on_new_pair(self, pair_id: str) -> None: ...


@dataclass(frozen=True)
class IncomingMessage:
    """A Discord message, reduced to what the logic actually needs."""

    message_id: str
    author_id: str
    content: str = ""
    channel_id: str = "0"
    guild_id: str | None = None
    author_is_bot: bool = False
    mentions_bot: bool = False
    reply_to_message_id: str | None = None
    reply_to_is_bot: bool = False
    attachment_urls: Sequence[str] = ()
    #: A direct message. Explicit rather than inferred from a missing guild id,
    #: so a message that simply forgot to say which guild it came from is never
    #: silently promoted to "addressed to the bot".
    is_dm: bool = False

    def addressed_to_bot(self) -> bool:
        """Did this person mean this message for the bot?

        The three ways to talk to it, and the boundary of the default grant:
        everything else someone says is theirs and is not collected.
        """
        return bool(self.mentions_bot or self.reply_to_is_bot or self.is_dm)

    def source(self) -> str:
        """Which of those three ways it was, for the corpus row's provenance."""
        if self.is_dm:
            return SOURCE_DM
        if self.mentions_bot:
            return SOURCE_MENTION
        return SOURCE_REPLY


@dataclass(frozen=True)
class ReactionEvent:
    message_id: str
    emoji: str
    user_id: str
    channel_id: str = "0"
    guild_id: str | None = None
    user_is_bot: bool = False


@dataclass(frozen=True)
class Generation:
    """What a generator hands back: the text plus how it was produced."""

    text: str
    step: int = 0
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
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


def is_correction(text: str) -> bool:
    """Does this reply claim to be teaching the bot something?"""
    return text.lstrip().startswith(CORRECTION_MARKER)


def strip_correction_marker(text: str) -> str:
    """The lesson without the marker that flagged it as one.

    One space after the marker is eaten so `>> hey!` and `>>hey!` teach the same
    thing. Anything beyond that first space is the person's own formatting and
    is left alone. The marker itself must never reach the corpus: it is
    addressed to the bot's dispatcher, not part of what it should have said.
    """
    body = text.lstrip()[len(CORRECTION_MARKER) :]
    return (body[1:] if body[:1] == " " else body).strip()


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
        corpus: CorpusStore | None = None,
        exchanges: ExchangeLog | None = None,
        ids: Pseudonymiser | None = None,
        log: EventLog | None = None,
        blocklist: Blocklist | None = None,
        bot_user_id: str | None = None,
        feed: _CollectionFeed | None = None,
        publisher: _Publisher | None = None,
        train_trigger: _TrainTrigger | None = None,
        post_trigger: _PostTrigger | None = None,
        augment_trigger: _AugmentTrigger | None = None,
    ) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.generator = generator
        self.ids = ids or Pseudonymiser.load(settings)
        self.consent = consent or ConsentStore(settings.consent_path)
        self.store = store or InteractionStore(settings.interactions_path)
        self.corpus = corpus or CorpusStore(settings.corpus_path)
        self.exchanges = exchanges or ExchangeLog(settings.exchanges_path)
        self.log = log or NullLog()
        self.blocklist = blocklist if blocklist is not None else Blocklist.load()
        self.bot_user_id = str(bot_user_id) if bot_user_id else None
        # The collection feed and the growth-based publisher. Both optional and
        # both no-ops when absent, so every existing test that builds a Babble
        # without them keeps working and stays token-free and network-free.
        self.feed = feed
        self.publisher = publisher
        self.train_trigger = train_trigger
        self.post_trigger = post_trigger
        self.augment_trigger = augment_trigger

    # --- entry points ---------------------------------------------------

    def handle_message(self, msg: IncomingMessage) -> list[Reply]:
        if msg.author_is_bot:
            self._log_drop(msg, "author_is_bot")
            return []

        command = _parse_command(msg.content)
        if command is not None:
            return self._handle_command(msg, *command)

        if not msg.addressed_to_bot():
            return self._handle_ambient(msg)

        self.log.event(
            "bot.ping",
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
            guild=self.log.guild(msg.guild_id),
            is_reply=msg.reply_to_is_bot,
            chars=len(msg.content),
            attachments=len(msg.attachment_urls) or None,
        )

        # A reply to something we said and still remember is a correction --
        # but only if it is marked as one. An unmarked reply is just someone
        # talking to the bot, and it gets answered like any other message
        # rather than quietly stored as the answer it should have given.
        if msg.reply_to_is_bot and msg.reply_to_message_id:
            exchange = self.exchanges.get(msg.reply_to_message_id)
            if exchange is not None:
                if is_correction(self._message_text(msg)):
                    return self._handle_correction(msg, exchange)
                # Not `bot.dropped`: the message is not dropped, it is answered
                # below like any other. What was declined is treating it as a
                # lesson, and that decision needs its own line in the log.
                self.log.event(
                    "capture.unmarked",
                    user=self.log.user(msg.author_id),
                    channel=self.log.channel(msg.channel_id),
                    guild=self.log.guild(msg.guild_id),
                    chars=len(msg.content),
                )

        return self._respond(msg)

    def _handle_ambient(self, msg: IncomingMessage) -> list[Reply]:
        """A message that wasn't for us. Normally we never see it again.

        The one exception is somebody who ran `!babble all` in this exact
        channel: they asked for everything they type here to be collected, so it
        is -- silently, with no reply, and only for them. Nobody else in the
        channel is affected and nothing is ever said that would prompt someone
        who has not opted in.
        """
        if not self.consent.may_capture_channel(msg.author_id, msg.channel_id):
            self._log_drop(msg, "not_addressed")
            return []
        self._capture_corpus(msg, self._message_text(msg), SOURCE_AMBIENT)
        return []

    def handle_reaction(self, evt: ReactionEvent) -> list[Reply]:
        if evt.user_is_bot or _normalise_emoji(evt.emoji) != THUMBS_UP:
            return []

        exchange = self.exchanges.get(evt.message_id)
        if exchange is None:
            # A 👍 on some older or unremembered message. Nothing to attach it to.
            self.log.event(
                "reaction.ignored",
                user=self.log.user(evt.user_id),
                channel=self.log.channel(evt.channel_id),
                guild=self.log.guild(evt.guild_id),
                reason="no_matching_exchange",
            )
            return []

        if not self.consent.may_capture(
            evt.user_id, exchange.prompt_author_id, scope=SCOPE_CORRECTIONS
        ):
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

    # --- corpus capture --------------------------------------------------

    def _capture_corpus(
        self,
        msg: IncomingMessage,
        text: str,
        source: str,
        *,
        author_id: object | None = None,
    ) -> bool:
        """File one piece of somebody's writing, if everything says we may.

        `author_id` names whose writing it is, defaulting to whoever sent the
        message. It differs only for a correction, where the prompt half belongs
        to the person who was answered, not to the person doing the correcting --
        and each half is gated on its own author's grant, because a corpus row
        has exactly one author and nobody else's consent is relevant to it.
        """
        author_id = msg.author_id if author_id is None else author_id
        text = text.strip()
        if not text:
            return False
        state = self.consent.decision(author_id, SCOPE_CORPUS)
        if state not in CAPTURE_OK:
            self.log.event(
                "capture.skipped",
                signal="corpus",
                scope=SCOPE_CORPUS,
                reason="no_consent",
                source=source,
                user=self.log.user(author_id),
                author_state=state,
            )
            return False
        if self.blocklist.matches(text):
            self._log_blocked("corpus", text, "", author_id)
            return False

        row = CorpusRow(
            id=make_corpus_id(text, self.ids.user(author_id)),
            text=text,
            author=self.ids.user(author_id),
            source=source,
            # Straight off the pseudonymiser, not off the log: a stored row must
            # not depend on which logger happens to be wired up.
            guild=self.ids.guild(msg.guild_id) if msg.guild_id else None,
            channel=self.ids.channel(msg.channel_id),
            created_at=utcnow_iso(),
        )
        fresh = self.corpus.append(row)
        totals = self.corpus.totals()
        self.log.event(
            "capture.corpus",
            row=row.id,
            user=self.log.user(author_id),
            source=source,
            duplicate=None if fresh else True,
            chars=len(text),
            total_rows=totals.rows,
        )
        if fresh:
            # The feed only ever sees the pseudonym and the already-filtered text
            # -- a blocked row never reaches here, it returned above -- so nothing
            # identifying and nothing withheld leaves through this path.
            if self.feed is not None:
                self.feed.row(
                    text=text,
                    source=source,
                    author=row.author,
                    rows=totals.rows,
                    chars=totals.chars,
                    contributors=totals.contributors,
                )
            # A new row may have pushed the corpus past the publish threshold.
            if self.publisher is not None:
                self.publisher.maybe_publish()
            # ...and past the training trigger: every N new rows, a fresh
            # pretrain run fires so the model picks up the newest human writing.
            if self.train_trigger is not None:
                self.train_trigger.maybe_run()
        return fresh

    # --- behaviour ------------------------------------------------------

    def _respond(self, msg: IncomingMessage) -> list[Reply]:
        corrections_state = self.consent.decision(msg.author_id, SCOPE_CORRECTIONS)
        corpus_state = self.consent.decision(msg.author_id, SCOPE_CORPUS)

        # First contact is the consent moment. No generation, no storage.
        if corrections_state == UNKNOWN and corpus_state == UNKNOWN:
            self.consent.mark_prompted(msg.author_id, SCOPE_CORRECTIONS)
            self.consent.mark_prompted(msg.author_id, SCOPE_CORPUS)
            self.log.event(
                "consent.prompt",
                user=self.log.user(msg.author_id),
                channel=self.log.channel(msg.channel_id),
                guild=self.log.guild(msg.guild_id),
                trigger="first_ping",
                scope="all",
            )
            return [Reply(CONSENT_NOTICE, reply_to=msg.message_id, kind="consent")]

        # Somebody who opted in before the corpus existed. They keep every right
        # the old notice gave them; what they have not agreed to is this, so they
        # get asked exactly once and nothing of theirs is collected meanwhile.
        #
        # Only for someone actually *granted* under the old notice: the re-ask
        # opens with "you opted in back when", which is not true of a person who
        # was shown the old notice and never answered it. They keep the standing
        # ask they already have, and `!babble consent` shows them today's terms.
        reask = (
            corrections_state in CAPTURE_OK
            and corpus_state == UNKNOWN
            and self.consent.mark_prompted(msg.author_id, SCOPE_CORPUS)
        )
        if reask:
            self.log.event(
                "consent.prompt",
                user=self.log.user(msg.author_id),
                channel=self.log.channel(msg.channel_id),
                guild=self.log.guild(msg.guild_id),
                trigger="corpus_reask",
                scope=SCOPE_CORPUS,
            )

        allowed = corrections_state in CAPTURE_OK
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
            guild=self.log.guild(msg.guild_id),
            step=generation.step,
            temperature=generation.temperature,
            top_k=generation.top_k,
            top_p=generation.top_p,
            repetition_penalty=generation.repetition_penalty,
            frequency_penalty=generation.frequency_penalty,
            presence_penalty=generation.presence_penalty,
            no_repeat_ngram_size=generation.no_repeat_ngram_size,
            max_new_tokens=generation.max_new_tokens,
            ms=round(generation.ms, 1),
            prompt_chars=preview["chars"],
            prompt=preview.get("text"),
            out_chars=len(body),
            output=truncate(body.replace("\n", "\\n"), self.settings.log_preview_chars),
            capturable=allowed,
        )

        # What they wrote is the corpus. Their own message, under their own
        # pseudonym, gated on their own grant -- the bot's reply is not stored
        # anywhere, because a corpus of what a random model emitted is not a
        # corpus of human writing.
        self._capture_corpus(msg, prompt, msg.source())

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
        if not allowed:
            footer = FOOTER_UNCONSENTED
        elif corpus_state in CAPTURE_OK:
            footer = FOOTER
        else:
            footer = FOOTER_CORPUS_PENDING
        replies = [
            Reply(
                f"{body}\n{footer}",
                reply_to=msg.message_id,
                kind="generation",
                exchange=exchange,
            )
        ]
        if reask:
            replies.append(Reply(CORPUS_NOTICE, reply_to=msg.message_id, kind="consent"))
        return replies

    def _handle_correction(self, msg: IncomingMessage, exchange: Exchange) -> list[Reply]:
        corrector = msg.author_id

        if self.consent.decision(corrector, SCOPE_CORRECTIONS) == UNKNOWN:
            self.consent.mark_prompted(corrector, SCOPE_CORRECTIONS)
            self.consent.mark_prompted(corrector, SCOPE_CORPUS)
            self.log.event(
                "consent.prompt",
                user=self.log.user(corrector),
                channel=self.log.channel(msg.channel_id),
                guild=self.log.guild(msg.guild_id),
                trigger="first_correction",
                scope="all",
            )
            return [Reply(CONSENT_NOTICE, reply_to=msg.message_id, kind="consent")]

        if not self.consent.may_capture(
            corrector, exchange.prompt_author_id, scope=SCOPE_CORRECTIONS
        ):
            self._log_skip(CORRECTION, corrector, exchange.prompt_author_id)
            return [
                Reply(
                    "-# noted, but not stored — someone in this thread hasn't opted in, "
                    "so i can't learn from it. `!babble consent`",
                    reply_to=msg.message_id,
                    kind="ack",
                )
            ]

        correction = strip_correction_marker(self._message_text(msg))
        if not correction:
            return [
                Reply(
                    f"-# `{CORRECTION_MARKER}` on its own doesn't teach me anything — put what i "
                    f"should have said after it, like `{CORRECTION_MARKER} hey, what's up`.",
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

        # A correction is human writing too, on both sides: what was asked and
        # what somebody typed as the better answer. Each half goes in under its
        # own author's corpus grant, or not at all.
        self._capture_corpus(msg, correction, SOURCE_CORRECTION)
        self._capture_corpus(
            msg, exchange.prompt, SOURCE_PROMPT, author_id=exchange.prompt_author_id
        )
        # ...and past the post-train trigger: every N new pairs, a fresh
        # supervised pass fires so the served model picks up the newest
        # correction. Only for a genuinely new pair -- a duplicate correction
        # never grows `pair_count`, so it must never be allowed to fire this.
        if fresh and self.post_trigger is not None:
            self.post_trigger.maybe_run()
        # ...and the pair-augmentation hook: paraphrase THIS correction into
        # extra variants right now, out of band, so the pair set compounds as
        # corrections keep arriving rather than waiting for a batch command.
        # Off by default (`Settings.post_augment_pairs`) -- see `pairaugment.
        # AutoAugmentTrigger`.
        if fresh and self.augment_trigger is not None:
            self.augment_trigger.on_new_pair(row.id)
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
            guild=self.log.guild(msg.guild_id),
        )
        reply = lambda text: [Reply(text, reply_to=msg.message_id, kind="command")]  # noqa: E731

        if verb in ("accept", "yes", "agree", "optin", "opt-in"):
            self.consent.grant(msg.author_id)
            self.log.event("consent.accept", user=self.log.user(msg.author_id), scope="all")
            if self.feed is not None:
                self.feed.consent_granted(author=self.ids.user(msg.author_id))
            return reply(
                "you're in — thank you. from now on the messages you send me go into the "
                "training corpus and the public dataset, and so do your corrections.\n"
                "-# `!babble all` in a channel widens that to *everything you say there*. "
                "`!babble forget` any time, which also deletes what i've kept."
            )

        if verb in ("decline", "no", "nope", "optout", "opt-out"):
            self.consent.decline(msg.author_id)
            self.log.event("consent.decline", user=self.log.user(msg.author_id), scope="all")
            if self.feed is not None:
                self.feed.consent_declined(author=self.ids.user(msg.author_id))
            return reply(
                "understood — nothing of yours will be stored, trained on, or published.\n"
                "-# i'll still reply if you ping me. `!babble accept` if you ever change your mind."
            )

        if verb in ("all", "everything", "all-in"):
            if args and args[0].lower() in ("off", "stop", "no", "none", "undo"):
                return self._narrow_channel(msg, reply)
            return self._widen_channel(msg, reply)

        if verb in ("pings", "only-pings", "just-pings", "notall", "not-all"):
            return self._narrow_channel(msg, reply)

        if verb in ("forget", "delete", "withdraw", "erase"):
            author = self.ids.user(msg.author_id)
            purged = self.store.purge_author(author)
            purged_corpus = self.corpus.purge_author(author)
            dropped = self.exchanges.forget_author(msg.author_id)
            self.consent.withdraw(msg.author_id)
            self.log.event(
                "consent.withdraw",
                user=author,
                rows_purged=purged,
                corpus_purged=purged_corpus,
                pending_dropped=dropped,
            )
            if self.feed is not None:
                self.feed.consent_withdrawn(
                    author=author, corpus_purged=purged_corpus, correction_purged=purged
                )
            total = purged + purged_corpus
            return reply(
                f"done. consent withdrawn and **{total}** stored "
                f"{'row' if total == 1 else 'rows'} of yours deleted "
                f"({purged_corpus} from the corpus, {purged} correction "
                f"{'row' if purged == 1 else 'rows'}).\n"
                "-# anything already published to HuggingFace in a previous export won't vanish "
                "from people's downloads, but it's gone from here and from every export after this."
            )

        if verb in ("consent", "privacy", "data"):
            return reply(f"{CONSENT_NOTICE}\n\n{self._describe_consent(msg)}")

        if verb in ("status", "stats"):
            from .stats import render_snapshot, snapshot  # local: keeps core import cheap

            return reply(
                f"{render_snapshot(snapshot(self.settings))}\n{self._describe_consent(msg)}"
            )

        return reply(HELP_TEXT)

    def _widen_channel(self, msg: IncomingMessage, reply: Callable[[str], list[Reply]]) -> list[Reply]:
        """`!babble all` — collect everything this person says in this channel."""
        if self.consent.decision(msg.author_id, SCOPE_CORPUS) not in CAPTURE_OK:
            # Widening a collection nobody has agreed to is not a thing. Show
            # them what they would be agreeing to first.
            self.consent.mark_prompted(msg.author_id, SCOPE_CORRECTIONS)
            self.consent.mark_prompted(msg.author_id, SCOPE_CORPUS)
            self.log.event(
                "consent.prompt",
                user=self.log.user(msg.author_id),
                channel=self.log.channel(msg.channel_id),
                guild=self.log.guild(msg.guild_id),
                trigger="widen_without_consent",
                scope=SCOPE_CORPUS,
            )
            return reply(
                f"{CONSENT_NOTICE}\n\n"
                "-# **`!babble accept` first** — then `!babble all` here again and i'll widen it."
            )

        added = self.consent.widen(msg.author_id, msg.channel_id)
        self.log.event(
            "consent.widen",
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
            guild=self.log.guild(msg.guild_id),
            already_on=None if added else True,
            channels=len(self.consent.wide_channels(msg.author_id)),
        )
        if not added:
            return reply(
                "already on in this channel — everything you say here is going into the corpus.\n"
                "-# `!babble pings` turns it back off here."
            )
        if self.feed is not None:
            self.feed.channel_widened(
                author=self.ids.user(msg.author_id), channel=self.ids.channel(msg.channel_id)
            )
        return reply(
            "done — from now on **every message you send in this channel** goes into the corpus, "
            "not just the ones aimed at me. it's only you and it's only here: nobody else in this "
            "channel is affected, and it doesn't follow you anywhere else.\n"
            "-# `!babble pings` turns it back off here, straight away. `!babble forget` opts you "
            "out entirely and deletes everything of yours."
        )

    def _narrow_channel(self, msg: IncomingMessage, reply: Callable[[str], list[Reply]]) -> list[Reply]:
        """`!babble pings` — back to collecting only what is aimed at the bot."""
        removed = self.consent.narrow(msg.author_id, msg.channel_id)
        self.log.event(
            "consent.narrow",
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
            guild=self.log.guild(msg.guild_id),
            was_on=removed or None,
            channels=len(self.consent.wide_channels(msg.author_id)),
        )
        if not removed:
            return reply(
                "that wasn't on here — in this channel i only keep what you send me directly.\n"
                "-# `!babble all` if you want me to keep everything you say here."
            )
        if self.feed is not None:
            self.feed.channel_narrowed(
                author=self.ids.user(msg.author_id), channel=self.ids.channel(msg.channel_id)
            )
        return reply(
            "done — from now on i only keep what you send me directly in this channel. "
            "that takes effect immediately.\n"
            "-# what's already stored stays until you `!babble forget`, which deletes all of it."
        )

    # --- helpers --------------------------------------------------------

    def _describe_consent(self, msg: IncomingMessage) -> str:
        """One line telling this person exactly where they stand, per grant."""
        corpus = _describe(self.consent.decision(msg.author_id, SCOPE_CORPUS))
        corrections = _describe(self.consent.decision(msg.author_id, SCOPE_CORRECTIONS))
        wide = str(msg.channel_id) in self.consent.wide_channels(msg.author_id)
        here = "everything you say here" if wide else "only what you send me"
        return (
            f"-# you: messages **{corpus}** · corrections **{corrections}** · "
            f"in this channel i keep **{here}**"
        )

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

    def _log_drop(self, msg: IncomingMessage, reason: str) -> None:
        """Record that an incoming message was not acted on, and why. Never the text."""
        self.log.event(
            "bot.dropped",
            reason=reason,
            user=self.log.user(msg.author_id),
            channel=self.log.channel(msg.channel_id),
            guild=self.log.guild(msg.guild_id),
            chars=len(msg.content),
        )

    def _log_skip(
        self,
        signal: str,
        signal_author: object,
        prompt_author: object,
        scope: str = SCOPE_CORRECTIONS,
    ) -> None:
        """Record that we threw data away, and why. Never what the data was."""
        blockers = {
            "signal_author": self.consent.decision(signal_author, scope),
            "prompt_author": self.consent.decision(prompt_author, scope),
        }
        missing = [role for role, state in blockers.items() if state not in CAPTURE_OK]
        self.log.event(
            "capture.skipped",
            signal=signal,
            scope=scope,
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
