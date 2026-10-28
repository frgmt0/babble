"""The jackpot's other half: asking Claude to write one reply in booper's voice.

`babble/slots.py` decides *whether* a reply is a jackpot; this module makes the
one API call that writes it. It is the only file in the repo that talks to the
Anthropic API, and it is built to be boring about failure: every way the call
can go wrong -- no SDK, no key, rate limit, timeout, refusal, empty answer --
comes back as an `OracleResult` with `ok=False` and a short reason, and the
caller (`Babble.cash_out`) simply sends the local model's reply it already had.

What is sent, in full: the user's own (mention-scrubbed) message, booper's
own draft reply to it (what the local model was about to send), booper's own
earlier replies in this conversation, and up to N of booper's *own* recent
replies as voice examples. No ids, no usernames, no other people's messages,
and none of the user's earlier messages -- only the one the notice discloses. What comes back is
never stored, never captured to the corpus and never trained on -- see
`Babble.cash_out`.

The `anthropic` SDK is an optional extra (`pip install -e ".[fable]"`) and is
imported lazily, so a box without it never pays for the import and the rest of
the bot runs exactly as before.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .slots import JackpotRequest

API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Server-side refusal fallback. `"default"` routes by refusal category, so
#: there is no fallback model list to keep current here.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACKS = "default"

#: Streaming, so a long max-effort turn never trips an HTTP read timeout.
MAX_TOKENS = 64_000

#: Target size of a jackpot reply. Comfortably under Discord's 2,000-character
#: message limit; `clean_for_discord` still hard-caps whatever comes back.
REPLY_CHAR_BUDGET = 1900

# Stable across every call (good for caching) and short on purpose: this model
# does worse when it is handed a wall of rules. Everything per-message goes in
# the user turn (`build_user_message`), so this block stays cacheable.
_VOICE = f"""You are writing one Discord reply as booper, a tiny chat bot. booper's \
real brain is a 34M-parameter model trained from scratch on Discord chat, so it \
mostly babbles. This one time, you write the reply instead.

You get the person's message, booper's earlier replies in this conversation (if \
any), and booper's draft: what the tiny model was about to say. Use the draft as \
the launchpad -- keep its vibe, its weird premise, its best broken phrase -- and \
spin it into something hella funny. Then land the turn: somewhere in there, say \
something genuinely sharp, true or useful about what they actually said, the \
kind of line that makes them go "wait, that's actually smart". Funny first, \
smart second, never a lecture. If the draft is pure nonsense, the nonsense can \
be the bit.

Voice: lowercase, casual Discord-chat register, playful, a little unhinged, \
confident. Usually a few lines; only go long if the bit earns it. No markdown \
headers or bullet lists, no explaining the joke. Never mention being Claude, an \
AI model, a draft, or a slot machine -- you are just booper.

Keep the whole reply under {REPLY_CHAR_BUDGET:,} characters; it has to fit in \
one Discord message."""


def build_system_prompt(voice_examples: tuple[str, ...] | list[str]) -> str:
    """The system prompt: fixed voice text, then the voice examples."""
    parts = [_VOICE]
    if voice_examples:
        parts.append(
            "booper's recent replies, to match how booper sounds (style only, not context):\n"
            f"<voice_examples>\n{_lines(voice_examples)}\n</voice_examples>"
        )
    return "\n\n".join(parts)


def build_user_message(request: JackpotRequest) -> str:
    """The per-message turn: what they said, the thread so far, booper's draft."""
    parts = [f"<their_message>\n{request.prompt}\n</their_message>"]
    if request.thread:
        parts.append(
            "booper's earlier replies in this conversation, oldest first:\n"
            f"<booper_earlier>\n{_lines(request.thread)}\n</booper_earlier>"
        )
    draft = request.draft.strip() or "(nothing -- the tiny model came up empty)"
    parts.append(f"<booper_draft>\n{draft}\n</booper_draft>")
    parts.append("Write booper's reply.")
    return "\n\n".join(parts)


def _lines(texts: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {text.replace(chr(10), ' ').strip()}" for text in texts)


@dataclass(frozen=True)
class OracleResult:
    """What one call produced. `ok=False` always carries a `reason`."""

    ok: bool
    text: str = ""
    reason: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    ms: float = 0.0


class Oracle:
    """One configured Anthropic client and how to ask it for a jackpot reply."""

    def __init__(self, client: Any, *, model: str, effort: str, max_tokens: int = MAX_TOKENS) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    @classmethod
    def from_settings(cls, settings, log=None) -> "Oracle | None":
        """A live oracle, or None (feature inert) with the reason logged once.

        None when the slots are switched off, when the `fable` extra is not
        installed, or when there is no `ANTHROPIC_API_KEY`. An explicit key is
        required on purpose: a bot on a shared box must not quietly pick up a
        developer's personal CLI login and start spending on it.
        """

        def inert(reason: str) -> None:
            if log is not None:
                log.event("slots.disabled", reason=reason)
            return None

        if not settings.slots_enabled:
            return None
        try:
            import anthropic
        except ImportError:
            return inert("anthropic_sdk_missing")
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            return inert("no_api_key")
        # max_retries=0: the timeout is then a real ceiling on how long a
        # jackpot can keep someone waiting, instead of timeout x (retries+1).
        client = anthropic.Anthropic(
            api_key=key, timeout=settings.slots_timeout_seconds, max_retries=0
        )
        if log is not None:
            log.event("slots.enabled", model=settings.slots_model, effort=settings.slots_effort)
        return cls(client, model=settings.slots_model, effort=settings.slots_effort)

    def ask(self, request: JackpotRequest) -> OracleResult:
        """Make the one streamed call. Never raises for an API-side failure."""
        import anthropic  # present: from_settings only builds an Oracle if it is

        started = time.monotonic()

        def failed(reason: str, **extra: Any) -> OracleResult:
            return OracleResult(ok=False, reason=reason, ms=(time.monotonic() - started) * 1000, **extra)

        # No `thinking` (always on for this model; anything explicit but
        # adaptive 400s), no temperature/top_p/top_k and no assistant prefill
        # (all rejected). Depth is `output_config.effort` alone.
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=build_system_prompt(request.voice_examples),
                messages=[{"role": "user", "content": build_user_message(request)}],
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks=FALLBACKS,
            ) as stream:
                message = stream.get_final_message()
        except anthropic.APITimeoutError:
            return failed("timeout")
        except anthropic.RateLimitError:
            return failed("rate_limited")
        except anthropic.APIStatusError as exc:
            return failed(f"http_{exc.status_code}")
        except anthropic.APIConnectionError:
            return failed("connection")
        except anthropic.AnthropicError as exc:
            return failed(f"sdk_{type(exc).__name__}")

        usage = getattr(message, "usage", None)
        meta = {
            "model": getattr(message, "model", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        # A refusal is HTTP 200; check it before reading any content.
        if getattr(message, "stop_reason", None) == "refusal":
            return failed("refusal", **meta)
        text = "".join(
            block.text for block in (message.content or []) if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            return failed("empty", **meta)
        return OracleResult(ok=True, text=text, ms=(time.monotonic() - started) * 1000, **meta)
