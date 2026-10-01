"""Shared, tokenizer-neutral formatting for multi-turn chat prompts.

The model still receives the same structural layout used by pair checkpoints::

    <bos> PROMPT <sep> RESPONSE <eos>

Only ``PROMPT`` changes for a multi-turn checkpoint.  It is a plain role
transcript, so no tokenizer migration or new special tokens are required::

    user: first message
    assistant: first reply
    user: follow-up

This module is deliberately independent of Discord and torch.  Runtime and SFT
can import the same formatter instead of maintaining two almost-identical
prompt conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

USER_PREFIX = "user: "
ASSISTANT_PREFIX = "assistant: "


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user/assistant turn retained as inference context."""

    user: str
    assistant: str

    def to_dict(self) -> dict[str, str]:
        return {"user": self.user, "assistant": self.assistant}

    @classmethod
    def from_dict(cls, raw: object) -> "ConversationTurn | None":
        if not isinstance(raw, dict):
            return None
        user = raw.get("user")
        assistant = raw.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            return None
        return cls(user=user, assistant=assistant)


def _serialize(history: Iterable[ConversationTurn], current_user: str) -> str:
    lines: list[str] = []
    for turn in history:
        lines.append(f"{USER_PREFIX}{turn.user}")
        lines.append(f"{ASSISTANT_PREFIX}{turn.assistant}")
    lines.append(f"{USER_PREFIX}{current_user}")
    return "\n".join(lines)


def _trim_to_floor(
    kept: list[ConversationTurn],
    fits_low: Callable[[str], bool],
) -> list[ConversationTurn]:
    """Drop oldest turns until the *history* fits the low watermark.

    Called only once the transcript has overflowed its real cap. The floor is
    measured on the history alone (serialized up to the trailing ``user: ``),
    not on the current message, so where the window lands after a trim does
    not depend on how long the message that triggered it was: the window is
    predictable before the next message arrives, which is what lets the bot
    pre-prefill it. The caller's ordinary one-turn-at-a-time trim then makes
    room for an unusually long current message, exactly as before.
    """

    trimmed = list(kept)
    while trimmed and not fits_low(_serialize(trimmed, "")):
        trimmed.pop(0)
    return trimmed


def bounded_history(
    history: Sequence[ConversationTurn],
    *,
    max_turns: int,
) -> tuple[ConversationTurn, ...]:
    """Keep the newest completed turns, with non-positive limits meaning none."""

    limit = max(0, int(max_turns))
    return tuple(history[-limit:]) if limit else ()


def overflow_floor(cap: int, overflow_keep: float) -> int:
    """The low watermark an overflowing window is trimmed down to.

    ``overflow_keep`` is the fraction of a cap kept after an overflow; 1.0 (or
    more) means "just under the cap", i.e. slide one turn at a time.
    """

    keep = float(overflow_keep)
    if keep >= 1.0 or cap <= 0:
        return cap
    return max(1, int(cap * keep)) if keep > 0 else 0


def windowed_turns(
    turns: Sequence[ConversationTurn], *, max_turns: int, overflow_keep: float = 1.0
) -> tuple[ConversationTurn, ...]:
    """Bound a chain's turns, trimming in one chunk when it overflows.

    With ``overflow_keep`` 1.0 this is exactly `bounded_history`: every turn
    past the cap drops the oldest one, so the transcript's prefix changes on
    every turn and a prefix KV cache can never hit. With e.g. 0.5, a chain
    that reaches ``max_turns + 1`` turns is cut back to ``max_turns // 2`` and
    then grows again, so the following turns extend a stable prefix.
    """

    limit = max(0, int(max_turns))
    if len(turns) <= limit:
        return tuple(turns)
    floor = overflow_floor(limit, overflow_keep)
    if floor >= limit:
        return bounded_history(turns, max_turns=limit)
    return tuple(turns[-floor:]) if floor else ()


def used_history(
    history: Sequence[ConversationTurn], current_user: str, prompt: str
) -> tuple[ConversationTurn, ...]:
    """The suffix of ``history`` a formatter actually serialized into ``prompt``.

    Formatters only ever drop whole turns from the front (and, as a last
    resort, crop the current message, in which case no history is used).
    """

    for start in range(len(history) + 1):
        if prompt == _serialize(history[start:], current_user):
            return tuple(history[start:])
    return ()


def conversation_prompt(
    history: Sequence[ConversationTurn],
    current_user: str,
    *,
    max_turns: int,
    max_chars: int,
    overflow_keep: float = 1.0,
) -> str:
    """Serialize a bounded chronological transcript ending at ``current_user``.

    Whole old turns are discarded first.  If the current message alone is too
    long, its newest characters are retained, matching the left-truncation used
    by both serving backends.  The ``user: `` prefix is always kept intact.
    ``max_chars <= 0`` means there is no character cap; turn bounding still
    applies.
    """

    kept = list(bounded_history(history, max_turns=max_turns))
    cap = int(max_chars)
    if cap <= 0:
        return _serialize(kept, current_user)
    if cap < len(USER_PREFIX):
        raise ValueError("conversation character budget is too small for the user role")

    if kept and len(_serialize(kept, current_user)) > cap:
        kept = _trim_to_floor(kept, lambda value: len(value) <= overflow_floor(cap, overflow_keep))
    while kept and len(_serialize(kept, current_user)) > cap:
        kept.pop(0)

    prompt = _serialize(kept, current_user)
    if len(prompt) <= cap:
        return prompt

    # Even the current turn is larger than the budget. Keep the structural
    # prefix and the newest part of the message; a tiny cap still returns a
    # valid role-labelled prompt rather than slicing through ``user: ``.
    body_budget = max(0, cap - len(USER_PREFIX))
    body = current_user[-body_budget:] if body_budget else ""
    return f"{USER_PREFIX}{body}"


def conversation_prompt_for_token_budget(
    history: Sequence[ConversationTurn],
    current_user: str,
    *,
    max_turns: int,
    max_chars: int,
    max_tokens: int,
    token_count: Callable[[str], int],
    overflow_keep: float = 1.0,
) -> str:
    """Fit the transcript without letting token truncation split role framing.

    Serving backends know their tokenizer and exact prompt budget; core does
    not. They use this helper through their optional ``conversation_prompt``
    method. Oldest complete turns are removed until the transcript fits both
    configured bounds. If the current message alone is oversized, only its
    body is left-cropped and the ``user: `` marker stays whole.
    """

    token_cap = int(max_tokens)
    if token_cap <= 0:
        raise ValueError("a conversation prompt needs a positive token budget")

    char_cap = int(max_chars)

    def fits(value: str) -> bool:
        return (char_cap <= 0 or len(value) <= char_cap) and token_count(value) <= token_cap

    kept = list(bounded_history(history, max_turns=max_turns))
    prompt = _serialize(kept, current_user)
    if kept and not fits(prompt):
        low_tokens = max(1, overflow_floor(token_cap, overflow_keep))
        low_chars = overflow_floor(char_cap, overflow_keep)

        def fits_low(value: str) -> bool:
            return (char_cap <= 0 or len(value) <= low_chars) and token_count(value) <= low_tokens

        kept = _trim_to_floor(kept, fits_low)
        prompt = _serialize(kept, current_user)
    while kept and not fits(prompt):
        kept.pop(0)
        prompt = _serialize(kept, current_user)
    if fits(prompt):
        return prompt

    if not fits(USER_PREFIX):
        raise ValueError("conversation token/character budget is too small for the user role")

    # Find the longest suffix that fits. Token counts are effectively monotonic
    # for suffix growth with the supported BPE/byte tokenizers; the final loops
    # make the boundary exact even if a merge changes around the cut point.
    low, high = 0, len(current_user)
    while low < high:
        size = (low + high + 1) // 2
        candidate = f"{USER_PREFIX}{current_user[-size:]}" if size else USER_PREFIX
        if fits(candidate):
            low = size
        else:
            high = size - 1
    while low < len(current_user):
        candidate = f"{USER_PREFIX}{current_user[-(low + 1):]}"
        if not fits(candidate):
            break
        low += 1
    return f"{USER_PREFIX}{current_user[-low:]}" if low else USER_PREFIX
