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


def bounded_history(
    history: Sequence[ConversationTurn],
    *,
    max_turns: int,
) -> tuple[ConversationTurn, ...]:
    """Keep the newest completed turns, with non-positive limits meaning none."""

    limit = max(0, int(max_turns))
    return tuple(history[-limit:]) if limit else ()


def conversation_prompt(
    history: Sequence[ConversationTurn],
    current_user: str,
    *,
    max_turns: int,
    max_chars: int,
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
