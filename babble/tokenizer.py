"""Byte-level tokenisation. No vocabulary file, no download, nothing learned.

Why bytes and not words: ro wants to be able to correct the bot with a gif, and a
gif is a URL. A word vocabulary would have to fall back to <unk> on
`https://tenor.com/view/...`, on 🫠, on `:3`, on Japanese, on a typo. UTF-8 bytes
cover all of it with a 260-token vocabulary, which is also why the model can stay
this small.

The cost is that the model must learn spelling one byte at a time. That is a real
cost and it is a large part of why this will babble for a long time.
"""

from __future__ import annotations

from dataclasses import dataclass

# 0..255 are the byte values themselves; the four specials sit above them.
PAD_ID = 256
BOS_ID = 257
SEP_ID = 258  # divides "what was said to it" from "what it should say"
EOS_ID = 259
VOCAB_SIZE = 260

SPECIAL_IDS = frozenset({PAD_ID, BOS_ID, SEP_ID, EOS_ID})
SPECIAL_NAMES = {PAD_ID: "<pad>", BOS_ID: "<bos>", SEP_ID: "<sep>", EOS_ID: "<eos>"}


def encode(text: str) -> list[int]:
    """Text -> byte ids. Total function: every str has a UTF-8 encoding."""
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    """Byte ids -> text, dropping specials.

    `errors="replace"` matters more than it looks: a freshly initialised model
    emits uniformly random bytes, most of which are not valid UTF-8. Without this
    the very first sample generation would raise instead of showing the noise,
    and the noise is the point.
    """
    raw = bytes(i for i in ids if i not in SPECIAL_IDS and 0 <= i < 256)
    return raw.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class Example:
    """One training row, already tokenised.

    `mask` is 1 exactly on the tokens the model is meant to produce -- the
    response and the final <eos>. The prompt is context, not a target: we are
    teaching it to answer, not to invent things people might say to it.
    """

    tokens: list[int]
    mask: list[int]
    weight: float = 1.0

    def __len__(self) -> int:
        return len(self.tokens)


def prompt_budget(block_size: int) -> int:
    """How many prompt bytes survive truncation, given only the block size.

    This deliberately does **not** depend on the response. It used to: the
    response was kept whole and the prompt got whatever was left. That is
    unusable at inference, where there is no response yet -- so the trainer and
    `prompt_context` disagreed about how far to trim a long prompt, which moved
    `<sep>` to a different position than the one the model was trained on.
    With learned position embeddings that is not a small discrepancy; it is a
    different input. A prompt-only rule makes the two byte-for-byte identical.

    A quarter of the block is held back for the response so that a very long
    prompt cannot squeeze the part we actually train on down to nothing.
    """
    reserved = max(1, block_size // 4)
    return max(1, block_size - 3 - reserved)


def build_example(prompt: str, response: str, block_size: int, weight: float = 1.0) -> Example:
    """Lay out `<bos> prompt <sep> response <eos>` and mask it for training."""
    budget = block_size - 3  # the three special tokens always cost their slots
    if budget < 1:
        raise ValueError(f"block_size {block_size} is too small to hold an example")

    # Trim the prompt by the same rule inference uses, keeping its tail: the end
    # of a long message is usually the part being responded to.
    prompt_ids = encode(prompt)[-prompt_budget(block_size) :]
    response_ids = encode(response)[: budget - len(prompt_ids)]

    tokens = [BOS_ID, *prompt_ids, SEP_ID, *response_ids, EOS_ID]
    mask = [0] * (len(prompt_ids) + 2) + [1] * (len(response_ids) + 1)
    assert len(tokens) == len(mask)
    return Example(tokens=tokens, mask=mask, weight=weight)


def prompt_context(prompt: str, block_size: int) -> list[int]:
    """The inference-time prefix: `<bos> prompt <sep>`, ready to continue from.

    Truncated by `prompt_budget`, which is exactly what `build_example` applies,
    so the prefix the model generates from is the prefix it was trained on.
    """
    return [BOS_ID, *encode(prompt)[-prompt_budget(block_size) :], SEP_ID]
