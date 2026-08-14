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


def build_example(prompt: str, response: str, block_size: int, weight: float = 1.0) -> Example:
    """Lay out `<bos> prompt <sep> response <eos>` and mask it for training."""
    budget = block_size - 3  # the three special tokens always cost their slots
    if budget < 1:
        raise ValueError(f"block_size {block_size} is too small to hold an example")

    response_ids = encode(response)[:budget]
    # Whatever the response left over goes to the prompt, keeping its tail: the
    # end of a long message is usually the part being responded to.
    room = budget - len(response_ids)
    prompt_ids = encode(prompt)[-room:] if room > 0 else []

    tokens = [BOS_ID, *prompt_ids, SEP_ID, *response_ids, EOS_ID]
    mask = [0] * (len(prompt_ids) + 2) + [1] * (len(response_ids) + 1)
    assert len(tokens) == len(mask)
    return Example(tokens=tokens, mask=mask, weight=weight)


def prompt_context(prompt: str, block_size: int) -> list[int]:
    """The inference-time prefix: `<bos> prompt <sep>`, ready to continue from."""
    room = max(0, block_size - 2)
    return [BOS_ID, *encode(prompt)[-room:], SEP_ID]
