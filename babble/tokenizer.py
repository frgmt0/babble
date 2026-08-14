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

    `mask` is 1 exactly on the tokens the model is meant to produce. For the
    corpus examples the trainer now builds that is *everything except the
    opening <bos>*, because plain next-token prediction has no part of the
    sequence it declines to learn. The paired `<bos> prompt <sep> response`
    layout below masks the prompt out instead; that layout is no longer the
    training objective, but it is still how a correction pair gets scored.
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


# --- the corpus layout ---------------------------------------------------
#
# `<bos> text <eos>`, and every token after the <bos> is a target. No <sep>,
# because there is nothing on either side of it: a corpus row is one piece of
# writing, not a question and its answer.


def text_budget(block_size: int) -> int:
    """How many bytes of text fit in one example, after <bos> and <eos>."""
    budget = block_size - 2
    if budget < 1:
        raise ValueError(f"block_size {block_size} is too small to hold an example")
    return budget


def build_text_example(chunk: list[int], *, final: bool = True) -> Example:
    """One `<bos> chunk [<eos>]` example with everything after <bos> trained.

    `final=False` leaves the <eos> off, for a chunk that is the middle of a
    longer row: <eos> means "the text ended here", and it only actually did at
    the last chunk. Position 0 carries no loss either way -- nothing precedes it
    to predict it from -- which is why the mask starts with a zero and not
    because any part of the text is being held back.
    """
    tokens = [BOS_ID, *chunk] + ([EOS_ID] if final else [])
    mask = [0] + [1] * (len(tokens) - 1)
    return Example(tokens=tokens, mask=mask)


def text_examples(text: str, block_size: int) -> list[Example]:
    """A corpus row, tokenised into as many examples as it takes.

    A row longer than the block is split into consecutive chunks rather than
    truncated, so a long message contributes all of itself instead of just its
    first few hundred bytes. Splitting mid-character is harmless here: the model
    is byte-level, and `decode` replaces any broken pair on the way out.
    """
    budget = text_budget(block_size)
    ids = encode(text)
    if not ids:
        return []
    chunks = [ids[i : i + budget] for i in range(0, len(ids), budget)]
    return [
        build_text_example(chunk, final=index == len(chunks) - 1)
        for index, chunk in enumerate(chunks)
    ]


def text_prefix_budget(block_size: int) -> int:
    """How many prefix bytes survive when something is generated after them.

    A quarter of the block is held back for the continuation, the same rule
    `prompt_budget` applies, so a very long prefix cannot squeeze the generated
    part down to nothing -- and so the context a continuation is *generated*
    from is byte-for-byte the context it is *scored* in.
    """
    reserved = max(1, block_size // 4)
    return max(1, text_budget(block_size) - reserved)


def build_continuation_example(prefix: str, continuation: str, block_size: int) -> Example:
    """`<bos> prefix continuation <eos>`, with only the continuation trained.

    The scoring counterpart of `text_examples`. Nothing builds these for
    training -- the corpus objective trains every token -- but comparing two
    candidate continuations has to mask the prefix they share, or the comparison
    is dominated by bytes neither candidate chose.
    """
    prefix_ids = encode(prefix)[-text_prefix_budget(block_size) :]
    cont_ids = encode(continuation)[: text_budget(block_size) - len(prefix_ids)]
    tokens = [BOS_ID, *prefix_ids, *cont_ids, EOS_ID]
    mask = [0] * (len(prefix_ids) + 1) + [1] * (len(cont_ids) + 1)
    assert len(tokens) == len(mask)
    return Example(tokens=tokens, mask=mask)


def text_context(prefix: str, block_size: int) -> list[int]:
    """The inference-time prefix for the corpus layout: `<bos> prefix`.

    Truncated by `text_prefix_budget`, keeping the tail, which is exactly what
    `build_continuation_example` applies -- so the prefix the model generates
    from is the prefix it gets scored on.
    """
    return [BOS_ID, *encode(prefix)[-text_prefix_budget(block_size) :]]
