"""Corpus-trained subword tokenizers: byte-pair encoding and word-level.

Experimental, off the production path -- `babble/tokenizer.py` (raw UTF-8
bytes) stays the tokenizer the live bot, `core.py`, `generate.py` and
`trainer.py` use. Nothing in this module is imported by any of them; it exists
for `experiments/tokenizer_sweep.py`, which asks whether spending fewer,
bigger tokens on the same 423-row corpus buys anything over spelling English
one byte at a time. See TOKENIZER_SWAP_REPORT.md.

Both tokenizers keep byte-level's one real virtue -- total coverage, no
`<unk>` -- by building on top of the same 256 raw byte ids rather than
replacing them: a merge or a word is just a longer id that decodes back to a
fixed byte string, and anything that was never seen in training falls back to
its constituent bytes. `encode(decode(encode(x))) == encode(x)` always holds;
`decode(encode(x)) == x` holds for any valid UTF-8 `x`, same as the byte
tokenizer.

Ids 0..255 are always the raw bytes. Ids 256..256+len(vocab)-1 are the
learned merges/words, in the order they were learned (BPE) or by descending
frequency (word). The four specials sit above that, at
`256 + len(vocab) + {0,1,2,3}` -- shifted up from `babble.tokenizer`'s fixed
256 because the learned vocab's size isn't fixed ahead of time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import tokenizer as bytetok
from .tokenizer import Example

# Splits text into whitespace runs and non-whitespace runs. BPE merges never
# cross a chunk boundary (standard GPT-2-style pretokenization): without this,
# frequent bigrams like "e " would happily merge across a word boundary,
# entangling a word's identity with whatever usually follows it.
_CHUNK_RE = re.compile(r"\s+|\S+")


def _chunks(text: str) -> list[str]:
    return _CHUNK_RE.findall(text)


@dataclass(frozen=True)
class SpecialIds:
    """The four structural ids, positioned above a tokenizer's learned vocab."""

    pad: int
    bos: int
    sep: int
    eos: int

    @classmethod
    def above(cls, vocab_len: int) -> "SpecialIds":
        base = 256 + vocab_len
        return cls(pad=base, bos=base + 1, sep=base + 2, eos=base + 3)


# --- byte-pair encoding ----------------------------------------------------


def _pair_counts(word_ids: dict[str, list[int]], freq: dict[str, int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for chunk, ids in word_ids.items():
        f = freq[chunk]
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + f
    return counts


def _merge_ids(ids: list[int], a: int, b: int, new_id: int) -> list[int]:
    if len(ids) < 2:
        return ids
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """Byte-level BPE: iteratively merge the most frequent adjacent byte pair.

    `merges` is the ordered list of `(a, b, new_id)` learned on the training
    corpus; encoding always replays them in that same order, which is what
    makes BPE encoding deterministic rather than dependent on merge-time state.
    """

    merges: list[tuple[int, int, int]]
    vocab: dict[int, bytes]
    specials: SpecialIds = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "specials", SpecialIds.above(len(self.merges)))

    @classmethod
    def train(cls, texts: list[str], num_merges: int) -> "BPETokenizer":
        """Learn `num_merges` merges from `texts` (pass TRAIN-side rows only --
        fitting on validation text would leak held-out phrasing into the
        vocabulary itself, before a single gradient step)."""
        freq: dict[str, int] = {}
        for text in texts:
            for chunk in _chunks(text):
                freq[chunk] = freq.get(chunk, 0) + 1
        word_ids = {chunk: list(chunk.encode("utf-8")) for chunk in freq}

        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[int, int, int]] = []
        next_id = 256
        for _ in range(num_merges):
            counts = _pair_counts(word_ids, freq)
            if not counts:
                break
            (a, b) = max(counts, key=lambda p: (counts[p], p))
            new_id = next_id
            next_id += 1
            merges.append((a, b, new_id))
            vocab[new_id] = vocab[a] + vocab[b]
            for chunk in word_ids:
                word_ids[chunk] = _merge_ids(word_ids[chunk], a, b, new_id)
        return cls(merges=merges, vocab=vocab)

    @property
    def vocab_size(self) -> int:
        """Total ids: 256 raw bytes + learned merges + 4 specials."""
        return 256 + len(self.merges) + 4

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        for a, b, new_id in self.merges:
            ids = _merge_ids(ids, a, b, new_id)
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _chunks(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for i in ids:
            piece = self.vocab.get(i)
            if piece is not None:
                raw.extend(piece)
        return bytes(raw).decode("utf-8", errors="replace")

    def to_json(self, path: Path) -> None:
        """Just the ordered merge list -- `vocab` is fully determined by
        replaying `merges` over the 256 raw bytes (see `__post_init__`), so
        nothing else needs to round-trip. Same schema `pretrain_hf.py` writes,
        on purpose: a checkpoint that script produces and the `tokenizer.json`
        next to it load back in here with no translation step."""
        path.write_text(json.dumps({"merges": [list(m) for m in self.merges]}))

    @classmethod
    def from_json(cls, path: Path) -> "BPETokenizer":
        raw = json.loads(Path(path).read_text())
        merges = [tuple(m) for m in raw["merges"]]
        vocab = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in merges:
            vocab[new_id] = vocab[a] + vocab[b]
        return cls(merges=merges, vocab=vocab)


# --- word-level tokenization -----------------------------------------------


@dataclass
class WordTokenizer:
    """Whole chunks (words, punctuation runs, whitespace runs) as single ids.

    The `max_words` most frequent chunks in the training text get their own
    id; anything else -- a typo, a URL, a rare word, an emoji -- falls back to
    its raw UTF-8 bytes, chunk by chunk. That fallback is what keeps this a
    total function instead of an `<unk>`-emitting one: nothing is ever thrown
    away, it just costs more tokens to say.
    """

    vocab: dict[str, int]  # chunk text -> id (256..)
    by_id: dict[int, str] = field(init=False)
    specials: SpecialIds = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", {v: k for k, v in self.vocab.items()})
        object.__setattr__(self, "specials", SpecialIds.above(len(self.vocab)))

    @classmethod
    def train(cls, texts: list[str], max_words: int) -> "WordTokenizer":
        """Learn a vocabulary of the `max_words` most frequent chunks in
        `texts` (TRAIN-side rows only, same leakage discipline as BPE)."""
        counts: dict[str, int] = {}
        for text in texts:
            for chunk in _chunks(text):
                counts[chunk] = counts.get(chunk, 0) + 1
        ranked = sorted(counts, key=lambda w: (-counts[w], w))[:max_words]
        vocab = {chunk: 256 + i for i, chunk in enumerate(ranked)}
        return cls(vocab=vocab)

    @property
    def vocab_size(self) -> int:
        """Total ids: 256 raw bytes + learned words + 4 specials."""
        return 256 + len(self.vocab) + 4

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _chunks(text):
            known = self.vocab.get(chunk)
            ids.append(known) if known is not None else ids.extend(chunk.encode("utf-8"))
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        buf = bytearray()

        def flush() -> None:
            if buf:
                parts.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()

        for i in ids:
            word = self.by_id.get(i)
            if word is not None:
                flush()
                parts.append(word)
            elif 0 <= i < 256:
                buf.append(i)
        flush()
        return "".join(parts)


@dataclass
class ByteTokenizer:
    """Adapts `babble.tokenizer`'s raw-byte functions to the same protocol as
    `BPETokenizer`/`WordTokenizer`, so `experiments/tokenizer_sweep.py` can
    train/eval all three through one code path instead of three. `specials`
    lands on 256..259 -- `SpecialIds.above(0)`, an empty learned vocab -- which
    is byte-for-byte `babble.tokenizer.PAD_ID`..`EOS_ID`, so this is not a
    reimplementation with room to drift, just the same ids under a shared
    interface.
    """

    specials: SpecialIds = field(default_factory=lambda: SpecialIds.above(0))

    @property
    def vocab_size(self) -> int:
        return bytetok.VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        return bytetok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return bytetok.decode(ids)


# --- the corpus layout, generalised over either tokenizer ------------------
#
# Mirrors `babble.tokenizer.text_examples` / `build_text_example` exactly,
# just parameterised by a tokenizer's `.encode()` and `.specials` instead of
# the fixed byte ids -- same `<bos> chunk [<eos>]` layout, same masking rule.

Tokenizer = BPETokenizer | WordTokenizer | ByteTokenizer


def text_budget(block_size: int) -> int:
    budget = block_size - 2
    if budget < 1:
        raise ValueError(f"block_size {block_size} is too small to hold an example")
    return budget


def build_text_example(tok: Tokenizer, chunk: list[int], *, final: bool = True) -> Example:
    tokens = [tok.specials.bos, *chunk] + ([tok.specials.eos] if final else [])
    mask = [0] + [1] * (len(tokens) - 1)
    return Example(tokens=tokens, mask=mask)


def text_examples(tok: Tokenizer, text: str, block_size: int) -> list[Example]:
    budget = text_budget(block_size)
    ids = tok.encode(text)
    if not ids:
        return []
    chunks = [ids[i : i + budget] for i in range(0, len(ids), budget)]
    return [
        build_text_example(tok, chunk, final=index == len(chunks) - 1)
        for index, chunk in enumerate(chunks)
    ]


def text_prefix_budget(block_size: int) -> int:
    reserved = max(1, block_size // 4)
    return max(1, text_budget(block_size) - reserved)


def text_context(tok: Tokenizer, prefix: str, block_size: int) -> list[int]:
    """The inference-time prefix: `<bos> prefix`, truncated the same way
    `text_examples` truncates a training example, keeping the tail."""
    ids = tok.encode(prefix)[-text_prefix_budget(block_size) :]
    return [tok.specials.bos, *ids]


def build_continuation_example(
    tok: Tokenizer, prefix: str, continuation: str, block_size: int
) -> Example:
    """`<bos> prefix continuation <eos>`, with only the continuation trained.

    Generalises `babble.tokenizer.build_continuation_example` over any
    `Tokenizer` (byte, BPE, or word) instead of the fixed byte ids, so a
    post-train stage can teach prompt/response pairs on top of a checkpoint
    that was pretrained with a different tokenizer than `babble.tokenizer`'s
    raw bytes -- same layout, same truncation rule, just parameterised.
    """
    prefix_ids = tok.encode(prefix)[-text_prefix_budget(block_size) :]
    cont_ids = tok.encode(continuation)[: text_budget(block_size) - len(prefix_ids)]
    tokens = [tok.specials.bos, *prefix_ids, *cont_ids, tok.specials.eos]
    mask = [0] * (len(prefix_ids) + 1) + [1] * (len(cont_ids) + 1)
    assert len(tokens) == len(mask)
    return Example(tokens=tokens, mask=mask)


def stack_examples(
    examples: list[Example], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad `examples` to the batch's longest, using `pad_id`.

    `babble.trainer._stack_examples` hardcodes byte-tokenizer `PAD_ID` (256),
    which is actively wrong for a learned tokenizer: BPE merge ids start at
    256 too, so padding with a bare 256 would pad with the first learned
    merge rather than a pad token. This takes the tokenizer's own pad id
    instead, so it works for any `Tokenizer`.
    """
    width = max(len(e.tokens) for e in examples)
    tokens = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, example in enumerate(examples):
        length = len(example.tokens)
        tokens[i, :length] = torch.as_tensor(example.tokens, dtype=torch.long)
        mask[i, :length] = torch.as_tensor(example.mask, dtype=torch.long)
    weights = torch.as_tensor([e.weight for e in examples], dtype=torch.float32)
    return tokens, mask, weights
