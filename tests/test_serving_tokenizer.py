"""Serving must load the tokenizer that ships with a checkpoint.

A byte-level `latest.pt` (no tokenizer.json) keeps working; a BPE checkpoint
is only servable when `tokenizer.json` sits beside it. Hardcoding a swap
would make rollback impossible.
"""

from __future__ import annotations

import pytest
import torch

from babble.generate import CheckpointGenerator, continue_text, load_model, tokenizer_for_checkpoint
from babble.model import Babbler, ModelConfig
from babble.subword import BPETokenizer, ByteTokenizer
from babble.tokenizer import VOCAB_SIZE


def _write_ckpt(path, model: Babbler, step: int = 1) -> None:
    torch.save(
        {"step": step, "config": model.config.to_dict(), "model": model.state_dict()},
        path,
    )


def test_byte_checkpoint_without_json_uses_byte_tokenizer(settings):
    model = Babbler(ModelConfig(vocab_size=VOCAB_SIZE, n_layer=2, n_head=2, n_embd=32, block_size=64))
    _write_ckpt(settings.latest_checkpoint, model, step=7)
    loaded, step = load_model(settings)
    assert step == 7
    assert isinstance(loaded.tokenizer, ByteTokenizer)
    text = continue_text(loaded, "hi", max_new_tokens=8, temperature=0.0)
    assert isinstance(text, str)


def test_bpe_checkpoint_loads_sidecar_tokenizer_and_decodes_merges(settings, tmp_path):
    tok = BPETokenizer.train(["hello there friend", "the cat sat"], num_merges=20)
    tok.to_json(settings.tokenizer_path)
    model = Babbler(
        ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=64)
    )
    _write_ckpt(settings.latest_checkpoint, model, step=99)
    loaded, step = load_model(settings)
    assert step == 99
    assert isinstance(loaded.tokenizer, BPETokenizer)
    assert loaded.tokenizer.vocab_size == tok.vocab_size
    text = continue_text(loaded, "hello", max_new_tokens=8, temperature=0.0)
    assert isinstance(text, str)
    # Byte decode would drop every merge id (>=256). Serving must keep them.
    assert tokenizer_for_checkpoint(settings.latest_checkpoint, tok.vocab_size).vocab_size == tok.vocab_size


def test_bpe_checkpoint_without_json_is_refused(settings):
    tok = BPETokenizer.train(["hello there friend"], num_merges=8)
    model = Babbler(
        ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=64)
    )
    _write_ckpt(settings.latest_checkpoint, model)
    with pytest.raises(ValueError, match="tokenizer.json"):
        load_model(settings)


def test_mismatched_tokenizer_json_is_refused(settings):
    tok = BPETokenizer.train(["hello there friend"], num_merges=20)
    other = BPETokenizer.train(["zzzz something else entirely"], num_merges=4)
    other.to_json(settings.tokenizer_path)
    model = Babbler(
        ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=64)
    )
    _write_ckpt(settings.latest_checkpoint, model)
    with pytest.raises(ValueError, match="vocab_size"):
        load_model(settings)


def test_byte_checkpoint_can_roll_back_after_bpe_sidecar(settings):
    """Putting a byte checkpoint back and removing tokenizer.json serves bytes again."""
    tok = BPETokenizer.train(["hello there friend"], num_merges=12)
    tok.to_json(settings.tokenizer_path)
    bpe = Babbler(
        ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=64)
    )
    _write_ckpt(settings.latest_checkpoint, bpe, step=2)
    gen = CheckpointGenerator(settings)
    assert gen.step == 2
    assert isinstance(gen._model.tokenizer, BPETokenizer)

    settings.tokenizer_path.unlink()
    byte = Babbler(ModelConfig(vocab_size=VOCAB_SIZE, n_layer=2, n_head=2, n_embd=32, block_size=64))
    _write_ckpt(settings.latest_checkpoint, byte, step=3)
    assert gen.step == 3
    assert isinstance(gen._model.tokenizer, ByteTokenizer)
