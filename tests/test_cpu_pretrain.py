"""Vocab-mismatch guard and Discord-Dialogues row flattening for cpu_pretrain."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babble.model import Babbler, ModelConfig
from babble.subword import BPETokenizer

from cpu_pretrain import (
    ResumeError,
    RESUME_REQUIRED_KEYS,
    VocabMismatch,
    assert_vocab_matches,
    row_to_text,
    save_checkpoint,
)


def test_row_to_text_prefers_text_field():
    assert row_to_text({"text": "hello there", "id": "x"}) == "hello there"


def test_row_to_text_flattens_chat_turns():
    row = {
        "conversations": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
    }
    # conversations is not in the preferred keys; messages/content/text are.
    # list under `messages`:
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
    }
    text = row_to_text(row)
    assert "hi" in text and "hey" in text


def test_save_refuses_vocab_mismatch(tmp_path):
    tok = BPETokenizer.train(["hello there friend " * 20], num_merges=8)
    model = Babbler(
        ModelConfig(vocab_size=260, n_layer=1, n_head=1, n_embd=16, block_size=32)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(VocabMismatch, match="refusing to write"):
        save_checkpoint(tmp_path, model, opt, tok, step=1, tokens_consumed=10, docs_consumed=1, loss=1.0, keep=1)
    assert not list(tmp_path.glob("*.pt"))


def test_save_writes_when_vocab_matches(tmp_path):
    tok = BPETokenizer.train(["hello there friend " * 20], num_merges=8)
    model = Babbler(
        ModelConfig(
            vocab_size=tok.vocab_size, n_layer=1, n_head=1, n_embd=16, block_size=32
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path, model, opt, tok, step=3, tokens_consumed=99, docs_consumed=2, loss=2.5, keep=2
    )
    assert path.exists()
    latest = tmp_path / "latest.pt"
    assert latest.exists()
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["step"] == 3
    assert payload["tokens_consumed"] == 99
    assert payload["config"]["vocab_size"] == tok.vocab_size


def test_assert_vocab_matches_ok():
    tok = BPETokenizer.train(["aa bb cc dd ee ff"], num_merges=4)
    model = Babbler(
        ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, n_embd=8, block_size=16)
    )
    assert_vocab_matches(model, tok)


def test_resume_required_keys_cover_served_payload():
    served_keys = {
        "config",
        "docs_consumed",
        "loss",
        "model",
        "optim",
        "saved_at",
        "step",
        "tokens_consumed",
        "torch_rng",
    }
    assert set(RESUME_REQUIRED_KEYS) <= served_keys
    assert ResumeError is ResumeError
