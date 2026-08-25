"""`Settings.serve_layout` picks which decode layout `CheckpointGenerator`
serves a checkpoint with -- "continuation" (`generate.best_continuation`, what
an ordinary corpus-trained checkpoint understands) or "pair"
(`generate.best_of`, what a checkpoint SFT'd on prompt/response pairs like
SSH's booper-chat understands). See `babble/config.py` and the
`CheckpointGenerator.__call__` dispatch in `babble/generate.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from babble import generate as generate_module
from babble.config import Settings
from babble.generate import CheckpointGenerator
from babble.model import Babbler, ModelConfig
from babble.tokenizer import VOCAB_SIZE


def _write_ckpt(path: Path, model: Babbler, step: int = 1) -> None:
    torch.save(
        {"step": step, "config": model.config.to_dict(), "model": model.state_dict()},
        path,
    )


def _seed_checkpoint(settings: Settings) -> None:
    model = Babbler(ModelConfig(vocab_size=VOCAB_SIZE, n_layer=2, n_head=2, n_embd=32, block_size=64))
    _write_ckpt(settings.latest_checkpoint, model, step=1)


def test_default_serve_layout_is_continuation() -> None:
    default = dataclasses.fields(Settings)
    field = next(f for f in default if f.name == "serve_layout")
    assert field.default == "continuation"


def test_serve_layout_env_override_reaches_settings(monkeypatch) -> None:
    monkeypatch.setenv("BABBLE_SERVE_LAYOUT", "pair")
    assert Settings.from_env(root=Path("/tmp/does-not-need-to-exist")).serve_layout == "pair"


def test_serve_layout_unset_env_defaults_to_continuation(monkeypatch) -> None:
    monkeypatch.delenv("BABBLE_SERVE_LAYOUT", raising=False)
    assert Settings.from_env(root=Path("/tmp/does-not-need-to-exist")).serve_layout == "continuation"


def test_continuation_layout_routes_to_best_continuation(settings, monkeypatch) -> None:
    _seed_checkpoint(settings)
    settings.serve_layout = "continuation"
    calls = []
    monkeypatch.setattr(
        generate_module,
        "best_continuation",
        lambda model, prompt, **kw: calls.append(("continuation", prompt)) or "reply",
    )
    monkeypatch.setattr(
        generate_module,
        "best_of",
        lambda model, prompt, **kw: calls.append(("pair", prompt)) or "reply",
    )
    gen = CheckpointGenerator(settings)
    result = gen("hello")
    assert result.text == "reply"
    assert calls == [("continuation", "hello")]


def test_pair_layout_routes_to_best_of(settings, monkeypatch) -> None:
    _seed_checkpoint(settings)
    settings.serve_layout = "pair"
    calls = []
    monkeypatch.setattr(
        generate_module,
        "best_continuation",
        lambda model, prompt, **kw: calls.append(("continuation", prompt)) or "reply",
    )
    monkeypatch.setattr(
        generate_module,
        "best_of",
        lambda model, prompt, **kw: calls.append(("pair", prompt)) or "reply",
    )
    gen = CheckpointGenerator(settings)
    result = gen("hello")
    assert result.text == "reply"
    assert calls == [("pair", "hello")]


def test_unknown_serve_layout_raises_with_the_offending_value(settings) -> None:
    _seed_checkpoint(settings)
    settings.serve_layout = "sideways"
    gen = CheckpointGenerator(settings)
    with pytest.raises(ValueError, match="sideways"):
        gen("hello")
