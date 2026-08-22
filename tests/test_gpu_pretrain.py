"""gpu-pretrain package: config, schema, resume-from-checkpoint, CPU smoke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu-pretrain"
sys.path.insert(0, str(GPU))

from train import (  # noqa: E402
    LOSS_KEYS,
    Babbler,
    BPETokenizer,
    CheckpointError,
    ModelConfig,
    RunConfig,
    assert_shape_matches,
    corpus_docs_offset,
    epoch_and_offset,
    load_payload,
    row_to_text,
    save_checkpoint,
)

HOST_PATH = "/home/beckett"


def test_config_parses():
    cfg = RunConfig.from_json(GPU / "config.json")
    assert cfg.vocab_size == 16384
    assert cfg.block_size == 1024
    assert cfg.n_layer == 8
    assert cfg.n_head == 8
    assert cfg.n_embd == 512
    assert cfg.dropout == 0.1
    assert cfg.token_budget == 1_000_000_000
    assert cfg.dataset_repo == "mookiezi/Discord-Dialogues"
    assert len(cfg.dataset_sha256) == 64


def test_shipped_files_have_no_host_absolute_paths():
    skip_suffixes = {".pt", ".png"}
    leaked = []
    for path in GPU.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in skip_suffixes:
            continue
        if any(p in {".venv", ".cache", "run-output", "__pycache__"} for p in path.parts):
            continue
        try:
            text = path.read_text(errors="replace")
        except UnicodeDecodeError:
            continue
        if HOST_PATH in text:
            leaked.append(str(path.relative_to(ROOT)))
    assert leaked == []


def test_row_to_text_matches_cpu_flattening():
    assert row_to_text({"text": "hello there", "id": "x"}) == "hello there"
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
    }
    text = row_to_text(row)
    assert "hi" in text and "hey" in text


def test_tokenizer_vocab_matches_config():
    raw = json.loads((GPU / "tokenizer.json").read_text())
    cfg = RunConfig.from_json(GPU / "config.json")
    assert 256 + len(raw["merges"]) + 4 == cfg.vocab_size


def test_param_count_is_34m_tied():
    cfg = RunConfig.from_json(GPU / "config.json")
    model = Babbler(
        ModelConfig(
            vocab_size=cfg.vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        )
    )
    assert model.num_params() == 34_096_128


def test_epoch_and_offset():
    assert epoch_and_offset(408_581, 7_300_966) == (0, 408_581)
    assert epoch_and_offset(7_300_966, 7_300_966) == (1, 0)
    assert epoch_and_offset(7_300_967, 7_300_966) == (1, 1)


def test_ultra_fineweb_docs_are_not_a_discord_skip():
    discord = "mookiezi/Discord-Dialogues"
    assert corpus_docs_offset(None, 408_581, discord) == 0
    assert corpus_docs_offset("", 408_581, discord) == 0
    assert corpus_docs_offset("openbmb/Ultra-FineWeb-L1", 408_581, discord) == 0
    assert corpus_docs_offset(discord, 12_345, discord) == 12_345


def test_shape_mismatch_is_readable():
    cfg = RunConfig.from_json(GPU / "config.json")
    with pytest.raises(CheckpointError, match="n_embd"):
        assert_shape_matches({**{k: getattr(cfg, k) for k in (
            "vocab_size", "block_size", "n_layer", "n_head", "n_embd"
        )}, "n_embd": 64}, cfg)


def test_missing_checkpoint_exits_nonzero(tmp_path):
    missing = tmp_path / "nope.pt"
    cmd = [
        sys.executable,
        str(GPU / "train.py"),
        "--config",
        str(GPU / "config.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--checkpoint",
        str(missing),
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    blob = proc.stdout + proc.stderr
    assert "start checkpoint not found" in blob
    assert "from scratch" in blob.lower()


def test_resume_payload_counters(tmp_path):
    tok = BPETokenizer.from_json(GPU / "tokenizer.json")
    cfg = RunConfig.from_json(GPU / "config.json")
    model = Babbler(
        ModelConfig(
            vocab_size=tok.vocab_size,
            block_size=32,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_dir = tmp_path / "ckpts"
    save_checkpoint(ckpt_dir, model, opt, tok, 3118, 600_206_202, 408_581, 2.53, 1, "x")
    payload = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 3118
    assert payload["tokens_consumed"] == 600_206_202
    assert payload["docs_consumed"] == 408_581
    with pytest.raises(CheckpointError, match="block_size"):
        assert_shape_matches(payload["config"], cfg)


def test_load_web_checkpoint_resets_discord_offset(tmp_path):
    tok = BPETokenizer.from_json(GPU / "tokenizer.json")
    cfg = RunConfig.from_json(GPU / "config.json")
    model = Babbler(
        ModelConfig(
            vocab_size=cfg.vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ckpt_dir = tmp_path / "ckpts"
    save_checkpoint(
        ckpt_dir, model, opt, tok, 3118, 600_206_202, 408_581, 2.53, 1, "openbmb/Ultra-FineWeb-L1"
    )
    path = ckpt_dir / "latest.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["dataset_id"]
    stripped = tmp_path / "start.pt"
    torch.save(payload, stripped)
    _m, _o, step, tokens, docs = load_payload(stripped, torch.device("cpu"), cfg)
    assert step == 3118
    assert tokens == 600_206_202
    assert docs == 0


@pytest.mark.slow
def test_cpu_smoke_writes_loss_schema(tmp_path):
    out = tmp_path / "smoke-out"
    cmd = [
        sys.executable,
        str(GPU / "train.py"),
        "--config",
        str(GPU / "config.json"),
        "--output-dir",
        str(out),
        "--smoke",
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    loss_path = out / "loss.jsonl"
    assert loss_path.is_file()
    lines = [json.loads(x) for x in loss_path.read_text().splitlines() if x.strip()]
    assert len(lines) >= 1
    last = lines[-1]
    for key in LOSS_KEYS:
        assert key in last, key
    assert last["step"] >= 1
    assert (out / "checkpoints" / "latest.pt").is_file()
    assert (out / "tokenizer.json").is_file()
    assert "send-back" in proc.stdout
    assert "fresh init" in proc.stdout
