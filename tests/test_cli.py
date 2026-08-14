"""The commands in the README's quickstart, exercised the way a person runs them."""

from __future__ import annotations

import pytest

from babble.cli import main


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A throwaway babble installation driven purely through the environment."""
    for name, sub in (
        ("BABBLE_DATA_DIR", "data"),
        ("BABBLE_CHECKPOINT_DIR", "checkpoints"),
        ("BABBLE_EXPORT_DIR", "export"),
        ("BABBLE_LOG_DIR", "logs"),
    ):
        monkeypatch.setenv(name, str(tmp_path / sub))
    monkeypatch.setenv("BABBLE_HASH_SALT", "cli-test-salt")
    for name, value in (
        ("BABBLE_N_LAYER", "2"),
        ("BABBLE_N_HEAD", "2"),
        ("BABBLE_N_EMBD", "32"),
        ("BABBLE_BLOCK_SIZE", "64"),
        ("BABBLE_BATCH_SIZE", "2"),
        ("BABBLE_CHECKPOINT_EVERY", "2"),
        ("BABBLE_MAX_NEW_TOKENS", "16"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("BABBLE_DISCORD_TOKEN", raising=False)
    return tmp_path


def test_no_arguments_prints_help(cli, capsys):
    assert main([]) == 0
    assert "babble" in capsys.readouterr().out


def test_summary_works_on_a_fresh_install(cli, capsys):
    assert main(["summary"]) == 0

    out = capsys.readouterr().out
    assert "step 0" in out


def test_the_readme_quickstart_runs_end_to_end(cli, capsys):
    assert main(["fake-data"]) == 0
    assert main(["train", "--steps", "4", "--seed", "1"]) == 0
    assert main(["sample", "--prompt", "hello"]) == 0
    assert main(["curve"]) == 0
    assert main(["summary"]) == 0
    assert main(["logs", "-n", "5"]) == 0
    assert main(["export"]) == 0

    out = capsys.readouterr().out
    assert "fake row" in out
    assert "checkpoint" in out
    assert "not pushed" in out
    assert (cli / "export" / "data" / "train.jsonl").exists()
    assert (cli / "export" / "README.md").exists()


def test_sample_works_before_any_training_has_happened(cli, capsys):
    assert main(["sample", "--prompt", "hello"]) == 0

    assert "random init" in capsys.readouterr().out


def test_the_bot_refuses_to_start_without_a_token_and_says_so(cli, capsys):
    assert main(["bot"]) == 2

    out = capsys.readouterr().out
    assert "BABBLE_DISCORD_TOKEN" in out
    assert "babble train" in out  # points at what does work


def test_training_with_nothing_to_learn_from_explains_itself(cli, capsys):
    assert main(["train", "--steps", "2"]) == 0

    assert "fake-data" in capsys.readouterr().out


def test_logs_are_written_and_readable_back(cli, capsys):
    main(["fake-data"])
    main(["train", "--steps", "2", "--seed", "1", "--quiet"])
    capsys.readouterr()

    assert main(["logs", "-n", "50", "--json"]) == 0

    out = capsys.readouterr().out
    assert "train.checkpoint" in out
    assert '"component": "trainer"' in out
