"""The commands in the README's quickstart, exercised the way a person runs them."""

from __future__ import annotations

import pytest

from babble.cli import build_parser, main
from babble.config import Settings
from babble.consent import ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.identity import Pseudonymiser
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id


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


def test_train_runs_end_to_end_through_the_cli_with_no_external_download(cli, capsys, monkeypatch):
    monkeypatch.setenv("BABBLE_TRAIN_TRIGGER_ROWS", "100")  # so nothing auto-fires without --force

    # No consented rows yet: a clean no-op, no external download, no base.pt.
    assert main(["train", "--force"]) == 0
    assert "Nothing to train on" in capsys.readouterr().out

    settings = Settings.from_env()
    assert not (settings.checkpoint_dir / "base.pt").exists()

    # Seed some consented human rows, then pretrain from random init -> latest.pt.
    ids = Pseudonymiser.load(settings)
    ConsentStore(settings.consent_path).grant("human-1", "corpus")
    store = CorpusStore(settings.corpus_path)
    author = ids.user("human-1")
    for text in ("hey there", "wacky wacky", "babble on"):
        store.append(CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=SOURCE_MENTION))

    assert main(["train", "--force", "--steps", "4"]) == 0
    assert settings.latest_checkpoint.exists()
    assert not (settings.checkpoint_dir / "base.pt").exists()

    assert main(["train-status"]) == 0
    out = capsys.readouterr().out
    assert "corpus rows" in out and "threshold" in out


def test_summary_works_on_a_fresh_install(cli, capsys):
    assert main(["summary"]) == 0

    out = capsys.readouterr().out
    assert "step 0" in out


def test_the_readme_quickstart_runs_end_to_end(cli, capsys):
    assert main(["fake-data"]) == 0
    assert main(["train", "--force", "--steps", "4", "--seed", "1"]) == 0
    assert main(["sample", "--prompt", "hello"]) == 0
    assert main(["curve"]) == 0
    assert main(["summary"]) == 0
    assert main(["logs", "-n", "5"]) == 0
    assert main(["export"]) == 0

    out = capsys.readouterr().out
    assert "fake row" in out
    assert "flattened" in out  # fake-data also seeds the corpus, not just the pairs
    assert "checkpoint" in out
    assert "corpus" in out  # the export breakdown names both configs
    assert "corrections" in out
    assert "not pushed" in out
    assert (cli / "export" / "data" / "corpus.jsonl").exists()
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
    assert main(["train", "--force", "--steps", "2"]) == 0

    assert "fake-data" in capsys.readouterr().out


def test_train_without_force_is_a_no_op_below_the_trigger_threshold(cli, capsys):
    main(["fake-data"])  # a handful of rows -- nowhere near the default threshold

    assert main(["train", "--steps", "2"]) == 0

    out = capsys.readouterr().out
    assert "Not due" in out
    assert "--force" in out


def test_logs_are_written_and_readable_back(cli, capsys):
    main(["fake-data"])
    main(["train", "--force", "--steps", "2", "--seed", "1", "--quiet"])
    capsys.readouterr()

    assert main(["logs", "-n", "50", "--json"]) == 0

    out = capsys.readouterr().out
    assert "train.checkpoint" in out
    assert '"component": "trainer"' in out


# --- backfill-corpus ------------------------------------------------------


def test_backfill_corpus_appears_in_the_parser():
    args = build_parser().parse_args(["backfill-corpus"])

    assert args.command == "backfill-corpus"


def _seed_one_correction_pair(cli) -> None:
    """A single consented correction row, filed directly -- not through fake-data,
    which already runs the backfill itself and would leave nothing left to add."""
    settings = Settings.from_env()
    settings.ensure_dirs()
    ids = Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    consent.grant("asker-1")
    consent.grant("helper-1")
    asker, helper = ids.user("asker-1"), ids.user("helper-1")
    InteractionStore(settings.interactions_path).append(
        Interaction(
            id=make_row_id(CORRECTION, "hi", "hey", asker, helper),
            signal=CORRECTION,
            prompt="hi",
            rejected="junk",
            chosen="hey",
            prompt_author=asker,
            signal_author=helper,
            weight=1.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )


def test_backfill_corpus_adds_rows_the_first_time_and_nothing_the_second(cli, capsys):
    _seed_one_correction_pair(cli)

    assert main(["backfill-corpus"]) == 0
    first = capsys.readouterr().out
    assert "added 2 corpus row(s)" in first  # the prompt half and the chosen half

    assert main(["backfill-corpus"]) == 0
    second = capsys.readouterr().out
    assert "added 0 corpus row(s)" in second
    assert "already in the corpus" in second


# --- rescan-blocklist -------------------------------------------------------


def test_rescan_blocklist_reports_both_stores(cli, capsys, monkeypatch):
    blocklist_path = cli / "blocklist.txt"
    blocklist_path.write_text("badword\n", encoding="utf-8")
    monkeypatch.setenv("BABBLE_BLOCKLIST_PATH", str(blocklist_path))

    settings = Settings.from_env()
    settings.ensure_dirs()
    ids = Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    consent.grant("alice-1")
    author = ids.user("alice-1")

    InteractionStore(settings.interactions_path).append(
        Interaction(
            id=make_row_id(CORRECTION, "hi", "a real badword", author, author),
            signal=CORRECTION,
            prompt="hi",
            rejected="junk",
            chosen="a real badword",
            prompt_author=author,
            signal_author=author,
            weight=1.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    CorpusStore(settings.corpus_path).append(
        CorpusRow(
            id=make_corpus_id("a real badword", author),
            text="a real badword",
            author=author,
            source=SOURCE_MENTION,
        )
    )

    assert main(["rescan-blocklist"]) == 0

    out = capsys.readouterr().out
    assert "purged 1 correction row(s) and 1 corpus row(s)" in out
