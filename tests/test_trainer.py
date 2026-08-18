"""The trainer: learns from random init on the human corpus, checkpoints, and
survives being killed. `force=True` is threaded through most calls here
because the +N-row trigger (tested in test_train_trigger.py) would otherwise
make a tiny test corpus a no-op."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time

import pytest
import torch

from babble.blocklist import Blocklist
from babble.consent import ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.discord_feed import TrainingFeed
from babble.fakedata import seed_fake_data
from babble.identity import Pseudonymiser
from babble.tokenizer import BOS_ID, EOS_ID, PAD_ID, build_example
from babble.trainer import (
    PROBE_FALLBACK,
    PROBE_PREFIX_BYTES,
    PROBE_TRAIN,
    SAMPLE_PREFIXES,
    SCRATCH_DIR,
    corpus_rows,
    dataset_stats,
    distinct_texts,
    leading_words,
    make_batch,
    probe_prefix,
    split_rows,
    sweep_scratch,
    to_examples,
    train,
)


def _row(text: str, *, row_id: str, author: str = "a") -> CorpusRow:
    """A bare corpus row, in memory only -- for tests that exercise pure
    functions over `list[CorpusRow]` and never touch a store on disk."""
    return CorpusRow(
        id=row_id,
        text=text,
        author=author,
        source=SOURCE_MENTION,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _seed_corpus_row(settings, text: str, author: str, *, source: str = SOURCE_MENTION) -> CorpusRow:
    """Write one row straight into the corpus store, content-addressed the same
    way the real capture path builds an id."""
    row = CorpusRow(
        id=make_corpus_id(text, author),
        text=text,
        author=author,
        source=source,
        created_at="2026-01-01T00:00:00+00:00",
    )
    CorpusStore(settings.corpus_path).append(row)
    return row


@pytest.fixture
def seeded(settings):
    seed_fake_data(settings)
    return settings


class FakeSender:
    """Records what the trainer tried to post; can be made to explode."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, url: str, content: str) -> None:
        if self.fail:
            raise ConnectionError("discord is unreachable")
        self.calls.append((url, content))


# --- batching -----------------------------------------------------------


def test_a_batch_is_padded_to_its_longest_example():
    examples = [build_example("a", "b", 64), build_example("a", "much longer answer", 64)]

    tokens, mask, weights = make_batch(examples, batch_size=4, rng=random.Random(0))

    assert tokens.shape == mask.shape == (4, max(len(e) for e in examples))
    assert weights.shape == (4,)
    for row, row_mask in zip(tokens, mask):
        padding = row == PAD_ID
        assert (row_mask[padding] == 0).all(), "padding must never be a target"


# --- the run --------------------------------------------------------------


def test_it_trains_from_random_init_and_writes_checkpoints(seeded):
    result = train(seeded, force=True, steps=6, echo=False, seed=1)

    assert result.steps_run == 6
    assert result.checkpoints_written >= 1
    assert seeded.latest_checkpoint.exists()
    assert list(seeded.checkpoint_dir.glob("ckpt-*.pt"))


def test_every_checkpoint_records_a_loss_and_a_sample(seeded):
    train(seeded, force=True, steps=6, echo=False, seed=1)

    entries = [
        json.loads(line)
        for line in seeded.loss_curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert entries
    for entry in entries:
        assert entry["step"] > 0
        assert isinstance(entry["loss"], float)
        assert "sample" in entry  # the babble is a first-class output
        assert "at" in entry


def test_the_loss_goes_down(seeded):
    seeded.checkpoint_every = 40
    train(seeded, force=True, steps=80, echo=False, seed=1)

    entries = [
        json.loads(line)
        for line in seeded.loss_curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert entries[-1]["loss"] < entries[0]["loss"]


def test_it_prints_the_sample_generation_per_checkpoint(seeded, capsys):
    train(seeded, force=True, steps=4, echo=True, seed=1)

    printed = capsys.readouterr().out
    assert "loss" in printed and "step" in printed and "->" in printed


def test_half_written_checkpoints_are_staged_out_of_the_checkpoint_directory(seeded):
    """`os.replace` was always atomic, so a torn file was never loadable -- but
    it was *visible*, one glob away from the real checkpoints. Staging keeps the
    checkpoint directory containing only whole files, however long a write takes
    or whenever the process dies during one.
    """
    train(seeded, force=True, steps=2, echo=False, seed=1)

    scratch = seeded.checkpoint_dir / SCRATCH_DIR
    assert scratch.is_dir(), "writes go through a scratch directory"
    assert list(scratch.iterdir()) == [], "and nothing is left in it afterwards"
    for entry in seeded.checkpoint_dir.iterdir():
        assert entry.is_dir() or entry.suffix == ".pt" or entry.name in ("loss.jsonl", "train_state.json")


def test_a_leftover_partial_write_is_swept_on_the_next_start(seeded):
    train(seeded, force=True, steps=2, echo=False, seed=1)
    scratch = seeded.checkpoint_dir / SCRATCH_DIR
    (scratch / "ckpt-0000099.pt.tmp").write_bytes(b"a write that was killed halfway")

    train(seeded, force=True, steps=2, echo=False, seed=1)

    assert list(scratch.iterdir()) == []
    assert seeded.latest_checkpoint.exists()


def test_sweeping_scratch_never_touches_a_real_checkpoint(seeded):
    train(seeded, force=True, steps=2, echo=False, seed=1)
    before = sorted(p.name for p in seeded.checkpoint_dir.glob("*.pt"))

    sweep_scratch(seeded)

    assert sorted(p.name for p in seeded.checkpoint_dir.glob("*.pt")) == before


def test_old_checkpoints_are_pruned_but_latest_survives(seeded):
    seeded.keep_checkpoints = 2
    seeded.checkpoint_every = 2

    train(seeded, force=True, steps=10, echo=False, seed=1)

    assert len(list(seeded.checkpoint_dir.glob("ckpt-*.pt"))) <= 2
    assert seeded.latest_checkpoint.exists()


def test_with_no_data_it_stops_politely_instead_of_crashing(settings):
    result = train(settings, force=True, steps=5, echo=False)

    assert result.stopped_because == "no_data"
    assert result.steps_run == 0


def test_killing_the_trainer_mid_run_leaves_a_loadable_checkpoint(settings, tmp_path):
    """The real thing: SIGKILL a separate process, then check the checkpoint on
    disk is whole -- there is no resume to check any more, every run starts
    from random init, so this is purely about atomic-write safety."""
    seed_fake_data(settings)
    env = {
        **os.environ,
        "BABBLE_DATA_DIR": str(settings.data_dir),
        "BABBLE_CHECKPOINT_DIR": str(settings.checkpoint_dir),
        "BABBLE_LOG_DIR": str(settings.log_dir),
        "BABBLE_HASH_SALT": settings.salt,
        "BABBLE_N_LAYER": "2",
        "BABBLE_N_HEAD": "2",
        "BABBLE_N_EMBD": "32",
        "BABBLE_BLOCK_SIZE": "64",
        "BABBLE_BATCH_SIZE": "2",
        "BABBLE_CHECKPOINT_EVERY": "2",
        # Validation off: a huge step budget is only a safe "stays alive until
        # killed" guarantee if best-val early stopping can never cut it short.
        "BABBLE_VAL_MIN_ROWS": "100000",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "babble", "train", "--force", "--quiet", "--steps", "100000"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 90
        while not settings.latest_checkpoint.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("trainer exited before writing a checkpoint")
            time.sleep(0.1)
        assert settings.latest_checkpoint.exists(), "no checkpoint appeared in time"
        time.sleep(0.5)  # let it get well into the next step
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()

    # The file must be whole despite being killed without warning.
    payload = torch.load(settings.latest_checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] > 0
    assert not list(settings.checkpoint_dir.glob("*.tmp")), "a torn temp file was left behind"
    assert not list(settings.checkpoint_dir.glob("ckpt-*.pt.tmp"))


# --- the discord training feed -------------------------------------------


def test_a_configured_feed_gets_a_post_per_checkpoint(seeded):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    result = train(seeded, force=True, steps=6, echo=False, seed=1, feed=feed)

    assert result.checkpoints_written >= 1
    checkpoint_posts = [c for c in sender.calls if "🔁" in c[1]]
    assert len(checkpoint_posts) == result.checkpoints_written


def test_a_failing_feed_post_never_breaks_training(seeded):
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=FakeSender(fail=True))

    result = train(seeded, force=True, steps=6, echo=False, seed=1, feed=feed)

    assert result.steps_run == 6
    assert result.checkpoints_written >= 1
    assert seeded.latest_checkpoint.exists()


def test_an_unconfigured_feed_makes_no_network_calls(seeded, monkeypatch):
    monkeypatch.delenv("BABBLE_LOG_WEBHOOK_URL", raising=False)

    def explode(*a, **k):
        raise AssertionError("should never be called when unconfigured")

    monkeypatch.setattr("babble.discord_feed.post_webhook", explode)

    result = train(seeded, force=True, steps=6, echo=False, seed=1)  # feed defaults from env: disabled

    assert result.steps_run == 6


def test_start_is_always_reported_as_started_never_resumed(seeded):
    """Every run is a fresh random-init model -- there is nothing to resume,
    so the start post must never claim otherwise."""
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, force=True, steps=2, echo=False, seed=1, feed=feed)

    starts = [c for c in sender.calls if "started" in c[1].lower() or "resum" in c[1].lower()]
    assert len(starts) == 1
    assert "started" in starts[0][1].lower()
    assert "resum" not in starts[0][1].lower()


def test_going_idle_posts_once_not_every_check(settings):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(settings, force=True, steps=2, echo=False, feed=feed)  # no data at all -> idle immediately

    idle_posts = [c for c in sender.calls if "idle" in c[1].lower()]
    assert len(idle_posts) == 1


def test_the_feed_carries_cycle_step_loss_delta_rows_and_sample(seeded):
    seeded.checkpoint_every = 3
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, force=True, steps=6, echo=False, seed=1, feed=feed)

    checkpoint_posts = [c for c in sender.calls if "loss" in c[1].lower()]
    assert len(checkpoint_posts) >= 2
    first, second = checkpoint_posts[0][1], checkpoint_posts[1][1]
    assert "rows" in first.lower()
    # the second post carries a delta against the first checkpoint's loss
    assert "(" in second and ("+" in second or "-" in second)


def test_a_cycle_start_post_shows_the_dataset_shape_and_hyperparams(seeded):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, force=True, steps=2, echo=False, seed=1, feed=feed)

    starts = [c[1] for c in sender.calls if "starting" in c[1].lower()]
    assert len(starts) == 1
    # 24 corpus rows: the 12 fake corrections flattened into a prompt row and a
    # chosen row apiece, all from consenting fake users, none blocklisted.
    assert "24 stored" in starts[0]
    assert "24 training" in starts[0]
    assert "0 dropped" in starts[0]
    assert "19 train" in starts[0]
    assert "5 val" in starts[0]
    assert "examples" in starts[0].lower()
    assert "tokens" in starts[0].lower()
    assert "batch" in starts[0].lower()


def test_a_cycle_end_post_shows_steps_and_duration(seeded):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, force=True, steps=3, echo=False, seed=1, feed=feed)

    ends = [c[1] for c in sender.calls if "done" in c[1].lower()]
    assert len(ends) == 1
    assert "3 steps" in ends[0]


def test_checkpoints_probe_different_real_prompts_across_a_cycle(seeded):
    seeded.checkpoint_every = 1
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, force=True, steps=4, echo=False, seed=1, feed=feed)

    checkpoint_posts = [c[1] for c in sender.calls if "🔁" in c[1]]
    assert len(checkpoint_posts) == 4
    probed = [post.split("`")[1] for post in checkpoint_posts]
    assert len(set(probed)) > 1  # not stuck on one or two hardcoded phrases
    for a, b in zip(probed, probed[1:]):
        assert a != b  # never the same prefix twice in a row
    # There is no answer in a corpus row, so the feed must never invent one --
    # "continuation" is what actually happened, "expected" would be a lie.
    assert all("continuation" in post.lower() for post in checkpoint_posts)
    assert not any("expected" in post.lower() for post in checkpoint_posts)


def test_the_probe_says_which_side_of_the_split_the_row_came_from(seeded):
    """Identical garbage means opposite things on a trained row and a held-out
    one. Without the label nobody reading the feed can tell which they are
    looking at, so the probe output answers nothing.
    """
    seeded.checkpoint_every = 1
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, force=True, steps=2, echo=False, seed=1, feed=feed)

    checkpoint_posts = [c[1] for c in sender.calls if "🔁" in c[1]]
    assert checkpoint_posts
    assert all(PROBE_TRAIN in post for post in checkpoint_posts)


def test_the_probe_only_ever_asks_about_rows_it_was_trained_on(seeded, read_log):
    """The probe walks the train split, so the memorisation question stays
    answerable at every single checkpoint.
    """
    seeded.checkpoint_every = 1
    seeded.val_min_rows = 5  # force a real held-out split on the fake corpus

    train(seeded, force=True, steps=6, echo=False, seed=1)

    checkpoints = read_log("train.checkpoint")
    assert checkpoints
    split = split_rows(corpus_rows(seeded), seeded)
    trained_texts = [row.text for row in split.train]
    held_out_texts = [row.text for row in split.val]
    assert held_out_texts, "this test is meaningless without a real holdout"
    for entry in checkpoints:
        assert entry["probe_side"] == PROBE_TRAIN
        prefix = entry["prefix"]
        assert any(text.startswith(prefix) for text in trained_texts)
        assert not any(text.startswith(prefix) for text in held_out_texts)


# --- dataset visibility ---------------------------------------------------


def test_probe_prefix_walks_the_whole_dataset_in_order_then_wraps():
    rows = [_row(f"row-text-{i}", row_id=str(i)) for i in range(4)]

    walked = [probe_prefix(rows, i) for i in range(4)]

    assert walked == [(f"row-text-{i}", PROBE_TRAIN) for i in range(4)]
    assert probe_prefix(rows, 4) == probe_prefix(rows, 0)
    assert probe_prefix(rows, 7) == probe_prefix(rows, 3)


def test_probe_prefix_never_repeats_two_checkpoints_in_a_row():
    rows = [_row(f"row-text-{i}", row_id=str(i)) for i in range(5)]

    probed = [probe_prefix(rows, i)[0] for i in range(20)]

    for a, b in zip(probed, probed[1:]):
        assert a != b


def test_probe_prefix_dedupes_a_row_whose_text_repeats():
    rows = [
        _row("boop", row_id="1", author="a"),
        _row("boop", row_id="2", author="b"),  # same writing, different author
        _row("git-good", row_id="3"),
    ]

    assert distinct_texts(rows) == ["boop", "git-good"]
    assert probe_prefix(rows, 0) == ("boop", PROBE_TRAIN)
    assert probe_prefix(rows, 1) == ("git-good", PROBE_TRAIN)
    assert probe_prefix(rows, 2) == ("boop", PROBE_TRAIN)  # wraps back around


def test_probe_prefix_falls_back_to_a_hardcoded_prefix_when_theres_nothing_to_probe():
    assert probe_prefix([], 0) == (SAMPLE_PREFIXES[0], PROBE_FALLBACK)
    assert probe_prefix([], 1) == (SAMPLE_PREFIXES[1], PROBE_FALLBACK)
    assert probe_prefix([], 2) == (SAMPLE_PREFIXES[0], PROBE_FALLBACK)


def test_the_probe_prefix_holds_a_word_back_so_there_is_something_to_continue():
    """A "continuation" of the whole row only shows the model agreeing it ended."""
    assert leading_words("hello there how are you") == "hello there how are"


@pytest.mark.parametrize(
    "text",
    [
        "a" * 500,  # one enormous word
        "日本語のテキストです これは長いです",  # multi-byte, over budget in bytes not chars
        "supercalifragilisticexpialidocious antidisestablishmentarianism",
    ],
)
def test_the_probe_prefix_respects_its_byte_budget_even_on_the_very_first_word(text):
    """The budget has to apply to the first word too.

    Exempting it -- which is what a `if prefix and over_budget` guard does --
    means a row that is one 500-byte word comes back whole: the "prefix" is the
    entire row, the model has nothing left to continue, and the feed line is a
    wall of text. Byte length, not character length, is what the budget is in.
    """
    prefix = leading_words(text)

    assert prefix
    assert len(prefix.encode("utf-8")) <= PROBE_PREFIX_BYTES
    assert text.startswith(prefix)


def test_a_row_with_nothing_usable_in_it_is_probed_as_a_fallback_not_as_trained():
    """`probe_side` is a claim about where the prefix came from, so it must not
    say "trained" when the prefix is a hardcoded string the model never saw.
    """
    prefix, side = probe_prefix([_row("   ", row_id="1")], 0)

    assert (prefix, side) == (SAMPLE_PREFIXES[0], PROBE_FALLBACK)


def test_dataset_stats_accounts_for_every_stored_row(settings):
    ids = Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    good = ids.user("good-user")
    stranger = ids.user("never-asked")
    consent.grant("good-user")  # grants both the corrections and corpus scopes

    for i in range(11):
        _seed_corpus_row(settings, f"row text {i}", good)
    _seed_corpus_row(settings, "hi from a stranger", stranger)
    _seed_corpus_row(settings, "a badword right here", good)

    blocklist = Blocklist(frozenset({"badword"}))
    stats = dataset_stats(settings, ids, blocklist)

    assert stats.stored == 13
    assert stats.trained == 11
    assert stats.dropped_consent == 1
    assert stats.dropped_blocklist == 1


# --- consent at training time -------------------------------------------


def test_rows_from_people_who_never_consented_are_not_trained_on(settings):
    ids = Pseudonymiser.load(settings)
    stranger = ids.user("someone-who-never-agreed")
    _seed_corpus_row(settings, "hi", stranger)

    assert corpus_rows(settings) == []
    assert train(settings, force=True, steps=2, echo=False).stopped_because == "no_data"


def test_withdrawing_consent_takes_rows_out_of_training(seeded):
    """A corpus row has exactly one author, so withdrawing takes out that one
    person's rows -- not the whole corpus, which the pair-level rule used to.
    """
    from babble.fakedata import FAKE_USER

    ids = Pseudonymiser.load(seeded)
    withdrawn_author = ids.user(FAKE_USER)
    before = corpus_rows(seeded)
    assert any(row.author == withdrawn_author for row in before)

    ConsentStore(seeded.consent_path).withdraw(FAKE_USER)

    after = corpus_rows(seeded)
    assert len(after) < len(before)
    assert all(row.author != withdrawn_author for row in after)


# --- the plain next-token objective ---------------------------------------


def test_to_examples_masks_nothing_after_the_bos_and_weights_every_row_equally():
    """No prompt to hold back, no per-row weighting: every token past <bos> is
    a target and every example counts the same as every other one.
    """
    rows = [
        _row("a short row", row_id="1", author="a"),
        _row("a somewhat longer row of plain text to tokenise", row_id="2", author="b"),
    ]

    examples = to_examples(rows, block_size=64)

    assert len(examples) == 2
    for example in examples:
        assert example.mask == [0] + [1] * (len(example.tokens) - 1)
        assert example.weight == 1.0


def test_a_row_longer_than_the_block_becomes_several_examples_covering_all_of_it():
    """A long row is chunked, not truncated: nothing about the input written to
    the bot is thrown away just because it did not fit in one block.
    """
    text = "abcdefghij " * 40  # far longer than any block used in these tests
    rows = [_row(text, row_id="1", author="a")]

    examples = to_examples(rows, block_size=64)

    assert len(examples) > 1
    rebuilt = bytearray()
    for index, example in enumerate(examples):
        tokens = example.tokens
        assert tokens[0] == BOS_ID
        body = tokens[1:]
        is_last = index == len(examples) - 1
        assert (body[-1] == EOS_ID) == is_last
        if is_last:
            body = body[:-1]
        rebuilt.extend(body)
    assert bytes(rebuilt) == text.encode("utf-8")
