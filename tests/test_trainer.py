"""The trainer: learns from random init, checkpoints, and survives being killed."""

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
from babble.discord_feed import TrainingFeed
from babble.fakedata import seed_fake_data
from babble.identity import Pseudonymiser
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.tokenizer import PAD_ID, build_example
from babble.trainer import (
    SAMPLE_PROMPTS,
    consented_rows,
    dataset_stats,
    distinct_prompts,
    make_batch,
    probe_prompt,
    train,
)


def _row(prompt: str, chosen: str, *, row_id: str, author: str = "a") -> Interaction:
    return Interaction(
        id=row_id,
        signal=CORRECTION,
        prompt=prompt,
        rejected="junk",
        chosen=chosen,
        prompt_author=author,
        signal_author=author,
        weight=1.0,
        created_at="2026-01-01T00:00:00+00:00",
    )


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


# --- the loop -----------------------------------------------------------


def test_it_trains_from_random_init_and_writes_checkpoints(seeded):
    result = train(seeded, steps=6, echo=False, seed=1)

    assert result.steps_run == 6
    assert result.final_step == 6
    assert result.checkpoints_written >= 1
    assert seeded.latest_checkpoint.exists()
    assert list(seeded.checkpoint_dir.glob("ckpt-*.pt"))


def test_every_checkpoint_records_a_loss_and_a_sample(seeded):
    train(seeded, steps=6, echo=False, seed=1)

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
    train(seeded, steps=80, echo=False, seed=1)

    entries = [
        json.loads(line)
        for line in seeded.loss_curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert entries[-1]["loss"] < entries[0]["loss"]


def test_it_prints_the_sample_generation_per_checkpoint(seeded, capsys):
    train(seeded, steps=4, echo=True, seed=1)

    printed = capsys.readouterr().out
    assert "loss" in printed and "step" in printed and "->" in printed


def test_old_checkpoints_are_pruned_but_latest_survives(seeded):
    seeded.keep_checkpoints = 2
    seeded.checkpoint_every = 2

    train(seeded, steps=10, echo=False, seed=1)

    assert len(list(seeded.checkpoint_dir.glob("ckpt-*.pt"))) <= 2
    assert seeded.latest_checkpoint.exists()


def test_with_no_data_it_stops_politely_instead_of_crashing(settings):
    result = train(settings, steps=5, echo=False)

    assert result.stopped_because == "no_data"
    assert result.steps_run == 0


def test_loop_mode_stops_after_the_cycle_budget(seeded):
    result = train(seeded, steps=2, loop=True, max_cycles=3, echo=False, seed=1)

    assert result.cycles == 3
    assert result.steps_run == 6


# --- resuming -----------------------------------------------------------


def test_a_second_run_resumes_where_the_first_stopped(seeded):
    first = train(seeded, steps=4, echo=False, seed=1)

    second = train(seeded, steps=4, echo=False, seed=1)

    assert first.final_step == 4
    assert second.final_step == 8
    assert second.steps_run == 4


def test_resuming_restores_the_optimiser_not_just_the_weights(seeded):
    train(seeded, steps=4, echo=False, seed=1)

    payload = torch.load(seeded.latest_checkpoint, map_location="cpu", weights_only=True)

    assert payload["step"] == 4
    assert payload["optim"]["state"], "AdamW moments must survive a restart"
    assert "torch_rng" in payload


def test_a_corrupt_checkpoint_falls_back_to_random_init(seeded):
    train(seeded, steps=2, echo=False, seed=1)
    seeded.latest_checkpoint.write_bytes(b"not a checkpoint at all")

    result = train(seeded, steps=2, echo=False, seed=1)

    assert result.final_step == 2  # started over rather than dying


def test_killing_the_trainer_mid_run_leaves_a_loadable_checkpoint(settings, tmp_path):
    """The real thing: SIGKILL a separate process, then resume from disk."""
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
        "BABBLE_STEPS_PER_CYCLE": "100000",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "babble", "train", "--loop", "--quiet"],
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
    killed_at = payload["step"]
    assert killed_at > 0
    assert not list(settings.checkpoint_dir.glob("*.tmp")), "a torn temp file was left behind"

    resumed = train(settings, steps=2, echo=False)
    assert resumed.final_step == killed_at + 2


# --- the discord training feed -------------------------------------------


def test_a_configured_feed_gets_a_post_per_checkpoint(seeded):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    result = train(seeded, steps=6, echo=False, seed=1, feed=feed)

    assert result.checkpoints_written >= 1
    checkpoint_posts = [c for c in sender.calls if "🔁" in c[1]]
    assert len(checkpoint_posts) == result.checkpoints_written


def test_a_failing_feed_post_never_breaks_training(seeded):
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=FakeSender(fail=True))

    result = train(seeded, steps=6, echo=False, seed=1, feed=feed)

    assert result.steps_run == 6
    assert result.checkpoints_written >= 1
    assert seeded.latest_checkpoint.exists()


def test_an_unconfigured_feed_makes_no_network_calls(seeded, monkeypatch):
    monkeypatch.delenv("BABBLE_LOG_WEBHOOK_URL", raising=False)

    def explode(*a, **k):
        raise AssertionError("should never be called when unconfigured")

    monkeypatch.setattr("babble.discord_feed.post_webhook", explode)

    result = train(seeded, steps=6, echo=False, seed=1)  # feed defaults from env: disabled

    assert result.steps_run == 6


def test_start_is_reported_fresh_then_resumed(seeded):
    sender = FakeSender()
    first_feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)
    train(seeded, steps=2, echo=False, seed=1, feed=first_feed)

    second_feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)
    train(seeded, steps=2, echo=False, seed=1, feed=second_feed)

    starts = [c for c in sender.calls if "started" in c[1].lower() or "resum" in c[1].lower()]
    assert "started" in starts[0][1].lower()
    assert "resum" in starts[1][1].lower()


def test_going_idle_posts_once_not_every_check(settings):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(settings, steps=2, echo=False, feed=feed)  # no data at all -> idle immediately, loop=False

    idle_posts = [c for c in sender.calls if "idle" in c[1].lower()]
    assert len(idle_posts) == 1


def test_the_feed_carries_cycle_step_loss_delta_rows_and_sample(seeded):
    seeded.checkpoint_every = 3
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, steps=6, echo=False, seed=1, feed=feed)

    checkpoint_posts = [c for c in sender.calls if "loss" in c[1].lower()]
    assert len(checkpoint_posts) >= 2
    first, second = checkpoint_posts[0][1], checkpoint_posts[1][1]
    assert "rows" in first.lower()
    # the second post carries a delta against the first checkpoint's loss
    assert "(" in second and ("+" in second or "-" in second)


def test_a_cycle_start_post_shows_the_dataset_shape_and_hyperparams(seeded):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, steps=2, echo=False, seed=1, feed=feed)

    starts = [c[1] for c in sender.calls if "starting" in c[1].lower()]
    assert len(starts) == 1
    assert "13 stored" in starts[0]
    assert "13 training" in starts[0]
    assert "0 dropped" in starts[0]
    assert "examples" in starts[0].lower()
    assert "batch" in starts[0].lower()


def test_a_cycle_end_post_shows_steps_and_duration(seeded):
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, steps=3, echo=False, seed=1, feed=feed)

    ends = [c[1] for c in sender.calls if "done" in c[1].lower()]
    assert len(ends) == 1
    assert "3 steps" in ends[0]


def test_checkpoints_probe_different_real_prompts_across_a_cycle(seeded):
    seeded.checkpoint_every = 1
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    train(seeded, steps=4, echo=False, seed=1, feed=feed)

    checkpoint_posts = [c[1] for c in sender.calls if "🔁" in c[1]]
    assert len(checkpoint_posts) == 4
    probed = [post.split("`")[1] for post in checkpoint_posts]
    assert len(set(probed)) > 1  # not stuck on one or two hardcoded phrases
    for a, b in zip(probed, probed[1:]):
        assert a != b  # never the same prompt twice in a row
    assert all("expected" in post.lower() for post in checkpoint_posts)


# --- dataset visibility ---------------------------------------------------


def test_probe_prompt_walks_the_whole_dataset_in_order_then_wraps():
    rows = [_row(f"prompt-{i}", f"answer-{i}", row_id=str(i)) for i in range(4)]

    walked = [probe_prompt(rows, i) for i in range(4)]

    assert walked == [(f"prompt-{i}", f"answer-{i}") for i in range(4)]
    assert probe_prompt(rows, 4) == probe_prompt(rows, 0)
    assert probe_prompt(rows, 7) == probe_prompt(rows, 3)


def test_probe_prompt_never_repeats_two_checkpoints_in_a_row():
    rows = [_row(f"prompt-{i}", f"answer-{i}", row_id=str(i)) for i in range(5)]

    probed = [probe_prompt(rows, i)[0] for i in range(20)]

    for a, b in zip(probed, probed[1:]):
        assert a != b


def test_probe_prompt_dedupes_a_repeated_prompt_to_its_latest_answer():
    rows = [
        _row("boop", "Boop", row_id="1"),
        _row("boop", "Beep", row_id="2"),  # a later correction supersedes the first
        _row("git good", "never", row_id="3"),
    ]

    assert distinct_prompts(rows) == [("boop", "Beep"), ("git good", "never")]
    assert probe_prompt(rows, 0) == ("boop", "Beep")
    assert probe_prompt(rows, 1) == ("git good", "never")
    assert probe_prompt(rows, 2) == ("boop", "Beep")  # wraps back around


def test_probe_prompt_falls_back_to_the_hardcoded_pair_when_theres_nothing_to_probe():
    assert probe_prompt([], 0) == (SAMPLE_PROMPTS[0], "")
    assert probe_prompt([], 1) == (SAMPLE_PROMPTS[1], "")
    assert probe_prompt([], 2) == (SAMPLE_PROMPTS[0], "")


def test_dataset_stats_accounts_for_every_stored_row(settings):
    ids = Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    store = InteractionStore(settings.interactions_path)
    good = ids.user("good-user")
    stranger = ids.user("never-asked")
    consent.grant("good-user")

    for i in range(11):
        store.append(_row(f"p{i}", f"c{i}", row_id=f"good-{i}", author=good))
    store.append(_row("hi", "hey", row_id="stranger-1", author=stranger))
    store.append(_row("slur prompt", "badword", row_id="blocked-1", author=good))

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
    InteractionStore(settings.interactions_path).append(
        Interaction(
            id=make_row_id(CORRECTION, "hi", "hey", stranger, stranger),
            signal=CORRECTION,
            prompt="hi",
            rejected="junk",
            chosen="hey",
            prompt_author=stranger,
            signal_author=stranger,
            weight=1.0,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )

    assert consented_rows(settings) == []
    assert train(settings, steps=2, echo=False).stopped_because == "no_data"


def test_withdrawing_consent_takes_rows_out_of_training(seeded):
    from babble.fakedata import FAKE_USER

    assert consented_rows(seeded)

    ConsentStore(seeded.consent_path).withdraw(FAKE_USER)

    assert consented_rows(seeded) == []
