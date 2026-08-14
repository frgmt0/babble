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

from babble.consent import ConsentStore
from babble.fakedata import seed_fake_data
from babble.identity import Pseudonymiser
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.tokenizer import PAD_ID, build_example
from babble.trainer import consented_rows, make_batch, train


@pytest.fixture
def seeded(settings):
    seed_fake_data(settings)
    return settings


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
