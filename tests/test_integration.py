"""The whole loop, with the real model wired in.

Everywhere else the generator is faked so the bot tests stay fast. Here the
actual byte-level transformer is plugged into the actual `Babble`, so that the
seam between them is covered too: random init answers a ping, a human corrects
it, the trainer learns from that correction, and the bot picks up the new
weights without restarting.
"""

from __future__ import annotations

import json

from babble.consent import ConsentStore
from babble.core import Babble
from babble.export_hf import DATA_FILE, build_export
from babble.generate import CheckpointGenerator
from babble.store import CORRECTION
from babble.trainer import train
from conftest import FakeDiscord

ALICE = "111111111111111111"


def test_ping_correct_train_reload_export(settings, log):
    generator = CheckpointGenerator(settings, log)
    brain = Babble(settings, generator=generator, log=log, bot_user_id="9999")
    gw = FakeDiscord(brain)

    # 1. With no checkpoint on disk it answers from pure noise.
    assert not settings.latest_checkpoint.exists()
    gw.onboard(ALICE)
    answer = gw.ping(ALICE, "hello")[0]
    assert answer.kind == "generation"
    assert generator.step == 0

    # 2. A human tells it what it should have said.
    gw.ping(ALICE, "hey!", reply_to=answer.id)
    (row,) = brain.store.all()
    assert row.signal == CORRECTION
    assert row.chosen == "hey!"

    # 3. The trainer learns from exactly that row.
    result = train(settings, steps=4, echo=False, seed=1, log=log)
    assert result.steps_run == 4
    assert settings.latest_checkpoint.exists()

    # 4. The bot hot-reloads the new weights with no restart.
    assert generator.step == 4
    second = gw.ping(ALICE, "hello")[0]
    assert second.kind == "generation"

    # 5. And the correction is publishable, pseudonymously.
    export = build_export(settings, log=log)
    assert export.rows == 1
    published = json.loads((export.path / DATA_FILE).read_text(encoding="utf-8").strip())
    assert published["prompt"] == "hello"
    assert published["chosen"] == "hey!"
    assert published["rejected"] == answer.content.split("\n")[0]
    assert ALICE not in json.dumps(published)


def test_withdrawing_removes_the_row_from_training_and_publishing(settings, log):
    brain = Babble(settings, generator=CheckpointGenerator(settings, log), log=log)
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)
    answer = gw.ping(ALICE, "hello")[0]
    gw.ping(ALICE, "hey!", reply_to=answer.id)
    assert build_export(settings, log=log).rows == 1

    gw.say(ALICE, "!babble forget")

    assert brain.store.count() == 0
    assert build_export(settings, log=log).rows == 0
    assert train(settings, steps=2, echo=False, log=log).stopped_because == "no_data"


def test_the_log_tells_the_whole_story_of_a_conversation(settings, log, read_log):
    brain = Babble(settings, generator=CheckpointGenerator(settings, log), log=log)
    gw = FakeDiscord(brain)

    gw.ping(ALICE, "hi")  # consent notice
    gw.accept(ALICE)
    answer = gw.ping(ALICE, "hello")[0]
    gw.ping(ALICE, "hey!", reply_to=answer.id)
    gw.react(ALICE, answer.id)
    train(settings, steps=2, echo=False, seed=1, log=log)

    seen = [e["event"] for e in read_log()]
    for expected in (
        "consent.prompt",
        "consent.accept",
        "bot.ping",
        "model.load",
        "bot.generate",
        "capture.correction",
        "capture.approval",
        "train.start",
        "train.checkpoint",
        "train.stop",
    ):
        assert expected in seen, f"{expected} was never logged"

    # The generation log carries the sampling params and the checkpoint it used.
    (generation,) = read_log("bot.generate")
    assert generation["temperature"] == settings.temperature
    assert generation["top_k"] == settings.top_k
    assert "step" in generation and "ms" in generation


def test_a_declined_capture_is_logged_with_its_reason_and_not_its_content(
    settings, log, read_log
):
    brain = Babble(settings, generator=CheckpointGenerator(settings, log), log=log)
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)
    answer = gw.ping(ALICE, "hello")[0]

    gw.ping("222222222222222222", "unlisted")  # notice
    gw.decline("222222222222222222")
    gw.ping("222222222222222222", "SECRETCORRECTION", reply_to=answer.id)

    (skip,) = read_log("capture.skipped")
    assert skip["reason"] == "no_consent"
    assert skip["missing"] == "signal_author"
    body = (settings.log_dir / "babble.jsonl").read_text(encoding="utf-8")
    assert "SECRETCORRECTION" not in body
