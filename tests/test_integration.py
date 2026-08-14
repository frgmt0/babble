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
from babble.core import FOOTER, Babble
from babble.export_hf import CORPUS_FILE, DATA_FILE, build_export
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
    # The ping itself -- addressed to the bot, from a consented person -- is
    # already a corpus row, independent of whatever happens to it next.
    assert brain.corpus.count() == 1

    # 2. A human tells it what it should have said.
    gw.correct(ALICE, "hey!", reply_to=answer.id)
    (row,) = brain.store.all()
    assert row.signal == CORRECTION
    assert row.chosen == "hey!"
    # The correction files its own text as a second corpus row; the original
    # prompt is a duplicate of the one already there, so the count rises by
    # exactly one, not two.
    assert brain.corpus.count() == 2

    # 3. The trainer learns from exactly that row.
    result = train(settings, steps=4, echo=False, seed=1, log=log)
    assert result.steps_run == 4
    assert settings.latest_checkpoint.exists()

    # 4. The bot hot-reloads the new weights with no restart.
    assert generator.step == 4
    second = gw.ping(ALICE, "hello")[0]
    assert second.kind == "generation"

    # 5. And both the correction and the corpus it fed are publishable,
    # pseudonymously.
    export = build_export(settings, log=log)
    assert export.correction_rows == 1
    assert export.corpus_rows == 2
    assert export.rows == 3
    published = json.loads((export.path / DATA_FILE).read_text(encoding="utf-8").strip())
    assert published["prompt"] == "hello"
    assert published["chosen"] == "hey!"
    # The stored `rejected` is the body the bot posted, footer removed. Strip
    # the footer rather than taking the first line: a random model happily
    # emits newlines, and this is asserting what was published, not how many
    # lines it happened to come out as.
    assert published["rejected"] == answer.content[: -(len(FOOTER) + 1)]
    assert ALICE not in json.dumps(published)

    corpus_body = (export.path / CORPUS_FILE).read_text(encoding="utf-8")
    corpus_texts = {json.loads(line)["text"] for line in corpus_body.splitlines()}
    assert corpus_texts == {"hello", "hey!"}
    assert ALICE not in corpus_body


def test_withdrawing_removes_the_row_from_training_and_publishing(settings, log):
    brain = Babble(settings, generator=CheckpointGenerator(settings, log), log=log)
    gw = FakeDiscord(brain)
    gw.onboard(ALICE)
    answer = gw.ping(ALICE, "hello")[0]
    gw.correct(ALICE, "hey!", reply_to=answer.id)
    assert brain.corpus.count() == 2
    assert brain.store.count() == 1
    result = build_export(settings, log=log)
    assert result.correction_rows == 1
    assert result.corpus_rows == 2

    gw.say(ALICE, "!babble forget")

    # `!babble forget` empties both stores, not just the corrections one.
    assert brain.store.count() == 0
    assert brain.corpus.count() == 0
    result = build_export(settings, log=log)
    assert result.rows == 0
    assert result.correction_rows == 0
    assert result.corpus_rows == 0
    assert train(settings, steps=2, echo=False, log=log).stopped_because == "no_data"


def test_the_log_tells_the_whole_story_of_a_conversation(settings, log, read_log):
    brain = Babble(settings, generator=CheckpointGenerator(settings, log), log=log)
    gw = FakeDiscord(brain)

    gw.ping(ALICE, "hi")  # consent notice
    gw.accept(ALICE)
    answer = gw.ping(ALICE, "hello")[0]
    gw.correct(ALICE, "hey!", reply_to=answer.id)
    gw.react(ALICE, answer.id)
    train(settings, steps=2, echo=False, seed=1, log=log)

    seen = [e["event"] for e in read_log()]
    for expected in (
        "consent.prompt",
        "consent.accept",
        "bot.ping",
        "model.load",
        "bot.generate",
        "capture.corpus",
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
    gw.correct("222222222222222222", "SECRETCORRECTION", reply_to=answer.id)

    (skip,) = read_log("capture.skipped")
    assert skip["reason"] == "no_consent"
    assert skip["missing"] == "signal_author"
    body = (settings.log_dir / "babble.jsonl").read_text(encoding="utf-8")
    assert "SECRETCORRECTION" not in body
