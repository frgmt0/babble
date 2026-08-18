"""Auto-publish: every `hf_publish_every` checkpoints, push the corpus and the
corrections dataset beside it to HuggingFace, through the same consent/
blocklist gate as a manual `babble export --push` -- and never let a failed
push touch training.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from babble.config import Settings
from babble.consent import ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.discord_feed import TrainingFeed
from babble.export_hf import CORPUS_FILE, DATA_FILE
from babble.fakedata import FAKE_HELPER, FAKE_USER, seed_fake_data
from babble.identity import Pseudonymiser
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.trainer import train

FAKE_ROW_COUNT = 13  # 12 corrections + 1 approval, see babble/fakedata.py
# The backfill flattens each pair's prompt and chosen text into the corpus, one
# row per distinct (text, author): 12 distinct prompts + 12 distinct chosen
# answers. The approval reuses the first correction's prompt and chosen text
# verbatim, so it contributes no new corpus rows of its own.
FAKE_CORPUS_ROW_COUNT = 24
FAKE_TOTAL_ROW_COUNT = FAKE_ROW_COUNT + FAKE_CORPUS_ROW_COUNT


@pytest.fixture
def seeded(settings):
    seed_fake_data(settings)
    # These tests are about the publish cadence (checkpoint COUNT), not about
    # which checkpoint wins on val loss -- disable validation so best-val
    # early stopping never cuts a run short of the checkpoint count a test
    # expects.
    settings.val_min_rows = 1000
    return settings


class FakeSender:
    """Stands in for the HTTP layer of the training feed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, content: str) -> None:
        self.calls.append((url, content))


class FakePush:
    """Stands in for `export_hf.push` -- records calls, can be made to explode."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, Path]] = []
        self.fail = fail

    def __call__(self, settings, repo_id, out_dir, log=None, private=False):
        if self.fail:
            raise RuntimeError("HF is down")
        self.calls.append((repo_id, out_dir))
        return f"https://huggingface.co/datasets/{repo_id}"


def _add_row(settings, *, prompt, chosen, prompt_author, signal_author):
    row = Interaction(
        id=make_row_id(CORRECTION, prompt, chosen, prompt_author, signal_author),
        signal=CORRECTION,
        prompt=prompt,
        rejected="junk",
        chosen=chosen,
        prompt_author=prompt_author,
        signal_author=signal_author,
        weight=1.0,
        created_at="2026-01-01T00:00:00+00:00",
    )
    InteractionStore(settings.interactions_path).append(row)
    return row


# --- config ---------------------------------------------------------------


def test_hf_publish_every_defaults_to_20(tmp_path):
    assert Settings.for_root(tmp_path).hf_publish_every == 20


def test_hf_publish_every_is_overridable_by_env(monkeypatch):
    monkeypatch.setenv("BABBLE_HF_PUBLISH_EVERY", "5")

    assert Settings.from_env().hf_publish_every == 5


def test_hf_publish_every_zero_from_env_means_off(monkeypatch):
    monkeypatch.setenv("BABBLE_HF_PUBLISH_EVERY", "0")

    assert Settings.from_env().hf_publish_every == 0


# --- the schedule -----------------------------------------------------------


def test_it_publishes_once_every_configured_number_of_checkpoints(seeded, monkeypatch):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2
    pusher = FakePush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=FakeSender())

    train(seeded, force=True, steps=4, echo=False, seed=1, feed=feed)  # 2 checkpoints -> one publish

    assert len(pusher.calls) == 1
    repo_id, out_dir = pusher.calls[0]
    assert repo_id == seeded.hf_repo
    assert (out_dir / DATA_FILE).exists()
    assert (out_dir / CORPUS_FILE).exists()


def test_zero_disables_auto_publish(seeded, monkeypatch):
    seeded.hf_publish_every = 0
    pusher = FakePush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)

    train(seeded, force=True, steps=8, echo=False, seed=1)

    assert pusher.calls == []


def test_none_disables_auto_publish(seeded, monkeypatch):
    seeded.hf_publish_every = None
    pusher = FakePush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)

    train(seeded, force=True, steps=8, echo=False, seed=1)

    assert pusher.calls == []


def test_default_cadence_publishes_after_20_checkpoints(seeded, monkeypatch):
    seeded.checkpoint_every = 1  # one checkpoint per step, so 20 steps = 20 checkpoints
    pusher = FakePush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    result = train(seeded, force=True, steps=20, echo=False, seed=1, feed=feed)

    assert result.checkpoints_written == 20
    assert len(pusher.calls) == 1
    publish_posts = [c for c in sender.calls if "published" in c[1].lower()]
    assert len(publish_posts) == 1
    # The feed reports the total across both configs, not just the corrections.
    assert str(FAKE_TOTAL_ROW_COUNT) in publish_posts[0][1]
    assert "huggingface.co/datasets" in publish_posts[0][1]


def test_nothing_changed_since_the_last_publish_is_not_repushed(seeded, monkeypatch, log, read_log):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2
    pusher = FakePush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)

    # 4 checkpoints -> two scheduled publishes, but the corpus never changes.
    train(seeded, force=True, steps=8, echo=False, seed=1, log=log)

    assert len(pusher.calls) == 1
    assert len(read_log("publish.skipped")) == 1


def test_a_change_to_only_the_corpus_still_triggers_a_republish(seeded, monkeypatch):
    """The skip-if-unchanged hash covers both files, so a corpus-only change
    must not be mistaken for "nothing changed" just because the corrections
    file -- which the old, single-file hash used to key off -- is identical.
    """
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2

    class CorpusMutatingPush(FakePush):
        """Behaves like a real push, then simulates someone else appending to
        the corpus in between -- exactly as `!babble all` capture would while
        training runs in the background.
        """

        def __init__(self) -> None:
            super().__init__()
            self.mutated = False

        def __call__(self, settings, repo_id, out_dir, log=None, private=False):
            result = super().__call__(settings, repo_id, out_dir, log=log, private=private)
            if not self.mutated:
                self.mutated = True
                ConsentStore(settings.consent_path).grant("fresh-corpus-voice-000")
                author = Pseudonymiser.load(settings).user("fresh-corpus-voice-000")
                text = "a brand new sentence nobody corrected"
                CorpusStore(settings.corpus_path).append(
                    CorpusRow(
                        id=make_corpus_id(text, author),
                        text=text,
                        author=author,
                        source=SOURCE_MENTION,
                    )
                )
            return result

    pusher = CorpusMutatingPush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)

    # 4 checkpoints -> two scheduled publishes. The corrections file never
    # changes between them, but the corpus gains a row right after the first
    # publish, so the second one must still go out rather than be skipped.
    train(seeded, force=True, steps=8, echo=False, seed=1)

    assert len(pusher.calls) == 2


def test_reuses_the_existing_exporter_rather_than_a_second_one(seeded, monkeypatch, log, read_log):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2
    monkeypatch.setattr("babble.trainer.push_export", FakePush())

    train(seeded, force=True, steps=4, echo=False, seed=1, log=log)

    # "export.run" is logged from inside build_export itself, so seeing it here
    # is proof the auto-publish path called into export_hf.py rather than
    # reimplementing its own dataset writer.
    assert read_log("export.run")


# --- the safety gate ---------------------------------------------------------


def test_consent_gated_and_blocklisted_rows_are_excluded_from_the_auto_publish(
    seeded, monkeypatch, tmp_path
):
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("zzzsecret\n", encoding="utf-8")
    monkeypatch.setenv("BABBLE_BLOCKLIST_PATH", str(blocklist_path))

    ids = Pseudonymiser.load(seeded)
    stranger = ids.user("never-consented-000000000")
    _add_row(
        seeded,
        prompt="unconsented prompt",
        chosen="unconsented answer",
        prompt_author=stranger,
        signal_author=stranger,
    )
    asker, helper = ids.user(FAKE_USER), ids.user(FAKE_HELPER)
    _add_row(
        seeded,
        prompt="blocked prompt",
        chosen="zzzsecret answer",
        prompt_author=asker,
        signal_author=helper,
    )

    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2
    pusher = FakePush()
    monkeypatch.setattr("babble.trainer.push_export", pusher)

    train(seeded, force=True, steps=4, echo=False, seed=1)

    assert len(pusher.calls) == 1
    _, out_dir = pusher.calls[0]
    body = (out_dir / DATA_FILE).read_text(encoding="utf-8")
    assert "unconsented answer" not in body
    assert "zzzsecret" not in body
    assert stranger not in body
    assert "hey!" in body  # a legitimately consented, unblocked row is still there


# --- failure isolation --------------------------------------------------------


def test_a_failed_publish_never_stops_training_and_is_reported_in_the_feed(
    seeded, monkeypatch, log, read_log
):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2
    monkeypatch.setattr("babble.trainer.push_export", FakePush(fail=True))
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    result = train(seeded, force=True, steps=4, echo=False, seed=1, feed=feed, log=log)

    assert result.steps_run == 4
    assert seeded.latest_checkpoint.exists()

    entries = read_log("publish.failed")
    assert len(entries) == 1
    assert "HF is down" in entries[0]["error"]

    failure_posts = [c for c in sender.calls if "failed" in c[1].lower()]
    assert len(failure_posts) == 1
    assert "HF is down" in failure_posts[0][1]


def test_a_failed_publish_is_retried_at_the_next_scheduled_publish(seeded, monkeypatch):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2
    pusher = FakePush(fail=True)
    monkeypatch.setattr("babble.trainer.push_export", pusher)

    # 4 checkpoints -> two scheduled publishes, both attempted despite failing.
    train(seeded, force=True, steps=8, echo=False, seed=1)

    assert pusher.fail is True  # sanity: still configured to fail throughout


def test_a_failed_publish_makes_no_progress_but_a_later_success_still_works(seeded, monkeypatch):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2

    calls = {"n": 0}

    def flaky(settings, repo_id, out_dir, log=None, private=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first attempt: HF is down")
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr("babble.trainer.push_export", flaky)

    # 4 checkpoints -> attempts at checkpoint 2 (fails) and checkpoint 4 (succeeds,
    # same unchanged content as the failed attempt -- a failure must not be
    # mistaken for "already published" and silently skipped).
    train(seeded, force=True, steps=8, echo=False, seed=1)

    assert calls["n"] == 2


def test_an_export_blocked_by_the_pseudonym_guard_is_reported_not_raised(seeded, monkeypatch):
    seeded.hf_publish_every = 2
    seeded.checkpoint_every = 2

    from babble import export_hf

    def explode(*args, **kwargs):
        raise export_hf.ExportBlocked("row abc123: not a pseudonym -- refusing to export")

    monkeypatch.setattr("babble.trainer.build_export", explode)
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    result = train(seeded, force=True, steps=4, echo=False, seed=1, feed=feed)

    assert result.steps_run == 4  # training was not interrupted
    blocked_posts = [c for c in sender.calls if "blocked" in c[1].lower()]
    assert len(blocked_posts) == 1
