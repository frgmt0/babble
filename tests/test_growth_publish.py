"""The growth-based publisher: push the dataset when the corpus has grown,
not when a trainer wrote a checkpoint -- because in the collection phase there
is no trainer. Same consent/blocklist gate as a manual `babble export --push`.
"""

from __future__ import annotations

from pathlib import Path


from babble import export_hf
from babble.consent import ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.discord_feed import CollectionFeed
from babble.export_hf import CORPUS_FILE, DATA_FILE
from babble.identity import Pseudonymiser
from babble.publish import GrowthPublisher

CONSENTED = "consented-voice-000000"
STRANGER = "stranger-voice-0000000"


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, content: str) -> None:
        self.calls.append((url, content))


class FakePush:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, Path]] = []
        self.fail = fail

    def __call__(self, settings, repo_id, out_dir, log=None, private=False):
        if self.fail:
            raise RuntimeError("HF is down")
        self.calls.append((repo_id, out_dir))
        return f"https://huggingface.co/datasets/{repo_id}"


def add_row(settings, text: str, *, user_id: str = CONSENTED, consent: bool = True) -> None:
    if consent:
        ConsentStore(settings.consent_path).grant(user_id)
    author = Pseudonymiser.load(settings).user(user_id)
    CorpusStore(settings.corpus_path).append(
        CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=SOURCE_MENTION)
    )


def make_publisher(settings, *, every_rows=3, every_chars=10_000, pusher=None, feed=None):
    return GrowthPublisher(
        settings,
        feed=feed,
        every_rows=every_rows,
        every_chars=every_chars,
        pusher=pusher or FakePush(),
    )


# --- the trigger -----------------------------------------------------------


def test_below_the_threshold_it_does_not_publish(settings):
    add_row(settings, "one")
    add_row(settings, "two")
    pusher = FakePush()
    make_publisher(settings, every_rows=3, pusher=pusher).maybe_publish()

    assert pusher.calls == []


def test_crossing_the_row_threshold_publishes(settings):
    for word in ("one", "two", "three"):
        add_row(settings, word)
    sender = FakeSender()
    feed = CollectionFeed(webhook_url="https://x/webhook", sender=sender)
    pusher = FakePush()
    make_publisher(settings, every_rows=3, pusher=pusher, feed=feed).maybe_publish()

    assert len(pusher.calls) == 1
    _, out_dir = pusher.calls[0]
    assert (out_dir / CORPUS_FILE).exists() and (out_dir / DATA_FILE).exists()
    published = [c[1] for c in sender.calls if "published" in c[1].lower()]
    assert len(published) == 1
    assert "+3" in published[0]
    assert "huggingface.co/datasets" in published[0]


def test_crossing_the_char_threshold_publishes_even_with_few_rows(settings):
    add_row(settings, "x" * 500)
    pusher = FakePush()
    # One row, but 500 chars > the 400-char threshold.
    make_publisher(settings, every_rows=1_000, every_chars=400, pusher=pusher).maybe_publish()

    assert len(pusher.calls) == 1


def test_both_thresholds_zero_disables_it(settings):
    for word in ("a", "b", "c", "d", "e"):
        add_row(settings, word)
    pusher = FakePush()
    make_publisher(settings, every_rows=0, every_chars=0, pusher=pusher).maybe_publish()

    assert pusher.calls == []


# --- the persisted baseline ------------------------------------------------


def test_a_fresh_publisher_after_a_publish_does_not_republish_unchanged(settings):
    for word in ("one", "two", "three"):
        add_row(settings, word)
    pusher = FakePush()
    make_publisher(settings, every_rows=3, pusher=pusher).maybe_publish()
    assert len(pusher.calls) == 1

    # A brand-new publisher, as if the bot restarted: the baseline is on disk, so
    # the same unchanged corpus is not pushed again.
    make_publisher(settings, every_rows=3, pusher=pusher).maybe_publish()
    assert len(pusher.calls) == 1


def test_more_growth_after_a_publish_publishes_again(settings):
    for word in ("one", "two", "three"):
        add_row(settings, word)
    pusher = FakePush()
    publisher = make_publisher(settings, every_rows=3, pusher=pusher)
    publisher.maybe_publish()
    assert len(pusher.calls) == 1

    for word in ("four", "five", "six"):
        add_row(settings, word)
    publisher.maybe_publish()
    assert len(pusher.calls) == 2


# --- the safety gate -------------------------------------------------------


def test_unconsented_growth_is_not_pushed_and_does_not_spin(settings, log, read_log):
    for word in ("one", "two", "three"):
        add_row(settings, word)
    pusher = FakePush()
    publisher = GrowthPublisher(settings, log, every_rows=3, pusher=pusher)
    publisher.maybe_publish()
    assert len(pusher.calls) == 1

    # Three more rows, but from someone who never consented: the corpus grew, so
    # the trigger fires, but the *publishable* bytes are identical, so nothing is
    # pushed -- and the baseline still advances, so it does not re-attempt.
    for word in ("four", "five", "six"):
        add_row(settings, word, user_id=STRANGER, consent=False)
    publisher.maybe_publish()

    assert len(pusher.calls) == 1
    assert read_log("publish.skipped")


def test_a_blocklisted_new_row_never_reaches_the_pushed_file(settings, monkeypatch, tmp_path):
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("badword\n", encoding="utf-8")
    monkeypatch.setenv("BABBLE_BLOCKLIST_PATH", str(blocklist_path))

    add_row(settings, "a clean sentence")
    add_row(settings, "another clean one")
    add_row(settings, "badword slips through")  # consented, but blocklisted
    pusher = FakePush()
    make_publisher(settings, every_rows=3, pusher=pusher).maybe_publish()

    assert len(pusher.calls) == 1
    body = (pusher.calls[0][1] / CORPUS_FILE).read_text(encoding="utf-8")
    assert "badword" not in body
    assert "a clean sentence" in body


# --- failure isolation -----------------------------------------------------


def test_a_failed_push_is_reported_in_the_feed_and_never_raises(settings, log, read_log):
    for word in ("one", "two", "three"):
        add_row(settings, word)
    sender = FakeSender()
    feed = CollectionFeed(webhook_url="https://x/webhook", sender=sender)
    publisher = GrowthPublisher(settings, log, feed=feed, every_rows=3, pusher=FakePush(fail=True))

    publisher.maybe_publish()  # must not raise

    assert read_log("publish.failed")
    failed = [c[1] for c in sender.calls if "failed" in c[1].lower()]
    assert len(failed) == 1
    assert "HF is down" in failed[0]


def test_a_failed_push_does_not_hammer_every_message(settings):
    for word in ("one", "two", "three"):
        add_row(settings, word)
    pusher = FakePush(fail=True)
    publisher = make_publisher(settings, every_rows=3, pusher=pusher)

    publisher.maybe_publish()  # attempt 1, fails
    publisher.maybe_publish()  # no new growth -> no second attempt

    assert len(pusher.calls) == 0  # both failed before recording, but only one attempt was made
    # Prove only one attempt happened by making the next push succeed and seeing
    # it needs fresh growth to fire.
    pusher.fail = False
    publisher.maybe_publish()
    assert pusher.calls == []  # still no growth since the failed attempt advanced the baseline


def test_an_export_blocked_is_reported_not_raised(settings, monkeypatch, log, read_log):
    for word in ("one", "two", "three"):
        add_row(settings, word)

    def explode(*args, **kwargs):
        raise export_hf.ExportBlocked("row abc: not a pseudonym -- refusing to export")

    sender = FakeSender()
    feed = CollectionFeed(webhook_url="https://x/webhook", sender=sender)
    publisher = GrowthPublisher(
        settings, log, feed=feed, every_rows=3, exporter=explode, pusher=FakePush()
    )

    publisher.maybe_publish()  # must not raise

    assert read_log("publish.blocked")
    blocked = [c[1] for c in sender.calls if "blocked" in c[1].lower()]
    assert len(blocked) == 1
