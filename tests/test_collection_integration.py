"""The collection feed wired into the brain: pinging, widening, withdrawing and
blocklist enforcement all the way through `core`, exactly as the bot drives it.
"""

from __future__ import annotations


from babble.blocklist import Blocklist
from babble.core import Babble
from babble.discord_feed import CollectionFeed
from conftest import FakeDiscord


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, content: str) -> None:
        self.calls.append((url, content))


class CountingPublisher:
    def __init__(self) -> None:
        self.n = 0

    def maybe_publish(self) -> None:
        self.n += 1


def build(settings, generator, log, *, blocklist=None, publisher=None):
    sender = FakeSender()
    feed = CollectionFeed(
        webhook_url="https://x/webhook",
        sender=sender,
        blocklist=blocklist if blocklist is not None else Blocklist(frozenset()),
    )
    brain = Babble(
        settings,
        generator=generator,
        log=log,
        bot_user_id="bot-9999",
        blocklist=blocklist if blocklist is not None else Blocklist(frozenset()),
        feed=feed,
        publisher=publisher,
    )
    return FakeDiscord(brain), feed, sender


def rows(sender: FakeSender) -> list[str]:
    return [c[1] for c in sender.calls if c[1].startswith("🌱")]


# --- done means: someone pings, the channel shows the collected row --------


def test_a_ping_shows_up_as_a_collected_row_with_totals(settings, generator, log):
    fake, feed, sender = build(settings, generator, log)
    fake.onboard("user-1")  # first ping + accept

    fake.ping("user-1", "hey there booper")
    feed.flush()

    posts = rows(sender)
    assert len(posts) == 1
    assert "hey there booper" in posts[0]
    assert "a ping" in posts[0]
    assert "1" in posts[0]  # one row so far
    # No raw id anywhere -- only the pseudonym.
    assert "user-1" not in posts[0]
    assert "u_" in posts[0]


def test_the_grant_itself_is_announced(settings, generator, log):
    fake, feed, sender = build(settings, generator, log)
    fake.onboard("user-1")

    granted = [c[1] for c in sender.calls if "opted in" in c[1]]
    assert len(granted) == 1
    assert "user-1" not in granted[0]


# --- done means: widen a channel, then subsequent messages are collected ---


def test_widening_a_channel_is_announced_and_then_collects_plain_messages(settings, generator, log):
    fake, feed, sender = build(settings, generator, log)
    fake.onboard("user-1")
    fake.collect_all("user-1", channel="chan-7")

    opened = [c[1] for c in sender.calls if "!babble all" in c[1] and "opened" in c[1]]
    assert len(opened) == 1

    # An ordinary message in that channel -- not addressed to the bot -- is now
    # collected and shows up as a row from the widened surface.
    fake.say("user-1", "just chatting in here", channel="chan-7")
    feed.flush()

    posts = rows(sender)
    assert any("just chatting in here" in p and "!babble all" in p for p in posts)


# --- done means: withdraw reports the purge count --------------------------


def test_withdrawing_reports_how_many_rows_were_purged(settings, generator, log):
    fake, feed, sender = build(settings, generator, log)
    fake.onboard("user-1")
    fake.ping("user-1", "a keepable sentence")
    feed.flush()

    fake.say("user-1", "!babble forget")

    withdrew = [c[1] for c in sender.calls if "withdrew" in c[1]]
    assert len(withdrew) == 1
    assert "1 corpus" in withdrew[0]  # the one row it had collected


# --- blocklist enforcement, all the way through core -----------------------


def test_a_blocklisted_message_is_never_collected_and_never_posted(settings, generator, log):
    blocklist = Blocklist(frozenset({"badword"}))
    fake, feed, sender = build(settings, generator, log, blocklist=blocklist)
    fake.onboard("user-1")

    fake.ping("user-1", "badword everywhere")
    feed.flush()

    # It was blocked at capture, so no corpus row and therefore no feed row --
    # the blocked text never reaches the channel at all.
    assert rows(sender) == []


# --- the publisher is poked on every fresh row -----------------------------


def test_a_fresh_row_pokes_the_publisher(settings, generator, log):
    publisher = CountingPublisher()
    fake, feed, sender = build(settings, generator, log, publisher=publisher)
    fake.onboard("user-1")

    before = publisher.n
    fake.ping("user-1", "something brand new")
    assert publisher.n == before + 1

    # A duplicate of the same text stores no new row, so the publisher is not poked.
    steady = publisher.n
    fake.ping("user-1", "something brand new")
    assert publisher.n == steady
