"""The collection feed: what the channel shows now that no trainer is running.

Rows arriving, consent changing, growth milestones, dataset publishes -- posted
to the same webhook the training feed uses, coalesced so a burst is one message,
and blocklist-filtered + neutered before any collected text reaches Discord.
"""

from __future__ import annotations

from babble.blocklist import Blocklist
from babble.corpus import SOURCE_AMBIENT, SOURCE_DM, SOURCE_MENTION
from babble.discord_feed import (
    CHAR_MILESTONES,
    ROW_MILESTONES,
    WITHHELD,
    CollectionFeed,
    milestone_interval,
)


class FakeSender:
    """Records posts; can be told to explode to prove the feed swallows it."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, url: str, content: str) -> None:
        if self.fail:
            raise TimeoutError("discord did not answer")
        self.calls.append((url, content))


class Clock:
    """A hand-cranked monotonic clock, so the coalescing window is deterministic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_feed(sender: FakeSender | None = None, *, blocklist: Blocklist | None = None, clock=None):
    return CollectionFeed(
        webhook_url="https://discord.example/webhook",
        sender=sender or FakeSender(),
        blocklist=blocklist if blocklist is not None else Blocklist(frozenset()),
        clock=clock or Clock(),
    )


def row_posts(sender: FakeSender) -> list[str]:
    """Just the corpus-row messages, filtering out any milestone posts."""
    return [content for _, content in sender.calls if content.startswith("🌱")]


# --- silent / fail-soft, exactly like the training feed ---------------------


def test_unconfigured_feed_never_calls_the_sender():
    sender = FakeSender()
    feed = CollectionFeed(webhook_url=None, sender=sender)

    feed.row(text="hi", source=SOURCE_MENTION, author="u_1", rows=1, chars=2, contributors=1)
    feed.flush()
    feed.consent_granted(author="u_1")

    assert sender.calls == []


def test_a_failing_post_never_raises_and_is_logged(read_log, log):
    feed = make_feed(FakeSender(fail=True))
    feed.log = log

    feed.consent_granted(author="u_1")  # does not raise

    entries = read_log("feed.post_failed")
    assert len(entries) == 1
    assert "TimeoutError" in entries[0]["error"]


# --- a row -----------------------------------------------------------------


def test_a_single_row_reports_text_surface_pseudonym_and_totals():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="hello there", source=SOURCE_MENTION, author="u_9f2c", rows=54, chars=1012, contributors=7)
    feed.flush()

    posts = row_posts(sender)
    assert len(posts) == 1
    content = posts[0]
    assert "hello there" in content
    assert "a ping" in content  # the surface, in words
    assert "u_9f2c" in content
    assert "54" in content and "1,012" in content and "7" in content and "contributors" in content


def test_row_surface_labels_distinguish_dm_and_widened_channel():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="a", source=SOURCE_DM, author="u_1", rows=1, chars=1, contributors=1)
    feed.flush()
    feed.row(text="b", source=SOURCE_AMBIENT, author="u_1", rows=2, chars=2, contributors=1)
    feed.flush()

    assert "a DM" in sender.calls[0][1]
    assert "!babble all" in sender.calls[1][1]


def test_a_lone_contributor_is_singular():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="a", source=SOURCE_MENTION, author="u_1", rows=1, chars=1, contributors=1)
    feed.flush()

    content = row_posts(sender)[0]
    assert "contributor" in content
    assert "contributors" not in content


# --- coalescing ------------------------------------------------------------


def test_a_burst_inside_the_window_coalesces_into_one_message():
    sender = FakeSender()
    clock = Clock()
    feed = make_feed(sender, clock=clock)

    for i in range(5):
        clock.advance(0.2)  # all well inside the 3s window
        feed.row(text=f"msg {i}", source=SOURCE_AMBIENT, author="u_1", rows=50 + i, chars=100, contributors=1)

    assert sender.calls == []  # nothing posted yet -- still coalescing

    clock.advance(5.0)
    feed.flush_due()

    posts = row_posts(sender)
    assert len(posts) == 1
    content = posts[0]
    assert "+5" in content
    for i in range(5):
        assert f"msg {i}" in content


def test_flush_due_does_nothing_until_the_window_elapses():
    sender = FakeSender()
    clock = Clock()
    feed = make_feed(sender, clock=clock)

    feed.row(text="a", source=SOURCE_MENTION, author="u_1", rows=1, chars=1, contributors=1)
    clock.advance(1.0)  # under the 3s window
    feed.flush_due()
    assert sender.calls == []

    clock.advance(3.0)  # now past it
    feed.flush_due()
    assert len(sender.calls) == 1


def test_a_row_after_the_window_flushes_the_previous_one_first():
    sender = FakeSender()
    clock = Clock()
    feed = make_feed(sender, clock=clock)

    feed.row(text="first", source=SOURCE_MENTION, author="u_1", rows=1, chars=1, contributors=1)
    clock.advance(10.0)  # long gap -- the trickle case
    feed.row(text="second", source=SOURCE_MENTION, author="u_1", rows=2, chars=2, contributors=1)

    # The first row's window had elapsed, so it posted on its own; the second is
    # still buffered.
    assert len(sender.calls) == 1
    assert "first" in sender.calls[0][1]
    assert "second" not in sender.calls[0][1]


def test_a_burst_larger_than_the_cap_flushes_without_waiting():
    sender = FakeSender()
    clock = Clock()
    feed = CollectionFeed(
        webhook_url="https://discord.example/webhook",
        sender=sender,
        blocklist=Blocklist(frozenset()),
        clock=clock,
        max_coalesce=3,
    )

    for i in range(3):
        feed.row(text=f"m{i}", source=SOURCE_AMBIENT, author="u_1", rows=i + 1, chars=1, contributors=1)

    # Hit the cap on the 3rd without any clock movement, so it must have flushed.
    assert len(sender.calls) == 1
    assert "+3" in sender.calls[0][1]


# --- milestones ------------------------------------------------------------


def test_milestone_interval_scales_with_size():
    assert milestone_interval(54, ROW_MILESTONES) == 25
    assert milestone_interval(500, ROW_MILESTONES) == 100
    assert milestone_interval(5_000, ROW_MILESTONES) == 500
    assert milestone_interval(50_000, ROW_MILESTONES) == 2_500
    assert milestone_interval(500, CHAR_MILESTONES) == 2_000
    assert milestone_interval(500_000, CHAR_MILESTONES) == 100_000


def test_a_small_corpus_gets_a_row_milestone_at_fifty():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="x", source=SOURCE_MENTION, author="u_1", rows=50, chars=120, contributors=3)
    feed.flush()

    milestones = [c[1] for c in sender.calls if "milestone" in c[1].lower()]
    assert len(milestones) == 1
    assert "50" in milestones[0]
    assert "rows" in milestones[0].lower()


def test_a_large_corpus_uses_the_coarse_interval_not_the_fine_one():
    sender = FakeSender()
    feed = make_feed(sender)

    # 10,499 rows: below no fine threshold, so the 2,500 interval applies and the
    # last milestone reached is 10,000 -- not 10,475 or any 25-spaced value.
    feed.row(text="x", source=SOURCE_MENTION, author="u_1", rows=10_499, chars=5, contributors=9)
    feed.flush()

    milestones = [c[1] for c in sender.calls if "milestone" in c[1].lower() and "rows" in c[1].lower()]
    assert len(milestones) == 1
    assert "10,000" in milestones[0]


def test_a_milestone_fires_once_and_then_only_on_the_next_threshold():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="a", source=SOURCE_MENTION, author="u_1", rows=50, chars=1, contributors=1)
    feed.flush()
    feed.row(text="b", source=SOURCE_MENTION, author="u_1", rows=60, chars=1, contributors=1)
    feed.flush()

    milestones = [c[1] for c in sender.calls if "milestone" in c[1].lower() and "rows" in c[1].lower()]
    assert len(milestones) == 1  # 50 fired, 60 did not (next is 75)


def test_a_milestone_reports_the_new_scale_after_crossing_a_boundary():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="a", source=SOURCE_MENTION, author="u_1", rows=99, chars=1, contributors=1)
    feed.flush()  # crosses 75 (fine 25-interval)
    feed.row(text="b", source=SOURCE_MENTION, author="u_1", rows=101, chars=1, contributors=1)
    feed.flush()  # now the 100-interval applies -> 100

    milestones = [c[1] for c in sender.calls if "milestone" in c[1].lower() and "rows" in c[1].lower()]
    assert any("75" in m for m in milestones)
    assert any("100" in m for m in milestones)


def test_priming_suppresses_re_announcing_the_last_milestone_after_a_restart():
    sender = FakeSender()
    feed = make_feed(sender)
    # As if the bot just restarted onto a corpus of 54 rows / 1,050 chars.
    feed.prime(rows=54, chars=1_050)

    # The next captured row must not re-fire the 50-row milestone already passed.
    feed.row(text="x", source=SOURCE_MENTION, author="u_1", rows=55, chars=1_060, contributors=8)
    feed.flush()
    assert not [c for c in sender.calls if "milestone" in c[1].lower()]

    # But a genuinely new milestone still fires.
    feed.row(text="y", source=SOURCE_MENTION, author="u_1", rows=75, chars=1_070, contributors=8)
    feed.flush()
    assert any("75" in c[1] for c in sender.calls if "milestone" in c[1].lower())


def test_a_character_milestone_fires_independently_of_rows():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="x", source=SOURCE_MENTION, author="u_1", rows=3, chars=2_000, contributors=1)
    feed.flush()

    milestones = [c[1] for c in sender.calls if "milestone" in c[1].lower() and "character" in c[1].lower()]
    assert len(milestones) == 1
    assert "2,000" in milestones[0]


# --- blocklist + neuter enforcement on feed text ---------------------------


def test_feed_text_is_neutered_so_a_row_can_never_ping():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.row(text="please @everyone look", source=SOURCE_MENTION, author="u_1", rows=1, chars=1, contributors=1)
    feed.flush()

    assert "@everyone" not in sender.calls[0][1]


def test_a_blocklisted_row_is_withheld_from_the_feed_text():
    sender = FakeSender()
    # A term with no repeated letters, so the blocklist's normalisation (which
    # collapses runs like "zzz" -> "z") leaves it matchable as written here.
    feed = make_feed(sender, blocklist=Blocklist(frozenset({"badword"})))

    feed.row(text="badword leaked", source=SOURCE_MENTION, author="u_1", rows=1, chars=1, contributors=1)
    feed.flush()

    content = row_posts(sender)[0]
    assert "badword" not in content
    assert WITHHELD in content


def test_feed_text_never_exceeds_discord_limit_even_in_a_big_burst():
    sender = FakeSender()
    clock = Clock()
    feed = CollectionFeed(
        webhook_url="https://discord.example/webhook",
        sender=sender,
        blocklist=Blocklist(frozenset()),
        clock=clock,
        max_coalesce=50,
    )

    for i in range(40):
        feed.row(text="x" * 200, source=SOURCE_AMBIENT, author="u_1", rows=i + 1, chars=200, contributors=1)
    feed.flush()

    for _, content in sender.calls:
        assert len(content) <= 2000


# --- consent events --------------------------------------------------------


def test_consent_events_are_posted_immediately_with_the_pseudonym_only():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.consent_granted(author="u_abc")
    feed.channel_widened(author="u_abc", channel="c_room")
    feed.channel_narrowed(author="u_abc", channel="c_room")
    feed.consent_declined(author="u_abc")
    feed.consent_withdrawn(author="u_abc", corpus_purged=12, correction_purged=3)

    joined = "\n".join(c[1] for c in sender.calls)
    assert "opted in" in joined
    assert "!babble all" in joined
    assert "opted out" in joined
    assert "withdrew" in joined
    assert "u_abc" in joined
    assert "c_room" in joined
    # The withdrawal reports how many rows the purge removed.
    assert "15" in sender.calls[-1][1]  # 12 corpus + 3 corrections
    assert "12 corpus" in sender.calls[-1][1]


# --- publish ---------------------------------------------------------------


def test_publish_and_failure_announcements():
    sender = FakeSender()
    feed = make_feed(sender)

    feed.published(rows=78, url="https://huggingface.co/datasets/kowo-co/babble", grew_rows=30, grew_chars=900)
    feed.publish_failed("RuntimeError: HF is down")

    assert "78" in sender.calls[0][1]
    assert "huggingface.co/datasets" in sender.calls[0][1]
    assert "+30" in sender.calls[0][1]
    assert "failed" in sender.calls[1][1].lower()
    assert "HF is down" in sender.calls[1][1]
