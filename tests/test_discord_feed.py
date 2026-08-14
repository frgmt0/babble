"""The Discord training feed: best-effort, silent unconfigured, never a ping."""

from __future__ import annotations

import io
import urllib.error

from babble import __version__
from babble.discord_feed import TrainingFeed, neuter_sample, post_webhook


class FakeSender:
    """Stands in for the HTTP layer -- records calls, can be made to explode."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, url: str, content: str) -> None:
        if self.fail:
            raise TimeoutError("discord did not answer")
        self.calls.append((url, content))


# --- sanitisation ---------------------------------------------------------


def test_neuter_sample_breaks_everyone_and_here():
    text = neuter_sample("please @everyone and @here look at this")

    assert "@everyone" not in text
    assert "@here" not in text
    assert "everyone" in text  # readable, just de-fanged
    assert "here" in text


def test_neuter_sample_breaks_raw_mention_markup():
    text = neuter_sample("hey <@123456789012345678> and <@!987654321098765432> and <@&555>")

    assert "<@123456789012345678>" not in text
    assert "<@!987654321098765432>" not in text
    assert "<@&555>" not in text


def test_neuter_sample_strips_control_bytes_and_collapses_newlines():
    text = neuter_sample("line one\nline two\x00\x07")

    assert "\n" not in text
    assert "\x00" not in text and "\x07" not in text
    assert "line one" in text and "line two" in text


def test_neuter_sample_truncates_long_output():
    text = neuter_sample("x" * 5000)

    assert len(text) <= 200


def test_neuter_sample_flattens_backticks_so_the_code_span_cannot_escape():
    text = neuter_sample("```js\nalert(1)```")

    assert "`" not in text


def test_post_webhook_sets_allowed_mentions_to_none(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    def fake_urlopen(request, timeout):
        import json

        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr("babble.discord_feed.urllib.request.urlopen", fake_urlopen)

    post_webhook("https://discord.example/webhook", "hello @everyone")

    assert captured["url"] == "https://discord.example/webhook"
    assert captured["body"]["allowed_mentions"] == {"parse": []}
    assert captured["body"]["content"] == "hello @everyone"


def test_post_webhook_sends_a_real_user_agent(monkeypatch):
    """Discord's edge 403s urllib's default UA before the request reaches the webhook."""
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    def fake_urlopen(request, timeout):
        captured["user_agent"] = request.get_header("User-agent")
        return FakeResponse()

    monkeypatch.setattr("babble.discord_feed.urllib.request.urlopen", fake_urlopen)

    post_webhook("https://discord.example/webhook", "hello")

    ua = captured["user_agent"]
    assert ua is not None
    assert not ua.startswith("Python-urllib")  # urllib's default -- the exact thing that 403s
    assert ua.startswith("DiscordBot")
    assert __version__ in ua


# --- TrainingFeed -----------------------------------------------------------


def test_unconfigured_feed_never_calls_the_sender():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url=None, sender=sender)

    feed.start(resumed=False, step=0)
    feed.idle()
    feed.checkpoint(cycle=1, step=50, loss=1.0, prev_loss=None, rows=3, prompt="hello", sample="hi")

    assert sender.calls == []


def test_a_checkpoint_produces_a_well_formed_post():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=3, step=650, loss=2.184, prev_loss=2.5, rows=128, prompt="hello", sample="heoll wrold"
    )

    assert len(sender.calls) == 1
    url, content = sender.calls[0]
    assert url == "https://discord.example/webhook"
    assert "3" in content and "650" in content
    assert "2.184" in content or "2.1840" in content
    assert "128" in content
    assert "heoll wrold" in content
    assert "-0.316" in content  # 2.184 - 2.5


def test_a_failing_post_never_raises_and_gets_logged(read_log, log):
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=FakeSender(fail=True), log=log)

    feed.checkpoint(cycle=1, step=50, loss=1.0, prev_loss=None, rows=3, prompt="hi", sample="hi")

    entries = read_log("feed.post_failed")
    assert len(entries) == 1
    assert "TimeoutError" in entries[0]["error"]


def test_a_failing_post_logs_the_response_body_but_never_the_url(read_log, log):
    def raise_http_error(url, content):
        raise urllib.error.HTTPError(
            url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"message": "401: Unauthorized", "code": 0}'),
        )

    feed = TrainingFeed(
        webhook_url="https://discord.example/webhooks/1/topsecrettoken",
        sender=raise_http_error,
        log=log,
    )

    feed.checkpoint(cycle=1, step=50, loss=1.0, prev_loss=None, rows=3, prompt="hi", sample="hi")

    entries = read_log("feed.post_failed")
    assert len(entries) == 1
    error = entries[0]["error"]
    assert "401: Unauthorized" in error
    assert "topsecrettoken" not in error
    assert "discord.example" not in error


def test_a_failing_post_with_a_long_body_is_truncated(read_log, log):
    long_body = ("x" * 500).encode("utf-8")

    def raise_http_error(url, content):
        raise urllib.error.HTTPError(
            url, 500, "Server Error", hdrs=None, fp=io.BytesIO(long_body)
        )

    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=raise_http_error, log=log)

    feed.checkpoint(cycle=1, step=50, loss=1.0, prev_loss=None, rows=3, prompt="hi", sample="hi")

    error = read_log("feed.post_failed")[0]["error"]
    assert "x" * 300 in error
    assert "x" * 301 not in error


def test_throttling_only_posts_every_nth_checkpoint():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", every_n=3, sender=sender)

    for step in (10, 20, 30, 40, 50, 60):
        feed.checkpoint(cycle=1, step=step, loss=1.0, prev_loss=None, rows=1, prompt="hi", sample="hi")

    assert len(sender.calls) == 2  # the 3rd and the 6th


def test_a_model_sample_containing_everyone_is_neutered_before_posting():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1, step=10, loss=1.0, prev_loss=None, rows=1, prompt="hi", sample="@everyone free stuff"
    )

    _, content = sender.calls[0]
    assert "@everyone" not in content


def test_start_reports_a_fresh_run_versus_a_resume():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.start(resumed=False, step=0)
    feed.start(resumed=True, step=400)

    assert "started" in sender.calls[0][1].lower()
    assert "400" in sender.calls[1][1]
    assert "resum" in sender.calls[1][1].lower()


def test_idle_posts_once_until_active_is_called():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.idle()
    feed.idle()
    feed.idle()
    assert len(sender.calls) == 1

    feed.active()
    feed.idle()
    assert len(sender.calls) == 2


def test_cycle_start_shows_the_dataset_shape_and_hyperparams():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.cycle_start(
        cycle=1, stored=13, trained=11, dropped_consent=1, dropped_blocklist=1,
        examples=11, batch_size=8, lr=0.0003,
    )

    assert len(sender.calls) == 1
    content = sender.calls[0][1]
    assert "13 stored" in content
    assert "11 training" in content
    assert "2 dropped" in content
    assert "1 no consent" in content
    assert "1 blocklist" in content
    assert "11 examples" in content
    assert "batch 8" in content


def test_cycle_start_omits_the_drop_note_when_nothing_was_dropped():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.cycle_start(
        cycle=1, stored=5, trained=5, dropped_consent=0, dropped_blocklist=0,
        examples=5, batch_size=8, lr=0.001,
    )

    content = sender.calls[0][1]
    assert "0 dropped" in content
    assert "no consent" not in content
    assert "blocklist" not in content


def test_cycle_end_shows_steps_and_duration():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.cycle_end(cycle=2, steps=40, seconds=12.3)

    assert len(sender.calls) == 1
    content = sender.calls[0][1]
    assert "40 steps" in content
    assert "12.3s" in content


def test_cycle_start_and_end_are_silent_when_unconfigured():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url=None, sender=sender)

    feed.cycle_start(
        cycle=1, stored=1, trained=1, dropped_consent=0, dropped_blocklist=0,
        examples=1, batch_size=1, lr=0.1,
    )
    feed.cycle_end(cycle=1, steps=1, seconds=1.0)

    assert sender.calls == []


def test_checkpoint_shows_the_expected_answer_alongside_the_sample():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1, step=10, loss=1.0, prev_loss=None, rows=3,
        prompt="boop", sample="garbled nonsense", expected="Beep",
    )

    content = sender.calls[0][1]
    assert "Beep" in content
    assert "expected" in content.lower()


def test_checkpoint_omits_the_expected_line_when_theres_no_answer_to_show():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1, step=10, loss=1.0, prev_loss=None, rows=3,
        prompt="hello", sample="hi", expected="",
    )

    assert "expected" not in sender.calls[0][1].lower()


def test_checkpoint_neuters_the_prompt_too_not_just_the_sample():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1, step=10, loss=1.0, prev_loss=None, rows=3,
        prompt="please @everyone look", sample="hi", expected="",
    )

    assert "@everyone" not in sender.calls[0][1]


def test_from_env_is_disabled_without_a_webhook_url(monkeypatch):
    monkeypatch.delenv("BABBLE_LOG_WEBHOOK_URL", raising=False)

    feed = TrainingFeed.from_env()

    assert not feed.enabled


def test_from_env_reads_the_webhook_and_throttle(monkeypatch):
    monkeypatch.setenv("BABBLE_LOG_WEBHOOK_URL", "https://discord.example/webhook")
    monkeypatch.setenv("BABBLE_LOG_EVERY_N", "5")

    feed = TrainingFeed.from_env()

    assert feed.enabled
    assert feed.webhook_url == "https://discord.example/webhook"
    assert feed.every_n == 5
