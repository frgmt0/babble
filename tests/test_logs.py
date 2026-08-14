"""The log has to be safe to read while the thing is running.

Append-only, never truncated, rotated by size, pseudonymous, and silent about the
content of anyone who has not opted in.
"""

from __future__ import annotations

import json

from babble.logs import EventLog, follow, tail


def test_events_land_in_both_the_structured_and_the_prose_log(settings, log):
    log.event("train.checkpoint", step=50, loss=1.25)
    log.flush()

    entry = json.loads((settings.log_dir / "babble.jsonl").read_text().strip())
    assert entry["event"] == "train.checkpoint"
    assert entry["step"] == 50
    assert entry["component"] == "test"
    assert "ts" in entry

    prose = (settings.log_dir / "babble.log").read_text()
    assert "train.checkpoint" in prose
    assert "step=50" in prose


def test_a_restart_appends_rather_than_clearing(settings, ids):
    first = EventLog(settings, ids)
    first.event("bot.start")
    first.close()

    second = EventLog(settings, ids)
    second.event("bot.stop")
    second.close()

    lines = (settings.log_dir / "babble.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["bot.start", "bot.stop"]


def test_reading_the_log_does_not_change_it(settings, log):
    for i in range(20):
        log.event("bot.ping", n=i)
    log.flush()
    path = settings.log_dir / "babble.log"
    before = (path.stat().st_size, path.stat().st_mtime_ns)

    assert len(tail(path, 5)) == 5

    assert (path.stat().st_size, path.stat().st_mtime_ns) == before


def test_tail_returns_the_last_lines_in_order(settings, log):
    for i in range(50):
        log.event("bot.ping", n=i)
    log.flush()

    lines = tail(settings.log_dir / "babble.jsonl", 3)

    assert [json.loads(line)["n"] for line in lines] == [47, 48, 49]


def test_it_rotates_by_size_and_keeps_the_old_file(settings, ids):
    settings.log_max_bytes = 400
    log = EventLog(settings, ids)

    for i in range(60):
        log.event("bot.ping", n=i, filler="x" * 40)
    log.flush()

    rotated = settings.log_dir / "babble.jsonl.1"
    assert rotated.exists(), "old log should be preserved, not discarded"
    assert rotated.stat().st_size > 0
    assert (settings.log_dir / "babble.jsonl").stat().st_size > 0
    # Rotation is bounded, so logging can never fill the disk.
    assert not (settings.log_dir / f"babble.jsonl.{settings.log_backups + 1}").exists()


def test_user_ids_are_pseudonymised_the_same_way_the_dataset_does(settings, log, ids):
    log.event("bot.ping", user=log.user("123456789012345678"))
    log.flush()

    body = (settings.log_dir / "babble.jsonl").read_text()
    assert "123456789012345678" not in body
    assert ids.user("123456789012345678") in body


def test_content_is_withheld_from_users_who_have_not_consented(log):
    allowed = log.preview("my secret message", allowed=True)
    withheld = log.preview("my secret message", allowed=False)

    assert allowed["text"] == "my secret message"
    assert withheld["chars"] == len("my secret message")
    assert "secret" not in withheld["text"]


def test_previews_are_clipped_so_one_message_cannot_flood_the_log(settings, log):
    settings.log_preview_chars = 20

    preview = log.preview("y" * 500, allowed=True)

    assert len(preview["text"]) <= 20
    assert preview["chars"] == 500


def test_follow_yields_lines_appended_after_it_caught_up(settings, log):
    log.event("bot.start")
    log.flush()
    path = settings.log_dir / "babble.log"

    stream = follow(path, from_start=True, poll=0.01)
    assert "bot.start" in next(stream)  # now parked at end of file

    log.event("bot.ready", guilds=1)
    log.flush()

    assert "bot.ready" in next(stream)
    stream.close()
