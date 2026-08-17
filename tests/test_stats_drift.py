"""Drift visibility: `babble summary` answering "is booper current?" without
shelling into the box -- reading the running commit fresh and comparing it
against whatever `deploy/update-live.sh` last recorded, never fetching."""

from __future__ import annotations

import json
from pathlib import Path

from babble.config import Settings
from babble.stats import render_drift, render_snapshot, running_commit, snapshot, update_status


def _write_state(settings: Settings, **fields) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.update_state_path.write_text(json.dumps(fields))


def test_running_commit_reads_this_checkout() -> None:
    commit = running_commit()
    assert commit is not None
    assert len(commit) >= 6


def test_running_commit_is_none_off_a_non_git_root(tmp_path: Path) -> None:
    assert running_commit(root=tmp_path) is None


def test_update_status_empty_when_never_checked(settings: Settings) -> None:
    assert update_status(settings) == {}


def test_update_status_empty_on_corrupt_state_file(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.update_state_path.write_text("not json")
    assert update_status(settings) == {}


def test_snapshot_current_when_running_matches_recorded_remote(settings: Settings) -> None:
    local = running_commit()
    assert local is not None
    _write_state(
        settings,
        checked_at="2026-08-17T07:15:23+00:00",
        local_commit=local + "0" * 33,
        remote_commit=local + "0" * 33,
        up_to_date=True,
        last_action="noop",
    )
    snap = snapshot(settings)
    assert snap.running_commit == local
    assert snap.update_current is True
    assert "current with origin/main" in render_drift(snap)


def test_snapshot_behind_when_recorded_remote_differs(settings: Settings) -> None:
    _write_state(
        settings,
        checked_at="2026-08-17T07:20:00+00:00",
        local_commit="a" * 40,
        remote_commit="b" * 40,
        up_to_date=False,
        last_action="skipped_dirty",
    )
    snap = snapshot(settings)
    assert snap.update_current is False
    rendered = render_drift(snap)
    assert "BEHIND origin/main" in rendered
    assert "b" * 12 in rendered


def test_snapshot_unknown_before_any_check(settings: Settings) -> None:
    snap = snapshot(settings)
    assert snap.update_checked_at is None
    assert snap.update_current is None
    assert "never checked" in render_drift(snap)


def test_render_snapshot_never_includes_drift(settings: Settings) -> None:
    """`render_snapshot` is also what `!babble status` sends to Discord --
    drift detail belongs only to `babble summary`'s separate `render_drift`
    line, never leaking into the bot's reply."""
    _write_state(
        settings,
        checked_at="2026-08-17T07:20:00+00:00",
        local_commit="a" * 40,
        remote_commit="b" * 40,
        up_to_date=False,
        last_action="skipped_dirty",
    )
    snap = snapshot(settings)
    rendered = render_snapshot(snap, markdown=False)
    assert "origin/main" not in rendered
    assert "BEHIND" not in rendered
    assert "code" not in rendered.lower()
