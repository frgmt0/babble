"""`deploy/update-live.sh` against a throwaway git fixture -- never the real
`~/babble-live`, which this test suite must never touch or even know about.

Every test builds a bare "origin" repo plus a clone that plays the live
install, runs the real script as a subprocess against them (with `git`
plumbing only -- no network), and reads back its exit code, the log lines it
appended, and `data/update_state.json`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "update-live.sh"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare 'origin' with one commit on main, and a clone of it -- the
    fixture every test starts from."""
    origin = tmp_path / "origin.git"
    _git("init", "--bare", "-q", str(origin), cwd=tmp_path)

    seed = tmp_path / "seed"
    _git("clone", "-q", str(origin), str(seed), cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=seed)
    _git("config", "user.name", "test", cwd=seed)
    (seed / "file.txt").write_text("hello\n")
    _git("add", "file.txt", cwd=seed)
    _git("commit", "-q", "-m", "first", cwd=seed)
    _git("branch", "-M", "main", cwd=seed)
    _git("push", "-q", "origin", "main", cwd=seed)

    live = tmp_path / "live"
    _git("clone", "-q", "-b", "main", str(origin), str(live), cwd=tmp_path)
    return origin, live


def _advance_origin(tmp_path: Path, message: str = "second") -> None:
    seed = tmp_path / "seed"
    with (seed / "file.txt").open("a") as fh:
        fh.write(f"{message}\n")
    _git("commit", "-aq", "-m", message, cwd=seed)
    _git("push", "-q", "origin", "main", cwd=seed)


def _run(tmp_path: Path, live: Path, origin: Path, *, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env.update(
        {
            "BABBLE_LIVE_DIR": str(live),
            "BABBLE_UPDATE_REMOTE": str(origin),
            "BABBLE_BOT_UNIT": "babble-bot-under-test",
            "BABBLE_UPDATE_RESTART_TIMEOUT": "3",
            "BABBLE_UPDATE_POLL_INTERVAL": "1",
        }
    )
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, env=full_env, capture_output=True, text=True
    )


def _state(live: Path) -> dict:
    return json.loads((live / "data" / "update_state.json").read_text())


def _log_text(live: Path) -> str:
    path = live / "logs" / "babble.log"
    return path.read_text() if path.exists() else ""


def test_already_current_is_a_silent_noop(tmp_path: Path) -> None:
    origin, live = _init_origin_and_clone(tmp_path)

    result = _run(tmp_path, live, origin)

    assert result.returncode == 0
    state = _state(live)
    assert state["up_to_date"] is True
    assert state["last_action"] == "noop"
    log = _log_text(live)
    assert "update.noop" in log
    assert "update.merged" not in log
    assert "update.restarted" not in log


def test_wrong_origin_fails_loudly_and_does_not_repoint(tmp_path: Path) -> None:
    origin, live = _init_origin_and_clone(tmp_path)
    wrong = str(tmp_path / "not-babble.git")
    _git("remote", "set-url", "origin", wrong, cwd=live)

    result = _run(tmp_path, live, origin)

    assert result.returncode != 0
    assert "origin" in result.stderr
    # never silently repointed -- the 2026-08-15 bug this guards against
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=live, capture_output=True, text=True
    ).stdout.strip()
    assert remote == wrong
    assert not (live / "data" / "update_state.json").exists()


def test_dirty_tree_refuses_to_merge(tmp_path: Path) -> None:
    origin, live = _init_origin_and_clone(tmp_path)
    _advance_origin(tmp_path)
    with (live / "file.txt").open("a") as fh:
        fh.write("uncommitted local edit\n")

    result = _run(tmp_path, live, origin)

    assert result.returncode != 0
    assert "uncommitted" in result.stderr
    state = _state(live)
    assert state["last_action"] == "skipped_dirty"
    assert state["up_to_date"] is False
    # HEAD never moved -- the merge itself was refused, not just logged
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=live, capture_output=True, text=True
    ).stdout.strip()
    assert head == state["local_commit"]
    assert "update.merged" not in _log_text(live)


def test_untracked_files_do_not_block_a_merge(tmp_path: Path) -> None:
    """Only *tracked* changes should refuse the merge -- an untracked scratch
    file lying around must not wedge the live install forever."""
    origin, live = _init_origin_and_clone(tmp_path)
    _advance_origin(tmp_path)
    (live / "scratch.tmp").write_text("not tracked by git\n")
    fakebin = _fake_systemctl(tmp_path, ready=True)

    result = _run(
        tmp_path,
        live,
        origin,
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "BABBLE_LOG_DIR": str(live / "logs")},
    )

    assert result.returncode == 0, result.stderr
    assert _state(live)["up_to_date"] is True


def test_training_in_flight_skips_the_restart(tmp_path: Path) -> None:
    origin, live = _init_origin_and_clone(tmp_path)
    _advance_origin(tmp_path)

    # Detection reads real argv out of /proc, token by token: an interpreter
    # as argv0 and a script *named* "babble" as argv1, mirroring what the
    # kernel actually leaves in /proc/<pid>/cmdline for the installed
    # console script once its shebang is resolved -- not just those words
    # somewhere in one long string.
    fake_bin = tmp_path / "fake-babble-bin"
    fake_bin.mkdir()
    fake_babble = fake_bin / "babble"
    fake_babble.write_text("import time\ntime.sleep(20)\n")
    marker = subprocess.Popen([sys.executable, str(fake_babble), "train"])
    try:
        time.sleep(0.3)
        result = _run(tmp_path, live, origin)
    finally:
        marker.terminate()
        marker.wait()

    assert result.returncode == 0
    state = _state(live)
    assert state["last_action"] == "skipped_training"
    assert state["up_to_date"] is False
    log = _log_text(live)
    assert "reason=training_in_flight" in log
    assert "update.merged" not in log
    assert "update.restarted" not in log
    # HEAD never moved -- the merge itself was skipped, not just the restart
    local_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=live, capture_output=True, text=True
    ).stdout.strip()
    assert local_sha == state["local_commit"]


def test_post_train_in_flight_skips_the_restart(tmp_path: Path) -> None:
    """`post-train` must count as "training in flight" under the default
    BABBLE_TRAIN_SUBCOMMANDS just like `train` does -- it's the stage-2 fine-tune
    and a mid-write restart during it is exactly as destructive."""
    origin, live = _init_origin_and_clone(tmp_path)
    _advance_origin(tmp_path)

    fake_bin = tmp_path / "fake-babble-bin"
    fake_bin.mkdir()
    fake_babble = fake_bin / "babble"
    fake_babble.write_text("import time\ntime.sleep(20)\n")
    marker = subprocess.Popen([sys.executable, str(fake_babble), "post-train"])
    try:
        time.sleep(0.3)
        result = _run(tmp_path, live, origin)
    finally:
        marker.terminate()
        marker.wait()

    assert result.returncode == 0
    state = _state(live)
    assert state["last_action"] == "skipped_training"
    assert state["up_to_date"] is False
    log = _log_text(live)
    assert "reason=training_in_flight" in log
    assert "update.merged" not in log
    assert "update.restarted" not in log
    local_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=live, capture_output=True, text=True
    ).stdout.strip()
    assert local_sha == state["local_commit"]


def _fake_systemctl(tmp_path: Path, *, ready: bool) -> Path:
    """A stand-in for `systemctl --user` that either logs a fresh `bot.ready`
    shortly after a restart, or never does -- so the verify step can be tested
    both ways without a real bot or a real systemd unit."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "systemctl"
    if ready:
        body = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            if [ "$1" = "--user" ] && [ "$2" = "restart" ]; then
              (
                sleep 0.3
                printf '%s  bot.ready              bot=test\\n' \\
                  "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> "$BABBLE_LOG_DIR/babble.log"
              ) &
              disown
              exit 0
            fi
            exit 0
            """
        )
    else:
        body = "#!/usr/bin/env bash\nexit 0\n"
    script.write_text(body)
    script.chmod(0o755)
    return bindir


def test_behind_merges_syncs_restarts_and_verifies(tmp_path: Path) -> None:
    origin, live = _init_origin_and_clone(tmp_path)
    _advance_origin(tmp_path)
    before_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=live, capture_output=True, text=True
    ).stdout.strip()
    fakebin = _fake_systemctl(tmp_path, ready=True)

    result = _run(
        tmp_path,
        live,
        origin,
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "BABBLE_LOG_DIR": str(live / "logs")},
    )

    assert result.returncode == 0, result.stderr
    state = _state(live)
    assert state["up_to_date"] is True
    assert state["last_action"] == "updated"
    assert state["local_commit"] != before_sha
    log = _log_text(live)
    assert "update.merged" in log
    assert "update.restarted" in log
    assert "update.done" in log
    new_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=live, capture_output=True, text=True
    ).stdout.strip()
    assert new_sha == state["local_commit"]


def test_bot_not_coming_back_fails_loudly(tmp_path: Path) -> None:
    origin, live = _init_origin_and_clone(tmp_path)
    _advance_origin(tmp_path)
    fakebin = _fake_systemctl(tmp_path, ready=False)

    result = _run(
        tmp_path,
        live,
        origin,
        env={"PATH": f"{fakebin}:{os.environ['PATH']}", "BABBLE_LOG_DIR": str(live / "logs")},
    )

    assert result.returncode != 0
    assert "bot.ready" in result.stderr or "did not log" in result.stderr
    state = _state(live)
    assert state["last_action"] == "failed_restart_verify"
    assert state["up_to_date"] is False
    # the merge itself still happened -- only the restart verification failed
    assert state["local_commit"] == state["remote_commit"]


def test_script_is_executable() -> None:
    assert SCRIPT.exists()
    assert os.access(SCRIPT, os.X_OK)


@pytest.mark.parametrize("unit_name", ["deploy/babble-update.service", "deploy/babble-update.timer"])
def test_systemd_units_exist(unit_name: str) -> None:
    path = Path(__file__).resolve().parent.parent / unit_name
    assert path.exists()
    text = path.read_text()
    assert "[Unit]" in text
