"""One-shot "how is it going" -- shared by `!babble status` and `babble summary`.

Deliberately torch-free: reading the state of the run should not cost 200ms of
importing a deep learning framework, and should work while the trainer is busy.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT, Settings
from .consent import SCOPE_CORPUS, SCOPE_CORRECTIONS, ConsentStore, CorpusConsent
from .corpus import CorpusStore
from .store import APPROVAL, CORRECTION, InteractionStore


@dataclass
class Snapshot:
    step: int
    last_loss: float | None
    first_loss: float | None
    checkpoints: int
    consented_users: int
    known_users: int
    stored_rows: int
    corrections: int
    approvals: int
    trainable_rows: int
    corpus_rows: int
    corpus_trainable: int
    corpus_chars: int
    log_bytes: int
    last_checkpoint_at: str | None
    running_commit: str | None
    update_checked_at: str | None
    update_remote_commit: str | None
    update_current: bool | None


def running_commit(root: Path = REPO_ROOT) -> str | None:
    """Short git SHA of the code actually running, or None off a real checkout
    (e.g. a tarball install, or the tmp dirs the tests build settings on)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def update_status(settings: Settings) -> dict:
    """The last drift check `deploy/update-live.sh` recorded, or an empty dict
    if the timer has never run here -- e.g. this checkout, or a box with the
    self-update timer not yet installed."""
    path = settings.update_state_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def loss_history(settings: Settings) -> list[dict]:
    """Every checkpoint ever written, oldest first. Read-only."""
    path = settings.loss_curve_path
    if not path.exists():
        return []
    history = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return history


def snapshot(settings: Settings) -> Snapshot:
    store = InteractionStore(settings.interactions_path)
    consent = ConsentStore(settings.consent_path)
    rows = store.all()
    corpus = CorpusStore(settings.corpus_path).all()
    history = loss_history(settings)
    tally = {}
    for row in rows:
        tally[row.signal] = tally.get(row.signal, 0) + 1

    granted = set(consent.granted_ids(SCOPE_CORRECTIONS))
    # "Rows the trainer would actually use", per store and per grant: a
    # correction needs both parties on the corrections grant, a corpus row needs
    # its one author on whichever grant governs where that row came from.
    allowed: set[str] = set()
    consented_users = 0
    corpus_trainable = 0
    if granted or consent.granted_ids(SCOPE_CORPUS):
        from .identity import Pseudonymiser

        ids = Pseudonymiser.load(settings)
        allowed = {ids.user(uid) for uid in granted}
        gate = CorpusConsent(consent, ids)
        consented_users = len(gate)
        corpus_trainable = sum(1 for r in corpus if gate.allows(r))
    trainable = sum(1 for r in rows if r.prompt_author in allowed and r.signal_author in allowed)

    log_bytes = 0
    if settings.log_dir.exists():
        log_bytes = sum(p.stat().st_size for p in settings.log_dir.glob("babble.*"))

    # Drift: the running commit is read fresh (cheap, local `git`); whether it
    # is current is judged against origin/main as of the last time
    # `deploy/update-live.sh` actually fetched, not by fetching here -- a
    # summary must never hit the network to answer "is booper current?".
    update = update_status(settings)
    local_commit = running_commit()
    remote_commit = update.get("remote_commit")
    update_current = None
    if local_commit and remote_commit:
        update_current = remote_commit.startswith(local_commit)

    return Snapshot(
        step=int(history[-1].get("step", 0)) if history else 0,
        last_loss=float(history[-1]["loss"]) if history and "loss" in history[-1] else None,
        first_loss=float(history[0]["loss"]) if history and "loss" in history[0] else None,
        checkpoints=len(list(settings.checkpoint_dir.glob("ckpt-*.pt")))
        if settings.checkpoint_dir.exists()
        else 0,
        consented_users=consented_users,
        known_users=len(consent.known_ids()),
        stored_rows=len(rows),
        corrections=tally.get(CORRECTION, 0),
        approvals=tally.get(APPROVAL, 0),
        trainable_rows=trainable,
        corpus_rows=len(corpus),
        corpus_trainable=corpus_trainable,
        corpus_chars=sum(len(r.text) for r in corpus),
        log_bytes=log_bytes,
        last_checkpoint_at=history[-1].get("at") if history else None,
        running_commit=local_commit,
        update_checked_at=update.get("checked_at"),
        update_remote_commit=remote_commit,
        update_current=update_current,
    )


def render_snapshot(snap: Snapshot, markdown: bool = True) -> str:
    b = "**" if markdown else ""
    loss = f"{snap.last_loss:.4f}" if snap.last_loss is not None else "—"
    drift = ""
    if snap.first_loss is not None and snap.last_loss is not None:
        drift = f" ({snap.last_loss - snap.first_loss:+.4f} since the first checkpoint)"
    return (
        f"{b}step{b} {snap.step:,} · {b}loss{b} {loss}{drift}\n"
        f"{b}corpus{b} {snap.corpus_rows} rows ({snap.corpus_trainable} training, "
        f"{snap.corpus_chars:,} chars) · {b}checkpoints{b} {snap.checkpoints}\n"
        f"{b}corrections{b} {snap.corrections} · {b}👍{b} {snap.approvals} · "
        f"{b}trainable pairs{b} {snap.trainable_rows}\n"
        f"{b}people opted in{b} {snap.consented_users} of {snap.known_users} asked"
    )


def render_drift(snap: Snapshot) -> str:
    """The `code` line `babble summary` appends: the commit actually running,
    and whether it matched `origin/main` as of the last check by
    `deploy/update-live.sh`. Deliberately separate from `render_snapshot` --
    that one is also what `!babble status` sends to Discord, which this drift
    detail is not part of."""
    running = snap.running_commit or "unknown (not a git checkout)"
    if snap.update_current is True:
        return f"{running} · current with origin/main (checked {snap.update_checked_at})"
    if snap.update_current is False:
        remote = (snap.update_remote_commit or "?")[:12]
        return f"{running} · BEHIND origin/main ({remote}) as of {snap.update_checked_at}"
    if snap.update_checked_at is not None:
        return f"{running} · origin/main: unknown (last check unreadable)"
    return f"{running} · origin/main: unknown (self-update timer has never checked)"


def render_curve(history: list[dict], width: int = 64, height: int = 14) -> str:
    """A loss curve you can read in a terminal or paste into Discord."""
    points = [(int(h.get("step", i)), float(h["loss"])) for i, h in enumerate(history) if "loss" in h]
    if not points:
        return "no checkpoints yet — run `babble train` and give it a minute."
    if len(points) == 1:
        step, loss = points[0]
        return f"one checkpoint so far: step {step}, loss {loss:.4f}"

    # Downsample into columns by averaging, so a long run still fits.
    buckets: list[list[float]] = [[] for _ in range(width)]
    span = len(points) - 1
    for index, (_, loss) in enumerate(points):
        buckets[min(width - 1, index * width // (span + 1))].append(loss)
    series = [(sum(b) / len(b)) for b in buckets if b]
    columns = len(series)

    hi, lo = max(series), min(series)
    rows = [[" "] * columns for _ in range(height)]
    for x, value in enumerate(series):
        y = 0 if hi == lo else int(round((hi - value) / (hi - lo) * (height - 1)))
        rows[y][x] = "•"

    label_width = 8
    lines = []
    for y, row in enumerate(rows):
        if y == 0:
            label = f"{hi:.3f}"
        elif y == height - 1:
            label = f"{lo:.3f}"
        else:
            label = ""
        lines.append(f"{label:>{label_width}} │{''.join(row)}")
    lines.append(f"{'':>{label_width}} └{'─' * columns}")
    lines.append(f"{'':>{label_width}}  step {points[0][0]:,}{' ' * max(1, columns - 20)}{points[-1][0]:,}")
    return "\n".join(lines)
