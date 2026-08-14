"""One-shot "how is it going" -- shared by `!babble status` and `babble summary`.

Deliberately torch-free: reading the state of the run should not cost 200ms of
importing a deep learning framework, and should work while the trainer is busy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .consent import ConsentStore
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
    log_bytes: int
    last_checkpoint_at: str | None


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
    history = loss_history(settings)
    tally = {}
    for row in rows:
        tally[row.signal] = tally.get(row.signal, 0) + 1

    granted = set(consent.granted_ids())
    # "Rows the trainer would actually use": both parties still consenting.
    allowed: set[str] = set()
    if granted:
        from .identity import Pseudonymiser

        ids = Pseudonymiser.load(settings)
        allowed = {ids.user(uid) for uid in granted}
    trainable = sum(1 for r in rows if r.prompt_author in allowed and r.signal_author in allowed)

    log_bytes = 0
    if settings.log_dir.exists():
        log_bytes = sum(p.stat().st_size for p in settings.log_dir.glob("babble.*"))

    return Snapshot(
        step=int(history[-1].get("step", 0)) if history else 0,
        last_loss=float(history[-1]["loss"]) if history and "loss" in history[-1] else None,
        first_loss=float(history[0]["loss"]) if history and "loss" in history[0] else None,
        checkpoints=len(list(settings.checkpoint_dir.glob("ckpt-*.pt")))
        if settings.checkpoint_dir.exists()
        else 0,
        consented_users=len(granted),
        known_users=len(consent.known_ids()),
        stored_rows=len(rows),
        corrections=tally.get(CORRECTION, 0),
        approvals=tally.get(APPROVAL, 0),
        trainable_rows=trainable,
        log_bytes=log_bytes,
        last_checkpoint_at=history[-1].get("at") if history else None,
    )


def render_snapshot(snap: Snapshot, markdown: bool = True) -> str:
    b = "**" if markdown else ""
    loss = f"{snap.last_loss:.4f}" if snap.last_loss is not None else "—"
    drift = ""
    if snap.first_loss is not None and snap.last_loss is not None:
        drift = f" ({snap.last_loss - snap.first_loss:+.4f} since the first checkpoint)"
    return (
        f"{b}step{b} {snap.step:,} · {b}loss{b} {loss}{drift}\n"
        f"{b}checkpoints{b} {snap.checkpoints} · {b}corrections{b} {snap.corrections} · "
        f"{b}👍{b} {snap.approvals} · {b}trainable rows{b} {snap.trainable_rows}\n"
        f"{b}people opted in{b} {snap.consented_users} of {snap.known_users} asked"
    )


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
