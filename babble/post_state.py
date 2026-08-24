"""The post-train trigger and pair filtering -- torch-free.

Split out of `posttrain.py` for one reason: `babble summary` reads this module
to show the post-train trigger state, and `stats.py` is deliberately
torch-free so a summary is cheap and works while a trainer is busy. Importing
`posttrain.py` there would drag in `torch` and the model just to read a JSON
file and count rows, so the pure-Python parts live here instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .blocklist import Blocklist
from .config import Settings
from .consent import SCOPE_CORRECTIONS, ConsentStore
from .identity import Pseudonymiser
from .store import Interaction, InteractionStore
from .util import atomic_write_text, utcnow_iso


@dataclass
class PostTrigger:
    current_pairs: int
    last_trained_pairs: int
    threshold: int
    has_pretrained: bool
    min_pairs: int = 0

    @property
    def new_pairs(self) -> int:
        return self.current_pairs - self.last_trained_pairs

    @property
    def due(self) -> bool:
        """Automatic firing: a pretrained checkpoint exists, the threshold is
        on, the correction pairs have grown by at least that many since the
        last post-train, AND the total is at or past the `post_min_pairs`
        floor. Without the floor here, a below-floor pair count kept the
        trigger armed and `AutoPostTrigger` spawned a subprocess after every
        new correction only for `post_train` to refuse each one."""
        return (
            self.has_pretrained
            and self.threshold > 0
            and self.new_pairs >= self.threshold
            and self.current_pairs >= self.min_pairs
        )


def trainable_pairs(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> list[Interaction]:
    """Correction/approval rows whose every party still consents, checked
    right now -- the same belt-and-braces re-check `corpus_rows` gives the
    corpus: withdrawal already purges rows, but "used to train the model" is
    enforced at the moment of training, not only at the moment of capture.

    Sorted by id (content-addressed, so stable) rather than file order, so the
    train/val split a post-train draws from this list is deterministic.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)
    allowed = {ids.user(uid) for uid in consent.granted_ids(SCOPE_CORRECTIONS)}
    rows = InteractionStore(settings.interactions_path).all()
    trainable = [
        r
        for r in rows
        if r.prompt_author in allowed
        and r.signal_author in allowed
        and not blocklist.matches(r.prompt, r.chosen)
    ]
    return sorted(trainable, key=lambda r: r.id)


def pair_count(settings: Settings) -> int:
    """Total stored interaction rows -- the number the +N-pair trigger
    measures growth against. The raw stored count, not the consent-filtered
    one, so a revocation cannot make the count appear to shrink below the
    last trigger, mirroring `corpus_row_count`."""
    return len(InteractionStore(settings.interactions_path).all())


def read_post_state(settings: Settings) -> dict:
    path = settings.post_state_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def file_hash(path: Path) -> str | None:
    """A content fingerprint for a checkpoint file, or None if it does not
    exist. Used to tell whether `latest.pt` is still the exact file a
    post-train wrote, or something else has landed there since (a fresh
    pretrain, most likely) -- see `pretrained_snapshot_stale`."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretrained_snapshot_stale(settings: Settings) -> bool:
    """Is `checkpoints/pretrained.pt` no longer a faithful snapshot of the
    current pretrain?

    `pretrained.pt` is meant to be refreshed exactly once per pretrain: the
    first post-train after a new `latest.pt` lands copies it, and every
    post-train after that reuses the copy so reruns never compound on a
    previous post-train's weights. Telling those two cases apart means
    recording, in `post_state.json`, the hash of `latest.pt` *at the moment
    post-train last wrote to it* -- if the file on disk still matches that
    hash, nothing has touched it since and the existing snapshot is still
    clean pretrain. If it does not match (or there is no state yet), whatever
    is in `latest.pt` now is not post-train's own output and the snapshot
    needs retaking.
    """
    if not settings.pretrained_checkpoint.exists():
        return True
    current = file_hash(settings.latest_checkpoint)
    if current is None:
        return False  # nothing to re-snapshot from; keep the existing snapshot
    return read_post_state(settings).get("latest_hash") != current


def write_post_state(settings: Settings, *, pairs: int, step: int, latest_hash: str | None) -> None:
    atomic_write_text(
        settings.post_state_path,
        json.dumps(
            {"last_trained_pairs": pairs, "step": step, "at": utcnow_iso(), "latest_hash": latest_hash},
            indent=2,
        ),
    )


def record_rollback(settings: Settings, *, restored_from: str, restored_step: int | None) -> None:
    """Note that `babble ab rollback` just restored an archived checkpoint
    into `latest.pt` -- called only there, never by post-train itself.

    Also rewrites `latest_hash` to the restored file's own hash. Without
    that, `pretrained_snapshot_stale` (which compares this field against the
    hash of whatever is on disk right now) would see `latest.pt` no longer
    matching what the last post-train itself wrote, read the rollback as a
    fresh `babble train` landing, and silently re-snapshot `pretrained.pt`
    from a *post-trained* checkpoint instead of a real pretrain -- exactly
    the compounding `pretrained.pt` exists to prevent. Recording the
    restored hash here keeps that snapshot exactly as it was before the
    promotion that just got rolled back, and updates the state file with the
    rollback timestamp so subsequent post-trains never miss it happened.
    """
    state = read_post_state(settings)
    state["latest_hash"] = file_hash(settings.latest_checkpoint)
    state["rolled_back_at"] = utcnow_iso()
    state["rolled_back_from"] = restored_from
    state["rolled_back_to_step"] = restored_step
    atomic_write_text(settings.post_state_path, json.dumps(state, indent=2))


def post_trigger(settings: Settings) -> PostTrigger:
    state = read_post_state(settings)
    return PostTrigger(
        current_pairs=pair_count(settings),
        last_trained_pairs=int(state.get("last_trained_pairs", 0)),
        threshold=settings.post_trigger_pairs,
        has_pretrained=settings.pretrained_checkpoint.exists() or settings.latest_checkpoint.exists(),
        min_pairs=max(0, settings.post_min_pairs),
    )
