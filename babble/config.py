"""Where things live and how hard the trainer is allowed to work.

Every value is overridable by environment variable, so the same code runs from a
checkout, from a systemd unit, or from a test pointed at a tmp directory without
anything being edited.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TOKEN_ENV = "BABBLE_DISCORD_TOKEN"
SALT_ENV = "BABBLE_HASH_SALT"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    """Paths and knobs. Construct with `Settings.from_env()` in real runs."""

    data_dir: Path
    checkpoint_dir: Path
    export_dir: Path
    log_dir: Path

    # Trainer politeness. These exist so the box stays usable while it learns.
    train_threads: int = 2
    train_nice: int = 19
    steps_per_cycle: int = 200
    checkpoint_every: int = 50
    rest_seconds: float = 60.0
    keep_checkpoints: int = 5

    # Model shape. Defaults are ~3.3M parameters; tests shrink these.
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 256

    # Optimisation.
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Sampling.
    temperature: float = 1.0
    top_k: int = 40
    max_new_tokens: int = 96

    # How much each kind of feedback is worth. A correction is the real signal;
    # a thumbs-up is a cheap nod, so it nudges the weights far less.
    correction_weight: float = 1.0
    approval_weight: float = 0.25

    # Held-out validation. A stable hash of each row's id decides its side of
    # the split, so the same row always lands on the same side as the corpus
    # grows and as the trainer restarts. Below `val_min_rows`, holding out
    # `val_fraction` of a handful of rows could starve training entirely, so
    # validation is skipped and reported as disabled instead.
    val_fraction: float = 0.2
    val_min_rows: int = 20

    # Logging.
    log_max_bytes: int = 8 * 1024 * 1024
    log_backups: int = 3
    log_flush_seconds: float = 1.0
    log_preview_chars: int = 200

    hf_repo: str = "kowo-co/babble-corrections"
    salt: str | None = None

    # Auto-publish: every this-many checkpoints written, push the corrections
    # dataset to `hf_repo` through the same consent/blocklist gate as a manual
    # `babble export --push`. 0 or None turns it off.
    hf_publish_every: int | None = 20

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = root or REPO_ROOT
        return cls(
            data_dir=_env_path("BABBLE_DATA_DIR", root / "data"),
            checkpoint_dir=_env_path("BABBLE_CHECKPOINT_DIR", root / "checkpoints"),
            export_dir=_env_path("BABBLE_EXPORT_DIR", root / "export"),
            log_dir=_env_path("BABBLE_LOG_DIR", root / "logs"),
            train_threads=_env_int("BABBLE_TRAIN_THREADS", 2),
            train_nice=_env_int("BABBLE_TRAIN_NICE", 19),
            steps_per_cycle=_env_int("BABBLE_STEPS_PER_CYCLE", 200),
            checkpoint_every=_env_int("BABBLE_CHECKPOINT_EVERY", 50),
            rest_seconds=_env_float("BABBLE_REST_SECONDS", 60.0),
            keep_checkpoints=_env_int("BABBLE_KEEP_CHECKPOINTS", 5),
            n_layer=_env_int("BABBLE_N_LAYER", 4),
            n_head=_env_int("BABBLE_N_HEAD", 4),
            n_embd=_env_int("BABBLE_N_EMBD", 256),
            block_size=_env_int("BABBLE_BLOCK_SIZE", 256),
            batch_size=_env_int("BABBLE_BATCH_SIZE", 8),
            learning_rate=_env_float("BABBLE_LEARNING_RATE", 1e-3),
            temperature=_env_float("BABBLE_TEMPERATURE", 1.0),
            top_k=_env_int("BABBLE_TOP_K", 40),
            max_new_tokens=_env_int("BABBLE_MAX_NEW_TOKENS", 96),
            val_fraction=_env_float("BABBLE_VAL_FRACTION", 0.2),
            val_min_rows=_env_int("BABBLE_VAL_MIN_ROWS", 20),
            hf_repo=os.environ.get("BABBLE_HF_REPO", "kowo-co/babble-corrections"),
            salt=os.environ.get(SALT_ENV) or None,
            hf_publish_every=_env_int("BABBLE_HF_PUBLISH_EVERY", 20),
        )

    @classmethod
    def for_root(cls, root: Path) -> "Settings":
        """A self-contained settings object under one directory. Used by tests."""
        return cls(
            data_dir=root / "data",
            checkpoint_dir=root / "checkpoints",
            export_dir=root / "export",
            log_dir=root / "logs",
        )

    # --- derived paths -------------------------------------------------

    @property
    def interactions_path(self) -> Path:
        return self.data_dir / "interactions.jsonl"

    @property
    def consent_path(self) -> Path:
        return self.data_dir / "consent.json"

    @property
    def exchanges_path(self) -> Path:
        return self.data_dir / "exchanges.json"

    @property
    def salt_path(self) -> Path:
        return self.data_dir / ".salt"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def loss_curve_path(self) -> Path:
        return self.checkpoint_dir / "loss.jsonl"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.checkpoint_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


def discord_token() -> str | None:
    """The bot token, or None. Never defaulted, never guessed, never committed."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None
