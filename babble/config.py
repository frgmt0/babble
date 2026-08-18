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

# A reply to one of the bot's messages only counts as a correction if it starts
# with this. Without it, every "lol" and "wrong" aimed at the bot was landing in
# the training corpus as the answer it should have given -- which is a corpus
# full of things nobody meant to teach it. Making the marker explicit is the
# difference between "someone replied" and "someone is teaching".
#
# It is stripped before the text is stored, so the marker never reaches the
# corpus, the dataset, or the model.
CORRECTION_MARKER = ">>"


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
    checkpoint_every: int = 50
    keep_checkpoints: int = 5

    # Model shape. Defaults are ~3.3M parameters; tests shrink these.
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    # 512, not the old 256: ro asked for a considerably wider context window. A
    # bigger `block_size` grows the learned positional-embedding table, so it
    # invalidates every checkpoint trained at the old size -- which is why it
    # lands together with the base retrain, never on its own.
    block_size: int = 512

    # Optimisation.
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Sampling. `temperature` was 1.0 and that was the single biggest reason a
    # model with a good-looking loss still answered "hi" with noise: at 1.0 the
    # sampler faithfully reproduces the model's residual uncertainty, and one
    # unlucky byte puts the rest of the response off the memorised path with no
    # way back. See "Why it babbled at loss 0.02" in the README.
    temperature: float = 0.5
    top_k: int = 40
    # 256, not the old 96: ro asked for a considerably longer reply. The model
    # still stops early on <eos>; this only raises the ceiling.
    max_new_tokens: int = 256

    # Best-of-n: draw this many candidates and keep whichever one the model
    # itself scores best. 1 turns it off. Kept small so a Discord reply stays
    # under a couple of seconds.
    best_of: int = 4

    # How much each kind of feedback is worth. These no longer touch training:
    # the objective is plain next-token prediction over unlabelled corpus text,
    # where there is no "chosen" answer to weight and every row counts the same.
    # They survive as metadata on the stored correction rows, which are still
    # captured and still published as their own dataset config.
    correction_weight: float = 1.0
    approval_weight: float = 0.25
    correction_boost: float = 3.0

    # Held-out validation. A stable hash of each corpus row's id decides its side
    # of the split, so the same row always lands on the same side as the corpus
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
    # `babble export --push`. 0 or None turns it off. This is the *trainer's*
    # cadence, and only fires while a trainer is running.
    hf_publish_every: int | None = 20

    # The collection phase has no trainer and so no checkpoints, so the bot
    # publishes on corpus *growth* instead: push once the corpus has gained this
    # many rows or this many characters since the last publish. Either 0 turns
    # that half off; both 0 turns the growth-based publish off entirely. See
    # `babble/publish.py`.
    hf_publish_every_rows: int = 10
    hf_publish_every_chars: int = 2_000

    # --- pretraining on the collected corpus ------------------------------
    # The only training path: random init, on the consented human corpus,
    # nothing else. `train_steps` is a ceiling, not a target -- training keeps
    # whichever checkpoint had the lowest val loss and stops early once
    # `train_patience` checkpoint intervals in a row fail to improve on it (0
    # turns early stopping off, always running the full step ceiling).
    train_steps: int = 400
    train_patience: int = 3

    # Re-fires every this-many new corpus rows since the last run -- a trigger,
    # not a loop. The count is persisted so a restart does not re-fire. 0 turns
    # the automatic trigger off (on-demand only, with `--force`).
    train_trigger_rows: int = 100

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
            checkpoint_every=_env_int("BABBLE_CHECKPOINT_EVERY", 50),
            keep_checkpoints=_env_int("BABBLE_KEEP_CHECKPOINTS", 5),
            n_layer=_env_int("BABBLE_N_LAYER", 4),
            n_head=_env_int("BABBLE_N_HEAD", 4),
            n_embd=_env_int("BABBLE_N_EMBD", 256),
            block_size=_env_int("BABBLE_BLOCK_SIZE", 512),
            batch_size=_env_int("BABBLE_BATCH_SIZE", 8),
            learning_rate=_env_float("BABBLE_LEARNING_RATE", 1e-3),
            temperature=_env_float("BABBLE_TEMPERATURE", 0.5),
            top_k=_env_int("BABBLE_TOP_K", 40),
            max_new_tokens=_env_int("BABBLE_MAX_NEW_TOKENS", 256),
            best_of=_env_int("BABBLE_BEST_OF", 4),
            correction_boost=_env_float("BABBLE_CORRECTION_BOOST", 3.0),
            val_fraction=_env_float("BABBLE_VAL_FRACTION", 0.2),
            val_min_rows=_env_int("BABBLE_VAL_MIN_ROWS", 20),
            hf_repo=os.environ.get("BABBLE_HF_REPO", "kowo-co/babble-corrections"),
            salt=os.environ.get(SALT_ENV) or None,
            hf_publish_every=_env_int("BABBLE_HF_PUBLISH_EVERY", 20),
            hf_publish_every_rows=_env_int("BABBLE_HF_PUBLISH_EVERY_ROWS", 10),
            hf_publish_every_chars=_env_int("BABBLE_HF_PUBLISH_EVERY_CHARS", 2_000),
            train_steps=_env_int("BABBLE_TRAIN_STEPS", 400),
            train_patience=_env_int("BABBLE_TRAIN_PATIENCE", 3),
            train_trigger_rows=_env_int("BABBLE_TRAIN_TRIGGER_ROWS", 100),
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
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

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

    @property
    def train_state_path(self) -> Path:
        """Persisted last-trained corpus row count, so the +N-row trigger does
        not re-fire on a restart."""
        return self.checkpoint_dir / "train_state.json"

    @property
    def update_state_path(self) -> Path:
        """Result of the last drift check by `deploy/update-live.sh` -- whether
        the running commit matched `origin/main` as of that check. Read-only
        from here; only the update script writes it."""
        return self.data_dir / "update_state.json"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.checkpoint_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


def discord_token() -> str | None:
    """The bot token, or None. Never defaulted, never guessed, never committed."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None
