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
    # Where the external base-stage corpus (dictionary words + simple stories) is
    # cached. This is NOT user data and never touches the consent path.
    external_dir: Path

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

    # --- two-stage pretraining -------------------------------------------
    # Stage 1 (BASE) trains from random init on an external corpus: a real
    # English word list for word shape, and simple short stories for grammar.
    # Neither is anybody's message, so neither goes through consent.
    wordlist_path: Path = Path("/usr/share/dict/cracklib-small")
    stories_repo: str = "roneneldan/TinyStories"
    stories_file: str = "TinyStoriesV2-GPT4-valid.txt"
    # How many characters of stories to keep. The valid split is ~22M chars; the
    # rough ~20-tokens-per-parameter rule wants tens of millions for a 3.3M model,
    # so the whole valid split is a sane default. Point `stories_file` at a train
    # split and raise this for more. 0 means "keep all of it".
    base_story_chars: int = 20_000_000
    base_word_limit: int = 0  # 0 = every usable word in the list
    base_steps: int = 3_000  # stage-1 step budget
    voice_steps: int = 400  # stage-2 step budget; cheap, seconds not minutes

    # Stage 2 (VOICE) re-fires every this-many new corpus rows since the last
    # voice pass -- a trigger, not a loop. The count is persisted so a restart
    # does not re-fire. 0 turns the automatic trigger off (on-demand only).
    voice_trigger_rows: int = 100
    # `voice_steps` is a ceiling, not a target: the voice pass keeps whichever
    # checkpoint had the lowest val loss, and stops early once this many
    # checkpoint intervals in a row fail to improve on it. 0 turns early
    # stopping off (always run the full step ceiling).
    voice_patience: int = 3

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = root or REPO_ROOT
        return cls(
            data_dir=_env_path("BABBLE_DATA_DIR", root / "data"),
            checkpoint_dir=_env_path("BABBLE_CHECKPOINT_DIR", root / "checkpoints"),
            export_dir=_env_path("BABBLE_EXPORT_DIR", root / "export"),
            log_dir=_env_path("BABBLE_LOG_DIR", root / "logs"),
            external_dir=_env_path("BABBLE_EXTERNAL_DIR", root / "external"),
            train_threads=_env_int("BABBLE_TRAIN_THREADS", 2),
            train_nice=_env_int("BABBLE_TRAIN_NICE", 19),
            steps_per_cycle=_env_int("BABBLE_STEPS_PER_CYCLE", 200),
            checkpoint_every=_env_int("BABBLE_CHECKPOINT_EVERY", 50),
            rest_seconds=_env_float("BABBLE_REST_SECONDS", 60.0),
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
            wordlist_path=_env_path("BABBLE_WORDLIST_PATH", Path("/usr/share/dict/cracklib-small")),
            stories_repo=os.environ.get("BABBLE_STORIES_REPO", "roneneldan/TinyStories"),
            stories_file=os.environ.get("BABBLE_STORIES_FILE", "TinyStoriesV2-GPT4-valid.txt"),
            base_story_chars=_env_int("BABBLE_BASE_STORY_CHARS", 20_000_000),
            base_word_limit=_env_int("BABBLE_BASE_WORD_LIMIT", 0),
            base_steps=_env_int("BABBLE_BASE_STEPS", 3_000),
            voice_steps=_env_int("BABBLE_VOICE_STEPS", 400),
            voice_trigger_rows=_env_int("BABBLE_VOICE_TRIGGER_ROWS", 100),
            voice_patience=_env_int("BABBLE_VOICE_PATIENCE", 3),
        )

    @classmethod
    def for_root(cls, root: Path) -> "Settings":
        """A self-contained settings object under one directory. Used by tests."""
        return cls(
            data_dir=root / "data",
            checkpoint_dir=root / "checkpoints",
            export_dir=root / "export",
            log_dir=root / "logs",
            external_dir=root / "external",
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
    def base_checkpoint(self) -> Path:
        """The frozen stage-1 base weights. Written by `base-pretrain`, read by
        `voice-pass`, and never overwritten by a voice pass."""
        return self.checkpoint_dir / "base.pt"

    @property
    def base_corpus_path(self) -> Path:
        """The prepared external base corpus (JSONL of `{"text": ...}` rows)."""
        return self.external_dir / "base_corpus.jsonl"

    @property
    def voice_state_path(self) -> Path:
        """Persisted last-trained corpus row count, so the +N-row voice trigger
        does not re-fire on a restart."""
        return self.checkpoint_dir / "voice_state.json"

    @property
    def checkpoint_archive_dir(self) -> Path:
        """Where a base retrain moves the now-incompatible old checkpoints."""
        return self.checkpoint_dir / "archive"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.checkpoint_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


def discord_token() -> str | None:
    """The bot token, or None. Never defaulted, never guessed, never committed."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None
