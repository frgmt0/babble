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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


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
    # Dropout was always a field on `ModelConfig` but never reachable from
    # here -- `config_from_settings` silently pinned it to 0.0, so no dropout
    # result could ever ship. Plumbed now (BABBLE_DROPOUT). Training-only:
    # eval and generation run in eval mode where dropout is a no-op. Default
    # 0.2 is the replicated sweep winner (with lr 3e-4 + cosine: best val
    # 2.618/2.639/2.634 across seeds vs 2.794 for the old recipe).
    dropout: float = 0.2
    # 512, not the old 256: ro asked for a considerably wider context window. A
    # bigger `block_size` grows the learned positional-embedding table, so it
    # invalidates every checkpoint trained at the old size -- which is why it
    # lands together with the base retrain, never on its own.
    block_size: int = 512

    # Optimisation.
    batch_size: int = 8
    # 3e-4, not the old 1e-3: the long-budget LR sweep put 3e-4 ahead by
    # ~0.14-0.18 nats of held-out corpus val, replicated across three seeds
    # (spread 0.02) -- larger than the measured split-noise IQR of 0.074.
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Cosine-anneal the LR from `learning_rate` down to a tenth of it across
    # the step budget (BABBLE_TRAIN_COSINE). A decaying LR settles the val
    # curve near its minimum instead of bouncing across it at full stride.
    # On by default: part of the winning sweep config (see PIPELINE_REVAMP).
    train_cosine: bool = True

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
    # 1600 matches the winning sweep lanes exactly (cosine schedule length is
    # part of the recipe; best val landed at steps 600-800 of a 1600 budget).
    train_steps: int = 1_600
    train_patience: int = 3

    # Early stopping was firing on noise: on this corpus the val estimate has
    # a measured spread of ~0.05 nats (std over thousands of resampled 81-row
    # holdouts, checkpoint held fixed), and the run that shipped the last live
    # checkpoint was killed at step 350 by a 0.075 val wobble while train loss
    # was still falling fast. Two guards, both measured rather than guessed:
    # `train_min_steps` is a floor before patience may fire at all, and a
    # checkpoint only counts as a patience "stall" when val exceeds the best
    # seen by more than `train_stall_margin` -- movement inside the noise band
    # neither advances best nor burns patience. Margin 0 restores the old
    # any-non-improvement behaviour.
    # The floor must sit inside the step budget or patience is dead code --
    # `train_steps` and this move together, never separately.
    train_min_steps: int = 600
    train_stall_margin: float = 0.05

    # Mix the labelled synthetic corpus rows (`babble synth-corpus`,
    # data/synthetic_corpus.jsonl) into the *train* side of the pretrain
    # split. Validation always stays 100% real held-out human rows, so this
    # flag can never flatter the val number it is judged by. On by default:
    # the ±synthetic comparison on the winner config was 2.581 (three synth
    # seeds) vs 2.647 (mean of the two clean baseline seeds) best val -- the
    # largest single lever in the sweep. If data/synthetic_corpus.jsonl is
    # absent the mix is silently empty; if it is stale (the corpus grew and
    # the val holdout moved), the trainer rebuilds it before mixing
    # (`synthcorpus.refresh_synthetic_corpus_if_stale`), so a file generated
    # against an older corpus can never leak now-held-out phrasing into
    # training.
    train_synthetic: bool = True

    # Re-fires every this-many new corpus rows since the last run -- a trigger,
    # not a loop. The count is persisted so a restart does not re-fire. 0 turns
    # the automatic trigger off (on-demand only, with `--force`).
    train_trigger_rows: int = 100

    # --- post-training on the correction pairs ---------------------------
    # A short supervised pass, run after pretraining, that fine-tunes the
    # served checkpoint on the `(prompt, chosen)` correction pairs so the
    # model learns to answer a prompt instead of merely continuing one. Small
    # by design: there are only a few dozen pairs, and the point is to prove
    # the mechanism, not to out-run the data.
    post_steps: int = 200  # step ceiling; the best-val checkpoint may win earlier
    post_patience: int = 3  # stop after this many non-improving checkpoints, 0 = never
    # Which layout post-train teaches. The bot serves plain continuations
    # (`<bos> text`, `generate.best_continuation`) and never emits `<sep>` at
    # inference -- so the historical "pair" layout (`<bos> prompt <sep>
    # response <eos>`) trained a format that was unreachable at serving time,
    # and the only thing that transferred was the damage. "continuation" lays
    # a pair out as `<bos> prompt response <eos>` with the prompt masked --
    # byte-identical to the context the bot generates from. "pair" restores
    # the old layout for experiments.
    post_layout: str = "continuation"
    # Fine-tuning a 3.3M-param model on a few dozen pairs at the pretrain LR
    # (1e-3) tore straight through the pretrain: pair-val rose from its very
    # first checkpoint while pair-train memorised. Post-train gets its own,
    # far lower LR. 0 or negative falls back to `learning_rate`.
    post_learning_rate: float = 1e-4
    # Refuse to *run* below this many trainable pairs unless forced -- there
    # is nothing a supervised pass can generalise from at a few dozen rows.
    post_min_pairs: int = 100
    # Promotion gate: after post-train, the candidate is scored against the
    # pretrain snapshot on the real-corpus validation split (the layout the
    # bot actually serves). If the candidate is worse by more than this
    # margin, `latest.pt` is left alone and the run reports itself gated.
    # The margin is the measured val noise band, same figure as
    # `train_stall_margin`. Negative disables the gate.
    post_gate_margin: float = 0.05
    # Fraction of each post-train batch drawn from plain corpus rows
    # (rehearsal) rather than pairs, so the fine-tune cannot drift the
    # weights off the corpus distribution unopposed. 0 disables.
    post_rehearsal: float = 0.5
    # Re-fires every this-many new correction pairs since the last post-train
    # -- a trigger, not a loop, same discipline as the voice pass. The count
    # is persisted so a restart does not re-fire. 0 turns it off (on-demand
    # only). Small by default: there are only ~36 pairs to begin with.
    post_trigger_pairs: int = 10

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
            dropout=_env_float("BABBLE_DROPOUT", 0.2),
            block_size=_env_int("BABBLE_BLOCK_SIZE", 512),
            batch_size=_env_int("BABBLE_BATCH_SIZE", 8),
            learning_rate=_env_float("BABBLE_LEARNING_RATE", 3e-4),
            weight_decay=_env_float("BABBLE_WEIGHT_DECAY", 0.01),
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
            train_steps=_env_int("BABBLE_TRAIN_STEPS", 1_600),
            train_patience=_env_int("BABBLE_TRAIN_PATIENCE", 3),
            train_min_steps=_env_int("BABBLE_TRAIN_MIN_STEPS", 600),
            train_cosine=_env_bool("BABBLE_TRAIN_COSINE", True),
            train_stall_margin=_env_float("BABBLE_TRAIN_STALL_MARGIN", 0.05),
            train_synthetic=_env_bool("BABBLE_TRAIN_SYNTHETIC", True),
            train_trigger_rows=_env_int("BABBLE_TRAIN_TRIGGER_ROWS", 100),
            post_steps=_env_int("BABBLE_POST_STEPS", 200),
            post_patience=_env_int("BABBLE_POST_PATIENCE", 3),
            post_layout=os.environ.get("BABBLE_POST_LAYOUT", "continuation"),
            post_learning_rate=_env_float("BABBLE_POST_LEARNING_RATE", 1e-4),
            post_min_pairs=_env_int("BABBLE_POST_MIN_PAIRS", 100),
            post_gate_margin=_env_float("BABBLE_POST_GATE_MARGIN", 0.05),
            post_rehearsal=_env_float("BABBLE_POST_REHEARSAL", 0.5),
            post_trigger_pairs=_env_int("BABBLE_POST_TRIGGER_PAIRS", 10),
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
    def synthetic_pairs_path(self) -> Path:
        """Synthetic (prompt, response) correction pairs, generated from the
        corpus rather than typed by a human -- kept in their own file, never
        appended to `interactions.jsonl`, so a synthetic pair can never be
        mistaken for a human correction. See `babble/synthetic.py`."""
        return self.data_dir / "synthetic_pairs.jsonl"

    @property
    def synthetic_corpus_path(self) -> Path:
        """Synthetic corpus-style rows recombined from the corpus itself --
        kept in their own file, never appended to `corpus.jsonl`, so a
        synthetic row can never be mistaken for something a human typed. See
        `babble/synthcorpus.py`."""
        return self.data_dir / "synthetic_corpus.jsonl"

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
    def pretrained_checkpoint(self) -> Path:
        """The pretrained weights post-train fine-tunes from, snapshotted from
        `latest.pt`. Post-train always restarts from here, never from a
        previous post-train's own output, so reruns never compound. The
        snapshot is retaken whenever `latest.pt` no longer matches what the
        last post-train itself wrote there -- see `pretrained_snapshot_stale`
        in `post_state.py`."""
        return self.checkpoint_dir / "pretrained.pt"

    @property
    def post_state_path(self) -> Path:
        """Persisted last-trained correction-pair count, so the +N-pair
        post-train trigger does not re-fire on a restart."""
        return self.checkpoint_dir / "post_state.json"

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
