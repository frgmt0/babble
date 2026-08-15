"""Two-stage pretraining: a frozen external base, then a human voice pass.

The model is too small and the human corpus too tiny for the old
`babble train --loop` to do anything but memorise a couple of thousand
characters. So training is split in two, and the loop is gone:

* **Stage 1 -- BASE** (`pretrain_base`). Train from *random init* on the external
  corpus (`external.py`: dictionary words + simple stories) at the current
  geometry, and freeze the result as `checkpoints/base.pt`. This is expensive and
  rare; it is a separate, deliberate run. `base.pt` is never overwritten by a
  voice pass.

* **Stage 2 -- VOICE** (`voice_pass`). Continue-train *from the frozen base* on
  the consented human corpus only, and write `latest.pt` (what the bot serves).
  Cheap -- seconds -- and it always restarts from the clean base, so nothing
  compounds across reruns and the human voice is always the last thing the model
  learned.

Stage 2 fires on a **trigger, not a loop**: every `voice_trigger_rows` new corpus
rows since the last pass, or on demand from the CLI. The last-trained row count
is persisted (`voice_state.json`) so a restart never re-fires.

The consent model is untouched: stage 1 reads the external corpus, which never
enters the consent path, and stage 2 reads the human rows through the exact same
consent + blocklist gate the old trainer used (`trainer.corpus_rows`).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Settings
from .corpus import CorpusStore
from .external import read_base_rows
from .logs import EventLog, NullLog
from .model import Babbler, ModelConfig, config_from_settings, sequence_loss
from .tokenizer import Example, text_examples
from .trainer import (
    SCRATCH_DIR,
    append_curve,
    be_polite,
    corpus_rows,
    eval_loss,
    make_batch,
    save_checkpoint,
    sweep_scratch,
)
from .util import atomic_write_text, utcnow_iso

# How much of the base corpus to hold out for a validation read, and a ceiling so
# a huge base corpus does not make each checkpoint's eval pass expensive.
_VAL_FRACTION = 0.02
_VAL_CAP = 512


@dataclass
class StageResult:
    stage: str
    steps_run: int
    final_step: int
    last_loss: float
    val_loss: float | None
    checkpoints_written: int
    path: Path


@dataclass
class VoiceResult:
    ran: bool
    reason: str
    rows_trained: int = 0
    current_rows: int = 0
    last_trained_rows: int = 0
    stage: StageResult | None = None


@dataclass
class TriggerStatus:
    current_rows: int
    last_trained_rows: int
    threshold: int
    has_base: bool

    @property
    def new_rows(self) -> int:
        return self.current_rows - self.last_trained_rows

    @property
    def due(self) -> bool:
        """Automatic firing: a base exists, the threshold is on, and the corpus
        has grown by at least that many rows since the last pass."""
        return self.has_base and self.threshold > 0 and self.new_rows >= self.threshold


# --- shared helpers -------------------------------------------------------


def _text_examples(rows: list[str], block_size: int) -> list[Example]:
    return [ex for text in rows for ex in text_examples(text, block_size)]


def _save_to(settings: Settings, path: Path, model: Babbler, optimizer, step: int, loss: float) -> Path:
    """Atomically write a checkpoint to an explicit path, staged in `.partial/`
    exactly like the trainer, so a half-written file is never at `path`."""
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scratch = settings.checkpoint_dir / SCRATCH_DIR
    scratch.mkdir(exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "saved_at": utcnow_iso(),
    }
    tmp = scratch / f"{path.name}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _run_stage(
    settings: Settings,
    model: Babbler,
    optimizer: torch.optim.Optimizer,
    examples: list[Example],
    val_examples: list[Example],
    *,
    stage: str,
    steps: int,
    seed: int,
    echo: bool,
    log: EventLog,
    save,
    start_step: int = 0,
) -> StageResult:
    """The inner training loop, shared by both stages.

    `save(step, loss)` writes the checkpoint for this stage -- to `base.pt` for
    stage 1, to `latest.pt` (+ an archive) for stage 2. Loss and validation are
    reported to `loss.jsonl`, the event log and stdout exactly as the old trainer
    reported them.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    window: list[float] = []
    checkpoints = 0
    step = start_step
    last_loss = float("nan")
    val = None

    def checkpoint(at_step: int) -> None:
        # The headline loss is the mean over the steps since the last checkpoint.
        # A base corpus is millions of examples, so a full `measure` pass over all
        # of them per checkpoint would be prohibitive -- validation runs over the
        # small, capped held-out set instead, exactly the number that matters.
        nonlocal checkpoints, val
        mean = sum(window) / len(window) if window else last_loss
        val = eval_loss(model, val_examples) if val_examples else None
        save(at_step, mean)
        append_curve(settings, at_step, mean, "", len(examples))
        checkpoints += 1
        log.event(
            f"{stage}.checkpoint",
            step=at_step,
            loss=mean,
            val_loss=val,
            examples=len(examples),
        )
        if echo:
            val_s = f"{val:.4f}" if val is not None else "   n/a"
            print(
                f"[{stage}] step {at_step:7d} | loss {mean:8.4f} | val {val_s}",
                flush=True,
            )

    for _ in range(steps):
        tokens, mask, weights = make_batch(examples, settings.batch_size, rng)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        step += 1
        last_loss = float(loss.item())
        window.append(last_loss)
        if step % settings.checkpoint_every == 0:
            checkpoint(step)
            window = []

    if window:  # flush a final checkpoint for the tail steps
        checkpoint(step)

    return StageResult(
        stage=stage,
        steps_run=steps,
        final_step=step,
        last_loss=last_loss,
        val_loss=val,
        checkpoints_written=checkpoints,
        path=Path(),  # filled in by the caller, which knows its own output path
    )


def _split_val(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    """Hold out a small, capped tail of examples for a validation read. Returns
    `(train, val)`; val is empty when there are too few examples to spare any."""
    if len(examples) < 4:
        return examples, []
    n_val = min(_VAL_CAP, max(1, int(len(examples) * _VAL_FRACTION)))
    n_val = min(n_val, len(examples) - 1)
    return examples[:-n_val], examples[-n_val:]


# --- stage 1: base -------------------------------------------------------


def archive_existing_checkpoints(settings: Settings) -> int:
    """Move (never delete) every existing checkpoint out of the way before a base
    retrain, into `checkpoints/archive/<timestamp>/`. A new `block_size` changes
    the positional-embedding shape, so the old checkpoints can no longer be
    loaded -- but they are kept, not destroyed."""
    ckpt_dir = settings.checkpoint_dir
    if not ckpt_dir.is_dir():
        return 0
    stamp = utcnow_iso().replace(":", "").replace("-", "")
    dest = settings.checkpoint_archive_dir / stamp
    moved = 0
    for item in ckpt_dir.iterdir():
        if item.name in {SCRATCH_DIR, "archive"}:
            continue  # scratch is disposable; archive/ holds prior archives
        if item.is_dir():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(dest / item.name))
        moved += 1
    return moved


def pretrain_base(
    settings: Settings,
    *,
    steps: int | None = None,
    seed: int = 1,
    echo: bool = True,
    log: EventLog | None = None,
) -> StageResult:
    """Stage 1. Train from random init on the external corpus and freeze
    `base.pt`. Existing checkpoints are archived first. Raises loudly (via
    `read_base_rows`) if the base corpus has not been prepared."""
    settings.ensure_dirs()
    log = log or NullLog()
    rows = read_base_rows(settings)  # raises EmptyCorpusError if not prepared
    be_polite(settings, log)

    archived = archive_existing_checkpoints(settings)
    sweep_scratch(settings)

    model = Babbler(config_from_settings(settings))  # random init, new geometry
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    examples = _text_examples(rows, model.config.block_size)
    train_examples, val_examples = _split_val(examples)
    budget = steps if steps is not None else settings.base_steps

    log.event(
        "base.start",
        rows=len(rows),
        examples=len(examples),
        block_size=model.config.block_size,
        params=model.num_params(),
        steps=budget,
        archived=archived,
    )
    result = _run_stage(
        settings,
        model,
        optimizer,
        train_examples,
        val_examples,
        stage="base",
        steps=budget,
        seed=seed,
        echo=echo,
        log=log,
        save=lambda step, loss: _save_to(settings, settings.base_checkpoint, model, optimizer, step, loss),
    )
    result.path = settings.base_checkpoint
    log.event("base.done", step=result.final_step, loss=result.last_loss, path=str(result.path))
    return result


# --- stage 2: voice ------------------------------------------------------


def corpus_row_count(settings: Settings) -> int:
    """Total stored corpus rows -- the number the +N-row trigger measures growth
    against. Uses the raw stored count, not the consent-filtered count, so a
    revocation cannot make the corpus appear to shrink below the last trigger."""
    return CorpusStore(settings.corpus_path).count()


def read_voice_state(settings: Settings) -> dict:
    path = settings.voice_state_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_voice_state(settings: Settings, *, rows: int, step: int) -> None:
    atomic_write_text(
        settings.voice_state_path,
        json.dumps({"last_trained_rows": rows, "step": step, "at": utcnow_iso()}, indent=2),
    )


def voice_trigger(settings: Settings) -> TriggerStatus:
    state = read_voice_state(settings)
    return TriggerStatus(
        current_rows=corpus_row_count(settings),
        last_trained_rows=int(state.get("last_trained_rows", 0)),
        threshold=settings.voice_trigger_rows,
        has_base=settings.base_checkpoint.exists(),
    )


def _load_base(settings: Settings) -> Babbler:
    """The frozen base weights, as a fresh model. Read-only: the voice pass never
    writes back to `base.pt`."""
    payload = torch.load(settings.base_checkpoint, map_location="cpu", weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    return model


def voice_pass(
    settings: Settings,
    *,
    force: bool = False,
    steps: int | None = None,
    seed: int = 1,
    echo: bool = True,
    log: EventLog | None = None,
    ids=None,
    blocklist=None,
) -> VoiceResult:
    """Stage 2. Continue-train from the frozen base on the consented human corpus
    and write `latest.pt`. Safe to rerun: it always starts from `base.pt`, never
    from the previous voice checkpoint, so nothing compounds.

    Fires only when `force` is set or the +N-row trigger is due; otherwise it is a
    no-op that reports why."""
    settings.ensure_dirs()
    log = log or NullLog()
    status = voice_trigger(settings)

    if not status.has_base:
        log.event("voice.skipped", reason="no_base")
        return VoiceResult(False, "no_base", current_rows=status.current_rows,
                           last_trained_rows=status.last_trained_rows)
    if not force and not status.due:
        log.event("voice.skipped", reason="not_due", new_rows=status.new_rows,
                  threshold=status.threshold)
        return VoiceResult(False, "not_due", current_rows=status.current_rows,
                           last_trained_rows=status.last_trained_rows)

    rows = corpus_rows(settings, ids, blocklist)  # consent + blocklist gated
    if not rows:
        log.event("voice.skipped", reason="no_data")
        return VoiceResult(False, "no_data", current_rows=status.current_rows,
                           last_trained_rows=status.last_trained_rows)

    be_polite(settings, log)
    sweep_scratch(settings)
    model = _load_base(settings)  # every run restarts from the clean base
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    examples = [ex for row in rows for ex in text_examples(row.text, model.config.block_size)]
    train_examples, val_examples = _split_val(examples)
    budget = steps if steps is not None else settings.voice_steps

    log.event("voice.start", rows=len(rows), examples=len(examples), steps=budget,
              from_base=str(settings.base_checkpoint))
    result = _run_stage(
        settings,
        model,
        optimizer,
        train_examples,
        val_examples,
        stage="voice",
        steps=budget,
        seed=seed,
        echo=echo,
        log=log,
        # save_checkpoint writes latest.pt (+ ckpt-*.pt); base.pt is never touched.
        save=lambda step, loss: save_checkpoint(settings, model, optimizer, step, loss),
    )
    result.path = settings.latest_checkpoint

    current = corpus_row_count(settings)
    write_voice_state(settings, rows=current, step=result.final_step)
    log.event("voice.done", step=result.final_step, loss=result.last_loss,
              rows_trained=len(rows), last_trained_rows=current)
    return VoiceResult(True, "trained", rows_trained=len(rows), current_rows=current,
                       last_trained_rows=current, stage=result)


# --- the trigger, wired into the bot -------------------------------------


class VoiceAutoTrigger:
    """Fires a voice pass when the corpus has grown enough -- a trigger, not a
    loop. The bot calls `maybe_run()` after each fresh corpus row (mirroring the
    growth publisher); when the trigger is due it launches `babble voice-pass` as
    a detached, low-priority subprocess so training never blocks the event loop,
    and the bot hot-reloads the new `latest.pt` on its own.
    """

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        self.settings = settings
        self.log = log or NullLog()
        self._proc: subprocess.Popen | None = None

    def maybe_run(self) -> None:
        if self.settings.voice_trigger_rows <= 0:
            return
        if self._proc is not None and self._proc.poll() is None:
            return  # a pass is already running; do not stack them
        if not voice_trigger(self.settings).due:
            return
        self._launch()

    def _launch(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "babble", "voice-pass", "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.log.event("voice.triggered", pid=self._proc.pid)
        except Exception as exc:  # a launch hiccup must never take the bot down
            self.log.event("voice.trigger_failed", error=f"{type(exc).__name__}: {exc}")
