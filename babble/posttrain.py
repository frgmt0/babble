"""Stage 2: a short supervised post-train on the correction pairs.

Pretraining (`babble train`) teaches the model to continue plain text -- there
is no prompt/response boundary in that objective, so it never learns to
*answer*. This stage fine-tunes the pretrained checkpoint on the
`(prompt, chosen)` correction pairs laid out as `<bos> prompt <sep> response
<eos>` (`tokenizer.build_example`, the same pair layout `generate.py` already
uses to score a correction against its rejected answer), so the model has at
least seen what answering looks like.

There are only a few dozen pairs against a ~3.3M-parameter model. It will
memorise those pairs and generalise to approximately nothing -- that is the
expected result of this stage, not a bug to engineer around. The `rejected`
half of each correction is captured and published, but it is not the
objective here: this is supervised fine-tuning on the chosen answer, not
preference optimisation.

Every run restarts from `checkpoints/pretrained.pt`, a snapshot of the
pretrained checkpoint taken the first time a post-train ever runs -- so a
rerun starts from a clean pretrain, never from a previous post-train's
weights, the same discipline the old base/voice split used. The snapshot is
not frozen forever, though: if `latest.pt` no longer holds what the last
post-train itself wrote there (a fresh `babble train` landed since), the
snapshot is stale and gets retaken before this run starts, so a post-train
never silently fine-tunes a leftover pretrain. Like the pretrainer, it keeps
the *best-validation* checkpoint rather than the last one and stops early once
val stops improving, and it fires on a **trigger, not a loop**: every
`post_trigger_pairs` new correction pairs since the last post-train (the
running bot watches for that crossing itself, deferring to a pretrain still in
flight -- see `AutoPostTrigger`), or on demand with `--force`. The last-trained
pair count is persisted (`checkpoints/post_state.json`) so a restart never
re-fires.
"""

from __future__ import annotations

import copy
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from .blocklist import Blocklist
from .config import Settings
from .cpu_runtime import force_cpu_device, maybe_compile, model_state_dict, uncompiled
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .model import Babbler, ModelConfig, sequence_loss
from .post_state import (
    PostTrigger,
    file_hash,
    pair_count,
    post_trigger,
    pretrained_snapshot_stale,
    read_post_state,
    trainable_pairs,
    write_post_state,
)
from .store import Interaction
from .tokenizer import Example, build_example
from .trainer import (
    SCRATCH_DIR,
    _build_optimizer,
    append_curve,
    be_polite,
    eval_loss,
    make_batch,
    save_checkpoint,
    sweep_scratch,
)

__all__ = [
    "PostTrainResult",
    "PostTrigger",
    "AutoPostTrigger",
    "pair_count",
    "post_train",
    "post_trigger",
    "read_post_state",
    "trainable_pairs",
    "write_post_state",
]

# How much of the pair set to hold out for a validation read. Capped low
# because the whole set is a few dozen pairs to begin with.
_VAL_FRACTION = 0.2
_VAL_CAP = 512


@dataclass
class PostTrainResult:
    ran: bool
    reason: str
    pairs_trained: int = 0
    current_pairs: int = 0
    last_trained_pairs: int = 0
    final_step: int = 0
    last_loss: float | None = None
    val_loss: float | None = None
    checkpoints_written: int = 0
    budget: int = 0
    stopped_early: bool = False
    path: Path | None = None


def _split_val(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    """Hold out a small, capped tail of examples for a validation read. Returns
    `(train, val)`; val is empty when there are too few examples to spare any."""
    if len(examples) < 4:
        return examples, []
    n_val = min(_VAL_CAP, max(1, int(len(examples) * _VAL_FRACTION)))
    n_val = min(n_val, len(examples) - 1)
    return examples[:-n_val], examples[-n_val:]


def _ensure_pretrained_snapshot(settings: Settings) -> Path:
    """The clean pretrain weights to post-train from.

    Snapshotted from `latest.pt` the first time a post-train ever runs, and
    reused after that for as long as `latest.pt` still holds exactly what the
    last post-train wrote there -- so a rerun restarts from the same clean
    pretrain rather than the previous post-train's own output. But `latest.pt`
    is also the file a fresh pretrain (or the bot's auto-retrain) overwrites,
    and when that happens the snapshot is stale: `pretrained_snapshot_stale`
    notices via the hash recorded in `post_state.json` and this retakes it, so
    a post-train after a re-pretrain always fine-tunes the *new* pretrain, not
    a leftover from before it.
    """
    target = settings.pretrained_checkpoint
    if pretrained_snapshot_stale(settings):
        settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        scratch = settings.checkpoint_dir / SCRATCH_DIR
        scratch.mkdir(exist_ok=True)
        tmp = scratch / f"{target.name}.tmp"
        shutil.copyfile(settings.latest_checkpoint, tmp)
        os.replace(tmp, target)
    return target


def _load_pretrained(path: Path) -> Babbler:
    """The frozen pretrained weights, as a fresh model. Read-only: post-train
    never writes back to `pretrained.pt`."""
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    return maybe_compile(model)


def _build_examples(pairs: list[Interaction], block_size: int) -> list[Example]:
    return [build_example(p.prompt, p.chosen, block_size) for p in pairs]


def post_train(
    settings: Settings,
    *,
    force: bool = False,
    steps: int | None = None,
    patience: int | None = None,
    seed: int = 1,
    echo: bool = True,
    log: EventLog | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> PostTrainResult:
    """Fine-tune the pretrained checkpoint on the correction pairs and write
    the winning step to `latest.pt` -- the checkpoint the bot serves.

    Fires only when `force` is set or the +N-pair trigger is due; otherwise a
    no-op that reports why. `steps` is a ceiling, not a target: the checkpoint
    with the lowest val loss wins and is what gets written, and the run stops
    early once `patience` checkpoint intervals in a row fail to beat it. Both
    are no-ops without a held-out validation set (too few pairs to spare
    any), in which case every checkpoint interval is written in turn, exactly
    like the pretrainer without validation.
    """
    settings.ensure_dirs()
    log = log or NullLog()
    status = post_trigger(settings)

    if not status.has_pretrained:
        log.event("post.skipped", reason="no_pretrain")
        return PostTrainResult(
            False, "no_pretrain",
            current_pairs=status.current_pairs, last_trained_pairs=status.last_trained_pairs,
        )
    if not force and not status.due:
        log.event(
            "post.skipped", reason="not_due", new_pairs=status.new_pairs, threshold=status.threshold
        )
        return PostTrainResult(
            False, "not_due",
            current_pairs=status.current_pairs, last_trained_pairs=status.last_trained_pairs,
        )

    pairs = trainable_pairs(settings, ids, blocklist)
    if not pairs:
        log.event("post.skipped", reason="no_data")
        return PostTrainResult(
            False, "no_data",
            current_pairs=status.current_pairs, last_trained_pairs=status.last_trained_pairs,
        )

    budget = steps if steps is not None else settings.post_steps
    if budget <= 0:
        # A non-positive ceiling trains nothing -- bail out before touching the
        # pretrained snapshot or the model so a stray `--steps 0` cannot burn
        # the trigger (advance `post_state.json`) without training anything.
        log.event("post.skipped", reason="no_steps", steps=budget)
        return PostTrainResult(
            False, "no_steps",
            current_pairs=status.current_pairs, last_trained_pairs=status.last_trained_pairs,
        )

    be_polite(settings, log)
    sweep_scratch(settings)
    pretrained_path = _ensure_pretrained_snapshot(settings)
    model = _load_pretrained(pretrained_path)
    optimizer = _build_optimizer(model, settings)

    examples = _build_examples(pairs, model.config.block_size)
    train_examples, val_examples = _split_val(examples)
    run_patience = patience if patience is not None else settings.post_patience
    rng = random.Random(seed)
    torch.manual_seed(seed)

    log.event(
        "post.start",
        pairs=len(pairs),
        examples=len(examples),
        block_size=model.config.block_size,
        steps=budget,
        patience=run_patience,
        from_pretrained=str(pretrained_path),
    )

    window: list[float] = []
    checkpoints = 0
    step = 0
    last_loss = float("nan")
    best: dict | None = None
    stalls = 0
    stop = False

    def checkpoint() -> None:
        nonlocal checkpoints, best, stalls, stop, window
        mean = sum(window) / len(window) if window else last_loss
        val = eval_loss(model, val_examples) if val_examples else None
        checkpoints += 1
        append_curve(settings, step, mean, "", len(examples))
        log.event("post.checkpoint", step=step, loss=mean, val_loss=val, examples=len(examples))
        if echo:
            val_s = f"{val:.4f}" if val is not None else "   n/a"
            print(f"[post] step {step:7d} | loss {mean:8.4f} | val {val_s}", flush=True)
        if val is not None:
            if best is None or val < best["val"]:
                best = {
                    "step": step,
                    "loss": mean,
                    "val": val,
                    "model": copy.deepcopy(model_state_dict(model)),
                    "optim": copy.deepcopy(optimizer.state_dict()),
                }
                stalls = 0
            else:
                stalls += 1
                if run_patience and stalls >= run_patience:
                    stop = True
        else:
            # No held-out set to compare against: every interval is written in
            # turn, exactly like the pretrainer without validation.
            save_checkpoint(settings, model, optimizer, step, mean)
        window = []

    model.train()
    for _ in range(budget):
        tokens, mask, weights = make_batch(train_examples, settings.batch_size, rng)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        step += 1
        last_loss = float(loss.item())
        window.append(last_loss)
        if step % settings.checkpoint_every == 0:
            checkpoint()
            if stop:
                break

    if window and not stop:
        checkpoint()

    final_step, final_loss, final_val = step, last_loss, None
    if best is not None:
        uncompiled(model).load_state_dict(best["model"])
        optimizer.load_state_dict(best["optim"])
        save_checkpoint(settings, model, optimizer, best["step"], best["loss"])
        final_step, final_loss, final_val = best["step"], best["loss"], best["val"]

    current = pair_count(settings)
    write_post_state(
        settings, pairs=current, step=final_step, latest_hash=file_hash(settings.latest_checkpoint)
    )
    log.event(
        "post.done",
        step=final_step,
        loss=round(final_loss, 6) if final_loss is not None else None,
        val_loss=round(final_val, 6) if final_val is not None else None,
        pairs_trained=len(pairs),
        last_trained_pairs=current,
        checkpoints=checkpoints,
        stopped_early=stop,
        budget=budget,
    )

    return PostTrainResult(
        True,
        "trained",
        pairs_trained=len(pairs),
        current_pairs=current,
        last_trained_pairs=current,
        final_step=final_step,
        last_loss=final_loss,
        val_loss=final_val,
        checkpoints_written=checkpoints,
        budget=budget,
        stopped_early=stop,
        path=settings.latest_checkpoint,
    )


class AutoPostTrigger:
    """Fires a post-train when the correction pairs have grown enough -- a
    trigger, not a loop, the same discipline `AutoTrainTrigger` gives the
    pretrainer. The bot calls `maybe_run()` after each fresh correction pair;
    when due it launches `babble post-train` as a detached, low-priority
    subprocess so post-training never blocks the event loop.

    Takes the bot's `AutoTrainTrigger` (optional) so it can defer to
    `is_running()`: firing a post-train while a pretrain is still writing
    `latest.pt` would race the two subprocesses over the same file, and even
    without an actual write collision, post-training now would fine-tune a
    pretrain that is about to be replaced anyway. Deferring costs nothing --
    the pair count that made it due keeps counting, and the next
    correction (or the next `--force`) tries again.
    """

    def __init__(
        self, settings: Settings, log: EventLog | None = None, *, train_trigger=None
    ) -> None:
        self.settings = settings
        self.log = log or NullLog()
        self.train_trigger = train_trigger
        self._proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def maybe_run(self) -> None:
        if self.settings.post_trigger_pairs <= 0:
            return
        if self.is_running():
            return  # a run is already in flight; do not stack them
        if self.train_trigger is not None and self.train_trigger.is_running():
            return  # let the pretrain finish first -- see class docstring
        if not post_trigger(self.settings).due:
            return
        self._launch()

    def _launch(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "babble", "post-train", "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.log.event("post.triggered", pid=self._proc.pid)
        except Exception as exc:  # a launch hiccup must never take the bot down
            self.log.event("post.trigger_failed", error=f"{type(exc).__name__}: {exc}")
