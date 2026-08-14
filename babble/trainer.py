"""The background trainer.

Three properties matter more than speed here:

1. **It must not make the machine unusable.** It runs at nice 19 on a capped
   number of threads, in cycles of a fixed step budget with a rest in between.
   The default duty cycle is 200 steps then a 60 second nap.
2. **It must be safe to kill at any moment.** Checkpoints are written to a temp
   file and renamed, so a `kill -9` mid-write leaves the previous checkpoint
   intact. SIGINT/SIGTERM finish the current step, checkpoint, and exit.
3. **Progress must be watchable.** Every checkpoint appends a line to
   `checkpoints/loss.jsonl` with its step, loss and a sample generation, and
   prints the same thing. The babble is the show.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import torch

<<<<<<< ours
from .config import Settings
from .consent import ConsentStore
=======
from .blocklist import Blocklist
from .config import Settings
from .consent import ConsentStore
from .discord_feed import TrainingFeed
>>>>>>> theirs
from .generate import sample
from .identity import Pseudonymiser
from .logs import EventLog
from .model import Babbler, ModelConfig, config_from_settings, sequence_loss
from .store import Interaction, InteractionStore
from .tokenizer import PAD_ID, Example, build_example
from .util import utcnow_iso

SAMPLE_PROMPTS = ("hello", "how are you")


def be_polite(settings: Settings, log: EventLog) -> None:
    """Give up priority and threads before touching a single tensor."""
    applied_nice = None
    try:
        os.nice(settings.train_nice)
        applied_nice = os.nice(0)
    except (OSError, AttributeError, PermissionError):
        pass
    threads = max(1, settings.train_threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialised; harmless
    log.event("train.polite", nice=applied_nice, threads=threads, cpus=os.cpu_count())


<<<<<<< ours
def consented_rows(settings: Settings, ids: Pseudonymiser | None = None) -> list[Interaction]:
=======
def consented_rows(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> list[Interaction]:
>>>>>>> theirs
    """Rows both of whose participants still consent, checked right now.

    Withdrawal already purges rows, so this is belt and braces -- but "used to
    train the model" is a promise made in the consent notice, and it should be
<<<<<<< ours
    enforced at the moment of training, not only at the moment of capture.
    """
    ids = ids or Pseudonymiser.load(settings)
    consent = ConsentStore(settings.consent_path)
    allowed = {ids.user(uid) for uid in consent.granted_ids()}
    rows = InteractionStore(settings.interactions_path).all()
    return [r for r in rows if r.prompt_author in allowed and r.signal_author in allowed]
=======
    enforced at the moment of training, not only at the moment of capture. The
    blocklist gets the same belt-and-braces treatment: a row stored before a
    term was added must not survive to be trained on once it is.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)
    allowed = {ids.user(uid) for uid in consent.granted_ids()}
    rows = InteractionStore(settings.interactions_path).all()
    return [
        r
        for r in rows
        if r.prompt_author in allowed
        and r.signal_author in allowed
        and not blocklist.matches(r.prompt, r.chosen, r.rejected)
    ]
>>>>>>> theirs


def to_examples(rows: list[Interaction], block_size: int) -> list[Example]:
    return [
        build_example(row.prompt, row.chosen, block_size, weight=max(0.0, row.weight))
        for row in rows
        if row.chosen
    ]


def make_batch(
    examples: list[Example], batch_size: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample with replacement and right-pad to the longest in the batch."""
    chosen = [rng.choice(examples) for _ in range(batch_size)]
    width = max(len(e) for e in chosen)
    tokens = torch.full((len(chosen), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(chosen), width), dtype=torch.long)
    for i, example in enumerate(chosen):
        tokens[i, : len(example)] = torch.tensor(example.tokens, dtype=torch.long)
        mask[i, : len(example)] = torch.tensor(example.mask, dtype=torch.long)
    weights = torch.tensor([e.weight for e in chosen], dtype=torch.float32)
    return tokens, mask, weights


# --- checkpoints ---------------------------------------------------------


def save_checkpoint(settings: Settings, model: Babbler, optimizer, step: int, loss: float) -> Path:
    """Write `ckpt-NNNNNNN.pt` and repoint `latest.pt`, both atomically."""
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "saved_at": utcnow_iso(),
    }
    archive = settings.checkpoint_dir / f"ckpt-{step:07d}.pt"
    tmp = archive.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, archive)

    # Copy-then-rename so `latest.pt` is never observed half-written.
    latest_tmp = settings.latest_checkpoint.with_suffix(".pt.tmp")
    shutil.copyfile(archive, latest_tmp)
    os.replace(latest_tmp, settings.latest_checkpoint)

    prune_checkpoints(settings)
    return archive


def prune_checkpoints(settings: Settings) -> int:
    keep = max(1, settings.keep_checkpoints)
    archives = sorted(settings.checkpoint_dir.glob("ckpt-*.pt"))
    doomed = archives[:-keep] if len(archives) > keep else []
    for path in doomed:
        path.unlink(missing_ok=True)
    return len(doomed)


def append_curve(settings: Settings, step: int, loss: float, sample_text: str, rows: int) -> None:
    entry = {
        "step": step,
        "loss": round(loss, 6),
        "rows": rows,
        "at": utcnow_iso(),
        "sample": sample_text,
    }
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.loss_curve_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- the run ------------------------------------------------------------


class Interruption:
    """SIGINT/SIGTERM set a flag; the loop finishes its step and checkpoints."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def install(self) -> "Interruption":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not the main thread; caller polls .requested itself
        return self

    def _handle(self, signum, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def sleep(self, seconds: float, tick: float = 0.25) -> None:
        """Rest, but wake up promptly if someone wants us gone."""
        deadline = time.monotonic() + seconds
        while not self.requested and time.monotonic() < deadline:
            time.sleep(min(tick, max(0.0, deadline - time.monotonic())))


@dataclass
class RunResult:
    steps_run: int
    final_step: int
    last_loss: float | None
    checkpoints_written: int
    cycles: int
    stopped_because: str


def train(
    settings: Settings,
    *,
    steps: int | None = None,
    loop: bool = False,
    max_cycles: int | None = None,
    log: EventLog | None = None,
<<<<<<< ours
=======
    feed: TrainingFeed | None = None,
>>>>>>> theirs
    echo: bool = True,
    seed: int | None = None,
) -> RunResult:
    settings.ensure_dirs()
    ids = Pseudonymiser.load(settings)
<<<<<<< ours
    owns_log = log is None  # if we opened it, we close it
    log = log or EventLog(settings, ids, component="trainer", echo=echo)
=======
    blocklist = Blocklist.load()
    owns_log = log is None  # if we opened it, we close it
    log = log or EventLog(settings, ids, component="trainer", echo=echo)
    feed = feed or TrainingFeed.from_env(log)
>>>>>>> theirs
    be_polite(settings, log)

    budget = steps if steps is not None else settings.steps_per_cycle
    interrupt = Interruption().install()

    model, optimizer, step, resumed = _resume_or_init(settings, log)
    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed if seed is not None else step or 0)

    log.event(
        "train.start",
        step=step,
        resumed=resumed,
        params=model.num_params(),
        budget=budget,
        loop=loop,
        batch_size=settings.batch_size,
        lr=settings.learning_rate,
    )
<<<<<<< ours
=======
    feed.start(resumed=resumed, step=step)
>>>>>>> theirs

    steps_run = 0
    checkpoints = 0
    cycles = 0
    last_loss: float | None = None
<<<<<<< ours
=======
    prev_checkpoint_loss: float | None = None
>>>>>>> theirs
    reason = "budget_exhausted"

    while True:
        if interrupt.requested:
            reason = f"signal:{interrupt.signal_name}"
            break
        if max_cycles is not None and cycles >= max_cycles:
            reason = "max_cycles"
            break

<<<<<<< ours
        rows = consented_rows(settings, ids)
        examples = to_examples(rows, model.config.block_size)
        if not examples:
            log.event("train.idle", reason="no_consented_rows", rows=len(rows))
=======
        rows = consented_rows(settings, ids, blocklist)
        examples = to_examples(rows, model.config.block_size)
        if not examples:
            log.event("train.idle", reason="no_consented_rows", rows=len(rows))
            feed.idle()
>>>>>>> theirs
            if not loop:
                reason = "no_data"
                break
            interrupt.sleep(settings.rest_seconds)
            continue

<<<<<<< ours
=======
        feed.active()
>>>>>>> theirs
        cycles += 1
        cycle_started = time.perf_counter()
        log.event(
            "train.cycle.start",
            cycle=cycles,
            step=step,
            rows=len(rows),
            examples=len(examples),
            planned_steps=budget,
        )

        window: list[float] = []
        cycle_steps = 0
        model.train()
        for _ in range(budget):
            if interrupt.requested:
                break
            tokens, mask, weights = make_batch(examples, settings.batch_size, rng)
            loss = sequence_loss(model, tokens, mask, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()

            step += 1
            steps_run += 1
            cycle_steps += 1
            value = float(loss.detach())
            window.append(value)
            last_loss = value

            if step % settings.checkpoint_every == 0:
<<<<<<< ours
                _checkpoint(settings, log, model, optimizer, step, window, len(rows), echo)
=======
                prev_checkpoint_loss = _checkpoint(
                    settings, log, feed, blocklist, model, optimizer, step, window, len(rows),
                    cycles, prev_checkpoint_loss, echo,
                )
>>>>>>> theirs
                checkpoints += 1
                window = []

        # Never end a cycle with unsaved work: a kill during the rest period
        # would otherwise throw away everything since the last checkpoint.
        if window:
<<<<<<< ours
            _checkpoint(settings, log, model, optimizer, step, window, len(rows), echo)
=======
            prev_checkpoint_loss = _checkpoint(
                settings, log, feed, blocklist, model, optimizer, step, window, len(rows),
                cycles, prev_checkpoint_loss, echo,
            )
>>>>>>> theirs
            checkpoints += 1

        log.event(
            "train.cycle.end",
            cycle=cycles,
            step=step,
            steps=cycle_steps,
            seconds=round(time.perf_counter() - cycle_started, 2),
            loss=round(last_loss, 6) if last_loss is not None else None,
        )

        if interrupt.requested:
            reason = f"signal:{interrupt.signal_name}"
            break
        if not loop:
            break
        log.event("train.rest", seconds=settings.rest_seconds, step=step)
        interrupt.sleep(settings.rest_seconds)

    if interrupt.requested:
        log.event("train.interrupt", signal=interrupt.signal_name, step=step)

    log.event(
        "train.stop",
        step=step,
        steps_run=steps_run,
        checkpoints=checkpoints,
        cycles=cycles,
        reason=reason,
        last_loss=round(last_loss, 6) if last_loss is not None else None,
    )
    if owns_log:
        log.close()
    else:
        log.flush()
    return RunResult(steps_run, step, last_loss, checkpoints, cycles, reason)


def _resume_or_init(settings: Settings, log: EventLog) -> tuple[Babbler, torch.optim.Optimizer, int, bool]:
    path = settings.latest_checkpoint
    model = Babbler(config_from_settings(settings))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    if not path.exists():
        log.event("train.init", source="random", params=model.num_params())
        return model, optimizer, 0, False

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = Babbler(ModelConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        optimizer.load_state_dict(payload["optim"])
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"].to(torch.uint8))
        step = int(payload.get("step", 0))
        log.event(
            "train.resume",
            step=step,
            loss=payload.get("loss"),
            saved_at=payload.get("saved_at"),
            params=model.num_params(),
        )
        return model, optimizer, step, True
    except Exception as exc:  # a truncated or foreign checkpoint must not be fatal
        log.event("train.resume_failed", error=f"{type(exc).__name__}: {exc}", path=str(path))
        model = Babbler(config_from_settings(settings))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        return model, optimizer, 0, False


def _checkpoint(
    settings: Settings,
    log: EventLog,
<<<<<<< ours
=======
    feed: TrainingFeed,
    blocklist: Blocklist,
>>>>>>> theirs
    model: Babbler,
    optimizer,
    step: int,
    window: list[float],
    rows: int,
<<<<<<< ours
    echo: bool,
) -> None:
=======
    cycle: int,
    prev_loss: float | None,
    echo: bool,
) -> float:
>>>>>>> theirs
    mean_loss = sum(window) / len(window) if window else float("nan")
    started = time.perf_counter()
    prompt = SAMPLE_PROMPTS[step // max(1, settings.checkpoint_every) % len(SAMPLE_PROMPTS)]
    text = sample(
        model,
        prompt,
        max_new_tokens=min(64, settings.max_new_tokens),
        temperature=settings.temperature,
        top_k=settings.top_k,
    )
    path = save_checkpoint(settings, model, optimizer, step, mean_loss)
    append_curve(settings, step, mean_loss, text, rows)

    log.event(
        "train.checkpoint",
        step=step,
        loss=round(mean_loss, 6),
        rows=rows,
        file=path.name,
        seconds=round(time.perf_counter() - started, 2),
        prompt=prompt,
        sample=text.replace("\n", "\\n")[:200],
    )
    if echo:
        shown = text.replace("\n", "\\n")
        print(f"step {step:>7,} | loss {mean_loss:8.4f} | {prompt!r} -> {shown!r}", flush=True)
<<<<<<< ours
=======

    # The sample is what leaves the machine via the feed -- filter it the same
    # way any other model output headed for Discord gets filtered.
    feed_sample = text if blocklist.hit(text) is None else "*(withheld — matched the content filter)*"
    feed.checkpoint(
        cycle=cycle,
        step=step,
        loss=mean_loss,
        prev_loss=prev_loss,
        rows=rows,
        prompt=prompt,
        sample=feed_sample,
    )
    return mean_loss
>>>>>>> theirs
