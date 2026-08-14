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

import hashlib
import json
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .blocklist import Blocklist
from .config import Settings
from .consent import ConsentStore
from .discord_feed import TrainingFeed
from .export_hf import DATA_FILE, ExportBlocked, build_export, push as push_export
from .generate import best_of
from .identity import Pseudonymiser
from .logs import EventLog
from .model import (
    Babbler,
    ModelConfig,
    config_from_settings,
    per_token_loss,
    sequence_loss,
)
from .store import CORRECTION, Interaction, InteractionStore
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


def consented_rows(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> list[Interaction]:
    """Rows both of whose participants still consent, checked right now.

    Withdrawal already purges rows, so this is belt and braces -- but "used to
    train the model" is a promise made in the consent notice, and it should be
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


def row_weight(row: Interaction, settings: Settings) -> float:
    """The row's stored weight, with corrections boosted on top of it.

    The stored weight already separates a correction from a thumbs-up. The boost
    is a second, separate knob for a different problem: with a corpus this small
    a brand new correction is one row among twenty and takes many checkpoints to
    show up in the model's behaviour. Multiplying it makes the thing ro actually
    typed count for more than the things the bot was merely not told off for.
    """
    weight = max(0.0, row.weight)
    if row.signal == CORRECTION:
        weight *= max(0.0, settings.correction_boost)
    return weight


def trainable_rows(rows: list[Interaction]) -> list[Interaction]:
    """The rows that become examples: a row with no `chosen` teaches nothing.

    Split out so anything that wants to line a row up with its example -- the
    worst-row report, for one -- filters by exactly the same rule `to_examples`
    does, instead of indexing into a list one filter out of step.
    """
    return [row for row in rows if row.chosen]


def to_examples(
    rows: list[Interaction], block_size: int, settings: Settings | None = None
) -> list[Example]:
    """Rows to supervised examples. Without `settings`, stored weights are used
    as-is -- that is what an eval pass wants, since boosting corrections in the
    held-out loss would make it incomparable to the training loss.
    """
    return [
        build_example(
            row.prompt,
            row.chosen,
            block_size,
            weight=row_weight(row, settings) if settings else max(0.0, row.weight),
        )
        for row in trainable_rows(rows)
    ]


# --- held-out validation --------------------------------------------------

_VAL_SALT = "babble-val-split"


def _val_bucket(row_id: str) -> float:
    """A stable float in [0, 1) derived from the row id.

    Hashing the id -- not the row's position in the file, not a shuffle -- is
    what makes the same row land on the same side of the split every time the
    trainer restarts and as more rows are appended around it.
    """
    digest = hashlib.sha256(f"{_VAL_SALT}\x1f{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def val_holdout_size(total: int, fraction: float) -> int:
    """How many rows to hold out of `total`, never all of them and never none.

    Both clamps matter at this corpus size: a 20% split of 21 rows is 4, and
    rounding or a mis-set fraction must not be allowed to leave zero rows on
    either side of the split.
    """
    holdout = round(max(0.0, min(1.0, fraction)) * total)
    return max(1, min(holdout, total - 1))


@dataclass
class Split:
    """Held-out split of consented rows.

    `enabled=False` means the corpus was too small to hold anything out
    without starving training: `train` is every row, `val` is empty, and
    `disabled_reason` says why.
    """

    train: list[Interaction]
    val: list[Interaction]
    enabled: bool
    disabled_reason: str | None = None


def split_rows(rows: list[Interaction], settings: Settings) -> Split:
    """Deterministically hold out `settings.val_fraction` of `rows` by row id.

    The split takes the `val_holdout_size` rows with the **lowest** hash bucket,
    rather than every row whose bucket happens to fall under `val_fraction`.
    The hash is uniform -- measured over ten thousand real content-addressed row
    ids, the deciles are flat and the realised fraction is 0.1958 against a
    target of 0.2 -- but thresholding each row independently is a binomial draw,
    and at this corpus size that draw is wild. Over four thousand simulated
    21-row corpora at `val_fraction=0.2` it held out anywhere from 0 to 12 rows.
    The live trainer drew 10, which halved a corpus that only had 21 rows in it.
    A mean of 4.2 is no comfort when any single run is the one you are training.

    Ranking keeps the properties the hash was chosen for:

    - **Deterministic.** Same rows in, same split out, restart after restart.
    - **Independent of file order.** A row's side is decided by its id, never by
      where it landed in the file or by a shuffle.
    - **Nearly stable as rows are appended.** Not perfectly: `k` grows with the
      corpus, and a new row that hashes below the boundary displaces the row
      that was sitting on it. That churn is a couple of rows near the cut, not a
      reshuffle, and it is the price of the split actually being the size it
      says it is -- which at 21 rows matters a great deal more.
    """
    if len(rows) < settings.val_min_rows:
        return Split(
            train=rows,
            val=[],
            enabled=False,
            disabled_reason=(
                f"only {len(rows)} consented rows, need at least {settings.val_min_rows} "
                f"(BABBLE_VAL_MIN_ROWS) before holding any out"
            ),
        )
    holdout = val_holdout_size(len(rows), settings.val_fraction)
    # `row.id` breaks ties so two rows that collide on the bucket still order
    # deterministically; position is only ever the last resort.
    ranked = sorted(range(len(rows)), key=lambda i: (_val_bucket(rows[i].id), rows[i].id, i))
    held_out = set(ranked[:holdout])
    return Split(
        train=[row for i, row in enumerate(rows) if i not in held_out],
        val=[row for i, row in enumerate(rows) if i in held_out],
        enabled=True,
    )


def _stack_examples(examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad every example to the batch's longest, same layout as `make_batch`."""
    width = max(len(e) for e in examples)
    tokens = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, example in enumerate(examples):
        tokens[i, : len(example)] = torch.tensor(example.tokens, dtype=torch.long)
        mask[i, : len(example)] = torch.tensor(example.mask, dtype=torch.long)
    weights = torch.tensor([e.weight for e in examples], dtype=torch.float32)
    return tokens, mask, weights


def _real_mask(examples: list[Example], width: int) -> torch.Tensor:
    """1 on every token that is really there, 0 on the right-padding."""
    real = torch.zeros((len(examples), width), dtype=torch.long)
    for i, example in enumerate(examples):
        real[i, : len(example)] = 1
    return real


@dataclass
class LossReport:
    """The checkpoint loss, taken apart into the numbers that mean something.

    `response` is what training optimises. `prompt` is diagnostic: nothing ever
    trains it, so it should stay high, and a prompt loss that starts falling
    means the mask has broken.

    `worst_row` is the one that actually explains this bug. The training loss is
    a mean over *tokens*, so a short row that the model has not learned barely
    moves it: four bad bytes among four hundred good ones is a rounding error in
    the average and a completely broken answer to whoever typed that prompt.
    Reporting the worst row alongside the mean makes "loss 0.02 but it babbles"
    visible instead of mysterious.
    """

    response: float | None = None
    prompt: float | None = None
    worst_row: float | None = None
    worst_prompt: str | None = None


@torch.no_grad()
def measure(
    model: Babbler, examples: list[Example], rows: list[Interaction] | None = None
) -> LossReport:
    """Response loss, prompt loss and the worst single row, in one forward pass.

    `rows` is only used to name the worst row, and must be the rows `examples`
    was built from -- pass it through `trainable_rows` first, the way
    `to_examples` does, or the name will belong to a different row than the
    number.
    """
    if not examples:
        return LossReport()
    was_training = model.training
    model.eval()
    try:
        tokens, mask, _ = _stack_examples(examples)
        per_token = per_token_loss(model, tokens)

        response_mask = mask[:, 1:].to(per_token.dtype)
        real = _real_mask(examples, tokens.size(1))
        prompt_mask = (real[:, 1:] - mask[:, 1:]).clamp(min=0).to(per_token.dtype)

        per_row = (per_token * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1e-8)
        worst = int(torch.argmax(per_row))
        return LossReport(
            response=float(
                (per_token * response_mask).sum() / response_mask.sum().clamp(min=1e-8)
            ),
            prompt=(
                float((per_token * prompt_mask).sum() / prompt_mask.sum())
                if float(prompt_mask.sum())
                else None
            ),
            worst_row=float(per_row[worst]),
            worst_prompt=rows[worst].prompt if rows and worst < len(rows) else None,
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def eval_loss(model: Babbler, examples: list[Example]) -> float | None:
    """Mean loss over held-out examples, or None if there are none to score.

    Eval mode, `no_grad`, no optimizer involved -- this can never mutate a
    weight or an optimizer moment. `model.training` is restored on the way
    out, so a validation pass never changes what the next training step sees.
    """
    if not examples:
        return None
    was_training = model.training
    model.eval()
    try:
        tokens, mask, weights = _stack_examples(examples)
        return float(sequence_loss(model, tokens, mask, weights))
    finally:
        model.train(was_training)


def overfit_signal(
    train_loss: float | None,
    prev_train_loss: float | None,
    val_loss: float | None,
    prev_val_loss: float | None,
) -> bool:
    """A plain, honest overfitting flag: val loss rose while train loss fell,
    checkpoint over checkpoint. Reporting only -- it never changes training.
    """
    if None in (train_loss, prev_train_loss, val_loss, prev_val_loss):
        return False
    return val_loss > prev_val_loss and train_loss < prev_train_loss


@dataclass
class DatasetStats:
    """The full picture behind `consented_rows` -- not just the count that made it."""

    stored: int
    trained: int
    dropped_consent: int
    dropped_blocklist: int


def dataset_stats(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> DatasetStats:
    """How much of the stored corpus actually reaches training, and why the rest doesn't.

    Mirrors `consented_rows`'s filter exactly, just bucketing the rejects instead
    of discarding them, so the two can never quietly disagree about who trains.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    consent = ConsentStore(settings.consent_path)
    allowed = {ids.user(uid) for uid in consent.granted_ids()}
    all_rows = InteractionStore(settings.interactions_path).all()

    trained = dropped_consent = dropped_blocklist = 0
    for row in all_rows:
        if row.prompt_author not in allowed or row.signal_author not in allowed:
            dropped_consent += 1
        elif blocklist.matches(row.prompt, row.chosen, row.rejected):
            dropped_blocklist += 1
        else:
            trained += 1
    return DatasetStats(
        stored=len(all_rows),
        trained=trained,
        dropped_consent=dropped_consent,
        dropped_blocklist=dropped_blocklist,
    )


def distinct_prompts(rows: list[Interaction]) -> list[tuple[str, str]]:
    """Ordered, de-duplicated (prompt, chosen) pairs.

    Order follows first appearance, so the rotation walks the dataset in a
    stable sequence as new rows are only ever appended. The `chosen` half uses
    the *latest* row for that prompt, since that is what the dataset currently
    says the right answer is.
    """
    order: list[str] = []
    latest: dict[str, str] = {}
    for row in rows:
        if not row.prompt:
            continue
        if row.prompt not in latest:
            order.append(row.prompt)
        latest[row.prompt] = row.chosen
    return [(prompt, latest[prompt]) for prompt in order]


def probe_prompt(rows: list[Interaction], index: int) -> tuple[str, str]:
    """The prompt (and its expected answer) for checkpoint `index`.

    Cycles deterministically through every distinct prompt in the consented
    dataset, one per checkpoint, wrapping around once it reaches the end.
    Falls back to the hardcoded pair only when there is nothing to probe with.
    """
    pairs = distinct_prompts(rows)
    if not pairs:
        return SAMPLE_PROMPTS[index % len(SAMPLE_PROMPTS)], ""
    prompt, chosen = pairs[index % len(pairs)]
    return prompt, chosen


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


SCRATCH_DIR = ".partial"


def save_checkpoint(settings: Settings, model: Babbler, optimizer, step: int, loss: float) -> Path:
    """Write `ckpt-NNNNNNN.pt` and repoint `latest.pt`, both atomically.

    Half-written files are staged in a `.partial/` scratch directory rather than
    alongside the real checkpoints. `os.replace` was always atomic, so a torn
    file was never *loadable* -- but it was visible, sitting next to the good
    checkpoints with a name one glob away from being picked up, and a `kill -9`
    landing mid-write left it there. Staging elsewhere makes "nothing
    half-written is ever in the checkpoint directory" true by construction
    instead of true by how fast the write happened to be.
    """
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
    archive = settings.checkpoint_dir / f"ckpt-{step:07d}.pt"
    tmp = scratch / f"{archive.name}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, archive)

    # Copy-then-rename so `latest.pt` is never observed half-written.
    latest_tmp = scratch / "latest.pt.tmp"
    shutil.copyfile(archive, latest_tmp)
    os.replace(latest_tmp, settings.latest_checkpoint)

    prune_checkpoints(settings)
    return archive


def sweep_scratch(settings: Settings) -> int:
    """Delete anything a killed write left staged. Returns how many went.

    Called at trainer start: a leftover is dead weight from a previous process,
    never something the current one is mid-way through writing.
    """
    scratch = settings.checkpoint_dir / SCRATCH_DIR
    if not scratch.is_dir():
        return 0
    swept = 0
    for stale in scratch.iterdir():
        if stale.is_file():
            stale.unlink(missing_ok=True)
            swept += 1
    return swept


def prune_checkpoints(settings: Settings) -> int:
    keep = max(1, settings.keep_checkpoints)
    archives = sorted(settings.checkpoint_dir.glob("ckpt-*.pt"))
    doomed = archives[:-keep] if len(archives) > keep else []
    for path in doomed:
        path.unlink(missing_ok=True)
    return len(doomed)


# --- auto-publish ---------------------------------------------------------


@dataclass
class PublishState:
    """What was last actually pushed, so an unchanged corpus is never re-pushed."""

    content_hash: str | None = None
    rows: int = 0


def _auto_publish(
    settings: Settings, log: EventLog, feed: TrainingFeed, state: PublishState
) -> PublishState:
    """Export and push, through the exact same consent/blocklist gate as a
    manual `babble export --push`. Never raises: a bad token, no network, or
    HF being down is logged and reported in the feed, and training continues
    -- the next scheduled publish just tries again.
    """
    try:
        result = build_export(settings, log=log)
    except ExportBlocked as exc:
        log.event("publish.blocked", error=str(exc))
        feed.publish_failed(f"export blocked: {exc}")
        return state

    content_hash = hashlib.sha256((result.path / DATA_FILE).read_bytes()).hexdigest()
    if content_hash == state.content_hash and result.rows == state.rows:
        log.event("publish.skipped", reason="unchanged", rows=result.rows)
        return state

    try:
        url = push_export(settings, settings.hf_repo, result.path, log=log)
    except Exception as exc:  # network, rate limit, bad token, HF down -- never fatal
        log.event("publish.failed", error=f"{type(exc).__name__}: {exc}")
        feed.publish_failed(f"{type(exc).__name__}: {exc}")
        return state

    log.event("publish.ok", rows=result.rows, url=url)
    feed.publish(rows=result.rows, url=url)
    return PublishState(content_hash=content_hash, rows=result.rows)


def _maybe_auto_publish(
    settings: Settings, log: EventLog, feed: TrainingFeed, checkpoints: int, state: PublishState
) -> PublishState:
    if not settings.hf_publish_every or checkpoints % settings.hf_publish_every != 0:
        return state
    return _auto_publish(settings, log, feed, state)


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
    feed: TrainingFeed | None = None,
    echo: bool = True,
    seed: int | None = None,
) -> RunResult:
    settings.ensure_dirs()
    ids = Pseudonymiser.load(settings)
    blocklist = Blocklist.load()
    owns_log = log is None  # if we opened it, we close it
    log = log or EventLog(settings, ids, component="trainer", echo=echo)
    feed = feed or TrainingFeed.from_env(log)
    be_polite(settings, log)

    budget = steps if steps is not None else settings.steps_per_cycle
    interrupt = Interruption().install()

    swept = sweep_scratch(settings)
    if swept:
        log.event("train.scratch_swept", files=swept)

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
    feed.start(resumed=resumed, step=step)

    steps_run = 0
    checkpoints = 0
    cycles = 0
    probe_index = 0  # never reset: a checkpoint counter, not a dataset index
    last_loss: float | None = None
    prev_checkpoint_loss: float | None = None
    prev_val_loss: float | None = None
    publish_state = PublishState()
    reason = "budget_exhausted"

    while True:
        if interrupt.requested:
            reason = f"signal:{interrupt.signal_name}"
            break
        if max_cycles is not None and cycles >= max_cycles:
            reason = "max_cycles"
            break

        rows = consented_rows(settings, ids, blocklist)
        split = split_rows(rows, settings)
        examples = to_examples(split.train, model.config.block_size, settings)
        # Held-out rows keep their stored weights: boosting corrections in the
        # eval set would make val loss incomparable to train loss.
        val_examples = to_examples(split.val, model.config.block_size)
        if not examples:
            log.event("train.idle", reason="no_consented_rows", rows=len(rows))
            feed.idle()
            if not loop:
                reason = "no_data"
                break
            interrupt.sleep(settings.rest_seconds)
            continue

        feed.active()
        cycles += 1
        cycle_started = time.perf_counter()
        stats = dataset_stats(settings, ids, blocklist)
        log.event(
            "train.cycle.start",
            cycle=cycles,
            step=step,
            rows=len(rows),
            examples=len(examples),
            planned_steps=budget,
            val_rows=len(split.val),
            val_enabled=split.enabled,
            val_disabled_reason=None if split.enabled else split.disabled_reason,
            stored=stats.stored,
            dropped_consent=stats.dropped_consent,
            dropped_blocklist=stats.dropped_blocklist,
        )
        feed.cycle_start(
            cycle=cycles,
            stored=stats.stored,
            trained=stats.trained,
            dropped_consent=stats.dropped_consent,
            dropped_blocklist=stats.dropped_blocklist,
            examples=len(examples),
            batch_size=settings.batch_size,
            lr=settings.learning_rate,
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
                prev_checkpoint_loss, prev_val_loss = _checkpoint(
                    settings, log, feed, blocklist, model, optimizer, step, window, split.train,
                    probe_index, cycles, prev_checkpoint_loss, echo,
                    train_examples=examples,
                    val_examples=val_examples,
                    val_enabled=split.enabled,
                    val_disabled_reason=split.disabled_reason,
                    val_rows=len(split.val),
                    prev_val_loss=prev_val_loss,
                )
                probe_index += 1
                checkpoints += 1
                window = []
                publish_state = _maybe_auto_publish(settings, log, feed, checkpoints, publish_state)

        # Never end a cycle with unsaved work: a kill during the rest period
        # would otherwise throw away everything since the last checkpoint.
        if window:
            prev_checkpoint_loss, prev_val_loss = _checkpoint(
                settings, log, feed, blocklist, model, optimizer, step, window, split.train,
                probe_index, cycles, prev_checkpoint_loss, echo,
                train_examples=examples,
                val_examples=val_examples,
                val_enabled=split.enabled,
                val_disabled_reason=split.disabled_reason,
                val_rows=len(split.val),
                prev_val_loss=prev_val_loss,
            )
            probe_index += 1
            checkpoints += 1
            publish_state = _maybe_auto_publish(settings, log, feed, checkpoints, publish_state)

        cycle_seconds = round(time.perf_counter() - cycle_started, 2)
        log.event(
            "train.cycle.end",
            cycle=cycles,
            step=step,
            steps=cycle_steps,
            seconds=cycle_seconds,
            loss=round(last_loss, 6) if last_loss is not None else None,
        )
        feed.cycle_end(cycle=cycles, steps=cycle_steps, seconds=cycle_seconds)

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


WITHHELD = "*(withheld — matched the content filter)*"

# Which side of the split a probe row came from. Identical bad output means
# opposite things depending on this: on a held-out row the model was never
# trained on that prompt and garbage is expected; on a trained row it is the
# memorisation bug. Nobody reading the feed could tell the two apart before.
PROBE_TRAIN = "trained"
PROBE_FALLBACK = "not in dataset"


def _checkpoint(
    settings: Settings,
    log: EventLog,
    feed: TrainingFeed,
    blocklist: Blocklist,
    model: Babbler,
    optimizer,
    step: int,
    window: list[float],
    rows: list[Interaction],
    probe_index: int,
    cycle: int,
    prev_loss: float | None,
    echo: bool,
    *,
    train_examples: list[Example],
    val_examples: list[Example],
    val_enabled: bool,
    val_disabled_reason: str | None,
    val_rows: int,
    prev_val_loss: float | None,
) -> tuple[float, float | None]:
    mean_loss = sum(window) / len(window) if window else float("nan")
    started = time.perf_counter()
    prompt, expected = probe_prompt(rows, probe_index)
    # The probe walks the train split, so it is always asking about a row the
    # model has actually been trained on. That is the only way the probe answers
    # the question anyone is asking it -- garbage on a held-out row means
    # nothing, garbage on a trained row is the bug.
    probe_side = PROBE_TRAIN if rows else PROBE_FALLBACK
    text = best_of(
        model,
        prompt,
        n=max(1, settings.best_of),
        max_new_tokens=min(64, settings.max_new_tokens),
        temperature=settings.temperature,
        top_k=settings.top_k,
    )
    # What the running mean hides: which row is worst, and whether the loss is
    # really over the response and not over the prompt. `trainable_rows` keeps
    # the row list lined up with `train_examples`, which was filtered the same
    # way, so the worst row is named correctly.
    report = measure(model, train_examples, trainable_rows(rows))
    # Eval-mode, no-grad, no optimizer step: this scores the held-out rows
    # without moving a single weight or optimizer moment.
    val_loss = eval_loss(model, val_examples) if val_enabled else None
    overfitting = val_enabled and overfit_signal(mean_loss, prev_loss, val_loss, prev_val_loss)

    path = save_checkpoint(settings, model, optimizer, step, mean_loss)
    append_curve(settings, step, mean_loss, text, len(rows))

    log.event(
        "train.checkpoint",
        step=step,
        loss=round(mean_loss, 6),
        response_loss=round(report.response, 6) if report.response is not None else None,
        prompt_loss=round(report.prompt, 6) if report.prompt is not None else None,
        worst_row_loss=round(report.worst_row, 6) if report.worst_row is not None else None,
        worst_row_prompt=report.worst_prompt,
        val_loss=round(val_loss, 6) if val_loss is not None else None,
        val_rows=val_rows,
        val_enabled=val_enabled,
        val_disabled_reason=None if val_enabled else val_disabled_reason,
        overfit_signal=overfitting,
        rows=len(rows),
        file=path.name,
        seconds=round(time.perf_counter() - started, 2),
        prompt=prompt,
        probe_side=probe_side,
        expected=expected,
        sample=text.replace("\n", "\\n")[:200],
    )
    if echo:
        shown = text.replace("\n", "\\n")
        if not val_enabled:
            val_part = "val disabled"
        elif val_loss is None:
            val_part = "val   n/a"
        else:
            val_part = f"val {val_loss:8.4f}"
        overfit_part = "  ⚠ overfitting" if overfitting else ""
        # The worst row is printed next to the mean on purpose: the mean is the
        # number that looked fine while the bot was still talking nonsense.
        worst_part = f" | worst {report.worst_row:6.3f}" if report.worst_row is not None else ""
        print(
            f"step {step:>7,} | loss {mean_loss:8.4f}{worst_part} | {val_part}{overfit_part} | "
            f"[{probe_side}] {prompt!r} -> {shown!r} (expected {expected!r})",
            flush=True,
        )

    # Everything below leaves the machine via the feed -- filter it the same
    # way any other model output headed for Discord gets filtered, prompt and
    # expected answer included, since both come straight out of the dataset.
    feed_sample = text if blocklist.hit(text) is None else WITHHELD
    feed_prompt = prompt if blocklist.hit(prompt) is None else WITHHELD
    feed_expected = expected if blocklist.hit(expected) is None else WITHHELD
    feed.checkpoint(
        cycle=cycle,
        step=step,
        loss=mean_loss,
        prev_loss=prev_loss,
        rows=len(rows),
        prompt=feed_prompt,
        sample=feed_sample,
        expected=feed_expected,
        probe_side=probe_side,
        val_loss=val_loss,
        prev_val_loss=prev_val_loss,
        val_rows=val_rows,
        val_enabled=val_enabled,
        val_disabled_reason=val_disabled_reason,
        overfit_signal=overfitting,
    )
    return mean_loss, val_loss
