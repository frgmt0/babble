"""The trainer: pretrains the model from random init on the consented human
corpus -- the only corpus there is. Nothing external, nothing frozen to
continue from.

The objective is plain next-token prediction over the unlabelled corpus: every
token of every row is a target, every row counts the same, and nothing is paired
with anything. There is no prompt to mask off and no chosen answer to upweight,
because a corpus row is one piece of writing rather than a question and its
answer. Correction pairs are still captured and still published -- they are just
not what the loss is computed over any more.

Three properties matter more than speed here:

1. **It must not make the machine unusable.** It runs at nice 19 on a capped
   number of threads.
2. **It must be safe to kill at any moment.** Checkpoints are written to a temp
   file and renamed, so a `kill -9` mid-write leaves the previous checkpoint
   intact. SIGINT/SIGTERM finish the current step, checkpoint, and exit.
3. **Progress must be watchable.** Every checkpoint appends a line to
   `checkpoints/loss.jsonl` with its step, loss and a sample generation, and
   prints the same thing. The babble is the show.

Every run starts from random init and keeps the *best-validation* checkpoint,
not the last one -- on a small corpus the model overfits long before the step
budget is spent, so `train()` tracks val loss at every checkpoint interval and
writes whichever step had the lowest val loss. Early stopping is noise-aware:
on a corpus this size the val estimate itself has a measured spread of ~0.05
nats (recompute val over thousands of resampled holdouts with the checkpoint
held fixed and look at the std -- `experiments/val_noise.py`), so a patience
"stall" only counts once the `train_min_steps` floor has passed AND val sits
more than `train_stall_margin` above the best seen; wobble inside the band is
neutral. The run stops early after `train_patience` such stalls. It fires on a
**trigger, not a loop**: every `train_trigger_rows` new corpus rows since the
last run, or on demand with `--force`. The last-trained row count is persisted
(`checkpoints/train_state.json`) so a restart never re-fires, and training is
never a continuous cycle -- each call runs once and returns.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .backfill import backfill_corpus
from .blocklist import Blocklist
from .config import Settings
from .consent import ConsentStore, CorpusConsent
from .corpus import CorpusRow, CorpusStore
from .cpu_runtime import (
    configure_cpu,
    force_cpu_device,
    maybe_compile,
    model_state_dict,
    uncompiled,
)
from .discord_feed import TrainingFeed
from .export_hf import (
    CORPUS_FILE,
    DATA_FILE,
    ExportBlocked,
    build_export,
    push as push_export,
)
from .generate import continue_text
from .identity import Pseudonymiser
from .logs import EventLog, NullLog
from .model import (
    Babbler,
    config_from_settings,
    per_token_loss,
    sequence_loss,
)
from .subword import BPETokenizer
from .tokenizer import PAD_ID, VOCAB_SIZE, Example, text_examples
from .util import atomic_write_text, truncate, utcnow_iso

from . import valsplit

#: Used only when the corpus is empty and there is nothing real to seed with.
SAMPLE_PREFIXES = ("hello", "how are you")

#: How many bytes of a real row to feed the model before asking it to carry on.
PROBE_PREFIX_BYTES = 24

# Where the probe prefix came from. Identical bad output means opposite things
# depending on this: seeded from a hardcoded string the model has never seen,
# nonsense is expected; seeded from a row it has trained on, nonsense is the bug.
PROBE_TRAIN = "trained"
PROBE_FALLBACK = "not in dataset"

#: Posted in place of anything that matched the content filter on its way out.
WITHHELD = "*(withheld — matched the content filter)*"


def be_polite(settings: Settings, log: EventLog) -> None:
    """Give up priority and threads before touching a single tensor.

    Also locks torch onto the CPU / oneDNN path so training never quietly
    offloads to a GPU even if one is visible on the box.
    """
    applied_nice = None
    try:
        os.nice(settings.train_nice)
        applied_nice = os.nice(0)
    except (OSError, AttributeError, PermissionError):
        pass
    threads = max(1, settings.train_threads)
    cpu = configure_cpu(threads)
    log.event(
        "train.polite",
        nice=applied_nice,
        threads=threads,
        cpus=os.cpu_count(),
        device=cpu.get("device", "cpu"),
        mkldnn=cpu.get("mkldnn"),
    )


def corpus_rows(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> list[CorpusRow]:
    """Corpus rows whose author still consents, checked right now.

    Withdrawal already purges rows, so this is belt and braces -- but "used to
    train the model" is a promise made in the consent notice, and it should be
    enforced at the moment of training, not only at the moment of capture. The
    blocklist gets the same belt-and-braces treatment: a row stored before a
    term was added must not survive to be trained on once it is.

    A corpus row has one author, so one grant decides it -- and which grant that
    is depends on where the row's text came from, per `consent.SCOPE_BY_SOURCE`.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    gate = CorpusConsent(ConsentStore(settings.consent_path), ids)
    rows = CorpusStore(settings.corpus_path).all()
    return [r for r in rows if gate.allows(r) and not blocklist.matches(r.text)]


def to_examples(rows: list[CorpusRow], block_size: int) -> list[Example]:
    """Corpus rows to training examples: plain text, plain next-token objective.

    There is no weight argument and no masking argument, because there is
    nothing left to weight or mask. Every row counts once, every token in every
    row is a target, and a row too long for one block becomes several examples
    rather than a truncated one.
    """
    return [example for row in rows for example in text_examples(row.text, block_size)]


def count_tokens(examples: list[Example]) -> int:
    """Total tokens across `examples`, padding excluded. What actually gets trained."""
    return sum(len(e) for e in examples)


# --- held-out validation --------------------------------------------------
# The split identity lives in `valsplit.py` (torch-free, shared with the
# synthetic-corpus generator so it can exclude val-side rows); these are the
# same functions under their historical names.

_val_bucket = valsplit.val_bucket
val_holdout_size = valsplit.val_holdout_size


@dataclass
class Split:
    """Held-out split of consented rows.

    `enabled=False` means the corpus was too small to hold anything out
    without starving training: `train` is every row, `val` is empty, and
    `disabled_reason` says why.
    """

    train: list[CorpusRow]
    val: list[CorpusRow]
    enabled: bool
    disabled_reason: str | None = None


def split_rows(rows: list[CorpusRow], settings: Settings) -> Split:
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
        length = len(example)
        tokens[i, :length] = torch.as_tensor(example.tokens, dtype=torch.long)
        mask[i, :length] = torch.as_tensor(example.mask, dtype=torch.long)
    weights = torch.as_tensor([e.weight for e in examples], dtype=torch.float32)
    return tokens, mask, weights


@dataclass
class LossReport:
    """The checkpoint loss, taken apart into the numbers that mean something.

    `mean` is what training optimises, over every token of every example --
    there is no held-back half of the sequence any more, so there is no second
    loss to compare it against.

    `worst_row` is the number that explains "loss 0.02 but it babbles". The mean
    is a mean over *tokens*, so a short example the model has not learned barely
    moves it: four bad bytes among four hundred good ones is a rounding error in
    the average and complete nonsense to whoever wrote that line. Reporting the
    worst example alongside the mean makes that visible instead of mysterious.
    """

    mean: float | None = None
    worst_row: float | None = None
    worst_text: str | None = None


@torch.inference_mode()
def measure(
    model: Babbler, examples: list[Example], rows: list[CorpusRow] | None = None
) -> LossReport:
    """Mean loss and the worst single example, in one forward pass.

    `rows` is only used to name the worst example. One row can produce several
    examples, so the two lists are lined up by walking the rows in the same order
    `to_examples` does rather than by assuming they are the same length.
    """
    if not examples:
        return LossReport()
    was_training = model.training
    model.eval()
    try:
        tokens, mask, _ = _stack_examples(examples)
        per_token = per_token_loss(model, tokens)

        target_mask = mask[:, 1:].to(per_token.dtype)
        per_row = (per_token * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(min=1e-8)
        worst = int(torch.argmax(per_row))
        return LossReport(
            mean=float((per_token * target_mask).sum() / target_mask.sum().clamp(min=1e-8)),
            worst_row=float(per_row[worst]),
            worst_text=_example_owner(rows, worst, model.config.block_size) if rows else None,
        )
    finally:
        model.train(was_training)


def _example_owner(rows: list[CorpusRow], index: int, block_size: int) -> str | None:
    """The text of the row that produced example `index`.

    Rebuilding the counts is cheap and, unlike indexing straight into `rows`,
    stays correct when a long row expands into several examples.
    """
    seen = 0
    for row in rows:
        produced = len(text_examples(row.text, block_size))
        if seen <= index < seen + produced:
            return row.text
        seen += produced
    return None


@torch.inference_mode()
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
    """The full picture behind `corpus_rows` -- not just the count that made it."""

    stored: int
    trained: int
    dropped_consent: int
    dropped_blocklist: int


def dataset_stats(
    settings: Settings, ids: Pseudonymiser | None = None, blocklist: Blocklist | None = None
) -> DatasetStats:
    """How much of the stored corpus actually reaches training, and why the rest doesn't.

    Mirrors `corpus_rows`'s filter exactly, just bucketing the rejects instead
    of discarding them, so the two can never quietly disagree about what trains.
    """
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    gate = CorpusConsent(ConsentStore(settings.consent_path), ids)
    all_rows = CorpusStore(settings.corpus_path).all()

    trained = dropped_consent = dropped_blocklist = 0
    for row in all_rows:
        if not gate.allows(row):
            dropped_consent += 1
        elif blocklist.matches(row.text):
            dropped_blocklist += 1
        else:
            trained += 1
    return DatasetStats(
        stored=len(all_rows),
        trained=trained,
        dropped_consent=dropped_consent,
        dropped_blocklist=dropped_blocklist,
    )


def distinct_texts(rows: list[CorpusRow]) -> list[str]:
    """Ordered, de-duplicated row texts.

    Order follows first appearance, so the probe rotation walks the corpus in a
    stable sequence as new rows are only ever appended.
    """
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.text or row.text in seen:
            continue
        seen.add(row.text)
        order.append(row.text)
    return order


def leading_words(text: str, budget: int = PROBE_PREFIX_BYTES) -> str:
    """The opening of `text`, at most `budget` bytes, cut at a word boundary.

    Two properties, both load-bearing:

    **It is a genuine byte-prefix of the row.** Rebuilding it from
    `text.split()` joined by single spaces is not: a row containing a newline,
    an indent or a double space comes back as a byte sequence that never
    appeared in training. The model duly produces nonsense, the feed labels that
    nonsense `trained`, and the reader concludes the model has forgotten a row it
    has in fact memorised -- manufacturing exactly the failure `probe_side`
    exists to detect. So this slices the row's own bytes and never rejoins them.

    **Something is always held back**, whenever there is anything to hold back,
    so the model has something to actually continue. A "continuation" of the
    whole row only shows the model agreeing that the row has ended.

    Cutting on a byte boundary can split a multi-byte character; `errors=
    "ignore"` drops the fragment, which keeps the result a valid prefix.
    """
    raw = text.encode("utf-8")
    if not raw.strip():
        # Nothing but whitespace: there is no real word to seed with.
        return ""
    if len(raw) <= budget:
        # The whole row fits. Hold back the last word so there is something to
        # continue -- but a single word (no interior space) is handed back whole
        # rather than truncated, since chopping it would corrupt the very text we
        # are claiming the model trained on.
        boundary = raw.rstrip().rfind(b" ")
        head = raw[:boundary] if boundary > 0 else raw.rstrip()
    else:
        # Over budget: cut to the budget and back up to the last word boundary
        # inside the slice. A single word longer than the budget has none, and is
        # cut mid-word rather than handed back whole.
        head = raw[:budget]
        boundary = head.rfind(b" ")
        if boundary > 0:
            head = head[:boundary]
    return head.decode("utf-8", errors="ignore")


def probe_prefix(rows: list[CorpusRow], index: int) -> tuple[str, str]:
    """The prefix to continue from at checkpoint `index`, and where it came from.

    Cycles deterministically through every distinct text in the training split,
    one per checkpoint, wrapping around once it reaches the end. Falls back to a
    hardcoded prefix only when there is nothing real to seed with.
    """
    texts = distinct_texts(rows)
    if texts:
        prefix = leading_words(texts[index % len(texts)])
        if prefix:
            return prefix, PROBE_TRAIN
    # Nothing real to seed with, or nothing usable left after trimming. Say so
    # rather than labelling a hardcoded string as a row the model trained on.
    return SAMPLE_PREFIXES[index % len(SAMPLE_PREFIXES)], PROBE_FALLBACK


def make_batch(
    examples: list[Example], batch_size: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample with replacement and right-pad to the longest in the batch."""
    chosen = [rng.choice(examples) for _ in range(batch_size)]
    width = max(len(e) for e in chosen)
    tokens = torch.full((len(chosen), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(chosen), width), dtype=torch.long)
    for i, example in enumerate(chosen):
        length = len(example)
        tokens[i, :length] = torch.as_tensor(example.tokens, dtype=torch.long)
        mask[i, :length] = torch.as_tensor(example.mask, dtype=torch.long)
    weights = torch.as_tensor([e.weight for e in chosen], dtype=torch.float32)
    return tokens, mask, weights


def _build_optimizer(
    model: Babbler, settings: Settings, lr: float | None = None
) -> torch.optim.Optimizer:
    """AdamW tuned for CPU: `foreach` batches the small-param updates.

    `lr` overrides `settings.learning_rate` -- post-train runs at its own,
    far lower rate (`post_learning_rate`) than the pretrain."""
    kwargs = dict(
        lr=settings.learning_rate if lr is None else lr, weight_decay=settings.weight_decay
    )
    try:
        return torch.optim.AdamW(model.parameters(), foreach=True, **kwargs)
    except (TypeError, ValueError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


# --- checkpoints ---------------------------------------------------------


SCRATCH_DIR = ".partial"


class TokenizerMismatch(RuntimeError):
    """A checkpoint write was refused because its vocab_size disagrees with
    the tokenizer promoted for serving next to it.

    Raised instead of silently writing the bad file: this is exactly how a
    fresh byte-level (vocab 260) pretrain clobbered the promoted BPE (vocab
    16384) `latest.pt` live on 2026-08-22 -- see DEFECT_3 in the
    reconciliation PR. `save_checkpoint` is shared by the pretrain loop and
    `post_train`/`post_train_from_checkpoint`, so this guard protects both.
    """


def served_tokenizer_vocab_size(settings: Settings) -> int | None:
    """Vocab size of whatever `tokenizer.json` is currently promoted next to
    `latest.pt`, or `None` if none is promoted (the byte tokenizer is live).

    Cheap: `tokenizer.json` is just the ordered merge list, so loading it to
    read `vocab_size` costs a JSON parse, not a checkpoint load.
    """
    path = settings.tokenizer_path
    if not path.is_file():
        return None
    return BPETokenizer.from_json(path).vocab_size


def save_checkpoint(settings: Settings, model: Babbler, optimizer, step: int, loss: float) -> Path:
    """Write `ckpt-NNNNNNN.pt` and repoint `latest.pt`, both atomically.

    Half-written files are staged in a `.partial/` scratch directory rather than
    alongside the real checkpoints. `os.replace` was always atomic, so a torn
    file was never *loadable* -- but it was visible, sitting next to the good
    checkpoints with a name one glob away from being picked up, and a `kill -9`
    landing mid-write left it there. Staging elsewhere makes "nothing
    half-written is ever in the checkpoint directory" true by construction
    instead of true by how fast the write happened to be.

    Refuses (`TokenizerMismatch`, nothing written) if a tokenizer is already
    promoted for serving and this model's own vocab_size disagrees with it --
    defense in depth alongside the fail-fast check in `train()`, so any other
    caller of this function (post-train included) gets the same protection.
    """
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scratch = settings.checkpoint_dir / SCRATCH_DIR
    scratch.mkdir(exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "config": uncompiled(model).config.to_dict(),
        "model": model_state_dict(model),
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "saved_at": utcnow_iso(),
    }
    served_vocab = served_tokenizer_vocab_size(settings)
    model_vocab = payload["config"].get("vocab_size")
    if served_vocab is not None and model_vocab != served_vocab:
        raise TokenizerMismatch(
            f"refusing to write checkpoint with vocab_size={model_vocab}: the "
            f"tokenizer promoted for serving at {settings.tokenizer_path} has "
            f"vocab_size={served_vocab} -- {settings.latest_checkpoint} was not touched"
        )
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

    # Both files, so a change to either one is a change worth pushing: the
    # corrections can move while the corpus stands still, and vice versa.
    digest = hashlib.sha256()
    for name in (CORPUS_FILE, DATA_FILE):
        path = result.path / name
        digest.update(path.read_bytes() if path.exists() else b"")
    content_hash = digest.hexdigest()
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


def append_curve(
    settings: Settings,
    step: int,
    loss: float,
    sample_text: str,
    *,
    stored_rows: int,
    train_rows: int,
    val_rows: int = 0,
    val_loss: float | None = None,
) -> None:
    """Append one checkpoint line to `loss.jsonl`.

    `loss` is the window-averaged *training* loss for the interval. When a
    held-out set exists, `val_loss` is logged alongside it so runs are
    comparable across dates — comparing train loss to someone else's val loss
    is how "2.44 looks worse than 1.61" happens when nothing regressed.

    Row counts are split on purpose: `stored_rows` is the raw corpus size the
    +N-row trigger measures (same as `train_state.json`), while `train_rows` /
    `val_rows` are the split sizes that actually fed this checkpoint. The legacy
    `rows` field mirrors `train_rows` so old readers keep working.
    """
    entry: dict = {
        "step": step,
        "loss": round(loss, 6),
        "rows": train_rows,
        "stored_rows": stored_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "at": utcnow_iso(),
        "sample": sample_text,
    }
    if val_loss is not None:
        entry["val_loss"] = round(val_loss, 6)
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
class TrainResult:
    """What a `train()` call did, or why it didn't.

    `ran` is False for the two no-op outcomes (`not_due`, `no_data`); every
    other field is 0/None in those cases. `stopped_because` is `"trained"` for
    a completed or early-stopped run, `"not_due"` / `"no_data"` for a skip, or
    `"signal:SIGINT"` / `"signal:SIGTERM"` if it was interrupted mid-run.
    """

    ran: bool
    stopped_because: str
    steps_run: int = 0
    final_step: int = 0
    last_loss: float | None = None
    val_loss: float | None = None
    checkpoints_written: int = 0
    budget: int = 0
    stopped_early: bool = False
    rows_trained: int = 0
    current_rows: int = 0
    last_trained_rows: int = 0


@dataclass
class TrainTrigger:
    current_rows: int
    last_trained_rows: int
    threshold: int

    @property
    def new_rows(self) -> int:
        return self.current_rows - self.last_trained_rows

    @property
    def due(self) -> bool:
        """Automatic firing: the threshold is on, and the corpus has grown by
        at least that many rows since the last run."""
        return self.threshold > 0 and self.new_rows >= self.threshold


def corpus_row_count(settings: Settings) -> int:
    """Total stored corpus rows -- the number the +N-row trigger measures growth
    against. Uses the raw stored count, not the consent-filtered count, so a
    revocation cannot make the corpus appear to shrink below the last trigger."""
    return CorpusStore(settings.corpus_path).count()


def read_train_state(settings: Settings) -> dict:
    path = settings.train_state_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_train_state(settings: Settings, *, rows: int, step: int, steps_run: int | None = None) -> None:
    payload: dict = {"last_trained_rows": rows, "step": step, "at": utcnow_iso()}
    # `step` is the best-val checkpoint written to `latest.pt`; `steps_run` is
    # how far the loop actually got (loss.jsonl may run past `step` before early
    # stop). Both belong in state so the two files are not read as contradictory.
    if steps_run is not None:
        payload["steps_run"] = steps_run
    atomic_write_text(settings.train_state_path, json.dumps(payload, indent=2))


def train_trigger(settings: Settings) -> TrainTrigger:
    state = read_train_state(settings)
    return TrainTrigger(
        current_rows=corpus_row_count(settings),
        last_trained_rows=int(state.get("last_trained_rows", 0)),
        threshold=settings.train_trigger_rows,
    )


def train(
    settings: Settings,
    *,
    steps: int | None = None,
    force: bool = False,
    patience: int | None = None,
    seed: int | None = None,
    log: EventLog | None = None,
    feed: TrainingFeed | None = None,
    echo: bool = True,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> TrainResult:
    """The only training path: random init, on the consented human corpus,
    writes `checkpoints/latest.pt`. Every run starts fresh -- there is no base
    to continue from and no previous run to resume, so nothing compounds
    across reruns.

    Fires only when `force` is set or the +N-row trigger is due; otherwise a
    no-op that reports why. `steps` is a ceiling, not a target: the checkpoint
    with the lowest val loss wins and is what gets written, and the run stops
    early once `patience` checkpoint intervals in a row fail to beat it. Both
    are no-ops without a held-out validation set (too little data to spare
    any), in which case every checkpoint interval is simply written in turn.
    """
    settings.ensure_dirs()
    ids = ids or Pseudonymiser.load(settings)
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    owns_log = log is None  # if we opened it, we close it
    log = log or EventLog(settings, ids, component="trainer", echo=echo)
    feed = feed or TrainingFeed.from_env(log)

    status = train_trigger(settings)
    if not force and not status.due:
        log.event("train.skipped", reason="not_due", new_rows=status.new_rows, threshold=status.threshold)
        if owns_log:
            log.close()
        else:
            log.flush()
        return TrainResult(
            False, "not_due", current_rows=status.current_rows, last_trained_rows=status.last_trained_rows
        )

    # This path always builds a fresh byte-level (vocab_size 260) model from
    # random init -- see the docstring above. If a BPE (or other learned)
    # tokenizer is already promoted for serving, that model is incompatible
    # with what `latest.pt` currently holds; writing over it is exactly the
    # DEFECT_3 incident (a byte-level checkpoint clobbered a promoted 34M BPE
    # model live on 2026-08-22). Refuse loudly instead, and advance the
    # trigger by a full cycle so this does not refire on every subsequent
    # corpus row -- same cadence as a real run, until a human either retires
    # this pretrain path or un-promotes the tokenizer.
    served_vocab = served_tokenizer_vocab_size(settings)
    if served_vocab is not None and served_vocab != VOCAB_SIZE:
        log.event(
            "train.skipped",
            reason="tokenizer_mismatch",
            pretrain_vocab=VOCAB_SIZE,
            served_vocab=served_vocab,
            tokenizer_path=str(settings.tokenizer_path),
        )
        write_train_state(
            settings,
            rows=status.current_rows,
            step=int(read_train_state(settings).get("step", 0)),
        )
        if owns_log:
            log.close()
        else:
            log.flush()
        return TrainResult(
            False,
            "tokenizer_mismatch",
            current_rows=status.current_rows,
            last_trained_rows=status.current_rows,
        )

    # Migrate before reading, so an install that predates the corpus starts
    # training without anyone running a command by hand. Idempotent and quiet:
    # it dedupes on the content-addressed row id and only logs when it
    # actually added rows.
    backfill_corpus(settings, log=log, ids=ids, blocklist=blocklist, log_noop=False)
    rows = corpus_rows(settings, ids, blocklist)
    if not rows:
        log.event("train.idle", reason="no_consented_rows", rows=0)
        feed.idle()
        if owns_log:
            log.close()
        else:
            log.flush()
        return TrainResult(
            False, "no_data", current_rows=status.current_rows, last_trained_rows=status.last_trained_rows
        )

    be_polite(settings, log)
    interrupt = Interruption().install()
    swept = sweep_scratch(settings)
    if swept:
        log.event("train.scratch_swept", files=swept)

    device = force_cpu_device()
    model = Babbler(config_from_settings(settings)).to(device)
    model = maybe_compile(model)
    optimizer = _build_optimizer(model, settings)
    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed if seed is not None else 0)
    budget_for_schedule = steps if steps is not None else settings.train_steps
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, budget_for_schedule),
            eta_min=settings.learning_rate * 0.1,
        )
        if settings.train_cosine
        else None
    )

    split = split_rows(rows, settings)
    # Same call on both sides of the split: with no weighting and no masking
    # left, a held-out example is built exactly like a trained one, which is
    # what makes val loss comparable to train loss at all.
    examples = to_examples(split.train, model.config.block_size)
    val_examples = to_examples(split.val, model.config.block_size)

    # Labelled synthetic rows join the batch pool only when the flag says so,
    # and only ever on the train side -- validation stays 100% real held-out
    # human text, so the number that judges a run cannot be flattered by the
    # generator that is trying to improve it.
    synthetic_examples: list[Example] = []
    if settings.train_synthetic:
        from .synthcorpus import refresh_synthetic_corpus_if_stale, trainable_synthetic_rows

        # The val holdout moves as the corpus grows, so a synthetic file
        # generated against an older corpus can contain splices of rows that
        # are val NOW. Rebuild it first or the leak comes back silently.
        refreshed = refresh_synthetic_corpus_if_stale(settings, ids=ids, blocklist=blocklist)
        if refreshed is not None:
            log.event(
                "train.synthetic.rebuilt",
                generated=refreshed.generated,
                source_rows=refreshed.source_rows,
                excluded_val_rows=refreshed.excluded_val_rows,
            )
        synth_rows = trainable_synthetic_rows(settings, blocklist)
        synthetic_examples = [
            ex for r in synth_rows for ex in text_examples(r.text, model.config.block_size)
        ]
    batch_pool = examples + synthetic_examples
    budget = steps if steps is not None else settings.train_steps
    run_patience = patience if patience is not None else settings.train_patience
    train_tokens = count_tokens(examples)
    stats = dataset_stats(settings, ids, blocklist)

    log.event(
        "train.start",
        step=0,
        resumed=False,
        params=model.num_params(),
        budget=budget,
        batch_size=settings.batch_size,
        lr=settings.learning_rate,
    )
    feed.start(resumed=False, step=0)
    feed.active()
    log.event(
        "train.cycle.start",
        cycle=1,
        step=0,
        rows=len(rows),
        examples=len(examples),
        synthetic_examples=len(synthetic_examples) or None,
        tokens=train_tokens,
        val_tokens=count_tokens(val_examples),
        planned_steps=budget,
        train_rows=len(split.train),
        val_rows=len(split.val),
        val_enabled=split.enabled,
        val_disabled_reason=None if split.enabled else split.disabled_reason,
        stored=stats.stored,
        dropped_consent=stats.dropped_consent,
        dropped_blocklist=stats.dropped_blocklist,
    )
    feed.cycle_start(
        cycle=1,
        stored=stats.stored,
        trained=stats.trained,
        dropped_consent=stats.dropped_consent,
        dropped_blocklist=stats.dropped_blocklist,
        examples=len(examples),
        tokens=train_tokens,
        train_rows=len(split.train),
        val_rows=len(split.val),
        batch_size=settings.batch_size,
        lr=settings.learning_rate,
    )

    window: list[float] = []
    checkpoints = 0
    probe_index = 0  # never reset: a checkpoint counter, not a dataset index
    step = 0
    last_loss: float | None = None
    prev_checkpoint_loss: float | None = None
    prev_val_loss: float | None = None
    publish_state = PublishState()
    best: dict | None = None
    stalls = 0
    stopped_early = False
    interrupted = False
    cycle_started = time.perf_counter()
    model.train()

    def checkpoint() -> None:
        nonlocal prev_checkpoint_loss, prev_val_loss, probe_index, checkpoints, window
        nonlocal publish_state, best, stalls, stopped_early
        prev_checkpoint_loss, val = _checkpoint(
            settings, log, feed, blocklist, model, optimizer, step, window, split.train,
            probe_index, 1, prev_checkpoint_loss, echo,
            train_examples=examples,
            val_examples=val_examples,
            val_enabled=split.enabled,
            val_disabled_reason=split.disabled_reason,
            val_rows=len(split.val),
            prev_val_loss=prev_val_loss,
        )
        prev_val_loss = val
        probe_index += 1
        checkpoints += 1
        window = []
        publish_state = _maybe_auto_publish(settings, log, feed, checkpoints, publish_state)
        if val is None:
            return  # nothing to compare -- every interval just gets written as it comes
        if best is None or val < best["val"]:
            best = {
                "step": step,
                "loss": prev_checkpoint_loss,
                "val": val,
                "model": copy.deepcopy(model_state_dict(model)),
                "optim": copy.deepcopy(optimizer.state_dict()),
            }
            stalls = 0
            return
        # The val estimate on a corpus this size has a measured noise band of
        # ~0.05 nats (see `train_stall_margin`), so "failed to improve" is not
        # evidence of anything. A checkpoint burns patience only once the
        # `train_min_steps` floor has passed AND val sits above the best seen
        # by more than the margin; movement inside the band is neutral --
        # neither a new best nor a stall.
        margin = max(0.0, settings.train_stall_margin)
        if step >= settings.train_min_steps and val > best["val"] + margin:
            stalls += 1
            if run_patience and stalls >= run_patience:
                stopped_early = True

    for _ in range(budget):
        if interrupt.requested:
            interrupted = True
            break
        tokens, mask, weights = make_batch(batch_pool, settings.batch_size, rng)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        step += 1
        value = float(loss.detach())
        window.append(value)
        last_loss = value

        if step % settings.checkpoint_every == 0:
            checkpoint()
            if stopped_early:
                break

    # Never leave unsaved work: a kill right after the loop would otherwise
    # throw away everything since the last checkpoint interval.
    if window and not stopped_early:
        checkpoint()

    final_step, final_loss, final_val = step, last_loss, prev_val_loss
    if best is not None and best["step"] != step:
        # Roll the live model/optimizer back to the winning step before the
        # final save -- which reads off the live model -- writes it to disk.
        uncompiled(model).load_state_dict(best["model"])
        optimizer.load_state_dict(best["optim"])
        save_checkpoint(settings, model, optimizer, best["step"], best["loss"])
        final_step, final_loss, final_val = best["step"], best["loss"], best["val"]
    elif best is not None:
        final_val = best["val"]

    if interrupted:
        log.event("train.interrupt", signal=interrupt.signal_name, step=step)

    current = corpus_row_count(settings)
    write_train_state(settings, rows=current, step=final_step, steps_run=step)

    cycle_seconds = round(time.perf_counter() - cycle_started, 2)
    log.event(
        "train.cycle.end", cycle=1, step=step, steps=step, seconds=cycle_seconds,
        loss=round(last_loss, 6) if last_loss is not None else None,
    )
    feed.cycle_end(cycle=1, steps=step, seconds=cycle_seconds)

    reason = f"signal:{interrupt.signal_name}" if interrupted else "trained"
    log.event(
        "train.stop",
        step=final_step,
        steps_run=step,
        checkpoints=checkpoints,
        reason=reason,
        stopped_early=stopped_early,
        last_loss=round(final_loss, 6) if final_loss is not None else None,
        val_loss=round(final_val, 6) if final_val is not None else None,
        rows_trained=len(rows),
        last_trained_rows=current,
        budget=budget,
    )
    if owns_log:
        log.close()
    else:
        log.flush()

    return TrainResult(
        True,
        reason,
        steps_run=step,
        final_step=final_step,
        last_loss=final_loss,
        val_loss=final_val,
        checkpoints_written=checkpoints,
        budget=budget,
        stopped_early=stopped_early,
        rows_trained=len(rows),
        current_rows=current,
        last_trained_rows=current,
    )


class AutoTrainTrigger:
    """Fires a training run when the corpus has grown enough -- a trigger, not
    a loop. The bot calls `maybe_run()` after each fresh corpus row (mirroring
    the growth publisher); when the trigger is due it launches `babble train`
    as a detached, low-priority subprocess so training never blocks the event
    loop, and the bot hot-reloads the new `latest.pt` on its own.
    """

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        self.settings = settings
        self.log = log or NullLog()
        self._proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        """Is a pretrain launched by this trigger still in flight? Read by
        `AutoPostTrigger` so a post-train never starts while a pretrain it
        would otherwise race is still writing `latest.pt`."""
        return self._proc is not None and self._proc.poll() is None

    def maybe_run(self) -> None:
        if self.settings.train_trigger_rows <= 0:
            return
        if self.is_running():
            return  # a run is already in flight; do not stack them
        if not train_trigger(self.settings).due:
            return
        self._launch()

    def _launch(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "babble", "train", "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.log.event("train.triggered", pid=self._proc.pid)
        except Exception as exc:  # a launch hiccup must never take the bot down
            self.log.event("train.trigger_failed", error=f"{type(exc).__name__}: {exc}")


def _checkpoint(
    settings: Settings,
    log: EventLog,
    feed: TrainingFeed,
    blocklist: Blocklist,
    model: Babbler,
    optimizer,
    step: int,
    window: list[float],
    rows: list[CorpusRow],
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
    # The probe walks the train split, so the prefix is always the opening of a
    # row the model has actually been trained on. What comes back is a
    # continuation of it, not an answer to it: there is no answer to show, and
    # printing an "expected" field would be inventing one.
    prefix, probe_side = probe_prefix(rows, probe_index)
    text = continue_text(
        model,
        prefix,
        max_new_tokens=min(64, settings.max_new_tokens),
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        repetition_penalty=settings.repetition_penalty,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
    )
    # What the running mean hides: which example is worst. `_example_owner`
    # walks the rows the same way `to_examples` did, so a long row that became
    # several examples still names the right one.
    report = measure(model, train_examples, rows)
    # Eval-mode, no-grad, no optimizer step: this scores the held-out rows
    # without moving a single weight or optimizer moment.
    val_loss = eval_loss(model, val_examples) if val_enabled else None
    overfitting = val_enabled and overfit_signal(mean_loss, prev_loss, val_loss, prev_val_loss)

    path = save_checkpoint(settings, model, optimizer, step, mean_loss)
    append_curve(
        settings,
        step,
        mean_loss,
        text,
        stored_rows=corpus_row_count(settings),
        train_rows=len(rows),
        val_rows=val_rows,
        val_loss=val_loss,
    )

    log.event(
        "train.checkpoint",
        step=step,
        loss=round(mean_loss, 6),
        mean_loss=round(report.mean, 6) if report.mean is not None else None,
        worst_row_loss=round(report.worst_row, 6) if report.worst_row is not None else None,
        worst_row_text=truncate(report.worst_text, 200) if report.worst_text else None,
        val_loss=round(val_loss, 6) if val_loss is not None else None,
        val_rows=val_rows,
        val_enabled=val_enabled,
        val_disabled_reason=None if val_enabled else val_disabled_reason,
        overfit_signal=overfitting,
        rows=len(rows),
        file=path.name,
        seconds=round(time.perf_counter() - started, 2),
        prefix=prefix,
        probe_side=probe_side,
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
            f"[{probe_side}] {prefix!r} -> {shown!r}",
            flush=True,
        )

    # Everything below leaves the machine via the feed -- filter it the same
    # way any other model output headed for Discord gets filtered, the prefix
    # included, since it comes straight out of the corpus.
    feed_sample = text if blocklist.hit(text) is None else WITHHELD
    feed_prefix = prefix if blocklist.hit(prefix) is None else WITHHELD
    feed.checkpoint(
        cycle=cycle,
        step=step,
        loss=mean_loss,
        prev_loss=prev_loss,
        rows=len(rows),
        prefix=feed_prefix,
        sample=feed_sample,
        probe_side=probe_side,
        val_loss=val_loss,
        prev_val_loss=prev_val_loss,
        val_rows=val_rows,
        val_enabled=val_enabled,
        val_disabled_reason=val_disabled_reason,
        overfit_signal=overfitting,
    )
    return mean_loss, val_loss
