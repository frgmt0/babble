"""Stage 2: a short supervised post-train on the correction pairs.

Pretraining (`babble train`) teaches the model to continue plain text -- there
is no prompt/response boundary in that objective, so it never learns to
*answer*. This stage fine-tunes the pretrained checkpoint on the
`(prompt, chosen)` correction pairs laid out as `<bos> prompt <sep> response
<eos>` (`tokenizer.build_example`, the same pair layout `generate.py` already
uses to score a correction against its rejected answer), so the model has at
least seen what answering looks like.

There are only a few dozen pairs against a ~3.3M-parameter model. It will
memorise those pairs and generalise to approximately nothing -- and measured
on the live corpus, the old settings did real damage on the way there: 38
pairs at the pretrain LR shipped a checkpoint whose held-out corpus loss was
+1.15 nats worse than the pretrain it started from. Four guardrails now stand
between a fine-tune and the served bot, each config-flippable:

1. **Its own learning rate** (`post_learning_rate`, default 1e-4) -- the
   pretrain LR tears through the weights in a handful of steps at this scale.
2. **Rehearsal** (`post_rehearsal`, default 0.5): that fraction of every
   batch is plain corpus text under the pretrain objective, so the fine-tune
   cannot drift off the corpus distribution unopposed.
3. **A pair floor** (`post_min_pairs`): below it the run refuses to start
   (`--force` overrides for experiments).
4. **A promotion gate** (`post_gate_margin`): the candidate is scored against
   the pretrain snapshot on the held-out corpus split -- the layout the bot
   actually serves -- and a candidate worse by more than the margin never
   touches `latest.pt`. Nothing writes `latest.pt` mid-run either, so a
   half-finished fine-tune can never ship.

The `rejected` half of each correction is captured and published, but it is
not the objective here: this is supervised fine-tuning on the chosen answer,
not preference optimisation.

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

`include_synthetic=True` (`--include-synthetic` on the CLI) additionally
trains on the postulated-prompt pairs in `data/synthetic_pairs.jsonl` -- see
`synthetic.py`. Off by default and never counted by the +N-pair trigger, so
generating synthetic pairs never changes when a post-train fires; it only
changes what an explicit `--include-synthetic` run trains on.
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
from .synthetic import SyntheticPair, trainable_synthetic_pairs
from .tokenizer import Example, build_continuation_example, build_example
from .trainer import (
    SCRATCH_DIR,
    _build_optimizer,
    _stack_examples,
    append_curve,
    be_polite,
    corpus_rows,
    eval_loss,
    save_checkpoint,
    split_rows,
    sweep_scratch,
    to_examples,
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
    synthetic_pairs_trained: int = 0
    current_pairs: int = 0
    last_trained_pairs: int = 0
    final_step: int = 0
    last_loss: float | None = None
    val_loss: float | None = None
    checkpoints_written: int = 0
    budget: int = 0
    stopped_early: bool = False
    path: Path | None = None
    # The promotion gate's verdict: `promoted` says whether `latest.pt` was
    # actually written, and the two corpus-val numbers are what decided it --
    # candidate vs the pretrain snapshot it started from, both scored on the
    # same held-out real corpus rows the bot's serving layout is judged by.
    promoted: bool = False
    gated: bool = False
    corpus_val_before: float | None = None
    corpus_val_after: float | None = None


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


def _example_builder(settings: Settings):
    """The layout post-train teaches, per `Settings.post_layout`.

    "continuation" is the default and the one the bot can actually reach:
    `<bos> prompt response <eos>` with the prompt masked, byte-identical to
    the `text_context` prefix serving generates from. "pair" is the
    historical `<bos> prompt <sep> response <eos>` -- a layout whose `<sep>`
    never appears at inference, kept for experiments.
    """
    if settings.post_layout == "pair":
        return build_example
    return build_continuation_example


def _build_examples(
    pairs: list[Interaction], block_size: int, builder=build_continuation_example
) -> list[Example]:
    return [builder(p.prompt, p.chosen, block_size) for p in pairs]


def _combined_examples(
    pairs: list[Interaction],
    synthetic: list[SyntheticPair],
    block_size: int,
    builder=build_continuation_example,
) -> list[Example]:
    """Human and synthetic pairs, interleaved by sorting on their (both
    content-hash) ids rather than concatenating the two lists -- so the
    train/val tail-slice in `_split_val` below draws from a mix of both kinds
    instead of `val` landing entirely inside whichever pool was appended
    last. Both input lists already come pre-sorted by id from
    `trainable_pairs` / `trainable_synthetic_pairs`; re-sorting the merge is
    what actually interleaves them.
    """
    items = [(p.id, p.prompt, p.chosen) for p in pairs] + [
        (p.id, p.prompt, p.response) for p in synthetic
    ]
    items.sort(key=lambda item: item[0])
    return [builder(prompt, response, block_size) for _, prompt, response in items]


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
    include_synthetic: bool = False,
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

    `include_synthetic=True` also trains on `trainable_synthetic_pairs` --
    the postulated-prompt pairs `babble synth-generate` writes to
    `data/synthetic_pairs.jsonl` (see `synthetic.py`). Off by default: a
    synthetic pair is never trained on unless this is explicitly set (or
    `--include-synthetic` is passed on the CLI), so generating them is always
    a separate, inspectable step from training on them. The +N-pair trigger
    still counts human pairs only -- generating synthetic pairs never makes a
    post-train due on its own.
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
    synthetic_pairs = trainable_synthetic_pairs(settings, ids, blocklist) if include_synthetic else []
    if not pairs and not synthetic_pairs:
        log.event("post.skipped", reason="no_data")
        return PostTrainResult(
            False, "no_data",
            current_pairs=status.current_pairs, last_trained_pairs=status.last_trained_pairs,
        )

    # A supervised pass over a few dozen rows memorises them and generalises
    # to nothing -- measured, not hypothetical: 38 pairs at the old settings
    # drove pair-val from 3.59 to 5.56 while pair-train fell to 0.32. Below
    # this floor the run refuses to start; `--force` overrides for an explicit
    # experiment, and the promotion gate below still decides what ships.
    total_pairs = len(pairs) + len(synthetic_pairs)
    if not force and total_pairs < settings.post_min_pairs:
        log.event(
            "post.skipped", reason="too_few_pairs",
            pairs=total_pairs, threshold=settings.post_min_pairs,
        )
        return PostTrainResult(
            False, "too_few_pairs",
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
    post_lr = settings.post_learning_rate if settings.post_learning_rate > 0 else None
    optimizer = _build_optimizer(model, settings, lr=post_lr)

    builder = _example_builder(settings)
    examples = (
        _combined_examples(pairs, synthetic_pairs, model.config.block_size, builder)
        if include_synthetic
        else _build_examples(pairs, model.config.block_size, builder)
    )
    train_examples, val_examples = _split_val(examples)
    run_patience = patience if patience is not None else settings.post_patience
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Rehearsal: a slice of every batch is plain corpus text, trained with the
    # same next-token objective the pretrain used, so the fine-tune cannot
    # drift the weights off the corpus distribution unopposed. The corpus val
    # rows are excluded, so the promotion gate below still scores the
    # candidate on text it never fine-tuned on.
    rehearsal = min(1.0, max(0.0, settings.post_rehearsal))
    corpus_split = split_rows(corpus_rows(settings, ids, blocklist), settings)
    rehearsal_examples = (
        to_examples(corpus_split.train, model.config.block_size) if rehearsal > 0 else []
    )
    corpus_val_examples = to_examples(corpus_split.val, model.config.block_size)

    log.event(
        "post.start",
        pairs=len(pairs),
        synthetic_pairs=len(synthetic_pairs),
        examples=len(examples),
        rehearsal=rehearsal if rehearsal_examples else 0.0,
        rehearsal_examples=len(rehearsal_examples),
        lr=post_lr if post_lr is not None else settings.learning_rate,
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
        pair_val = eval_loss(model, val_examples) if val_examples else None
        corpus_val = eval_loss(model, corpus_val_examples) if corpus_val_examples else None
        # Selection runs on corpus val whenever it exists. The pair holdout is
        # a handful of rows (its standard error at the live pair count is on
        # the order of half a nat), so an argmin over it is close to random --
        # and worse, a pair-val winner can fail the promotion gate while an
        # earlier checkpoint would have passed. Corpus val is better measured
        # AND the number the gate judges, so selecting on it means the gate
        # scores the best candidate the run actually produced. Pair val is
        # still computed and logged.
        metric = corpus_val if corpus_val is not None else pair_val
        checkpoints += 1
        append_curve(
            settings,
            step,
            mean,
            "",
            stored_rows=pair_count(settings),
            train_rows=len(train_examples),
            val_rows=len(val_examples),
            val_loss=pair_val,
        )
        log.event(
            "post.checkpoint",
            step=step, loss=mean, val_loss=pair_val,
            corpus_val=round(corpus_val, 6) if corpus_val is not None else None,
            examples=len(examples),
        )
        if echo:
            val_s = f"{pair_val:.4f}" if pair_val is not None else "   n/a"
            cval_s = f"{corpus_val:.4f}" if corpus_val is not None else "   n/a"
            print(
                f"[post] step {step:7d} | loss {mean:8.4f} | pair val {val_s} | corpus val {cval_s}",
                flush=True,
            )
        if metric is not None:
            if best is None or metric < best["metric"]:
                best = {
                    "step": step,
                    "loss": mean,
                    "val": pair_val,
                    "corpus_val": corpus_val,
                    "metric": metric,
                    "model": copy.deepcopy(model_state_dict(model)),
                    "optim": copy.deepcopy(optimizer.state_dict()),
                }
                stalls = 0
            else:
                stalls += 1
                if run_patience and stalls >= run_patience:
                    stop = True
        # With no held-out examples at all the candidate is simply the final
        # weights -- nothing is written mid-run either way: the promotion gate
        # below is the only thing that ever writes `latest.pt`, so a fine-tune
        # can never ship without being scored first.
        window = []

    def make_mixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A batch of pair examples with a rehearsal slice of plain corpus
        text mixed in, per `post_rehearsal`."""
        chosen: list[Example] = []
        for _ in range(settings.batch_size):
            if rehearsal_examples and rng.random() < rehearsal:
                chosen.append(rng.choice(rehearsal_examples))
            else:
                chosen.append(rng.choice(train_examples))
        return _stack_examples(chosen)

    model.train()
    for _ in range(budget):
        tokens, mask, weights = make_mixed_batch()
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
    candidate_corpus_val: float | None = None
    if best is not None:
        uncompiled(model).load_state_dict(best["model"])
        optimizer.load_state_dict(best["optim"])
        final_step, final_loss, final_val = best["step"], best["loss"], best["val"]
        candidate_corpus_val = best.get("corpus_val")

    # --- promotion gate ---------------------------------------------------
    # The bot serves plain-text continuations, so the number that decides
    # promotion is corpus val loss: the candidate against the very pretrain it
    # started from, on the same held-out real rows. A candidate that gave up
    # more than `post_gate_margin` of corpus ability does not ship, no matter
    # how good its pair loss looks -- pair loss on a few dozen rows is mostly
    # a measure of memorisation.
    corpus_val_before: float | None = None
    corpus_val_after: float | None = None
    promoted = True
    if corpus_val_examples and settings.post_gate_margin >= 0:
        corpus_val_after = (
            candidate_corpus_val
            if candidate_corpus_val is not None
            else eval_loss(model, corpus_val_examples)
        )
        pretrain_model = _load_pretrained(pretrained_path)
        corpus_val_before = eval_loss(pretrain_model, corpus_val_examples)
        del pretrain_model
        promoted = corpus_val_after <= corpus_val_before + settings.post_gate_margin
    if promoted:
        save_checkpoint(settings, model, optimizer, final_step, final_loss)

    current = pair_count(settings)
    write_post_state(
        settings, pairs=current, step=final_step, latest_hash=file_hash(settings.latest_checkpoint)
    )
    log.event(
        "post.done",
        step=final_step,
        loss=round(final_loss, 6) if final_loss is not None else None,
        val_loss=round(final_val, 6) if final_val is not None else None,
        corpus_val_before=round(corpus_val_before, 6) if corpus_val_before is not None else None,
        corpus_val_after=round(corpus_val_after, 6) if corpus_val_after is not None else None,
        promoted=promoted,
        pairs_trained=len(pairs),
        synthetic_pairs_trained=len(synthetic_pairs),
        last_trained_pairs=current,
        checkpoints=checkpoints,
        stopped_early=stop,
        budget=budget,
    )

    return PostTrainResult(
        True,
        "trained" if promoted else "gated",
        pairs_trained=len(pairs),
        synthetic_pairs_trained=len(synthetic_pairs),
        current_pairs=current,
        last_trained_pairs=current,
        final_step=final_step,
        last_loss=final_loss,
        val_loss=final_val,
        checkpoints_written=checkpoints,
        budget=budget,
        stopped_early=stop,
        path=settings.latest_checkpoint if promoted else None,
        promoted=promoted,
        gated=not promoted,
        corpus_val_before=corpus_val_before,
        corpus_val_after=corpus_val_after,
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
