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
5. **A live-checkpoint gate** (`post_live_gate_margin`): the same held-out
   split is also scored on whatever `latest.pt` is currently served. A
   candidate that beats its own starting checkpoint but is worse than live
   by more than the margin does not ship. `--force-promote` overrides.

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

`include_pair_augmentation=True` (`--augment-pairs` on the CLI, or
`Settings.post_augment_pairs`) additionally trains on the LLM-paraphrased
variants in `data/augmented_pairs.jsonl` -- see `pairaugment.py`. Same
discipline: off by default, never counted by the trigger, re-checked for
consent/blocklist/train-side membership at train time
(`trainable_augmented_pairs`). The pair validation split
(`pairsplit.pair_split`) is computed once, from the real pairs only, BEFORE
any synthetic or augmented pairs are mixed in -- so val always stays 100%
real, held-out pairs no matter which of the two optional pools are enabled.
"""

from __future__ import annotations

import copy
import json
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
from .pairaugment import AugmentedPair, trainable_augmented_pairs
from .pairsplit import pair_split
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
from .subword import BPETokenizer
from .subword import build_continuation_example as _bpe_continuation_example
from .subword import stack_examples as _bpe_stack_examples
from .subword import text_examples as _bpe_text_examples
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
from .util import atomic_write_text, utcnow_iso, utcnow_stamp

__all__ = [
    "PostTrainResult",
    "PostTrigger",
    "AutoPostTrigger",
    "LiveScore",
    "decide_promotion",
    "pair_count",
    "post_train",
    "post_train_from_checkpoint",
    "post_trigger",
    "read_post_state",
    "score_live_checkpoint",
    "trainable_pairs",
    "write_post_state",
]

@dataclass
class PostTrainResult:
    ran: bool
    reason: str
    pairs_trained: int = 0
    synthetic_pairs_trained: int = 0
    augmented_pairs_trained: int = 0
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
    live_corpus_val: float | None = None
    live_compare_reason: str | None = None
    gate_reason: str | None = None
    force_promote: bool = False
    candidate_layout: str | None = None
    live_layout: str | None = None
    serve_layout_mismatch: bool = False


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


#: Timestamped predecessor copies, alongside the stable `previous.pt` pointer.
ARCHIVE_DIR = "archive"


def _archive_outgoing_checkpoint(settings: Settings) -> None:
    """Snapshot `latest.pt` before a promotion overwrites it.

    Nothing here touches `pretrained.pt` or its staleness discipline -- this
    is a *serving-layer* archive, one level up. `babble ab run` compares the
    freshly promoted checkpoint against exactly this snapshot by default, and
    `babble ab rollback` restores it if the humans say the promotion made
    things worse. A no-op the first time a post-train ever promotes anything
    -- there is no outgoing checkpoint yet.

    Both copies are written atomically (temp file in the same `.partial`
    scratch dir `save_checkpoint` uses, then `os.replace`), so a kill mid-copy
    can never leave a half-written archive next to the good ones. Timestamped
    copies are pruned to `keep_checkpoints`, the same cadence
    `prune_checkpoints` already gives `ckpt-*.pt` -- these are full ~400MB
    checkpoints too, and unbounded growth here would be its own incident.
    """
    current = settings.latest_checkpoint
    if not current.exists():
        return
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scratch = settings.checkpoint_dir / SCRATCH_DIR
    scratch.mkdir(exist_ok=True)
    archive_dir = settings.checkpoint_dir / ARCHIVE_DIR
    archive_dir.mkdir(exist_ok=True)

    step = None
    try:
        payload = torch.load(current, map_location="cpu", weights_only=True)
        step = payload.get("step")
    except Exception:
        pass  # archiving must never block a promotion over a bad read of the OLD file

    timestamped = archive_dir / f"previous-{utcnow_stamp()}.pt"
    tmp_timestamped = scratch / f"{timestamped.name}.tmp"
    shutil.copyfile(current, tmp_timestamped)
    os.replace(tmp_timestamped, timestamped)

    tmp_previous = scratch / "previous.pt.tmp"
    shutil.copyfile(current, tmp_previous)
    os.replace(tmp_previous, settings.previous_checkpoint)

    keep = max(1, settings.keep_checkpoints)
    archived = sorted(archive_dir.glob("previous-*.pt"))
    for stale in archived[:-keep] if len(archived) > keep else []:
        stale.unlink(missing_ok=True)

    atomic_write_text(
        settings.previous_meta_path,
        json.dumps(
            {
                "archived_at": utcnow_iso(),
                "step": step,
                "hash": file_hash(current),
                "timestamped_path": str(timestamped),
            },
            indent=2,
        ),
    )


def _load_pretrained(path: Path) -> Babbler:
    """The frozen pretrained weights, as a fresh model. Read-only: post-train
    never writes back to `pretrained.pt`."""
    device = force_cpu_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    return maybe_compile(model)


@dataclass(frozen=True)
class LiveScore:
    """How currently served `latest.pt` scored on the candidate's val split."""

    corpus_val: float | None = None
    skip_reason: str | None = None
    layout: str | None = None


def checkpoint_layout(payload: object) -> str | None:
    """Serve/train layout stamped on a checkpoint, or None if the file predates it."""
    if not isinstance(payload, dict):
        return None
    layout = payload.get("layout")
    return layout if isinstance(layout, str) and layout else None


def score_live_checkpoint(
    settings: Settings,
    examples: list[Example],
    *,
    candidate_vocab: int,
    eval_fn,
) -> LiveScore:
    """Score currently served `latest.pt` on the candidate's held-out examples.

    Always reports `layout` when the file is readable. `skip_reason` is set
    when the live checkpoint cannot be compared (missing, unreadable, vocab
    mismatch). Layout mismatch is the caller's decision: this still scores
    when vocab matches so both numbers can be logged.
    """
    path = settings.latest_checkpoint
    if not path.is_file():
        return LiveScore(skip_reason="live_missing")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return LiveScore(skip_reason="live_unreadable")
    cfg = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(cfg, dict):
        return LiveScore(skip_reason="live_unreadable")
    layout = checkpoint_layout(payload)
    live_vocab = cfg.get("vocab_size")
    if live_vocab != candidate_vocab:
        return LiveScore(skip_reason="vocab_mismatch", layout=layout)
    try:
        live_model = _load_pretrained(path)
        loss = eval_fn(live_model, examples)
        del live_model
    except Exception:
        return LiveScore(skip_reason="eval_failed", layout=layout)
    if loss is None:
        return LiveScore(skip_reason="live_unscored", layout=layout)
    return LiveScore(corpus_val=float(loss), layout=layout)


def decide_promotion(
    *,
    corpus_val_before: float | None,
    corpus_val_after: float | None,
    live_corpus_val: float | None,
    live_compare_reason: str | None,
    lineage_margin: float,
    live_margin: float,
    force_promote: bool,
    candidate_layout: str | None = None,
    live_layout: str | None = None,
) -> tuple[bool, str]:
    """Lineage gate plus live-checkpoint gate. A candidate must clear both.

    Live corpus_val is only decisive when candidate and live share a layout.
    corpus_val is continuation-format, so a pair-SFT checkpoint's number is
    not evidence it is worse than a continuation checkpoint. Different or
    unknown layouts skip the live numeric refuse (caller still logs both
    numbers). `force_promote` ships anyway after both verdicts are recorded.
    """
    lineage_ok = True
    if (
        corpus_val_after is not None
        and corpus_val_before is not None
        and lineage_margin >= 0
    ):
        lineage_ok = corpus_val_after <= corpus_val_before + lineage_margin
        lineage_reason = "cleared_lineage" if lineage_ok else "worse_than_start"
    elif lineage_margin < 0:
        lineage_reason = "lineage_gate_disabled"
    else:
        lineage_reason = "no_lineage_score"

    live_ok = True
    if live_compare_reason in (
        "live_missing",
        "live_unreadable",
        "vocab_mismatch",
        "eval_failed",
        "live_unscored",
        "live_gate_disabled",
    ):
        live_reason = live_compare_reason
    elif candidate_layout and live_layout and candidate_layout != live_layout:
        live_reason = "layout_mismatch"
    elif live_layout is None or candidate_layout is None:
        live_reason = "layout_unknown"
    elif live_margin < 0:
        live_reason = "live_gate_disabled"
    elif live_corpus_val is not None and corpus_val_after is not None:
        live_ok = corpus_val_after <= live_corpus_val + live_margin
        live_reason = "cleared_live" if live_ok else "worse_than_live"
    else:
        live_reason = live_compare_reason or "live_unscored"

    would_promote = lineage_ok and live_ok
    if force_promote and not would_promote:
        return True, "force_promote"
    if would_promote:
        if live_reason in (
            "cleared_live",
            "layout_mismatch",
            "layout_unknown",
            "vocab_mismatch",
            "live_missing",
            "live_unreadable",
            "eval_failed",
            "live_unscored",
            "live_gate_disabled",
        ):
            return True, live_reason
        return True, lineage_reason
    return False, live_reason if not live_ok else lineage_reason


def _live_gate(
    settings: Settings,
    examples: list[Example],
    *,
    candidate_vocab: int,
    candidate_layout: str,
    eval_fn,
    corpus_val_before: float | None,
    corpus_val_after: float | None,
    force_promote: bool,
) -> tuple[bool, float | None, str | None, str, str | None, bool]:
    live_corpus_val: float | None = None
    live_compare_reason: str | None = None
    live_layout: str | None = None
    if examples:
        scored = score_live_checkpoint(
            settings, examples, candidate_vocab=candidate_vocab, eval_fn=eval_fn
        )
        live_corpus_val = scored.corpus_val
        live_layout = scored.layout
        live_compare_reason = scored.skip_reason
        if settings.post_live_gate_margin < 0 and live_compare_reason is None:
            live_compare_reason = "live_gate_disabled"
    promoted, gate_reason = decide_promotion(
        corpus_val_before=corpus_val_before,
        corpus_val_after=corpus_val_after,
        live_corpus_val=live_corpus_val,
        live_compare_reason=live_compare_reason,
        lineage_margin=settings.post_gate_margin,
        live_margin=settings.post_live_gate_margin,
        force_promote=force_promote,
        candidate_layout=candidate_layout,
        live_layout=live_layout,
    )
    if gate_reason in ("layout_mismatch", "layout_unknown") and live_compare_reason is None:
        live_compare_reason = gate_reason
    serve_mismatch = candidate_layout != settings.serve_layout
    return promoted, live_corpus_val, live_compare_reason, gate_reason, live_layout, serve_mismatch


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


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


def _pair_examples(
    pairs: list[Interaction], block_size: int, builder=build_continuation_example
) -> list[Example]:
    return [builder(p.prompt, p.chosen, block_size) for p in pairs]


def _synthetic_examples(
    pairs: list[SyntheticPair], block_size: int, builder=build_continuation_example
) -> list[Example]:
    return [builder(p.prompt, p.response, block_size) for p in pairs]


def _augmented_examples(
    pairs: list[AugmentedPair], block_size: int, builder=build_continuation_example
) -> list[Example]:
    return [builder(p.prompt, p.chosen, block_size) for p in pairs]


def post_train(
    settings: Settings,
    *,
    force: bool = False,
    force_promote: bool = False,
    steps: int | None = None,
    patience: int | None = None,
    seed: int = 1,
    echo: bool = True,
    log: EventLog | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
    include_synthetic: bool = False,
    include_pair_augmentation: bool = False,
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

    `include_pair_augmentation=True` (`--augment-pairs` on the CLI, or
    `Settings.post_augment_pairs`) additionally trains on the LLM-paraphrased
    variants in `data/augmented_pairs.jsonl` (`pairaugment.py`). Same
    discipline as `include_synthetic`, and the two are independent: either,
    neither, or both may be on.
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
    augmented_pairs = (
        trainable_augmented_pairs(settings, ids, blocklist) if include_pair_augmentation else []
    )
    if not pairs and not synthetic_pairs and not augmented_pairs:
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
    total_pairs = len(pairs) + len(synthetic_pairs) + len(augmented_pairs)
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

    # Val split happens on the REAL pairs only, before anything synthetic or
    # augmented joins the pool -- so the held-out set stays 100% real
    # correction pairs no matter which optional pools are enabled, the same
    # discipline `trainer.py` gives corpus rows vs synthetic corpus rows.
    builder = _example_builder(settings)
    real_train_pairs, real_val_pairs = pair_split(pairs)
    train_examples = _pair_examples(real_train_pairs, model.config.block_size, builder)
    val_examples = _pair_examples(real_val_pairs, model.config.block_size, builder)
    if synthetic_pairs:
        train_examples += _synthetic_examples(synthetic_pairs, model.config.block_size, builder)
    if augmented_pairs:
        train_examples += _augmented_examples(augmented_pairs, model.config.block_size, builder)
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
        augmented_pairs=len(augmented_pairs),
        train_examples=len(train_examples),
        val_examples=len(val_examples),
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
            train_examples=len(train_examples),
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
    if corpus_val_examples and settings.post_gate_margin >= 0:
        corpus_val_after = (
            candidate_corpus_val
            if candidate_corpus_val is not None
            else eval_loss(model, corpus_val_examples)
        )
        pretrain_model = _load_pretrained(pretrained_path)
        corpus_val_before = eval_loss(pretrain_model, corpus_val_examples)
        del pretrain_model
    elif corpus_val_examples:
        corpus_val_after = (
            candidate_corpus_val
            if candidate_corpus_val is not None
            else eval_loss(model, corpus_val_examples)
        )
    promoted, live_corpus_val, live_compare_reason, gate_reason, live_layout, serve_mismatch = _live_gate(
        settings,
        corpus_val_examples,
        candidate_vocab=uncompiled(model).config.vocab_size,
        candidate_layout=settings.post_layout,
        eval_fn=eval_loss,
        corpus_val_before=corpus_val_before,
        corpus_val_after=corpus_val_after,
        force_promote=force_promote,
    )
    if serve_mismatch:
        log.event(
            "post.serve_layout_mismatch",
            candidate_layout=settings.post_layout,
            serve_layout=settings.serve_layout,
            promoted=promoted,
            reason="checkpoint_layout_disagrees_with_BABBLE_SERVE_LAYOUT",
        )
    if promoted:
        _archive_outgoing_checkpoint(settings)
        save_checkpoint(
            settings, model, optimizer, final_step, final_loss, layout=settings.post_layout
        )

    current = pair_count(settings)
    write_post_state(
        settings, pairs=current, step=final_step, latest_hash=file_hash(settings.latest_checkpoint)
    )
    log.event(
        "post.done",
        step=final_step,
        loss=round(final_loss, 6) if final_loss is not None else None,
        val_loss=round(final_val, 6) if final_val is not None else None,
        corpus_val_before=_round_or_none(corpus_val_before),
        corpus_val_after=_round_or_none(corpus_val_after),
        live_corpus_val=_round_or_none(live_corpus_val),
        live_compare_reason=live_compare_reason,
        candidate_layout=settings.post_layout,
        live_layout=live_layout,
        serve_layout=settings.serve_layout,
        serve_layout_mismatch=True if serve_mismatch else None,
        post_gate_margin=settings.post_gate_margin,
        post_live_gate_margin=settings.post_live_gate_margin,
        gate_reason=gate_reason,
        force_promote=force_promote,
        promoted=promoted,
        pairs_trained=len(pairs),
        synthetic_pairs_trained=len(synthetic_pairs),
        augmented_pairs_trained=len(augmented_pairs),
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
        augmented_pairs_trained=len(augmented_pairs),
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
        live_corpus_val=live_corpus_val,
        live_compare_reason=live_compare_reason,
        gate_reason=gate_reason,
        force_promote=force_promote,
        candidate_layout=settings.post_layout,
        live_layout=live_layout,
        serve_layout_mismatch=serve_mismatch,
    )


#: Output of `post_train_from_checkpoint` is pair-SFT (prompt/response), even
#: when corpus_val is still scored in continuation format. Stamped on the
#: promoted checkpoint so a later continuation post-train cannot "beat" it
#: on an incomparable corpus_val.
FROM_CKPT_LAYOUT = "pair"


def post_train_from_checkpoint(
    settings: Settings,
    checkpoint_path: Path,
    tokenizer_path: Path,
    *,
    force: bool = False,
    force_promote: bool = False,
    steps: int | None = None,
    patience: int | None = None,
    seed: int = 1,
    echo: bool = True,
    log: EventLog | None = None,
    ids: Pseudonymiser | None = None,
    blocklist: Blocklist | None = None,
) -> PostTrainResult:
    """Stage 2 against an EXTERNALLY supplied pretrain checkpoint -- e.g. one
    `pretrain_hf.py` produced on a GPU we don't have (see
    `docs/reports/HF_PRETRAIN_PIPELINE.md`), rather than the checkpoint `babble train`
    wrote locally.

    Same guardrails as `post_train()`: post-train's own (lower) learning
    rate, corpus rehearsal, a pair floor, best-val checkpoint selection, and
    a promotion gate that refuses to touch `latest.pt` if the candidate is
    worse than the checkpoint it started from on held-out corpus text. It
    exists as a separate function, not a branch inside `post_train()`,
    because `post_train()` hardcodes `babble.tokenizer`'s raw-byte layout
    (`build_example`/`build_continuation_example`/`_stack_examples`'s fixed
    `PAD_ID=256`) -- fine for a checkpoint `babble train` produced, actively
    wrong for one pretrained with a different tokenizer, because a learned
    BPE vocab's own merge ids can collide with byte-tokenizer's hardcoded
    specials. `tokenizer_path` must be the `tokenizer.json` that shipped
    alongside `checkpoint_path`; passing the wrong one silently reinterprets
    every id as a different byte/merge without ever raising.

    Unlike `post_train()`, this never checks or advances the +N-pair trigger
    (`post_state.json`) -- it is an on-demand run against a checkpoint the
    normal trigger has no way to know about, always acts as if `--force` was
    passed, and never fires the `has_pretrained`/pair-floor trigger gate.
    A promoted run does still overwrite `checkpoints/latest.pt`, so a
    subsequent ordinary `babble post-train` snapshots from it as usual.
    """
    settings.ensure_dirs()
    log = log or NullLog()
    tok = BPETokenizer.from_json(tokenizer_path)

    pairs = trainable_pairs(settings, ids, blocklist)
    if not pairs:
        log.event("post_from_ckpt.skipped", reason="no_data")
        return PostTrainResult(False, "no_data", current_pairs=pair_count(settings))

    total_pairs = len(pairs)
    if not force and total_pairs < settings.post_min_pairs:
        log.event(
            "post_from_ckpt.skipped", reason="too_few_pairs",
            pairs=total_pairs, threshold=settings.post_min_pairs,
        )
        return PostTrainResult(False, "too_few_pairs", current_pairs=pair_count(settings))

    budget = steps if steps is not None else settings.post_steps
    if budget <= 0:
        log.event("post_from_ckpt.skipped", reason="no_steps", steps=budget)
        return PostTrainResult(False, "no_steps", current_pairs=pair_count(settings))

    be_polite(settings, log)
    sweep_scratch(settings)

    device = force_cpu_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = ModelConfig.from_dict(payload["config"])
    if model_cfg.vocab_size != tok.vocab_size:
        raise ValueError(
            f"checkpoint vocab_size {model_cfg.vocab_size} does not match tokenizer "
            f"vocab_size {tok.vocab_size} ({tokenizer_path}) -- wrong tokenizer.json "
            f"for this checkpoint"
        )
    model = Babbler(model_cfg).to(device)
    model.load_state_dict(payload["model"])
    model = maybe_compile(model)

    post_lr = settings.post_learning_rate if settings.post_learning_rate > 0 else None
    optimizer = _build_optimizer(model, settings, lr=post_lr)

    block_size = model.config.block_size
    real_train_pairs, real_val_pairs = pair_split(pairs)
    train_examples = [_bpe_continuation_example(tok, p.prompt, p.chosen, block_size) for p in real_train_pairs]
    val_examples = [_bpe_continuation_example(tok, p.prompt, p.chosen, block_size) for p in real_val_pairs]

    run_patience = patience if patience is not None else settings.post_patience
    rng = random.Random(seed)
    torch.manual_seed(seed)

    rehearsal = min(1.0, max(0.0, settings.post_rehearsal))
    corpus_split = split_rows(corpus_rows(settings, ids, blocklist), settings)
    rehearsal_examples = (
        [ex for row in corpus_split.train for ex in _bpe_text_examples(tok, row.text, block_size)]
        if rehearsal > 0
        else []
    )
    corpus_val_examples = [
        ex for row in corpus_split.val for ex in _bpe_text_examples(tok, row.text, block_size)
    ]

    log.event(
        "post_from_ckpt.start",
        pairs=len(pairs),
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        rehearsal=rehearsal if rehearsal_examples else 0.0,
        rehearsal_examples=len(rehearsal_examples),
        lr=post_lr if post_lr is not None else settings.learning_rate,
        block_size=block_size,
        steps=budget,
        patience=run_patience,
        from_checkpoint=str(checkpoint_path),
        tokenizer=str(tokenizer_path),
    )

    def eval_examples(examples: list[Example]) -> float | None:
        if not examples:
            return None
        was_training = model.training
        model.eval()
        try:
            tokens, mask, weights = _bpe_stack_examples(examples, tok.specials.pad)
            with torch.inference_mode():
                return float(sequence_loss(model, tokens, mask, weights))
        finally:
            model.train(was_training)

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
        pair_val = eval_examples(val_examples)
        corpus_val = eval_examples(corpus_val_examples)
        metric = corpus_val if corpus_val is not None else pair_val
        checkpoints += 1
        append_curve(
            settings, step, mean, "",
            stored_rows=pair_count(settings), train_rows=len(train_examples),
            val_rows=len(val_examples), val_loss=pair_val,
        )
        log.event(
            "post_from_ckpt.checkpoint", step=step, loss=mean, val_loss=pair_val,
            corpus_val=round(corpus_val, 6) if corpus_val is not None else None,
        )
        if echo:
            val_s = f"{pair_val:.4f}" if pair_val is not None else "   n/a"
            cval_s = f"{corpus_val:.4f}" if corpus_val is not None else "   n/a"
            print(
                f"[post-ckpt] step {step:7d} | loss {mean:8.4f} | pair val {val_s} | corpus val {cval_s}",
                flush=True,
            )
        if metric is not None:
            if best is None or metric < best["metric"]:
                best = {
                    "step": step, "loss": mean, "val": pair_val, "corpus_val": corpus_val,
                    "metric": metric,
                    "model": copy.deepcopy(model_state_dict(model)),
                    "optim": copy.deepcopy(optimizer.state_dict()),
                }
                stalls = 0
            else:
                stalls += 1
                if run_patience and stalls >= run_patience:
                    stop = True
        window = []

    def make_mixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen: list[Example] = []
        for _ in range(settings.batch_size):
            if rehearsal_examples and rng.random() < rehearsal:
                chosen.append(rng.choice(rehearsal_examples))
            else:
                chosen.append(rng.choice(train_examples))
        return _bpe_stack_examples(chosen, tok.specials.pad)

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

    corpus_val_before: float | None = None
    corpus_val_after: float | None = None
    if corpus_val_examples and settings.post_gate_margin >= 0:
        corpus_val_after = (
            candidate_corpus_val if candidate_corpus_val is not None else eval_examples(corpus_val_examples)
        )
        pretrain_model = Babbler(model_cfg).to(device)
        pretrain_model.load_state_dict(payload["model"])
        corpus_val_before = eval_examples_with_model(pretrain_model, corpus_val_examples, tok.specials.pad)
        del pretrain_model
    elif corpus_val_examples:
        corpus_val_after = (
            candidate_corpus_val if candidate_corpus_val is not None else eval_examples(corpus_val_examples)
        )
    pad_id = tok.specials.pad
    candidate_layout = FROM_CKPT_LAYOUT
    promoted, live_corpus_val, live_compare_reason, gate_reason, live_layout, serve_mismatch = _live_gate(
        settings,
        corpus_val_examples,
        candidate_vocab=model_cfg.vocab_size,
        candidate_layout=candidate_layout,
        eval_fn=lambda m, ex: eval_examples_with_model(m, ex, pad_id),
        corpus_val_before=corpus_val_before,
        corpus_val_after=corpus_val_after,
        force_promote=force_promote,
    )
    if serve_mismatch:
        log.event(
            "post_from_ckpt.serve_layout_mismatch",
            candidate_layout=candidate_layout,
            serve_layout=settings.serve_layout,
            promoted=promoted,
            reason="checkpoint_layout_disagrees_with_BABBLE_SERVE_LAYOUT",
        )

    # Persist the best candidate for inspection even when the gate leaves
    # latest.pt untouched. This file is never served.
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scratch = settings.checkpoint_dir / SCRATCH_DIR
    scratch.mkdir(exist_ok=True)
    candidate_path = settings.checkpoint_dir / "post_candidate.pt"
    tmp = scratch / "post_candidate.pt.tmp"
    torch.save(
        {
            "step": final_step,
            "loss": final_loss,
            "config": uncompiled(model).config.to_dict(),
            "model": model_state_dict(model),
            "optim": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "layout": candidate_layout,
        },
        tmp,
    )
    os.replace(tmp, candidate_path)

    if promoted:
        _archive_outgoing_checkpoint(settings)
        save_checkpoint(
            settings, model, optimizer, final_step, final_loss, layout=candidate_layout
        )

    log.event(
        "post_from_ckpt.done",
        step=final_step,
        loss=round(final_loss, 6) if final_loss is not None else None,
        val_loss=round(final_val, 6) if final_val is not None else None,
        corpus_val_before=_round_or_none(corpus_val_before),
        corpus_val_after=_round_or_none(corpus_val_after),
        live_corpus_val=_round_or_none(live_corpus_val),
        live_compare_reason=live_compare_reason,
        candidate_layout=candidate_layout,
        live_layout=live_layout,
        serve_layout=settings.serve_layout,
        serve_layout_mismatch=True if serve_mismatch else None,
        post_gate_margin=settings.post_gate_margin,
        post_live_gate_margin=settings.post_live_gate_margin,
        gate_reason=gate_reason,
        force_promote=force_promote,
        promoted=promoted,
        pairs_trained=len(pairs),
        checkpoints=checkpoints,
        stopped_early=stop,
        budget=budget,
    )

    return PostTrainResult(
        True,
        "trained" if promoted else "gated",
        pairs_trained=len(pairs),
        current_pairs=pair_count(settings),
        last_trained_pairs=pair_count(settings),
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
        live_corpus_val=live_corpus_val,
        live_compare_reason=live_compare_reason,
        gate_reason=gate_reason,
        force_promote=force_promote,
        candidate_layout=candidate_layout,
        live_layout=live_layout,
        serve_layout_mismatch=serve_mismatch,
    )


def eval_examples_with_model(model: Babbler, examples: list[Example], pad_id: int) -> float | None:
    """`eval_examples` for a model that is not the one `post_train_from_checkpoint`
    is actively training -- used once, to score the frozen pretrain snapshot
    for the promotion gate, without needing a second closure over `model`."""
    if not examples:
        return None
    was_training = model.training
    model.eval()
    try:
        tokens, mask, weights = _bpe_stack_examples(examples, pad_id)
        with torch.inference_mode():
            return float(sequence_loss(model, tokens, mask, weights))
    finally:
        model.train(was_training)


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
