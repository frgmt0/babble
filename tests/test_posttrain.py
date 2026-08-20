"""Stage 2's post-train: the pretrained/post-trained checkpoint split, the
best-val checkpoint, the +N-pair trigger, consent, and the zero-pairs no-op."""

from __future__ import annotations

import json

import pytest
import torch

from babble.cli import build_parser
from babble.config import Settings
from babble.consent import ConsentStore
from babble.cpu_runtime import force_cpu_device
from babble.model import Babbler, config_from_settings
from babble.post_state import post_trigger
from babble.posttrain import AutoPostTrigger, post_train
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.trainer import _build_optimizer, save_checkpoint

# --- helpers ---------------------------------------------------------------


def _seed_pretrain(settings: Settings, *, step: int = 1) -> None:
    """A fixture pretrained checkpoint, standing in for `babble train`'s output."""
    device = force_cpu_device()
    model = Babbler(config_from_settings(settings)).to(device)
    optimizer = _build_optimizer(model, settings)
    save_checkpoint(settings, model, optimizer, step, 4.2)


def _seed_pairs(settings, ids, pairs, *, asker="asker-raw", helper="helper-raw"):
    ConsentStore(settings.consent_path).grant(asker)
    ConsentStore(settings.consent_path).grant(helper)
    store = InteractionStore(settings.interactions_path)
    asker_p, helper_p = ids.user(asker), ids.user(helper)
    for prompt, chosen in pairs:
        row = Interaction(
            id=make_row_id(CORRECTION, prompt, chosen, asker_p, helper_p),
            signal=CORRECTION,
            prompt=prompt,
            rejected="wrong answer",
            chosen=chosen,
            prompt_author=asker_p,
            signal_author=helper_p,
            weight=settings.correction_weight,
        )
        store.append(row)


def _model_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def _states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


PAIRS = [
    (f"question number {i}", f"answer number {i} with a little more text to it")
    for i in range(20)
]


# --- the pretrain / post-train split --------------------------------------


def test_post_train_fine_tunes_from_the_pretrained_checkpoint(settings, ids):
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)

    result = post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert result.ran and result.reason == "trained"
    assert result.pairs_trained == len(PAIRS)
    assert settings.latest_checkpoint.exists()
    assert settings.pretrained_checkpoint.exists()


def test_post_train_leaves_the_pretrained_checkpoint_unmodified(settings, ids):
    _seed_pretrain(settings)
    pretrain_bytes_before = settings.latest_checkpoint.read_bytes()
    _seed_pairs(settings, ids, PAIRS)

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    # pretrained.pt is the snapshot taken from the pretrain output -- unchanged
    # by the post-train that follows it, even though latest.pt now holds the
    # post-trained weights instead.
    assert settings.pretrained_checkpoint.read_bytes() == pretrain_bytes_before


def test_post_train_rerun_always_restarts_from_pretrained_not_previous_post_train(settings, ids):
    """Two reruns with nothing else changed must be deterministic and
    identical: if a rerun ever resumed from its own previous output in
    `latest.pt` instead of the frozen pretrained snapshot, the two runs would
    diverge (the second would be fine-tuning an already fine-tuned model)."""
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    first = _model_state(settings.latest_checkpoint)

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    second = _model_state(settings.latest_checkpoint)

    assert _states_equal(first, second)


def test_pretrained_snapshot_refreshes_when_a_new_pretrain_lands(settings, ids):
    """The snapshot is reused across post-trains (previous test) but must not
    be reused forever: once a fresh pretrain overwrites `latest.pt` -- a
    manual `babble train --force`, or the bot's own auto-retrain -- the next
    post-train must fine-tune *that* pretrain, not silently keep training on
    top of a stale one and overwrite `latest.pt` with the old lineage."""
    _seed_pretrain(settings, step=1)
    _seed_pairs(settings, ids, PAIRS)
    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    stale_snapshot_bytes = settings.pretrained_checkpoint.read_bytes()

    _seed_pretrain(settings, step=99)  # a fresh pretrain overwrites latest.pt
    fresh_pretrain_bytes = settings.latest_checkpoint.read_bytes()
    assert fresh_pretrain_bytes != stale_snapshot_bytes

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    # The snapshot was retaken from the new pretrain, verbatim, before any
    # post-train step ran against it.
    assert settings.pretrained_checkpoint.read_bytes() == fresh_pretrain_bytes


def test_pretrained_snapshot_is_reused_when_latest_is_still_its_own_output(settings, ids):
    """The other half of the same guarantee: a rerun must not mistake its own
    post-trained `latest.pt` for a fresh pretrain and re-snapshot from it --
    that would compound post-train output on post-train output."""
    _seed_pretrain(settings, step=1)
    _seed_pairs(settings, ids, PAIRS)
    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    snapshot_bytes = settings.pretrained_checkpoint.read_bytes()

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert settings.pretrained_checkpoint.read_bytes() == snapshot_bytes


def test_post_train_with_non_positive_steps_is_a_no_op(settings, ids):
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)
    latest_before = settings.latest_checkpoint.read_bytes()

    result = post_train(settings, force=True, steps=0, echo=False, ids=ids)

    assert not result.ran and result.reason == "no_steps"
    # Nothing trained: no snapshot taken, latest.pt untouched, and the
    # trigger not consumed -- a stray `--steps 0` must not burn it.
    assert not settings.pretrained_checkpoint.exists()
    assert settings.latest_checkpoint.read_bytes() == latest_before
    assert not settings.post_state_path.exists()


def test_post_train_with_no_pretrain_reports_no_pretrain(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    result = post_train(settings, force=True, steps=2, echo=False, ids=ids)
    assert not result.ran and result.reason == "no_pretrain"


def test_post_train_with_zero_pairs_reports_no_data_and_does_not_crash(settings, ids):
    _seed_pretrain(settings)
    result = post_train(settings, force=True, steps=2, echo=False, ids=ids)
    assert not result.ran and result.reason == "no_data"
    # No side effects from a no-op: no snapshot taken.
    assert not settings.pretrained_checkpoint.exists()


# --- best-val checkpoint selection + early stopping ------------------------


def test_post_train_writes_the_best_val_checkpoint_not_the_last(settings, ids, monkeypatch):
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)

    # A synthetic val curve that bottoms out at the 2nd of 4 checkpoints
    # (checkpoint_every=2, steps=8 -> checkpoints at steps 2, 4, 6, 8).
    val_curve = iter([2.0, 1.0, 1.5, 1.8])
    monkeypatch.setattr("babble.posttrain.eval_loss", lambda model, examples: next(val_curve))

    result = post_train(settings, force=True, steps=8, patience=10, seed=1, echo=False, ids=ids)

    assert result.ran
    assert result.val_loss == 1.0
    assert result.final_step == 4  # the step with the lowest val loss, not the last (8)
    assert result.checkpoints_written == 4  # the full budget ran; only the write picked the best
    payload = torch.load(settings.latest_checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] == 4


def test_post_train_stops_early_after_patience_non_improving_checkpoints(settings, ids, monkeypatch):
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)

    # Best is at the 2nd checkpoint (step 4); the run should stop 2 non-improving
    # checkpoints later (step 8), long before the step-20 budget.
    val_curve = iter([2.0, 1.0, 1.5, 1.8, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    monkeypatch.setattr("babble.posttrain.eval_loss", lambda model, examples: next(val_curve))

    result = post_train(settings, force=True, steps=20, patience=2, seed=1, echo=False, ids=ids)

    assert result.stopped_early is True
    assert result.checkpoints_written == 4  # did not run the full 20-step budget
    assert result.final_step == 4
    assert result.val_loss == 1.0


def test_post_done_log_names_the_winning_step_and_val_loss(settings, ids, monkeypatch):
    from babble.logs import EventLog

    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)

    val_curve = iter([2.0, 1.0, 1.5, 1.8])
    monkeypatch.setattr("babble.posttrain.eval_loss", lambda model, examples: next(val_curve))

    log = EventLog(settings, ids, component="post")
    post_train(settings, force=True, steps=8, patience=10, seed=1, echo=False, ids=ids, log=log)
    log.close()

    events = [
        json.loads(line)
        for line in (settings.log_dir / "babble.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    (started,) = [e for e in events if e["event"] == "post.start"]
    assert started["pairs"] == len(PAIRS)
    (done,) = [e for e in events if e["event"] == "post.done"]
    assert done["step"] == 4
    assert done["val_loss"] == 1.0


# --- the +N-pair trigger ----------------------------------------------------


def test_trigger_fires_only_after_the_threshold_of_new_pairs(settings, ids):
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 5

    _seed_pairs(settings, ids, PAIRS[:4])  # 4 < 5
    assert post_trigger(settings).due is False

    _seed_pairs(settings, ids, PAIRS[4:5])  # now 5 >= 5
    assert post_trigger(settings).due is True


def test_post_train_respects_the_trigger_and_persists_last_count(settings, ids):
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 3
    _seed_pairs(settings, ids, PAIRS[:2])  # 2 < 3

    # Not forced and not due -> a no-op that says why, and writes no post-train.
    skipped = post_train(settings, steps=2, echo=False, ids=ids)
    assert not skipped.ran and skipped.reason == "not_due"
    assert not settings.pretrained_checkpoint.exists()

    _seed_pairs(settings, ids, PAIRS[2:3])  # 3 >= 3 -> due
    assert post_trigger(settings).due is True

    ran = post_train(settings, steps=2, echo=False, ids=ids)  # fires without --force
    assert ran.ran

    # The last-trained count is persisted, so it does not re-fire on the next call.
    assert post_trigger(settings).due is False
    again = post_train(settings, steps=2, echo=False, ids=ids)
    assert not again.ran and again.reason == "not_due"


def test_trigger_off_when_threshold_is_zero(settings, ids):
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 0
    _seed_pairs(settings, ids, PAIRS)
    assert post_trigger(settings).due is False


def test_trigger_needs_a_pretrained_checkpoint(settings, ids):
    settings.post_trigger_pairs = 1
    _seed_pairs(settings, ids, PAIRS)
    status = post_trigger(settings)
    assert status.has_pretrained is False
    assert status.due is False


# --- consent ----------------------------------------------------------------


def test_post_train_only_trains_on_consented_pairs(settings, ids):
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS, asker="alice-raw", helper="bob-raw")

    # A pair whose corrector never consented must not be trained on.
    store = InteractionStore(settings.interactions_path)
    stranger = ids.user("stranger-raw")
    store.append(
        Interaction(
            id=make_row_id(CORRECTION, "q", "a", ids.user("alice-raw"), stranger),
            signal=CORRECTION,
            prompt="q",
            rejected="x",
            chosen="a",
            prompt_author=ids.user("alice-raw"),
            signal_author=stranger,
            weight=1.0,
        )
    )

    result = post_train(settings, force=True, steps=2, echo=False, ids=ids)
    assert result.pairs_trained == len(PAIRS)


# --- synthetic pairs (opt-in, separate from human corrections) -------------


def _seed_synthetic(settings, ids, pairs, *, author="synth-author-raw"):
    """A synthetic pair *and* the consented corpus row it claims to be built
    from -- `trainable_synthetic_pairs` re-checks the source row's consent at
    train time (see test_synthetic.py), so a pair with no matching consented
    row is correctly untrainable and would make this helper misleading."""
    from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
    from babble.synthetic import SyntheticPair, SyntheticPairStore, make_synthetic_id

    ConsentStore(settings.consent_path).grant(author)
    author_p = ids.user(author)
    corpus = CorpusStore(settings.corpus_path)
    store = SyntheticPairStore(settings.synthetic_pairs_path)
    for i, (prompt, response) in enumerate(pairs):
        row_text = f"source row {i} {response}"
        row_id = make_corpus_id(row_text, author_p)
        corpus.append(
            CorpusRow(id=row_id, text=row_text, author=author_p, source=SOURCE_MENTION)
        )
        store.append(
            SyntheticPair(
                id=make_synthetic_id(row_id, prompt, "generic"),
                prompt=prompt,
                response=response,
                source_row_id=row_id,
                method="generic",
            )
        )


def test_post_train_ignores_synthetic_pairs_by_default(settings, ids):
    """`include_synthetic` defaults to False: a synthetic pair sitting in
    `synthetic_pairs.jsonl` must never silently get trained on."""
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)
    _seed_synthetic(settings, ids, [("made up prompt", "made up response")])

    result = post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert result.ran
    assert result.pairs_trained == len(PAIRS)
    assert result.synthetic_pairs_trained == 0


def test_post_train_with_include_synthetic_trains_on_both(settings, ids):
    _seed_pretrain(settings)
    _seed_pairs(settings, ids, PAIRS)
    synthetic = [(f"synthetic prompt {i}", f"synthetic response {i}") for i in range(5)]
    _seed_synthetic(settings, ids, synthetic)

    result = post_train(
        settings, force=True, steps=4, seed=1, echo=False, ids=ids, include_synthetic=True
    )

    assert result.ran
    assert result.pairs_trained == len(PAIRS)
    assert result.synthetic_pairs_trained == len(synthetic)


def test_post_train_with_only_synthetic_pairs_still_trains(settings, ids):
    """No human corrections at all, but synthetic pairs exist and
    `include_synthetic` is set -- this must run, not report `no_data`."""
    _seed_pretrain(settings)
    _seed_synthetic(settings, ids, [("synthetic prompt", "synthetic response")])

    result = post_train(
        settings, force=True, steps=2, echo=False, ids=ids, include_synthetic=True
    )

    assert result.ran
    assert result.pairs_trained == 0
    assert result.synthetic_pairs_trained == 1


def test_post_train_without_include_synthetic_and_zero_human_pairs_is_no_data(settings, ids):
    """Synthetic pairs exist, but `include_synthetic` was not passed --
    behaviour must be identical to there being no data at all."""
    _seed_pretrain(settings)
    _seed_synthetic(settings, ids, [("synthetic prompt", "synthetic response")])

    result = post_train(settings, force=True, steps=2, echo=False, ids=ids)

    assert not result.ran and result.reason == "no_data"


def test_post_train_synthetic_inclusion_does_not_affect_the_pair_trigger(settings, ids):
    """The +N-pair trigger counts human corrections only -- generating (or
    including) synthetic pairs must never make a post-train due on its own."""
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 100
    _seed_synthetic(settings, ids, [(f"p{i}", f"r{i}") for i in range(200)])

    assert post_trigger(settings).due is False


# --- CLI wiring --------------------------------------------------------------


@pytest.mark.parametrize("command", ["post-train", "post-status"])
def test_new_subcommands_are_registered(command):
    args = build_parser().parse_args([command])
    assert args.command == command


# --- the automatic trigger --------------------------------------------------


class _FakePopen:
    def __init__(self, argv, **kw):
        self.argv = argv
        self.pid = 4321

    def poll(self):
        return None  # still running


def test_the_automatic_post_trigger_launches_post_train(settings, ids, monkeypatch):
    """`AutoPostTrigger` is what the bot calls after every fresh correction
    pair -- mirrors `AutoTrainTrigger`'s own wiring test."""
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 2
    launches = []
    monkeypatch.setattr(
        "babble.posttrain.subprocess.Popen", lambda argv, **kw: launches.append(argv) or _FakePopen(argv)
    )
    trigger = AutoPostTrigger(settings)

    trigger.maybe_run()  # 0 pairs -> not due -> no launch
    assert launches == []

    _seed_pairs(settings, ids, PAIRS[:2])  # 2 >= 2 -> due
    trigger.maybe_run()

    assert len(launches) == 1
    argv = launches[0]
    assert argv[1:4] == ["-m", "babble", "post-train"]
    assert "--force" not in argv  # post_train() re-checks the trigger itself

    trigger.maybe_run()  # a run is still "in flight" -> do not stack a second
    assert len(launches) == 1


def test_the_automatic_post_trigger_defers_while_a_pretrain_is_in_flight(settings, ids, monkeypatch):
    """A post-train must never start against a pretrain that is about to be
    replaced -- see the pretrained/post-trained split above. `AutoPostTrigger`
    takes the bot's `AutoTrainTrigger` for exactly this: while it reports a
    pretrain still running, the post-train trigger stays quiet even though its
    own threshold is due."""
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 1
    _seed_pairs(settings, ids, PAIRS[:1])
    launches = []
    monkeypatch.setattr(
        "babble.posttrain.subprocess.Popen", lambda argv, **kw: launches.append(argv) or _FakePopen(argv)
    )

    class _PretrainStillRunning:
        def is_running(self) -> bool:
            return True

    trigger = AutoPostTrigger(settings, train_trigger=_PretrainStillRunning())
    trigger.maybe_run()

    assert launches == []


def test_the_automatic_post_trigger_off_when_threshold_is_zero(settings, ids, monkeypatch):
    _seed_pretrain(settings)
    settings.post_trigger_pairs = 0
    _seed_pairs(settings, ids, PAIRS)
    launches = []
    monkeypatch.setattr(
        "babble.posttrain.subprocess.Popen", lambda argv, **kw: launches.append(argv) or _FakePopen(argv)
    )

    AutoPostTrigger(settings).maybe_run()

    assert launches == []
