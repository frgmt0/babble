"""The single training path: best-validation checkpoint selection, patience
early stopping, and the +N-row trigger -- with no external corpus and no
`base.pt` anywhere in the loop. This reverses the two-stage (base + voice
pass) design: `babble train` is now the only command, and it pretrains
straight from random init on the corpus people actually gave it.
"""

from __future__ import annotations

import json

import pytest
import torch

from babble.cli import build_parser
from babble.config import Settings
from babble.consent import SCOPE_CORPUS, ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.logs import EventLog
from babble.trainer import AutoTrainTrigger, train, train_trigger


def _seed_human(settings, ids, texts, *, author_raw="alice-raw"):
    ConsentStore(settings.consent_path).grant(author_raw, SCOPE_CORPUS)
    store = CorpusStore(settings.corpus_path)
    author = ids.user(author_raw)
    for text in texts:
        store.append(
            CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=SOURCE_MENTION)
        )


def _seed_plenty_of_human_rows(settings, ids):
    # Enough rows/text that split_rows holds out a non-empty val set, so
    # eval_loss actually fires at every checkpoint interval.
    _seed_human(settings, ids, [f"row number {i} with a bit more text to chew on" for i in range(20)])


# --- no base.pt anywhere -----------------------------------------------


def test_train_needs_no_base_checkpoint_and_none_is_created(settings, ids):
    """The whole point of the reversal: random init, straight to the human
    corpus, and nothing external or frozen involved at any point."""
    _seed_plenty_of_human_rows(settings, ids)
    assert not (settings.checkpoint_dir / "base.pt").exists()

    result = train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert result.ran and result.stopped_because == "trained"
    assert settings.latest_checkpoint.exists()
    # Still no base.pt -- nothing in the path reads or writes one.
    assert not (settings.checkpoint_dir / "base.pt").exists()
    assert not hasattr(settings, "base_checkpoint")
    assert not hasattr(settings, "base_corpus_path")


def test_the_automatic_trigger_path_trains_from_the_corpus_alone(settings, ids, monkeypatch):
    """`AutoTrainTrigger` is what the bot calls after every fresh row -- this
    is the *only* automatic path, and it must fire `babble train` (which
    starts from random init on the corpus) without ever touching base.pt."""
    settings.train_trigger_rows = 2
    launches = []

    class FakePopen:
        def __init__(self, argv, **kw):
            launches.append(argv)
            self.pid = 4321

        def poll(self):
            return None  # still running

    monkeypatch.setattr("babble.trainer.subprocess.Popen", FakePopen)
    trigger = AutoTrainTrigger(settings)

    trigger.maybe_run()  # 0 rows -> not due -> no launch
    assert launches == []

    _seed_human(settings, ids, ["one one", "two two"])  # 2 >= 2 -> due
    trigger.maybe_run()

    assert len(launches) == 1
    argv = launches[0]
    assert argv[1:4] == ["-m", "babble", "train"]
    assert "--force" not in argv  # train() re-checks the trigger itself
    assert not (settings.checkpoint_dir / "base.pt").exists()

    trigger.maybe_run()  # a run is still "in flight" -> do not stack a second
    assert len(launches) == 1


# --- best-val checkpoint selection + early stopping -----------------------


def test_train_writes_the_best_val_checkpoint_not_the_last(settings, ids, monkeypatch):
    _seed_plenty_of_human_rows(settings, ids)

    # A synthetic val curve that bottoms out at the 2nd of 4 checkpoints
    # (checkpoint_every=2, steps=8 -> checkpoints at steps 2, 4, 6, 8).
    val_curve = iter([2.0, 1.0, 1.5, 1.8])
    monkeypatch.setattr("babble.trainer.eval_loss", lambda model, examples: next(val_curve))

    result = train(settings, force=True, steps=8, patience=10, seed=1, echo=False, ids=ids)

    assert result.ran
    assert result.val_loss == 1.0
    assert result.final_step == 4  # the step with the lowest val loss, not the last step (8)
    assert result.steps_run == 8  # the full budget ran; only the *write* picked the best
    payload = torch.load(settings.latest_checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] == 4


def test_train_stops_early_after_patience_non_improving_checkpoints(settings, ids, monkeypatch):
    _seed_plenty_of_human_rows(settings, ids)

    # Best is at the 2nd checkpoint (step 4); the run should stop 2 non-improving
    # checkpoints later (step 8), long before the step-20 budget.
    val_curve = iter([2.0, 1.0, 1.5, 1.8, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    monkeypatch.setattr("babble.trainer.eval_loss", lambda model, examples: next(val_curve))

    result = train(settings, force=True, steps=20, patience=2, seed=1, echo=False, ids=ids)

    assert result.ran
    assert result.stopped_early is True
    assert result.budget == 20
    assert result.steps_run == 8  # did not run the full 20-step budget
    assert result.final_step == 4
    assert result.val_loss == 1.0


def test_train_stop_log_names_the_winning_step_and_val_loss(settings, ids, monkeypatch):
    _seed_plenty_of_human_rows(settings, ids)

    val_curve = iter([2.0, 1.0, 1.5, 1.8])
    monkeypatch.setattr("babble.trainer.eval_loss", lambda model, examples: next(val_curve))

    log = EventLog(settings, ids, component="trainer")
    train(settings, force=True, steps=8, patience=10, seed=1, echo=False, ids=ids, log=log)
    log.close()

    events = [
        json.loads(line)
        for line in (settings.log_dir / "babble.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    (done,) = [e for e in events if e["event"] == "train.stop"]
    assert done["step"] == 4
    assert done["val_loss"] == 1.0
    assert done["steps_run"] == 8
    assert done["budget"] == 8


# --- the +N-row trigger --------------------------------------------------


def test_trigger_fires_only_after_the_threshold_of_new_rows(settings, ids):
    settings.train_trigger_rows = 5

    _seed_human(settings, ids, [f"row number {i}" for i in range(4)])  # 4 < 5
    assert train_trigger(settings).due is False

    _seed_human(settings, ids, ["row number 4"])  # now 5 >= 5
    assert train_trigger(settings).due is True


def test_train_respects_the_trigger_and_persists_last_count(settings, ids):
    settings.train_trigger_rows = 3
    _seed_human(settings, ids, ["a a a", "b b b"])  # 2 < 3

    # Not forced and not due -> a no-op that says why, and writes no latest.pt.
    skipped = train(settings, steps=2, echo=False, ids=ids)
    assert not skipped.ran and skipped.stopped_because == "not_due"
    assert not settings.latest_checkpoint.exists()

    _seed_human(settings, ids, ["c c c"])  # 3 >= 3 -> due
    assert train_trigger(settings).due is True

    ran = train(settings, steps=2, echo=False, ids=ids)  # fires without --force
    assert ran.ran

    # The last-trained count is persisted, so it does not re-fire on the next call.
    assert train_trigger(settings).due is False
    state = json.loads(settings.train_state_path.read_text(encoding="utf-8"))
    assert state["last_trained_rows"] == CorpusStore(settings.corpus_path).count()
    assert "steps_run" in state
    again = train(settings, steps=2, echo=False, ids=ids)
    assert not again.ran and again.stopped_because == "not_due"


def test_trigger_off_when_threshold_is_zero(settings, ids):
    settings.train_trigger_rows = 0
    _seed_human(settings, ids, [f"row {i}" for i in range(50)])
    assert train_trigger(settings).due is False


# --- CLI wiring ----------------------------------------------------------


def test_train_subcommands_are_registered():
    args = build_parser().parse_args(["train", "--force"])
    assert args.command == "train"
    assert args.force is True

    status_args = build_parser().parse_args(["train-status"])
    assert status_args.command == "train-status"


def test_prepare_base_and_voice_pass_are_gone_from_the_cli():
    """The retired two-stage commands must not linger as dead entry points a
    future trigger could pick up by accident."""
    for command in ("prepare-base", "base-pretrain", "voice-pass", "voice-status"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command])
