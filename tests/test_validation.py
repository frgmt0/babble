"""The held-out validation loop: deterministic split, small-corpus guard,
non-mutating eval pass, and the overfit signal surfacing in the log and feed.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from babble.blocklist import Blocklist
from babble.config import Settings
from babble.corpus import SOURCE_AMBIENT, CorpusRow, CorpusStore, make_corpus_id
from babble.discord_feed import TrainingFeed
from babble.fakedata import seed_fake_data
from babble.identity import Pseudonymiser
from babble.logs import EventLog
from babble.model import Babbler, config_from_settings, sequence_loss
from babble.tokenizer import build_example
from babble.trainer import (
    _checkpoint,
    corpus_rows,
    eval_loss,
    overfit_signal,
    split_rows,
    to_examples,
    train,
)


def make_row(n: int) -> CorpusRow:
    """A distinct corpus row -- only its id needs to vary for split tests."""
    return CorpusRow(
        id=f"row-{n:05d}",
        text=f"row {n} content",
        author="u_helper",
        source=SOURCE_AMBIENT,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def many_rows() -> list[CorpusRow]:
    return [make_row(i) for i in range(200)]


@pytest.fixture
def seeded(settings):
    seed_fake_data(settings)
    return settings


@pytest.fixture
def bare_settings(tmp_path):
    """A `Settings` with no model or trainer needs -- just for exercising
    `split_rows`, which only reads `val_min_rows` / `val_fraction`."""
    s = Settings.for_root(tmp_path)
    s.val_min_rows = 20
    s.val_fraction = 0.2
    return s


# --- deterministic split --------------------------------------------------


def test_the_same_row_lands_on_the_same_side_every_time(many_rows, bare_settings):
    first = split_rows(many_rows, bare_settings)
    second = split_rows(many_rows, bare_settings)

    assert {r.id for r in first.val} == {r.id for r in second.val}
    assert {r.id for r in first.train} == {r.id for r in second.train}


def test_the_split_does_not_depend_on_row_order(many_rows, bare_settings):
    ordered = split_rows(many_rows, bare_settings)

    shuffled = many_rows[:]
    random.Random(7).shuffle(shuffled)
    reshuffled = split_rows(shuffled, bare_settings)

    assert {r.id for r in ordered.val} == {r.id for r in reshuffled.val}


def test_growing_the_corpus_barely_moves_existing_rows_across_the_split(many_rows, bare_settings):
    """Appending rows must not reshuffle the split.

    The holdout is the lowest-hashed `round(fraction * n)` rows, so a handful of
    rows near the boundary can change sides when the corpus grows and a new row
    hashes below them. That churn is bounded and has nothing to do with where a
    row sits in the file -- which is the property that actually matters, and the
    reason this hashes ids rather than shuffling.
    """
    before = split_rows(many_rows, bare_settings)
    before_val_ids = {r.id for r in before.val}

    grown = many_rows + [make_row(1000 + i) for i in range(20)]
    after = split_rows(grown, bare_settings)
    after_val_ids = {r.id for r in after.val}

    churn = before_val_ids - after_val_ids
    assert len(churn) <= 3, f"{len(churn)} rows changed sides; the split is reshuffling"
    # ...and the overwhelming majority stayed put.
    assert len(before_val_ids & after_val_ids) > 0.9 * len(before_val_ids)


def test_the_split_is_the_configured_fraction_not_a_coin_flip_per_row(bare_settings):
    """The bug this replaced: thresholding each row's hash independently is a
    binomial draw, and at this corpus size the draw is wild. The live trainer
    held out 10 of 21 rows at `val_fraction=0.2` -- half the corpus, when the
    corpus had 21 rows in it. The hash is uniform (measured: 0.1958 realised
    against a 0.2 target over 10,500 real ids); the *scheme* was the problem.

    Taking the lowest-hashed `round(fraction * n)` rows makes the size exact at
    every corpus size, which is the only thing that helps at n=21.
    """
    for total in (20, 21, 25, 40, 137, 200):
        rows = [make_row(i) for i in range(total)]

        split = split_rows(rows, bare_settings)

        assert len(split.val) == round(0.2 * total), f"{total} rows held out {len(split.val)}"
        assert len(split.train) + len(split.val) == total
        assert not ({r.id for r in split.train} & {r.id for r in split.val})


def test_neither_side_of_the_split_is_ever_emptied(bare_settings):
    """A rounded fraction must not starve either side outright."""
    bare_settings.val_min_rows = 2

    for fraction in (0.0, 0.01, 0.5, 0.99, 1.0):
        bare_settings.val_fraction = fraction
        for total in (2, 3, 21):
            split = split_rows([make_row(i) for i in range(total)], bare_settings)

            assert split.val, f"fraction {fraction}, {total} rows: nothing held out"
            assert split.train, f"fraction {fraction}, {total} rows: nothing left to train on"


# --- small-corpus guard ----------------------------------------------------


def test_below_the_minimum_validation_is_disabled_and_everything_trains(bare_settings):
    rows = [make_row(i) for i in range(5)]

    split = split_rows(rows, bare_settings)

    assert split.enabled is False
    assert split.val == []
    assert {r.id for r in split.train} == {r.id for r in rows}
    assert split.disabled_reason and "5" in split.disabled_reason and "20" in split.disabled_reason


def test_at_or_above_the_minimum_validation_is_enabled(bare_settings):
    rows = [make_row(i) for i in range(20)]

    split = split_rows(rows, bare_settings)

    assert split.enabled is True
    assert split.disabled_reason is None


def test_val_min_rows_and_val_fraction_are_configurable_via_env(monkeypatch):
    monkeypatch.setenv("BABBLE_VAL_MIN_ROWS", "7")
    monkeypatch.setenv("BABBLE_VAL_FRACTION", "0.5")

    settings = Settings.from_env()

    assert settings.val_min_rows == 7
    assert settings.val_fraction == 0.5


def test_val_settings_default_when_unset(monkeypatch):
    monkeypatch.delenv("BABBLE_VAL_MIN_ROWS", raising=False)
    monkeypatch.delenv("BABBLE_VAL_FRACTION", raising=False)

    settings = Settings.from_env()

    assert settings.val_min_rows == 20
    assert settings.val_fraction == 0.2


# --- held-out rows still respect consent and the blocklist -----------------


def test_held_out_rows_are_drawn_only_from_already_consented_rows(settings):
    settings.val_min_rows = 1
    settings.val_fraction = 0.9
    ids = Pseudonymiser.load(settings)
    stranger = ids.user("someone-who-never-agreed")
    CorpusStore(settings.corpus_path).append(
        CorpusRow(
            id=make_corpus_id("hi", stranger),
            text="hi",
            author=stranger,
            source=SOURCE_AMBIENT,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    seed_fake_data(settings)  # grants both scopes to the fake users, never to `stranger`

    rows = corpus_rows(settings)  # already filters consent + blocklist
    split = split_rows(rows, settings)

    combined = split.train + split.val
    assert stranger not in {r.author for r in combined}
    assert {r.id for r in combined} == {r.id for r in rows}


# --- eval pass never mutates the model or the optimizer --------------------


def test_eval_loss_returns_none_for_no_held_out_examples(settings):
    model = Babbler(config_from_settings(settings))
    assert eval_loss(model, []) is None


def test_eval_loss_never_touches_weights_grads_or_optimizer_state(settings):
    model = Babbler(config_from_settings(settings))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    tokens = torch.randint(0, 50, (2, 8))
    mask = torch.ones_like(tokens)
    weights = torch.ones(2)
    loss = sequence_loss(model, tokens, mask, weights)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()  # give the optimizer real moments, the way a resumed run would

    before_weights = {k: v.clone() for k, v in model.state_dict().items()}
    before_grads = {
        name: (p.grad.clone() if p.grad is not None else None) for name, p in model.named_parameters()
    }
    before_moments = sorted(
        float(v.sum()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)
    )
    model.train()

    examples = [
        build_example("hi", "hey", settings.block_size),
        build_example("yo", "hey there", settings.block_size),
    ]
    result = eval_loss(model, examples)

    assert result is not None and result == result  # not NaN
    assert model.training is True  # restored to what it was before

    after_weights = model.state_dict()
    for key, value in before_weights.items():
        assert torch.equal(value, after_weights[key]), f"{key} changed during eval"

    for name, p in model.named_parameters():
        if before_grads[name] is None:
            assert p.grad is None
        else:
            assert torch.equal(p.grad, before_grads[name])

    after_moments = sorted(
        float(v.sum()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)
    )
    assert before_moments == after_moments


def test_eval_loss_restores_eval_mode_if_the_model_was_already_in_eval(settings):
    model = Babbler(config_from_settings(settings))
    model.eval()

    eval_loss(model, [build_example("hi", "hey", settings.block_size)])

    assert model.training is False


# --- the overfit signal is a plain, pure comparison -------------------------


def test_overfit_signal_true_when_val_rises_and_train_falls():
    assert overfit_signal(train_loss=1.0, prev_train_loss=2.0, val_loss=3.0, prev_val_loss=2.0) is True


@pytest.mark.parametrize(
    "train_loss,prev_train_loss,val_loss,prev_val_loss",
    [
        (2.0, 1.0, 3.0, 2.0),  # train also rose -- not overfitting, just bad
        (1.0, 2.0, 2.0, 3.0),  # val fell too -- learning fine
        (1.0, 2.0, 2.0, 2.0),  # val flat -- no signal
        (1.0, None, 2.0, None),  # no history yet
        (1.0, 2.0, None, None),  # validation disabled
    ],
)
def test_overfit_signal_false_otherwise(train_loss, prev_train_loss, val_loss, prev_val_loss):
    assert overfit_signal(train_loss, prev_train_loss, val_loss, prev_val_loss) is False


# --- checkpoint log lines carry val loss (or say why not) -------------------


def test_checkpoint_log_reports_validation_disabled_for_a_small_corpus(seeded, read_log):
    """`seeded` backfills the 12 fake corrections into 24 corpus rows (a prompt
    row and a correction row per interaction) -- comfortably below a
    `val_min_rows` set above that, so validation must be explicitly reported
    off, never a number."""
    seeded.val_min_rows = 30

    train(seeded, force=True, steps=4, echo=False, seed=1)

    entries = read_log("train.checkpoint")
    assert entries
    for entry in entries:
        assert entry["val_enabled"] is False
        assert "val_disabled_reason" in entry
        assert "val_loss" not in entry


def test_checkpoint_log_carries_val_loss_when_the_corpus_is_big_enough(seeded, read_log):
    seeded.val_min_rows = 4
    seeded.val_fraction = 0.5
    seeded.checkpoint_every = 3

    train(seeded, force=True, steps=6, echo=False, seed=1)

    entries = read_log("train.checkpoint")
    enabled = [e for e in entries if e["val_enabled"] is True]
    assert enabled
    for entry in enabled:
        assert "val_disabled_reason" not in entry
        assert "val_rows" in entry


def test_validation_does_not_perturb_the_checkpoint_format(seeded):
    """Best-val bookkeeping rolls the live model back to an earlier step before
    the final save -- that must not corrupt what actually lands on disk."""
    seeded.val_min_rows = 4
    seeded.val_fraction = 0.5

    result = train(seeded, force=True, steps=4, echo=False, seed=1)

    payload = torch.load(seeded.latest_checkpoint, map_location="cpu", weights_only=True)
    assert payload["step"] == result.final_step
    assert payload["optim"]["state"]


def test_loss_jsonl_carries_val_loss_and_row_counts(seeded, monkeypatch):
    """`loss.jsonl` must log the same val metric the trainer picks checkpoints
    on, plus both stored and split row counts, so a future reader is not comparing
    train loss from one run to val loss from another."""
    seeded.val_min_rows = 4
    seeded.val_fraction = 0.5
    seeded.checkpoint_every = 3

    val_curve = iter([2.0, 1.5, 1.0])
    monkeypatch.setattr("babble.trainer.eval_loss", lambda model, examples: next(val_curve))

    train(seeded, force=True, steps=6, echo=False, seed=1)

    entries = [
        json.loads(line)
        for line in seeded.loss_curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    enabled = [e for e in entries if "val_loss" in e]
    assert enabled
    for entry in enabled:
        assert entry["stored_rows"] >= entry["train_rows"] + entry["val_rows"]
        assert entry["rows"] == entry["train_rows"]


# --- the feed post carries the same information ------------------------


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, content: str) -> None:
        self.calls.append((url, content))


def test_feed_post_is_unchanged_when_no_validation_state_is_passed():
    """Backward compatibility: callers that don't pass val kwargs (as the
    pre-existing feed tests do) must get exactly the old post shape."""
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(cycle=1, step=50, loss=1.0, prev_loss=None, rows=3, prefix="hi", sample="hi")

    _, content = sender.calls[0]
    assert "val" not in content.lower()


def test_feed_post_reports_val_loss_when_enabled():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1,
        step=50,
        loss=1.0,
        prev_loss=None,
        rows=30,
        prefix="hi",
        sample="hi",
        val_loss=1.5,
        prev_val_loss=1.2,
        val_rows=6,
        val_enabled=True,
    )

    _, content = sender.calls[0]
    assert "1.5000" in content
    assert "6" in content


def test_feed_post_reports_disabled_and_why():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1,
        step=50,
        loss=1.0,
        prev_loss=None,
        rows=8,
        prefix="hi",
        sample="hi",
        val_enabled=False,
        val_disabled_reason="only 8 consented rows, need at least 20",
    )

    _, content = sender.calls[0]
    assert "disabled" in content.lower()
    assert "only 8 consented rows" in content


def test_feed_post_flags_val_rising_while_train_falls():
    sender = FakeSender()
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=sender)

    feed.checkpoint(
        cycle=1,
        step=50,
        loss=1.0,
        prev_loss=1.5,
        rows=30,
        prefix="hi",
        sample="hi",
        val_loss=2.0,
        prev_val_loss=1.5,
        val_rows=6,
        val_enabled=True,
        overfit_signal=True,
    )

    _, content = sender.calls[0]
    assert "rising" in content.lower() or "⚠" in content


# --- the overfit flag is wired all the way through _checkpoint -------------


def test_checkpoint_wiring_flags_overfitting_across_two_checkpoints(seeded, monkeypatch):
    """Drive `_checkpoint` directly with a controlled window (-> train loss)
    and a monkeypatched `eval_loss` (-> val loss), so the rise/fall trend is
    exact instead of hoping real training happens to produce it in a few
    steps."""
    model = Babbler(config_from_settings(seeded))
    optimizer = torch.optim.AdamW(model.parameters(), lr=seeded.learning_rate)
    feed = TrainingFeed(webhook_url="https://discord.example/webhook", sender=FakeSender())
    blocklist = Blocklist.load()
    log = EventLog(seeded, Pseudonymiser.load(seeded), component="test")

    val_losses = iter([1.0, 2.0])  # rising
    monkeypatch.setattr("babble.trainer.eval_loss", lambda *a, **k: next(val_losses))

    rows = [make_row(i) for i in range(3)]
    examples = to_examples(rows, seeded.block_size)

    mean1, val1 = _checkpoint(
        seeded, log, feed, blocklist, model, optimizer,
        step=1, window=[5.0], rows=rows, probe_index=0, cycle=1, prev_loss=None, echo=False,
        train_examples=examples, val_examples=examples, val_enabled=True, val_disabled_reason=None,
        val_rows=6, prev_val_loss=None,
    )
    mean2, val2 = _checkpoint(
        seeded, log, feed, blocklist, model, optimizer,
        step=2, window=[3.0], rows=rows, probe_index=1, cycle=1, prev_loss=mean1, echo=False,
        train_examples=examples, val_examples=examples, val_enabled=True, val_disabled_reason=None,
        val_rows=6, prev_val_loss=val1,
    )

    assert mean1 == 5.0 and mean2 == 3.0  # train falling
    assert val1 == 1.0 and val2 == 2.0  # val rising

    log.flush()
    lines = seeded.log_dir.joinpath("babble.jsonl").read_text(encoding="utf-8").splitlines()
    checkpoints = [json.loads(l) for l in lines if l.strip() and json.loads(l).get("event") == "train.checkpoint"]

    assert checkpoints[0]["overfit_signal"] is False  # nothing to compare against yet
    assert checkpoints[1]["overfit_signal"] is True  # falling train, rising val

    overfit_posts = [c for c in feed.sender.calls if "rising" in c[1].lower() or "⚠" in c[1]]
    assert len(overfit_posts) == 1
