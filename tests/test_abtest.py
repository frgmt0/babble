"""Blind A/B rating: prompt selection, the blind mapping, resumable rating,
report unblinding + the hand-rolled binomial sign test, checkpoint archiving
on promotion, and rollback."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from babble.abtest import (
    DEFAULT_PROMPT_COUNT,
    EVERYDAY_PROMPTS,
    MIN_DECISIVE_VOTES,
    ABItem,
    ABSession,
    apply_vote,
    binomial_sign_test,
    build_report,
    load_session,
    rate_interactive,
    render_report,
    rollback,
    run_ab,
    save_session,
    select_prompts,
    unvoted_items,
)
from babble.cli import build_parser
from babble.config import Settings
from babble.consent import ConsentStore
from babble.cpu_runtime import force_cpu_device
from babble.model import Babbler, config_from_settings
from babble.post_state import read_post_state
from babble.posttrain import post_train
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id
from babble.trainer import _build_optimizer, save_checkpoint

# --- helpers ---------------------------------------------------------------


def _seed_checkpoint(settings: Settings, *, step: int = 1) -> None:
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
        store.append(
            Interaction(
                id=make_row_id(CORRECTION, prompt, chosen, asker_p, helper_p),
                signal=CORRECTION,
                prompt=prompt,
                rejected="wrong answer",
                chosen=chosen,
                prompt_author=asker_p,
                signal_author=helper_p,
                weight=settings.correction_weight,
            )
        )


PAIRS = [(f"question number {i}", f"answer number {i} with a little more text") for i in range(20)]


# --- select_prompts ----------------------------------------------------------


def test_select_prompts_includes_everyday_and_fills_with_correction_prompts():
    held_out = [f"held out prompt {i}" for i in range(30)]
    selected = select_prompts(held_out, 20)

    assert len(selected) == 20
    sources = [s for _, s in selected]
    assert sources.count("everyday") == len(EVERYDAY_PROMPTS)
    assert sources.count("correction") == 20 - len(EVERYDAY_PROMPTS)
    # everyday prompts come from the fixed list, never held-out text
    everyday_prompts = {p for p, s in selected if s == "everyday"}
    assert everyday_prompts == set(EVERYDAY_PROMPTS)


def test_select_prompts_dedupes_and_caps_at_count():
    held_out = list(EVERYDAY_PROMPTS) + ["fresh one"]  # first N collide with the fixed set
    selected = select_prompts(held_out, 3)
    assert len(selected) == 3
    texts = [p for p, _ in selected]
    assert len(texts) == len(set(texts))


def test_select_prompts_with_no_held_out_pairs_is_just_everyday():
    selected = select_prompts([], 20)
    assert [p for p, _ in selected] == EVERYDAY_PROMPTS


# --- binomial_sign_test -------------------------------------------------------


@pytest.mark.parametrize(
    "wins, losses, expected",
    [
        (0, 0, 1.0),
        (5, 5, 1.0),
        (10, 0, 2 * (1 / 1024)),
        (8, 2, 2 * (1 + 10 + 45) / 1024),
        (1, 0, 1.0),  # n=1, k=0: 2 * C(1,0)/2 = 1.0
    ],
)
def test_binomial_sign_test_matches_hand_computed_values(wins, losses, expected):
    assert binomial_sign_test(wins, losses) == pytest.approx(expected, abs=1e-9)


def test_binomial_sign_test_is_symmetric_in_wins_and_losses():
    assert binomial_sign_test(8, 2) == binomial_sign_test(2, 8)


# --- run_ab: session file, determinism, seed -------------------------------


def test_run_ab_writes_a_session_with_the_seed_and_deterministic_identical_sampling(settings, ids):
    """Same checkpoint on both sides must produce byte-identical responses --
    the whole point of feeding both sides the same per-prompt seed."""
    _seed_checkpoint(settings)

    session, out_path = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        count=6,
        seed=42,
        ids=ids,
    )

    assert out_path.exists()
    assert session.seed == 42
    assert len(session.items) == 6
    for item in session.items:
        assert item.response_candidate == item.response_baseline

    reloaded = load_session(out_path)
    assert reloaded.seed == 42
    assert [i.prompt for i in reloaded.items] == [i.prompt for i in session.items]


def test_run_ab_refuses_when_baseline_checkpoint_is_missing(settings, ids):
    _seed_checkpoint(settings)
    assert not settings.previous_checkpoint.exists()

    with pytest.raises(FileNotFoundError):
        run_ab(settings, count=4, ids=ids)


def test_run_ab_default_prompt_count(settings, ids):
    _seed_checkpoint(settings)
    session, _ = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        ids=ids,
    )
    assert len(session.items) == min(DEFAULT_PROMPT_COUNT, len(EVERYDAY_PROMPTS))


def test_run_ab_pulls_held_out_correction_prompts(settings, ids):
    """Correction prompts in the session must be val-side (held out), never
    a prompt the post-train would have trained on."""
    from babble.pairsplit import pair_split
    from babble.post_state import trainable_pairs

    _seed_checkpoint(settings)
    _seed_pairs(settings, ids, PAIRS)

    session, _ = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        count=20,
        ids=ids,
    )

    train_pairs, val_pairs = pair_split(trainable_pairs(settings, ids))
    train_prompts = {p.prompt for p in train_pairs}
    val_prompts = {p.prompt for p in val_pairs}
    correction_prompts = {i.prompt for i in session.items if i.prompt_source == "correction"}
    assert correction_prompts <= val_prompts
    assert not (correction_prompts & train_prompts)


# --- the blind coin flip: per-prompt, not constant, never leaked -----------


def test_coin_flip_is_per_prompt_and_matches_the_recorded_seed(settings, ids):
    _seed_checkpoint(settings)
    session, _ = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        count=6,
        seed=1,
        ids=ids,
    )

    expected_rng = random.Random(1)
    expected_flips = [expected_rng.random() < 0.5 for _ in session.items]

    assert [item.a_is_candidate for item in session.items] == expected_flips
    # Not a constant assignment -- seed 1 over 6 draws is a genuine mix.
    assert len(set(expected_flips)) > 1


def test_rate_interactive_never_prints_the_true_mapping(settings, ids, tmp_path):
    _seed_checkpoint(settings)
    session, out_path = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        count=4,
        seed=7,
        ids=ids,
    )
    # Response text deliberately avoids the words "candidate"/"baseline" so
    # those words appearing in the printed output can only mean the UI
    # labelled a side, not that it's quoting response content.
    raw = json.loads(out_path.read_text())
    for i, item in enumerate(raw["items"]):
        item["response_candidate"] = f"response-one-{i}"
        item["response_baseline"] = f"response-two-{i}"
    out_path.write_text(json.dumps(raw))

    printed = []
    answers = iter(["a", "b", "tie", "skip"])
    rate_interactive(out_path, input_fn=lambda _prompt: next(answers), print_fn=printed.append)

    blob = "\n".join(printed)
    assert "a_is_candidate" not in blob
    assert "candidate" not in blob.lower()
    assert "baseline" not in blob.lower()
    # The actual response content must still reach the rater.
    assert "response-one-0" in blob or "response-two-0" in blob

    reloaded = load_session(out_path)
    assert [i.vote for i in reloaded.items] == ["a", "b", "tie", "skip"]


def test_rate_interactive_resumes_after_a_quit(settings, ids):
    _seed_checkpoint(settings)
    session, out_path = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        count=3,
        seed=3,
        ids=ids,
    )

    first_answers = iter(["a", "q"])
    rate_interactive(out_path, input_fn=lambda _p: next(first_answers), print_fn=lambda *_: None)

    resumed = load_session(out_path)
    votes = [i.vote for i in resumed.items]
    assert votes[0] == "a"
    assert votes[1] is None and votes[2] is None

    second_answers = iter(["b", "tie"])
    rate_interactive(out_path, input_fn=lambda _p: next(second_answers), print_fn=lambda *_: None)

    finished = load_session(out_path)
    assert [i.vote for i in finished.items] == ["a", "b", "tie"]


def test_rate_interactive_with_nothing_left_says_so(settings, ids):
    _seed_checkpoint(settings)
    session, out_path = run_ab(
        settings,
        checkpoint_a=settings.latest_checkpoint,
        checkpoint_b=settings.latest_checkpoint,
        count=1,
        ids=ids,
    )
    printed = []
    rate_interactive(out_path, input_fn=lambda _p: "a", print_fn=printed.append)
    printed.clear()
    rate_interactive(out_path, input_fn=lambda _p: "a", print_fn=printed.append)
    assert any("nothing left" in line.lower() for line in printed)


# --- report: unblind + tally + sign test ------------------------------------


def _make_session(votes_and_mapping):
    """`votes_and_mapping`: list of (vote, a_is_candidate) tuples."""
    items = [
        ABItem(
            index=i,
            prompt=f"p{i}",
            prompt_source="everyday",
            response_candidate=f"cand{i}",
            response_baseline=f"base{i}",
            a_is_candidate=a_is_candidate,
            vote=vote,
        )
        for i, (vote, a_is_candidate) in enumerate(votes_and_mapping)
    ]
    return ABSession(
        session_id="s1", created_at="now", seed=1,
        candidate_checkpoint="cand.pt", baseline_checkpoint="base.pt",
        candidate_step=1, baseline_step=0, sampling={}, items=items,
    )


def test_build_report_tally_matches_votes_after_unblinding():
    # a=candidate wins twice, b=baseline is shown as candidate once (a vote
    # for "a" there is actually a baseline win) -- exercises both mappings.
    session = _make_session(
        [
            ("a", True),  # A is candidate, voted A -> candidate win
            ("b", True),  # A is candidate, voted B -> baseline win
            ("a", False),  # A is baseline, voted A -> baseline win
            ("b", False),  # A is baseline, voted B -> candidate win
            ("tie", True),
            ("skip", False),
        ]
    )
    report = build_report(session)
    assert report.candidate_wins == 2
    assert report.baseline_wins == 2
    assert report.ties == 1
    assert report.skips == 1
    assert report.decisive == 4
    assert report.candidate_win_rate == pytest.approx(0.5)
    assert report.p_value == pytest.approx(binomial_sign_test(2, 2))


def test_build_report_ignores_unrated_items():
    session = _make_session([("a", True), (None, True)])
    report = build_report(session)
    assert report.candidate_wins == 1
    assert report.decisive == 1
    assert len(report.rows) == 1


def test_render_report_flags_a_too_small_sample():
    session = _make_session([("a", True)] * 3 + [("b", True)] * 2)  # 5 decisive < MIN_DECISIVE_VOTES
    report = build_report(session)
    assert report.decisive < MIN_DECISIVE_VOTES
    text = render_report(session, report)
    assert "too small" in text.lower()


def test_render_report_unblinds_true_labels():
    session = _make_session([("a", False)])  # A is baseline; voting "a" is really a baseline win
    report = build_report(session)
    text = render_report(session, report)
    assert "cand0" in text and "base0" in text
    assert "vote: a -> baseline" in text


# --- apply_vote / unvoted_items ----------------------------------------------


def test_apply_vote_rejects_unknown_vote():
    session = _make_session([(None, True)])
    with pytest.raises(ValueError):
        apply_vote(session, 0, "maybe")


def test_apply_vote_rejects_unknown_index():
    session = _make_session([(None, True)])
    with pytest.raises(KeyError):
        apply_vote(session, 99, "a")


def test_unvoted_items_only_returns_unrated():
    session = _make_session([("a", True), (None, True), (None, False)])
    assert [i.index for i in unvoted_items(session)] == [1, 2]


# --- archive on promotion (posttrain.py) ------------------------------------


def test_promotion_archives_the_outgoing_checkpoint(settings, ids):
    _seed_checkpoint(settings, step=1)
    pretrain_bytes = settings.latest_checkpoint.read_bytes()
    _seed_pairs(settings, ids, PAIRS)
    # No corpus rows seeded -- the promotion gate has nothing held-out to
    # score against and is skipped entirely (same as
    # test_gate_skipped_when_there_are_no_corpus_val_rows in test_posttrain.py),
    # so this always promotes regardless of learning rate.

    result = post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert result.ran and result.promoted
    assert settings.previous_checkpoint.exists()
    assert settings.previous_checkpoint.read_bytes() == pretrain_bytes
    assert settings.previous_meta_path.exists()
    meta = json.loads(settings.previous_meta_path.read_text())
    assert meta["step"] == 1
    assert meta["hash"] is not None


def test_first_ever_promotion_when_nothing_to_archive_is_a_quiet_no_op(settings):
    """Archiving must never blow up on the very first promotion -- there was
    never an outgoing checkpoint before it (covered indirectly by every other
    post-train test, this pins the "no latest.pt yet" branch directly)."""
    from babble.posttrain import _archive_outgoing_checkpoint

    assert not settings.latest_checkpoint.exists()
    _archive_outgoing_checkpoint(settings)  # must not raise
    assert not settings.previous_checkpoint.exists()


def test_a_second_promotion_overwrites_previous_pt_with_the_first_candidate(settings, ids):
    _seed_checkpoint(settings, step=1)
    _seed_pairs(settings, ids, PAIRS)
    settings.post_learning_rate = 1e-5
    settings.post_trigger_pairs = 0

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    first_candidate_bytes = settings.latest_checkpoint.read_bytes()

    _seed_pairs(settings, ids, [(f"more q {i}", f"more a {i}") for i in range(20)], asker="asker2-raw", helper="helper2-raw")
    post_train(settings, force=True, steps=4, seed=2, echo=False, ids=ids)

    # previous.pt now holds what latest.pt was *before this second promotion*
    # -- i.e. the first post-train's own promoted output, not the original
    # pretrain.
    assert settings.previous_checkpoint.read_bytes() == first_candidate_bytes


# --- rollback ----------------------------------------------------------------


def test_rollback_refuses_with_no_archived_predecessor(settings):
    with pytest.raises(FileNotFoundError):
        rollback(settings)


def test_rollback_restores_byte_identical_weights_and_records_it(settings, ids):
    _seed_checkpoint(settings, step=1)
    pretrain_bytes = settings.latest_checkpoint.read_bytes()
    _seed_pairs(settings, ids, PAIRS)
    settings.post_learning_rate = 1e-5

    result = post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    assert result.ran and result.promoted
    promoted_bytes = settings.latest_checkpoint.read_bytes()
    assert promoted_bytes != pretrain_bytes  # the promotion actually changed latest.pt

    out = rollback(settings)

    assert settings.latest_checkpoint.read_bytes() == pretrain_bytes
    assert out["restored_step"] == 1

    state = read_post_state(settings)
    assert state["rolled_back_from"] == str(settings.previous_checkpoint)
    assert state["rolled_back_to_step"] == 1
    # latest_hash now matches what's actually on disk, so the NEXT post-train's
    # pretrained.pt snapshot logic treats these restored weights as valid
    # prior post-train output rather than mistaking them for a fresh pretrain.
    from babble.post_state import file_hash

    assert state["latest_hash"] == file_hash(settings.latest_checkpoint)


def test_rollback_then_post_train_does_not_resnapshot_pretrained_from_the_rollback(settings, ids):
    """The real-world guarantee behind `record_rollback`: after a rollback,
    the next post-train must still fine-tune from the SAME `pretrained.pt`
    it always has -- not silently re-snapshot it from the just-restored
    (post-trained) weights, which would compound post-train on post-train."""
    _seed_checkpoint(settings, step=1)
    _seed_pairs(settings, ids, PAIRS)
    settings.post_learning_rate = 1e-5
    settings.post_trigger_pairs = 0

    post_train(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    pretrained_snapshot_before = settings.pretrained_checkpoint.read_bytes()

    rollback(settings)
    _seed_pairs(settings, ids, [(f"q2 {i}", f"a2 {i}") for i in range(20)], asker="asker3-raw", helper="helper3-raw")
    post_train(settings, force=True, steps=4, seed=2, echo=False, ids=ids)

    assert settings.pretrained_checkpoint.read_bytes() == pretrained_snapshot_before


# --- config default ----------------------------------------------------------


def test_post_min_pairs_default_is_50():
    s = Settings.for_root(Path("/tmp/does-not-need-to-exist"))
    assert s.post_min_pairs == 50


def test_post_min_pairs_env_default_is_50(monkeypatch):
    monkeypatch.delenv("BABBLE_POST_MIN_PAIRS", raising=False)
    s = Settings.from_env(root=Path("/tmp/does-not-need-to-exist"))
    assert s.post_min_pairs == 50


# --- CLI wiring --------------------------------------------------------------


@pytest.mark.parametrize("subcommand", ["run", "rate", "report", "rollback"])
def test_ab_subcommands_are_registered(subcommand):
    argv = ["ab", subcommand]
    if subcommand in ("rate", "report"):
        argv.append("session.json")
    args = build_parser().parse_args(argv)
    assert args.command == "ab"
    assert args.ab_command == subcommand
