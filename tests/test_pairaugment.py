"""Correction-pair augmentation: the generator only ever derives variants
from TRAIN-side real pairs, a bad model response never silently becomes a
stored variant, one failed pair never kills the batch, and the leakage check
fails loudly on anything that slipped through. Every test here uses a fake
`LLMClient` -- nothing in this file shells out to the real `claude` binary."""

from __future__ import annotations

import json

import pytest

from babble.consent import ConsentStore
from babble.llm import LLMError
from babble.pairaugment import (
    AugmentedPair,
    AugmentedPairStore,
    AutoAugmentTrigger,
    LeakageError,
    ParaphraseError,
    _parse_variants,
    assert_no_leakage,
    augmented_pair_count,
    check_leakage,
    generate_augmented_pairs,
    make_augmented_id,
    register_comparison,
    trainable_augmented_pairs,
)
from babble.pairsplit import pair_split
from babble.post_state import trainable_pairs
from babble.store import CORRECTION, Interaction, InteractionStore, make_row_id


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


PAIRS = [(f"question number {i} about stuff", f"answer number {i} with real content") for i in range(30)]


class FakeLLMClient:
    """Returns `n_variants` distinct, deterministic paraphrases per call.
    `fail_markers` / `malformed_markers` let a test target one specific
    source pair's call by a substring that only appears in its prompt (the
    original prompt/response text, embedded verbatim by `_build_prompt`)."""

    def __init__(self, n_variants: int = 3, fail_markers=(), malformed_markers=()):
        self.calls: list[str] = []
        self.n_variants = n_variants
        self.fail_markers = fail_markers
        self.malformed_markers = malformed_markers
        self._counter = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for marker in self.fail_markers:
            if marker in prompt:
                raise LLMError(f"simulated failure ({marker})")
        for marker in self.malformed_markers:
            if marker in prompt:
                return "this is not json"
        self._counter += 1
        variants = [
            {"prompt": f"variant prompt {self._counter}-{i}", "chosen": f"variant chosen {self._counter}-{i}"}
            for i in range(self.n_variants)
        ]
        return json.dumps(variants)


# --- _parse_variants: fail loudly, never store garbage ---------------------


def test_parse_variants_accepts_a_well_formed_array():
    raw = json.dumps([{"prompt": "p1", "chosen": "c1"}, {"prompt": "p2", "chosen": "c2"}])
    out = _parse_variants(raw, "orig prompt", "orig chosen", 2)
    assert out == [("p1", "c1"), ("p2", "c2")]


def test_parse_variants_strips_a_markdown_fence():
    raw = "```json\n" + json.dumps([{"prompt": "p1", "chosen": "c1"}]) + "\n```"
    assert _parse_variants(raw, "orig", "orig", 1) == [("p1", "c1")]


def test_parse_variants_rejects_non_json():
    with pytest.raises(ParaphraseError):
        _parse_variants("not json at all", "orig prompt", "orig chosen", 3)


def test_parse_variants_rejects_wrong_shape():
    with pytest.raises(ParaphraseError):
        _parse_variants(json.dumps({"prompt": "p", "chosen": "c"}), "orig", "orig", 1)


def test_parse_variants_rejects_an_empty_half():
    raw = json.dumps([{"prompt": "", "chosen": "c1"}])
    with pytest.raises(ParaphraseError):
        _parse_variants(raw, "orig", "orig", 1)


def test_parse_variants_rejects_an_all_echo_response():
    raw = json.dumps([{"prompt": "orig prompt", "chosen": "orig chosen"}])
    with pytest.raises(ParaphraseError):
        _parse_variants(raw, "orig prompt", "orig chosen", 1)


def test_parse_variants_drops_duplicates_but_keeps_the_rest():
    raw = json.dumps(
        [
            {"prompt": "p1", "chosen": "c1"},
            {"prompt": "p1", "chosen": "c1"},
            {"prompt": "p2", "chosen": "c2"},
        ]
    )
    assert _parse_variants(raw, "orig", "orig", 3) == [("p1", "c1"), ("p2", "c2")]


def test_parse_variants_rejects_an_implausibly_long_response():
    raw = json.dumps([{"prompt": "p1", "chosen": "c" * 10_000}])
    with pytest.raises(ParaphraseError):
        _parse_variants(raw, "orig", "orig", 1)


# --- content-addressed ids ---------------------------------------------------


def test_ids_are_content_addressed():
    a = make_augmented_id("src-1", 0, "p", "c")
    b = make_augmented_id("src-1", 0, "p", "c")
    assert a == b
    assert a != make_augmented_id("src-1", 1, "p", "c")
    assert a != make_augmented_id("src-2", 0, "p", "c")


# --- the train-only guarantee ------------------------------------------------


def test_generate_only_reads_train_side_pairs(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    train_pairs, val_pairs = pair_split(pairs)
    assert train_pairs and val_pairs  # the fixture is large enough to hold some out

    client = FakeLLMClient(n_variants=2)
    result = generate_augmented_pairs(settings, n=2, client=client, ids=ids)

    assert result.train_side_pairs == len(train_pairs)
    assert result.val_side_pairs == len(val_pairs)
    assert len(client.calls) == len(train_pairs)  # never called for a val-side pair

    stored = AugmentedPairStore(settings.augmented_pairs_path).all()
    train_ids = {p.id for p in train_pairs}
    val_ids = {p.id for p in val_pairs}
    assert stored  # something was actually generated
    for row in stored:
        assert row.source_pair_id in train_ids
        assert row.source_pair_id not in val_ids


def test_generate_never_touches_interactions_jsonl(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    before = InteractionStore(settings.interactions_path).all()

    generate_augmented_pairs(settings, n=2, client=FakeLLMClient(n_variants=2), ids=ids)

    after = InteractionStore(settings.interactions_path).all()
    assert before == after
    assert settings.augmented_pairs_path != settings.interactions_path


def test_stored_rows_are_labelled_synthetic(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    generate_augmented_pairs(settings, n=2, client=FakeLLMClient(n_variants=2), ids=ids)
    for line in settings.augmented_pairs_path.read_text().splitlines():
        raw = json.loads(line)
        assert raw["synthetic"] is True
        assert raw["method"] == "llm_paraphrase"
        assert raw["source_pair_id"]


# --- one bad pair never kills the batch -------------------------------------


def test_a_failed_pair_is_reported_and_does_not_stop_the_rest(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    train_pairs, _ = pair_split(pairs)
    doomed = train_pairs[0]

    client = FakeLLMClient(n_variants=2, fail_markers=[f'"{doomed.prompt}"'])
    result = generate_augmented_pairs(settings, n=2, client=client, ids=ids)

    assert result.failed_pairs == 1
    assert any(doomed.id in f for f in result.failures)
    # everyone else still got variants
    stored = AugmentedPairStore(settings.augmented_pairs_path).all()
    assert stored
    assert all(row.source_pair_id != doomed.id for row in stored)
    assert result.generated == len(stored)


def test_a_malformed_response_is_reported_not_stored(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    train_pairs, _ = pair_split(pairs)
    target = train_pairs[0]

    client = FakeLLMClient(n_variants=2, malformed_markers=[f'"{target.prompt}"'])
    result = generate_augmented_pairs(settings, n=2, client=client, ids=ids)

    assert result.failed_pairs == 1
    stored = AugmentedPairStore(settings.augmented_pairs_path).all()
    assert all(row.source_pair_id != target.id for row in stored)


def test_a_client_broken_for_every_pair_still_reports_loudly_without_crashing(settings, ids):
    """A single flaky pair should not take the whole batch down (see the
    per-pair failure tests above) -- and neither should a client that is
    broken outright: every pair ends up in `.failures`, nothing is silently
    stored, and the call does not raise (important for `AutoAugmentTrigger`,
    which fires this from a background subprocess nobody is watching)."""
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    train_pairs, _ = pair_split(pairs)

    class BrokenClient:
        def complete(self, prompt: str) -> str:
            raise LLMError("binary not found")

    result = generate_augmented_pairs(settings, n=2, client=BrokenClient(), ids=ids)

    assert result.failed_pairs == len(train_pairs)
    assert result.generated == 0
    assert AugmentedPairStore(settings.augmented_pairs_path).all() == []


# --- idempotency / cheap reruns ----------------------------------------------


def test_rerun_skips_pairs_already_covered(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    client = FakeLLMClient(n_variants=2)
    generate_augmented_pairs(settings, n=2, client=client, ids=ids)
    first_calls = len(client.calls)
    assert first_calls > 0

    result = generate_augmented_pairs(settings, n=2, client=client, ids=ids)
    assert len(client.calls) == first_calls  # no new calls -- everything already covered
    assert result.skipped_already_covered == result.train_side_pairs
    assert result.generated == 0


def test_pair_id_filter_targets_exactly_one_pair(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    train_pairs, _ = pair_split(pairs)
    target = train_pairs[0]

    client = FakeLLMClient(n_variants=2)
    result = generate_augmented_pairs(settings, n=2, client=client, ids=ids, pair_ids=[target.id])

    assert len(client.calls) == 1
    stored = AugmentedPairStore(settings.augmented_pairs_path).all()
    assert stored and all(row.source_pair_id == target.id for row in stored)
    assert result.generated == len(stored)


def test_pair_id_filter_on_a_val_side_pair_generates_nothing(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    _, val_pairs = pair_split(pairs)
    assert val_pairs

    client = FakeLLMClient(n_variants=2)
    result = generate_augmented_pairs(settings, n=2, client=client, ids=ids, pair_ids=[val_pairs[0].id])

    assert client.calls == []
    assert result.generated == 0


# --- trainable_augmented_pairs: belt-and-braces re-check --------------------


def test_trainable_augmented_pairs_drops_a_row_forged_against_a_val_side_pair(settings, ids):
    """Even if something bypassed the generator and wrote a row derived from
    a val-side pair directly, training-time re-derivation must still exclude
    it -- the same defence-in-depth every other trainable_* function gives."""
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    _, val_pairs = pair_split(pairs)
    assert val_pairs
    forged = AugmentedPair(
        id=make_augmented_id(val_pairs[0].id, 0, "p", "c"),
        prompt="p", chosen="c", source_pair_id=val_pairs[0].id, variant_index=0,
    )
    AugmentedPairStore(settings.augmented_pairs_path).append(forged)

    assert trainable_augmented_pairs(settings, ids) == []


def test_trainable_augmented_pairs_respects_the_blocklist(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    generate_augmented_pairs(settings, n=1, client=FakeLLMClient(n_variants=1), ids=ids)
    assert trainable_augmented_pairs(settings, ids)

    class _BlockAll:
        def matches(self, *texts) -> bool:
            return True

    assert trainable_augmented_pairs(settings, ids, blocklist=_BlockAll()) == []


def test_augmented_pair_count_matches_the_store(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    generate_augmented_pairs(settings, n=1, client=FakeLLMClient(n_variants=1), ids=ids)
    assert augmented_pair_count(settings) == AugmentedPairStore(settings.augmented_pairs_path).count() > 0


# --- the leakage check --------------------------------------------------------


def test_leakage_check_passes_on_a_clean_generation(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    generate_augmented_pairs(settings, n=2, client=FakeLLMClient(n_variants=2), ids=ids)
    report = check_leakage(settings, ids)
    assert report.ok
    assert report.leaked == 0
    assert report.checked > 0
    assert_no_leakage(settings, ids)  # does not raise


def test_leakage_check_fails_loudly_on_a_val_derived_row(settings, ids):
    _seed_pairs(settings, ids, PAIRS)
    pairs = trainable_pairs(settings, ids)
    _, val_pairs = pair_split(pairs)
    assert val_pairs
    leaker = AugmentedPair(
        id=make_augmented_id(val_pairs[0].id, 0, "leak prompt", "leak chosen"),
        prompt="leak prompt", chosen="leak chosen",
        source_pair_id=val_pairs[0].id, variant_index=0,
    )
    AugmentedPairStore(settings.augmented_pairs_path).append(leaker)

    report = check_leakage(settings, ids)
    assert not report.ok
    assert report.leaked == 1
    assert leaker.id in report.leaked_ids

    with pytest.raises(LeakageError):
        assert_no_leakage(settings, ids)


def test_leakage_check_reports_orphaned_rows_without_failing(settings, ids):
    """A source pair that no longer resolves (withdrawn consent, purged) is
    suspicious but not a train/val leak -- it must be visible, not fatal."""
    _seed_pairs(settings, ids, PAIRS)
    orphan = AugmentedPair(
        id=make_augmented_id("no-such-pair", 0, "p", "c"),
        prompt="p", chosen="c", source_pair_id="no-such-pair", variant_index=0,
    )
    AugmentedPairStore(settings.augmented_pairs_path).append(orphan)

    report = check_leakage(settings, ids)
    assert report.ok  # orphaned, not leaked
    assert report.orphaned == 1
    assert orphan.id in report.orphaned_ids
    assert_no_leakage(settings, ids)  # does not raise


# --- register / voice-drift comparison ---------------------------------------


def test_register_comparison_reports_real_and_variant_stats(settings, ids):
    lowercase_pairs = [(f"hey q{i}", f"lol yeah {i} idk") for i in range(10)]
    _seed_pairs(settings, ids, lowercase_pairs)
    generate_augmented_pairs(settings, n=1, client=FakeLLMClient(n_variants=1), ids=ids)

    report = register_comparison(settings, ids)
    assert report.real_count == len(lowercase_pairs)
    assert report.variant_count > 0
    assert 0.0 <= report.vocab_overlap <= 1.0
    assert 0.0 <= report.real_lowercase_rate <= 1.0
    assert 0.0 <= report.variant_lowercase_rate <= 1.0


def test_register_comparison_flags_obvious_drift(settings, ids):
    lowercase_pairs = [(f"hey q{i}", f"lol yeah {i} idk") for i in range(10)]
    _seed_pairs(settings, ids, lowercase_pairs)

    class AssistantVoiceClient:
        def complete(self, prompt: str) -> str:
            return json.dumps(
                [
                    {
                        "prompt": "Certainly, I would be delighted to help.",
                        "chosen": (
                            "Certainly! I would be delighted to assist you with this "
                            "inquiry. Here is a thorough, well-structured response "
                            "that fully addresses your question in formal register."
                        ),
                    }
                ]
            )

    generate_augmented_pairs(settings, n=1, client=AssistantVoiceClient(), ids=ids)
    report = register_comparison(settings, ids)
    assert report.drifted is True


# --- the auto-fire hook -------------------------------------------------------


def test_auto_trigger_is_a_no_op_when_the_knob_is_off(settings, monkeypatch):
    settings.post_augment_pairs = False
    launched = []
    monkeypatch.setattr(
        "subprocess.Popen", lambda *a, **k: launched.append((a, k)) or object()
    )
    AutoAugmentTrigger(settings).on_new_pair("some-pair-id")
    assert launched == []


def test_auto_trigger_launches_a_detached_subprocess_when_on(settings, monkeypatch):
    settings.post_augment_pairs = True
    launched = []

    class _FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        launched.append((cmd, kwargs))
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    AutoAugmentTrigger(settings).on_new_pair("some-pair-id")

    assert len(launched) == 1
    cmd, kwargs = launched[0]
    assert "augment-pairs" in cmd
    assert "--pair-id" in cmd and "some-pair-id" in cmd
    assert kwargs.get("start_new_session") is True


def test_auto_trigger_swallows_a_launch_failure(settings, monkeypatch):
    settings.post_augment_pairs = True

    def broken_popen(*a, **k):
        raise OSError("no fork slots")

    monkeypatch.setattr("subprocess.Popen", broken_popen)
    AutoAugmentTrigger(settings).on_new_pair("some-pair-id")  # must not raise
