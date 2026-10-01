"""Multi-turn prompt construction, isolation, consent, and persistence."""

from __future__ import annotations

import json

from babble.conversation import (
    ConversationTurn,
    conversation_prompt,
    conversation_prompt_for_token_budget,
)
from babble.core import Babble
from babble.consent import SCOPE_CORRECTIONS
from babble.exchanges import Exchange, ExchangeLog
from conftest import FakeDiscord

ALICE = "111111111111111111"
BOB = "222222222222222222"


def _enable(settings) -> None:
    settings.conversation_context = True
    settings.conversation_max_turns = 6
    settings.conversation_max_tokens = 512
    settings.conversation_max_chars = 6_000


def test_existing_checkpoints_keep_the_single_turn_prompt_by_default(fake, generator):
    fake.onboard(ALICE)
    first = fake.ping(ALICE, "hello")[0]
    fake.ping(ALICE, "still there?", reply_to=first.id)

    assert generator.prompts == ["hello", "still there?"]


def test_enabled_context_follows_an_explicit_reply_chain(settings, generator, log):
    _enable(settings)
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)

    first = gateway.ping(ALICE, "hello")[0]
    second = gateway.ping(ALICE, "how are you?", reply_to=first.id)[0]
    gateway.ping(ALICE, "tell me more", reply_to=second.id)

    assert generator.prompts == [
        "user: hello",
        "user: hello\nassistant: wug wug blorp\nuser: how are you?",
        (
            "user: hello\nassistant: wug wug blorp\n"
            "user: how are you?\nassistant: wug wug blorp\nuser: tell me more"
        ),
    ]


def test_replying_to_an_older_answer_forks_from_that_exact_point(settings, generator, log):
    _enable(settings)
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)
    first = gateway.ping(ALICE, "start")[0]
    gateway.ping(ALICE, "first branch", reply_to=first.id)

    gateway.ping(ALICE, "other branch", reply_to=first.id)

    assert generator.prompts[-1] == (
        "user: start\nassistant: wug wug blorp\nuser: other branch"
    )
    assert "first branch" not in generator.prompts[-1]


def test_runtime_retains_only_the_configured_number_of_completed_turns(
    settings, generator, log
):
    _enable(settings)
    settings.conversation_max_turns = 1
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)
    first = gateway.ping(ALICE, "one")[0]
    second = gateway.ping(ALICE, "two", reply_to=first.id)[0]

    gateway.ping(ALICE, "three", reply_to=second.id)

    assert generator.prompts[-1] == (
        "user: two\nassistant: wug wug blorp\nuser: three"
    )


def test_conversation_context_does_not_enter_human_corpus_or_correction_prompt(
    settings, generator, log
):
    _enable(settings)
    brain = Babble(settings, generator=generator, log=log)
    gateway = FakeDiscord(brain)
    gateway.onboard(ALICE)

    first = gateway.ping(ALICE, "hello")[0]
    second = gateway.ping(ALICE, "how are you?", reply_to=first.id)[0]
    gateway.correct(ALICE, "better second answer", reply_to=second.id)

    assert {row.text for row in brain.corpus.all()} == {
        "hello",
        "how are you?",
        "better second answer",
    }
    (correction,) = brain.store.all()
    assert correction.prompt == "how are you?"
    assert correction.chosen == "better second answer"
    assert "assistant:" not in correction.prompt


def test_history_never_crosses_users(settings, generator, log):
    _enable(settings)
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)
    gateway.onboard(BOB)
    alice_reply = gateway.ping(ALICE, "alice topic")[0]

    gateway.ping(BOB, "bob jumps in", reply_to=alice_reply.id)

    assert generator.prompts[-1] == "user: bob jumps in"


def test_history_never_crosses_channels(settings, generator, log):
    _enable(settings)
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)
    reply = gateway.ping(ALICE, "one channel", channel="chan-1")[0]

    gateway.ping(ALICE, "another channel", reply_to=reply.id, channel="chan-2")

    assert generator.prompts[-1] == "user: another channel"


def test_a_marked_reply_remains_a_correction_not_a_conversation_turn(
    settings, generator, log
):
    _enable(settings)
    brain = Babble(settings, generator=generator, log=log)
    gateway = FakeDiscord(brain)
    gateway.onboard(ALICE)
    reply = gateway.ping(ALICE, "hello")[0]

    gateway.correct(ALICE, "say hi", reply_to=reply.id)

    assert generator.prompts == ["user: hello"]
    assert brain.store.all()[0].prompt == "hello"


def test_reply_chain_survives_a_restart(settings, generator, log):
    _enable(settings)
    before = Babble(settings, generator=generator, log=log)
    gateway = FakeDiscord(before)
    gateway.onboard(ALICE)
    first = gateway.ping(ALICE, "remember this")[0]

    after = Babble(settings, generator=generator, log=log)
    FakeDiscord(after).ping(ALICE, "after restart", reply_to=first.id)

    assert generator.prompts[-1] == (
        "user: remember this\nassistant: wug wug blorp\nuser: after restart"
    )


def test_legacy_exchange_records_load_but_do_not_cross_unknown_channel(settings):
    settings.ensure_dirs()
    settings.exchanges_path.write_text(
        json.dumps(
            {
                "old": {
                    "prompt": "old prompt",
                    "response": "old response",
                    "prompt_author_id": ALICE,
                }
            }
        ),
        encoding="utf-8",
    )

    log = ExchangeLog(settings.exchanges_path)

    assert log.get("old") == Exchange(
        prompt="old prompt", response="old response", prompt_author_id=ALICE
    )


def test_formatter_drops_oldest_whole_turns_before_truncating_current_message():
    history = (
        ConversationTurn("first", "one"),
        ConversationTurn("second", "two"),
        ConversationTurn("third", "three"),
    )
    expected = "user: third\nassistant: three\nuser: current"

    assert conversation_prompt(
        history,
        "current",
        max_turns=2,
        max_chars=len(expected),
    ) == expected

    assert conversation_prompt((), "0123456789", max_turns=6, max_chars=10) == "user: 6789"


def test_token_aware_formatter_never_left_truncates_through_a_role_boundary():
    history = (
        ConversationTurn("old question", "old answer"),
        ConversationTurn("recent question", "recent answer"),
    )

    prompt = conversation_prompt_for_token_budget(
        history,
        "0123456789",
        max_turns=6,
        max_chars=0,
        max_tokens=12,
        token_count=len,  # one-character toy tokenizer makes the boundary exact
    )

    assert prompt == "user: 456789"
    assert prompt.startswith("user: ")


def test_core_prefers_a_generators_token_aware_conversation_formatter(settings, log):
    _enable(settings)

    class TokenAwareGenerator:
        def __init__(self):
            self.prompts = []

        def conversation_prompt(
            self, history, current_user, *, max_turns, max_tokens, max_chars
        ):
            assert max_turns == settings.conversation_max_turns
            assert max_tokens == settings.conversation_max_tokens
            assert max_chars == settings.conversation_max_chars
            return "backend-fitted-prompt"

        def __call__(self, prompt):
            self.prompts.append(prompt)
            return "answer"

    generator = TokenAwareGenerator()
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)
    gateway.ping(ALICE, "hello")

    assert generator.prompts == ["backend-fitted-prompt"]


def test_checkpoint_backend_rejects_context_with_a_continuation_checkpoint(settings):
    from babble.generate import CheckpointGenerator

    settings.conversation_context = True
    settings.serve_layout = "continuation"
    generator = CheckpointGenerator(settings)

    try:
        generator.conversation_prompt(
            (), "hello", max_turns=6, max_tokens=512, max_chars=6_000
        )
    except ValueError as exc:
        assert "BABBLE_SERVE_LAYOUT=pair" in str(exc)
    else:
        raise AssertionError(
            "a continuation checkpoint cannot understand the transcript pair format"
        )


def test_checkpoint_backend_fits_whole_roles_before_its_pair_prompt_truncation(settings):
    from babble.generate import CheckpointGenerator, _serving_tokenizer

    settings.serve_layout = "pair"
    settings.max_new_tokens = 16
    generator = CheckpointGenerator(settings)
    history = tuple(
        ConversationTurn(f"question {i} " * 8, f"answer {i} " * 8) for i in range(4)
    )

    prompt = generator.conversation_prompt(
        history,
        "current question",
        max_turns=6,
        max_tokens=30,
        max_chars=6_000,
    )

    model = generator._model
    assert model is not None
    tokenizer = _serving_tokenizer(model)
    reserved = max(1, model.config.block_size // 4)
    budget = model.config.block_size - 3 - reserved
    assert len(tokenizer.encode(prompt)) <= min(budget, 30)
    assert prompt.startswith("user: ")
    assert prompt.endswith("user: current question")


def test_unconsented_messages_get_no_retained_conversation(settings, generator, log):
    _enable(settings)
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.ping(ALICE)  # notice
    gateway.decline(ALICE)

    first = gateway.ping(ALICE, "not retained")[0]
    gateway.ping(ALICE, "still not retained", reply_to=first.id)

    assert generator.prompts[-2:] == ["user: not retained", "user: still not retained"]
    assert not settings.exchanges_path.exists()


def test_legacy_corrections_only_consent_does_not_authorize_context_reuse(
    settings, generator, log
):
    _enable(settings)
    brain = Babble(settings, generator=generator, log=log)
    gateway = FakeDiscord(brain)
    brain.consent.grant(ALICE, SCOPE_CORRECTIONS)
    first = gateway.ping(ALICE, "kept only for a possible correction")[0]

    gateway.ping(ALICE, "do not reuse it", reply_to=first.id)

    assert generator.prompts[-1] == "user: do not reuse it"


def test_conversation_settings_are_explicit_environment_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("BABBLE_CONVERSATION_CONTEXT", "1")
    monkeypatch.setenv("BABBLE_CONVERSATION_MAX_TURNS", "3")
    monkeypatch.setenv("BABBLE_CONVERSATION_MAX_TOKENS", "512")
    monkeypatch.setenv("BABBLE_CONVERSATION_MAX_CHARS", "2048")

    settings = __import__("babble.config", fromlist=["Settings"]).Settings.from_env(root=tmp_path)

    assert settings.conversation_context is True
    assert settings.conversation_max_turns == 3
    assert settings.conversation_max_tokens == 512
    assert settings.conversation_max_chars == 2048


# --- chunked overflow trimming (prefix-cache friendly windows) ----------------


class _CountingFormatterGenerator:
    """A generator with a token-aware formatter over a 1-char toy tokenizer."""

    def __init__(self):
        self.prompts = []
        self.warmed = []

    def conversation_prompt(self, history, current_user, *, max_turns, max_tokens, max_chars, overflow_keep=1.0):
        return conversation_prompt_for_token_budget(
            history, current_user, max_turns=max_turns, max_chars=max_chars,
            max_tokens=max_tokens, token_count=len, overflow_keep=overflow_keep,
        )

    def prewarm(self, text):
        self.warmed.append(text)
        return {"tokens": len(text)}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "wug wug blorp"


def _chain(settings, log, generator, messages):
    gateway = FakeDiscord(Babble(settings, generator=generator, log=log))
    gateway.onboard(ALICE)
    last = None
    for text in messages:
        last = gateway.ping(ALICE, text, reply_to=last.id if last else None)[0]
    return gateway


def _capture_replies(brain):
    replies = []
    remember = brain.remember

    def wrapped(mid, reply):
        replies.append(reply)
        remember(mid, reply)

    brain.remember = wrapped
    return replies


def test_windowed_turns_matches_bounded_history_until_overflow():
    from babble.conversation import bounded_history, windowed_turns

    turns = tuple(ConversationTurn(f"q{i}", f"a{i}") for i in range(9))
    for n in range(10):
        assert windowed_turns(turns[:n], max_turns=8, overflow_keep=1.0) == bounded_history(turns[:n], max_turns=8)
        if n <= 8:
            assert windowed_turns(turns[:n], max_turns=8, overflow_keep=0.5) == turns[:n]
    assert windowed_turns(turns, max_turns=8, overflow_keep=0.5) == turns[-4:]
    # A one-turn window cannot drop "half a turn": it slides like before.
    assert windowed_turns(turns, max_turns=1, overflow_keep=0.5) == turns[-1:]
    assert windowed_turns(turns, max_turns=0, overflow_keep=0.5) == ()


def test_turn_overflow_trims_in_one_chunk_then_extends_a_stable_prefix(settings, log):
    from conftest import FakeGenerator

    _enable(settings)
    settings.conversation_max_turns = 4
    messages = [f"m{i}" for i in range(12)]

    settings.conversation_overflow_keep = 1.0
    sliding = FakeGenerator()
    _chain(settings, log, sliding, messages)
    settings.conversation_overflow_keep = 0.5
    chunked = FakeGenerator()
    _chain(settings, log, chunked, messages)
    chunked.prompts.remove("user: hello")  # ALICE is already onboarded the second time

    # Byte-identical to the sliding window until the window first overflows.
    assert chunked.prompts[:5] == sliding.prompts[:5]
    visible = [p.count("assistant: ") for p in chunked.prompts]
    assert visible == [0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2]
    # Every turn that is not a trim extends the previous prompt verbatim.
    for prev, cur, n in zip(chunked.prompts, chunked.prompts[1:], visible[1:]):
        if n != 2:
            assert cur.startswith(prev[: prev.rindex("user: ")])
    # The sliding window never does once it is full.
    assert not sliding.prompts[6].startswith(sliding.prompts[5][:12])


def test_token_overflow_trims_to_the_low_watermark_and_remembers_it(settings, log):
    _enable(settings)
    settings.conversation_max_turns = 50
    settings.conversation_max_chars = 0
    settings.conversation_max_tokens = 200
    settings.conversation_overflow_keep = 0.5
    gen = _CountingFormatterGenerator()
    _chain(settings, log, gen, [f"message number {i:02d}" for i in range(20)])

    lengths = [len(p) for p in gen.prompts]
    assert max(lengths) <= 200

    def extends(prev, cur):
        return cur.startswith(prev[: prev.rindex("user: ")])

    trims = [i for i in range(1, len(gen.prompts)) if not extends(gen.prompts[i - 1], gen.prompts[i])]
    assert trims, "the transcript should have overflowed"
    for i in trims:
        assert lengths[i] <= 100  # trimmed to the low watermark in one step
    # A trim buys several turns of a stable, growing prefix.
    assert all(b - a >= 3 for a, b in zip(trims, trims[1:]))


def test_token_trim_does_not_starve_history_for_a_huge_current_message():
    history = tuple(ConversationTurn("q" * 5, "a" * 5) for _ in range(6))
    current = "x" * 45  # alone (51 tokens) it misses the 50-token floor
    prompt = conversation_prompt_for_token_budget(
        history, current, max_turns=10, max_chars=0, max_tokens=100, token_count=len, overflow_keep=0.5,
    )
    sliding = conversation_prompt_for_token_budget(
        history, current, max_turns=10, max_chars=0, max_tokens=100, token_count=len,
    )
    assert prompt == sliding and "assistant: " in prompt


def test_char_formatter_overflow_keep_matches_until_overflow():
    history = tuple(ConversationTurn(f"q{i}", f"a{i}") for i in range(6))
    full = conversation_prompt(history, "now", max_turns=10, max_chars=0)
    for keep in (1.0, 0.5):
        assert conversation_prompt(history, "now", max_turns=10, max_chars=len(full), overflow_keep=keep) == full
    trimmed = conversation_prompt(history, "now", max_turns=10, max_chars=len(full) - 1, overflow_keep=0.5)
    assert len(trimmed) <= (len(full) - 1) // 2


def test_prewarm_text_is_the_exact_prefix_of_the_next_prompt(settings, log):
    _enable(settings)
    settings.conversation_max_turns = 3
    settings.conversation_max_tokens = 10_000  # far from the cap: one warm per reply
    settings.conversation_overflow_keep = 0.5
    gen = _CountingFormatterGenerator()
    brain = Babble(settings, generator=gen, log=log)
    gateway = FakeDiscord(brain)
    gateway.onboard(ALICE)
    replies = _capture_replies(brain)

    last = None
    for i in range(8):
        last = gateway.ping(ALICE, f"turn {i}", reply_to=last.id if last else None)[0]
        reply = replies[-1]
        assert reply.continues
        assert brain.prewarm(reply) == {"tokens": len(gen.warmed[-1])}
        if i:
            # what was warmed after the previous reply is how this prompt starts
            assert gen.prompts[-1] == gen.warmed[-2] + f"turn {i}"


def test_prewarm_near_the_token_cap_also_warms_the_trimmed_window(settings, log):
    from babble.core import PREWARM_PROBE

    _enable(settings)
    settings.conversation_max_turns = 50
    settings.conversation_max_chars = 0
    settings.conversation_max_tokens = 2000
    settings.conversation_overflow_keep = 0.5
    gen = _CountingFormatterGenerator()
    brain = Babble(settings, generator=gen, log=log)
    gateway = FakeDiscord(brain)
    gateway.onboard(ALICE)
    replies = _capture_replies(brain)

    last, alts = None, 0
    for i in range(60):
        # every fifth message is long enough to force a trim on its own
        text = f"turn {i}" + (" " + "x" * 400 if i % 5 == 4 else "")
        if i:
            warmed = list(gen.warmed[-2:]) if "alt" in info else [gen.warmed[-1]]
        last = gateway.ping(ALICE, text, reply_to=last.id if last else None)[0]
        if i:
            # whichever way the window went, one of the warms is its prefix
            assert any(gen.prompts[-1] == w + text for w in warmed), i
        info = brain.prewarm(replies[-1])
        if "alt" in info:
            alts += 1
            assert len(gen.warmed[-1]) < len(gen.warmed[-2])
    trims = sum(not b.startswith(a[: a.rindex("user: ")]) for a, b in zip(gen.prompts, gen.prompts[1:]))
    assert alts and trims >= 2 and len(PREWARM_PROBE) > 300


def test_prewarm_is_skipped_without_retained_context(settings, log):
    _enable(settings)
    gen = _CountingFormatterGenerator()
    brain = Babble(settings, generator=gen, log=log)
    gateway = FakeDiscord(brain)
    brain.consent.grant(ALICE, SCOPE_CORRECTIONS)  # legacy grant: no context reuse
    replies = _capture_replies(brain)
    gateway.ping(ALICE, "hello")
    assert replies and not replies[-1].continues
    assert brain.prewarm(replies[-1]) is None and gen.warmed == []


def test_overflow_keep_is_an_environment_setting(monkeypatch, tmp_path):
    Settings = __import__("babble.config", fromlist=["Settings"]).Settings
    assert Settings.from_env(root=tmp_path).conversation_overflow_keep == 0.5
    monkeypatch.setenv("BABBLE_CONVERSATION_OVERFLOW_KEEP", "1")
    assert Settings.from_env(root=tmp_path).conversation_overflow_keep == 1.0
