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
