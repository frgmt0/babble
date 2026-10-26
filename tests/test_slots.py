"""The hidden slot machine: odds, pity, cooldown, cap, consent, and cashing out.

No network anywhere. The reels are exercised through the pure digest functions
and a `SlotMachine` with an injected secret and clock; the Claude call is a fake
oracle in the brain tests and a mocked `anthropic` client in the oracle tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from babble.blocklist import Blocklist
from babble.bot import BabbleClient
from babble.consent import (
    CLAUDE_NOTICE_VERSION,
    NOTICE_VERSION,
    SCOPE_CORPUS,
    SCOPE_CORRECTIONS,
    ConsentStore,
)
from babble.core import (
    CONSENT_NOTICE,
    CORPUS_NOTICE,
    EMPTY_REPLY,
    HELP_TEXT,
    Babble,
    IncomingMessage,
    Reply,
)
from babble.exchanges import Exchange, ExchangeLog
from babble.oracle import FALLBACK_BETA, REPLY_CHAR_BUDGET, Oracle, OracleResult, build_system_prompt
from babble.slots import (
    JACKPOT_ODDS,
    OBSESSIONS,
    STRIP,
    VOID_COOLDOWN,
    VOID_DAILY_CAP,
    WEIGHTS,
    JackpotRequest,
    SlotMachine,
    SpinResult,
    digest_for,
    reels_from_digest,
)
from conftest import FakeDiscord, FakeGenerator

ALICE = "111111111111111111"
BOB = "222222222222222222"
SECRET = "fixed-secret-for-tests"


class Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


def machine(tmp_path, **kw) -> SlotMachine:
    kw.setdefault("secret", SECRET)
    kw.setdefault("pity", 0)
    kw.setdefault("cooldown_minutes", 0)
    kw.setdefault("daily_cap", 1000)
    return SlotMachine(tmp_path / "slots.json", **kw)


# --- the reels ------------------------------------------------------------


def test_the_strip_has_100_stops_matching_the_weights():
    assert len(STRIP) == 100
    counts = Counter(STRIP)
    assert dict(counts) == dict(WEIGHTS)
    assert set(OBSESSIONS) == set(counts)
    assert all(len(v) >= 4 for v in OBSESSIONS.values())
    assert JACKPOT_ODDS == pytest.approx(0.05299, abs=1e-5)


def test_jackpot_rate_and_symbol_frequencies_track_the_weights():
    n = 60_000
    jackpots = 0
    symbols = Counter()
    for i in range(n):
        reels = reels_from_digest(digest_for(SECRET, f"u_{i % 97}", i, f"hello {i}"))
        symbols.update(reels)
        jackpots += reels[0] == reels[1] == reels[2]
    rate = jackpots / n
    # ~5.3% expected; sigma at n=60k is ~0.09 percentage points.
    assert 0.047 < rate < 0.059
    total = sum(symbols.values())
    for symbol, weight in WEIGHTS:
        assert symbols[symbol] / total == pytest.approx(weight / 100, abs=0.01)


def test_rolls_are_deterministic_given_a_fixed_secret(tmp_path):
    a = SlotMachine(tmp_path / "a.json", secret=SECRET, pity=0)
    b = SlotMachine(tmp_path / "b.json", secret=SECRET, pity=0)
    other = SlotMachine(tmp_path / "c.json", secret="another", pity=0)
    prompts = [f"message {i}" for i in range(40)]
    seq_a = [a.roll("u_x", p).symbols for p in prompts]
    seq_b = [b.roll("u_x", p).symbols for p in prompts]
    seq_c = [other.roll("u_x", p).symbols for p in prompts]
    assert seq_a == seq_b
    assert seq_a != seq_c


def test_a_roll_matches_the_pure_digest_and_depends_on_the_message(tmp_path):
    m = machine(tmp_path)
    expected = reels_from_digest(digest_for(SECRET, "u_x", 0, "hi"))
    assert m.roll("u_x", "hi").symbols == expected
    # Same message again is a different spin (spin count is in the seed).
    assert m.spins("u_x") == 1


def test_the_default_secret_is_random_per_machine(tmp_path):
    a = SlotMachine(tmp_path / "a.json", pity=0)
    b = SlotMachine(tmp_path / "b.json", pity=0)
    prompts = [f"m{i}" for i in range(30)]
    assert [a.roll("u", p).symbols for p in prompts] != [b.roll("u", p).symbols for p in prompts]


def test_a_natural_jackpot_carries_an_obsession_for_its_symbol(tmp_path):
    m = machine(tmp_path)
    for i in range(2000):
        spin = m.roll("u_x", f"p{i}")
        if spin.jackpot:
            assert not spin.forced_by_pity
            assert spin.obsession in OBSESSIONS[spin.symbols[0]]
            assert spin.paid
            return
        assert spin.obsession is None
    pytest.fail("no natural jackpot in 2000 rolls")


# --- pity, cooldown, cap ----------------------------------------------------


def test_pity_forces_a_jackpot_exactly_at_the_configured_streak(tmp_path):
    pity = 5
    m = machine(tmp_path, pity=pity)
    streak = 0
    forced = 0
    for i in range(400):
        spin = m.roll("u_x", f"p{i}")
        if streak == pity - 1:
            assert spin.jackpot, f"roll {i} should have been forced"
        else:
            assert not spin.forced_by_pity, f"roll {i} forced early at streak {streak}"
        if spin.forced_by_pity:
            forced += 1
            assert len(set(spin.symbols)) == 1
            assert spin.obsession in OBSESSIONS[spin.symbols[0]]
        streak = 0 if spin.jackpot else streak + 1
        assert m.miss_streak("u_x") == streak
    assert forced > 0


def test_cooldown_voids_a_jackpot_without_counting_it_as_a_miss(tmp_path):
    clock = Clock()
    m = machine(tmp_path, pity=1, cooldown_minutes=30, clock=clock)  # pity=1: every roll pays

    first = m.roll("u_x", "a")
    assert first.paid
    clock.advance(minutes=10)
    second = m.roll("u_x", "b")
    assert second.jackpot and second.voided == VOID_COOLDOWN and not second.paid
    assert m.miss_streak("u_x") == 0
    # Somebody else is not on that person's cooldown.
    assert m.roll("u_y", "c").paid
    clock.advance(minutes=21)
    assert m.roll("u_x", "d").paid


def test_daily_cap_voids_and_resets_on_a_new_utc_day(tmp_path):
    clock = Clock(datetime(2026, 10, 25, 23, 0, tzinfo=timezone.utc))
    m = machine(tmp_path, pity=1, daily_cap=2, clock=clock)

    assert m.roll("u_a", "x").paid
    assert m.roll("u_b", "x").paid
    capped = m.roll("u_c", "x")
    assert capped.jackpot and capped.voided == VOID_DAILY_CAP
    assert m.miss_streak("u_c") == 0
    assert m.payouts_today == 2

    clock.advance(hours=1, minutes=1)  # past midnight UTC
    assert m.payouts_today == 0
    assert m.roll("u_c", "x").paid


def test_a_zero_daily_cap_pays_nothing(tmp_path):
    m = machine(tmp_path, pity=1, daily_cap=0)
    spin = m.roll("u_a", "x")
    assert spin.jackpot and spin.voided == VOID_DAILY_CAP


# --- persistence ------------------------------------------------------------


def test_state_persists_and_reloads(tmp_path):
    clock = Clock()
    m = machine(tmp_path, pity=3, clock=clock)
    for i in range(7):
        m.roll("u_x", f"p{i}")
    again = machine(tmp_path, pity=3, clock=clock)
    assert again.spins("u_x") == 7
    assert again.miss_streak("u_x") == m.miss_streak("u_x")
    assert again.payouts_today == m.payouts_today
    raw = json.loads((tmp_path / "slots.json").read_text())
    assert set(raw) == {"day", "payouts_today", "users"}
    assert list(raw["users"]) == ["u_x"]


def test_a_corrupt_state_file_starts_fresh(tmp_path):
    (tmp_path / "slots.json").write_text("{nope")
    m = machine(tmp_path)
    assert m.spins("u_x") == 0
    m.roll("u_x", "hi")
    assert m.spins("u_x") == 1


def test_forget_removes_a_user(tmp_path):
    m = machine(tmp_path)
    m.roll("u_x", "hi")
    m.roll("u_y", "hi")
    assert m.forget("u_x") is True
    assert m.forget("u_x") is False
    reloaded = machine(tmp_path)
    assert not reloaded.knows("u_x")
    assert reloaded.knows("u_y")


# --- the brain: eligibility, attaching, forgetting ----------------------------


class FakeOracle:
    model = "fake-fable"

    def __init__(self, result: OracleResult | None = None, exc: Exception | None = None) -> None:
        self.result = result or OracleResult(ok=True, text="the frogs have set rates at 4%", model="fake-fable")
        self.exc = exc
        self.requests: list[JackpotRequest] = []

    def ask(self, request: JackpotRequest) -> OracleResult:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return self.result


class StubSlots:
    """A machine whose next spin is whatever the test says."""

    def __init__(self, spin: SpinResult) -> None:
        self.spin = spin
        self.rolled: list[tuple[str, str]] = []
        self.forgotten: list[str] = []

    def roll(self, user: str, prompt: str) -> SpinResult:
        self.rolled.append((user, prompt))
        return self.spin

    def forget(self, user: str) -> bool:
        self.forgotten.append(user)
        return True


@pytest.fixture
def slots_settings(settings):
    settings.slots_enabled = True
    settings.slots_voice_examples = 20
    return settings


@pytest.fixture
def jackpot_machine(slots_settings):
    # pity=1: every eligible roll is a paid jackpot.
    return SlotMachine(
        slots_settings.slots_path, pity=1, daily_cap=1000, cooldown_minutes=0, secret=SECRET
    )


@pytest.fixture
def oracle():
    return FakeOracle()


@pytest.fixture
def sbrain(slots_settings, generator, log, jackpot_machine, oracle):
    return Babble(
        slots_settings,
        generator=generator,
        log=log,
        bot_user_id="bot-9999",
        slots=jackpot_machine,
        oracle=oracle,
    )


def _ping(brain: Babble, user: str, text: str = "hello", mid: str = "m1") -> list[Reply]:
    return brain.handle_message(
        IncomingMessage(message_id=mid, author_id=user, content=text, channel_id="chan-1", mentions_bot=True)
    )


def _write_consent(path, user, *, corpus="granted", corrections="granted", version=2, with_version=True):
    def rec(decision):
        entry = {"decision": decision, "updated_at": "2026-01-01T00:00:00+00:00"}
        if with_version:
            entry["notice_version"] = version
        return entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({user: {"scopes": {"corpus": rec(corpus), "corrections": rec(corrections)}}}))


def test_a_v3_grantee_rolls_and_gets_a_jackpot_request(sbrain, jackpot_machine, read_log):
    fake = FakeDiscord(sbrain)
    fake.onboard(ALICE)
    assert sbrain.consent.notice_version(ALICE, SCOPE_CORPUS) == NOTICE_VERSION == CLAUDE_NOTICE_VERSION

    (reply,) = _ping(sbrain, ALICE, "what is the meaning of soup")

    assert reply.kind == "generation"
    assert reply.jackpot is not None
    assert reply.jackpot.prompt == "what is the meaning of soup"
    assert reply.jackpot.obsession in OBSESSIONS[reply.jackpot.symbol]
    assert jackpot_machine.knows(sbrain.ids.user(ALICE))
    (roll,) = read_log("slots.roll")
    assert roll["user"] == sbrain.ids.user(ALICE)
    assert roll["jackpot"] is True
    assert "soup" not in json.dumps(roll)


@pytest.mark.parametrize("how", ["unknown", "pending", "declined", "withdrawn"])
def test_ineligible_people_never_roll_and_leave_no_state(sbrain, slots_settings, read_log, how):
    fake = FakeDiscord(sbrain)
    if how == "pending":
        fake.ping(ALICE)  # notice shown, never answered
    elif how == "declined":
        fake.ping(ALICE)
        fake.decline(ALICE)
    elif how == "withdrawn":
        fake.onboard(ALICE)
        fake.say(ALICE, "!babble forget")

    for i in range(5):
        for reply in _ping(sbrain, ALICE, f"hi {i}", mid=f"m{i}"):
            assert reply.jackpot is None

    assert read_log("slots.roll") == []
    assert not slots_settings.slots_path.exists()


def test_a_v2_grantee_keeps_their_consent_but_never_rolls(slots_settings, generator, log, jackpot_machine, oracle, read_log):
    _write_consent(slots_settings.consent_path, ALICE, version=2)
    before = slots_settings.consent_path.read_text()
    brain = Babble(slots_settings, generator=generator, log=log, slots=jackpot_machine, oracle=oracle)

    replies = _ping(brain, ALICE, "hello")

    # No re-prompt: the v3 bump did not invalidate their grant.
    assert [r.kind for r in replies] == ["generation"]
    assert replies[0].jackpot is None
    assert replies[0].exchange is not None  # still gradeable, as before
    assert read_log("consent.prompt") == []
    assert read_log("slots.roll") == []
    assert not slots_settings.slots_path.exists()
    # Their stored record is untouched -- still v2, still granted.
    assert json.loads(slots_settings.consent_path.read_text()) == json.loads(before)
    # And their message was still captured to the corpus, exactly as before.
    assert brain.corpus.totals().rows == 1


def test_a_v2_grantee_becomes_eligible_by_accepting_again(slots_settings, generator, log, jackpot_machine, oracle):
    _write_consent(slots_settings.consent_path, ALICE, version=2)
    brain = Babble(slots_settings, generator=generator, log=log, slots=jackpot_machine, oracle=oracle)
    FakeDiscord(brain).accept(ALICE)

    assert brain.consent.notice_version(ALICE, SCOPE_CORPUS) == 3
    assert _ping(brain, ALICE, "hello")[0].jackpot is not None


def test_a_scoped_record_without_a_version_loads_as_v2_not_current(slots_settings):
    _write_consent(slots_settings.consent_path, ALICE, with_version=False)
    store = ConsentStore(slots_settings.consent_path)
    assert store.notice_version(ALICE, SCOPE_CORPUS) == 2
    assert store.notice_version(ALICE, SCOPE_CORRECTIONS) == 2


def test_corpus_only_grant_is_not_enough(slots_settings, generator, log, jackpot_machine, oracle):
    _write_consent(slots_settings.consent_path, ALICE, corrections="declined", version=3)
    brain = Babble(slots_settings, generator=generator, log=log, slots=jackpot_machine, oracle=oracle)
    assert _ping(brain, ALICE)[0].jackpot is None


def test_feature_is_inert_when_disabled_or_missing_a_collaborator(settings, generator, log, jackpot_machine, oracle):
    fake_settings = settings
    fake_settings.slots_enabled = False
    brain = Babble(fake_settings, generator=generator, log=log, slots=jackpot_machine, oracle=oracle)
    FakeDiscord(brain).onboard(ALICE)
    assert _ping(brain, ALICE)[0].jackpot is None

    fake_settings.slots_enabled = True
    no_oracle = Babble(fake_settings, generator=generator, log=log, slots=jackpot_machine, oracle=None)
    assert _ping(no_oracle, ALICE)[0].jackpot is None
    assert not fake_settings.slots_path.exists()


def test_only_paid_jackpots_attach_a_request(slots_settings, generator, log, oracle):
    miss = StubSlots(SpinResult(symbols=("🍒", "🍋", "🐸"), jackpot=False))
    voided = StubSlots(SpinResult(symbols=("🐸",) * 3, jackpot=True, voided=VOID_DAILY_CAP, obsession="x"))
    paid = StubSlots(SpinResult(symbols=("🐸",) * 3, jackpot=True, obsession="frog-based macroeconomics"))
    for stub, expect in ((miss, False), (voided, False), (paid, True)):
        brain = Babble(slots_settings, generator=generator, log=log, slots=stub, oracle=oracle)
        FakeDiscord(brain).onboard(ALICE)
        reply = _ping(brain, ALICE, "hi")[0]
        assert (reply.jackpot is not None) is expect
        assert stub.rolled[-1][1] == "hi"  # consent is shared across loop brains
    assert reply.jackpot.symbols == ("🐸",) * 3
    assert reply.jackpot.obsession == "frog-based macroeconomics"


def test_a_blocklisted_prompt_never_rolls(slots_settings, generator, log, oracle):
    stub = StubSlots(SpinResult(symbols=("🐸",) * 3, jackpot=True, obsession="x"))
    brain = Babble(
        slots_settings, generator=generator, log=log, slots=stub, oracle=oracle,
        blocklist=Blocklist(frozenset({"badword"})),
    )
    FakeDiscord(brain).onboard(ALICE)
    assert _ping(brain, ALICE, "say badword")[0].jackpot is None
    assert stub.rolled == []


def test_voice_examples_are_only_the_bots_own_responses(sbrain, generator):
    fake = FakeDiscord(sbrain)
    fake.onboard(ALICE)
    fake.onboard(BOB)
    sent = []
    for i, (user, text) in enumerate([(ALICE, "PROMPT-ALPHA"), (BOB, "PROMPT-BRAVO"), (ALICE, "PROMPT-CHARLIE")]):
        generator.text = f"booper says {i}"
        fake.ping(user, text)
        sent.append(generator.text)

    generator.text = "latest"
    request = _ping(sbrain, ALICE, "PROMPT-DELTA", mid="m-final")[0].jackpot

    assert request is not None
    assert list(request.voice_examples) == sent  # oldest first, newest last
    blob = json.dumps([request.voice_examples, request.obsession, request.symbol])
    for secret in ("PROMPT-ALPHA", "PROMPT-BRAVO", "PROMPT-CHARLIE", ALICE, BOB):
        assert secret not in blob


def test_voice_examples_respect_the_configured_count(sbrain, slots_settings, generator):
    slots_settings.slots_voice_examples = 2
    fake = FakeDiscord(sbrain)
    fake.onboard(ALICE)
    for i in range(5):
        generator.text = f"reply {i}"
        fake.ping(ALICE, f"q{i}")
    request = _ping(sbrain, ALICE, "go", mid="m-final")[0].jackpot
    assert request.voice_examples == ("reply 3", "reply 4")


def test_recent_responses_skips_placeholders_and_empties(tmp_path):
    log = ExchangeLog(tmp_path / "ex.json")
    for i, response in enumerate(["one", "", EMPTY_REPLY, "two", "  three  "]):
        log.record(f"b{i}", Exchange(prompt=f"secret {i}", response=response, prompt_author_id=ALICE))
    assert log.recent_responses(10, exclude=(EMPTY_REPLY,)) == ["one", "two", "three"]
    assert log.recent_responses(2, exclude=(EMPTY_REPLY,)) == ["two", "three"]
    assert log.recent_responses(0) == []


def test_babble_forget_clears_slots_state(sbrain, jackpot_machine):
    fake = FakeDiscord(sbrain)
    fake.onboard(ALICE)
    _ping(sbrain, ALICE)
    pseudonym = sbrain.ids.user(ALICE)
    assert jackpot_machine.knows(pseudonym)

    fake.say(ALICE, "!babble forget")

    assert not jackpot_machine.knows(pseudonym)
    assert pseudonym not in sbrain.settings.slots_path.read_text()


def test_forget_clears_slots_even_with_the_feature_off(settings, generator, log, jackpot_machine):
    jackpot_machine.roll("u_whatever", "x")
    settings.slots_enabled = False
    brain = Babble(settings, generator=generator, log=log, slots=jackpot_machine, oracle=None)
    jackpot_machine.roll(brain.ids.user(ALICE), "x")
    FakeDiscord(brain).say(ALICE, "!babble forget")
    assert not jackpot_machine.knows(brain.ids.user(ALICE))


# --- cash_out ---------------------------------------------------------------


def _jackpot_reply(brain) -> Reply:
    FakeDiscord(brain).onboard(ALICE)
    reply = _ping(brain, ALICE, "should i eat soup", mid="m-jp")[0]
    assert reply.jackpot is not None and reply.exchange is not None
    return reply


def test_cash_out_success_swaps_in_the_claude_text(sbrain, oracle, read_log):
    reply = _jackpot_reply(sbrain)

    out = sbrain.cash_out(reply)

    assert out.content.startswith("the frogs have set rates at 4%")
    assert f"-# 🎰 {''.join(reply.jackpot.symbols)} · {reply.jackpot.obsession}" in out.content
    assert out.exchange is None and out.continues is False and out.jackpot is None
    assert out.reply_to == reply.reply_to and out.kind == "generation"
    assert oracle.requests == [reply.jackpot]
    (event,) = read_log("slots.cashout")
    assert event["status"] == "ok" and event["model"] == "fake-fable"
    assert "soup" not in json.dumps(event)


def test_cash_out_success_is_never_remembered_or_captured(sbrain):
    reply = _jackpot_reply(sbrain)
    rows_before = sbrain.corpus.totals().rows
    out = sbrain.cash_out(reply)
    sbrain.remember("bot-msg-1", out)
    assert sbrain.exchanges.get("bot-msg-1") is None
    assert sbrain.corpus.totals().rows == rows_before
    assert "frogs" not in sbrain.settings.corpus_path.read_text()


def test_cash_out_footer_can_be_turned_off(sbrain, slots_settings):
    reply = _jackpot_reply(sbrain)
    slots_settings.slots_footer = False
    out = sbrain.cash_out(reply)
    assert out.content == "the frogs have set rates at 4%"


@pytest.mark.parametrize(
    "oracle_kwargs, reason",
    [
        ({"result": OracleResult(ok=False, reason="refusal")}, "refusal"),
        ({"result": OracleResult(ok=False, reason="timeout")}, "timeout"),
        ({"result": OracleResult(ok=True, text="   \x00  ")}, "empty"),
        ({"result": OracleResult(ok=True, text="you badword")}, "blocked"),
        ({"exc": RuntimeError("boom")}, "error:RuntimeError"),
    ],
)
def test_cash_out_failures_return_the_local_reply_unchanged(
    slots_settings, generator, log, jackpot_machine, read_log, oracle_kwargs, reason
):
    brain = Babble(
        slots_settings, generator=generator, log=log, slots=jackpot_machine,
        oracle=FakeOracle(**oracle_kwargs), blocklist=Blocklist(frozenset({"badword"})),
    )
    reply = _jackpot_reply(brain)

    assert brain.cash_out(reply) is reply
    (event,) = read_log("slots.cashout")
    assert event["status"] == "fallback" and event["reason"] == reason


def test_cash_out_without_a_jackpot_is_a_no_op(sbrain, oracle):
    reply = Reply("plain")
    assert sbrain.cash_out(reply) is reply
    assert oracle.requests == []


def test_cash_out_keeps_the_reply_within_discords_limit(slots_settings, generator, log, jackpot_machine):
    brain = Babble(
        slots_settings, generator=generator, log=log, slots=jackpot_machine,
        oracle=FakeOracle(OracleResult(ok=True, text="frog " * 1000)),
    )
    out = brain.cash_out(_jackpot_reply(brain))
    assert len(out.content) <= 2000
    assert "-# 🎰" in out.content


# --- disclosure copy --------------------------------------------------------


def test_every_route_to_accept_discloses_claude():
    for text in (CONSENT_NOTICE, CORPUS_NOTICE, HELP_TEXT):
        assert "claude" in text and "anthropic" in text


def test_accept_confirmation_discloses_claude_and_the_re_accept(sbrain):
    reply = FakeDiscord(sbrain).accept(ALICE)[0]
    assert "claude" in reply.content and "anthropic" in reply.content
    assert "accepting again" in reply.content


# --- the oracle -------------------------------------------------------------


class _Stream:
    def __init__(self, message=None, exc=None) -> None:
        self.message, self.exc = message, exc

    def __enter__(self):
        if self.exc is not None:
            raise self.exc
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeAnthropicClient:
    def __init__(self, message=None, exc=None) -> None:
        self.calls: list[dict] = []
        outer = self

        class _Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return _Stream(message, exc)

        self.beta = SimpleNamespace(messages=_Messages())


def _message(stop_reason="end_turn", texts=("hi there",)):
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        model="claude-fable-5-1",
        usage=SimpleNamespace(input_tokens=120, output_tokens=4000),
    )


REQUEST = JackpotRequest(
    prompt="is soup a drink",
    symbol="💎",
    obsession="whether soup is a beverage, as a 4-hour treatise",
    symbols=("💎", "💎", "💎"),
    voice_examples=("boop", "beep\nbop"),
)


def test_oracle_sends_exactly_the_documented_request():
    client = FakeAnthropicClient(_message())
    result = Oracle(client, model="claude-fable-5-1", effort="max").ask(REQUEST)

    assert result.ok and result.text == "hi there"
    assert result.model == "claude-fable-5-1"
    assert (result.input_tokens, result.output_tokens) == (120, 4000)
    (kwargs,) = client.calls
    assert kwargs["model"] == "claude-fable-5-1"
    assert kwargs["max_tokens"] == 64000
    assert kwargs["output_config"] == {"effort": "max"}
    assert kwargs["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["messages"] == [{"role": "user", "content": "is soup a drink"}]
    for banned in ("thinking", "temperature", "top_p", "top_k"):
        assert banned not in kwargs
    assert kwargs["system"] == build_system_prompt(REQUEST.obsession, REQUEST.voice_examples)
    assert "<voice_examples>" in kwargs["system"] and "- beep bop" in kwargs["system"]
    assert REQUEST.obsession in kwargs["system"]
    assert set(kwargs) == {"model", "max_tokens", "system", "messages", "output_config", "betas", "fallbacks"}


def test_system_prompt_asks_for_an_existential_crisis_and_never_holds_the_user_message():
    prompt = build_system_prompt(REQUEST.obsession, REQUEST.voice_examples)
    assert REPLY_CHAR_BUDGET == 1900
    assert "existential crisis" in prompt
    assert "1,900" in prompt
    assert "one to three sentences" not in prompt
    assert f"Today's obsession: {REQUEST.obsession}" in prompt
    assert "<voice_examples>\n- boop\n- beep bop\n</voice_examples>" in prompt
    assert "style only, not context" in prompt
    assert REQUEST.prompt not in prompt
    # the prompt is stable apart from the examples and the obsession
    assert build_system_prompt("x", ()).split("Today's")[0] == build_system_prompt("y", ()).split("Today's")[0]


def test_oracle_joins_only_text_blocks():
    client = FakeAnthropicClient(_message(texts=("one ", "two")))
    assert Oracle(client, model="m", effort="max").ask(REQUEST).text == "one two"


def test_oracle_treats_a_refusal_as_a_failure_before_reading_content():
    client = FakeAnthropicClient(_message(stop_reason="refusal", texts=("partial",)))
    result = Oracle(client, model="m", effort="max").ask(REQUEST)
    assert not result.ok and result.reason == "refusal" and result.text == ""


def test_oracle_treats_empty_text_as_a_failure():
    client = FakeAnthropicClient(_message(texts=()))
    assert Oracle(client, model="m", effort="max").ask(REQUEST).reason == "empty"


_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "exc, reason",
    [
        (anthropic.APITimeoutError(request=_REQ), "timeout"),
        (anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=_REQ), body=None), "rate_limited"),
        (anthropic.InternalServerError("oops", response=httpx2.Response(500, request=_REQ), body=None), "http_500"),
        (anthropic.APIConnectionError(request=_REQ), "connection"),
    ],
)
def test_oracle_converts_sdk_errors_to_failures(exc, reason):
    client = FakeAnthropicClient(exc=exc)
    result = Oracle(client, model="m", effort="max").ask(REQUEST)
    assert not result.ok and result.reason == reason


def test_from_settings_is_none_without_a_key(settings, log, monkeypatch, read_log):
    settings.slots_enabled = True
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Oracle.from_settings(settings, log) is None
    (event,) = read_log("slots.disabled")
    assert event["reason"] == "no_api_key"


def test_from_settings_is_none_when_disabled(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    settings.slots_enabled = False
    assert Oracle.from_settings(settings) is None


def test_from_settings_is_none_without_the_sdk(settings, log, monkeypatch, read_log):
    settings.slots_enabled = True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import raises ImportError
    assert Oracle.from_settings(settings, log) is None
    (event,) = read_log("slots.disabled")
    assert event["reason"] == "anthropic_sdk_missing"


def test_from_settings_builds_a_client_with_the_configured_timeout(settings, monkeypatch):
    settings.slots_enabled = True
    settings.slots_timeout_seconds = 123.0
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    oracle = Oracle.from_settings(settings)
    assert oracle is not None
    assert oracle.model == "claude-fable-5-1" and oracle.effort == "max"
    assert oracle.client.timeout == 123.0
    assert oracle.client.max_retries == 0


def test_settings_read_the_slots_env(monkeypatch, tmp_path):
    from babble.config import Settings

    monkeypatch.setenv("BABBLE_SLOTS_ENABLED", "1")
    monkeypatch.setenv("BABBLE_SLOTS_PITY", "7")
    monkeypatch.setenv("BABBLE_SLOTS_DAILY_CAP", "0")
    monkeypatch.setenv("BABBLE_SLOTS_FOOTER", "0")
    s = Settings.from_env(tmp_path)
    assert s.slots_enabled and s.slots_pity == 7 and s.slots_daily_cap == 0 and not s.slots_footer
    assert s.slots_model == "claude-fable-5-1" and s.slots_effort == "max"
    assert s.slots_cooldown_minutes == 30 and s.slots_voice_examples == 20
    assert s.slots_timeout_seconds == 900.0
    assert s.slots_path == s.data_dir / "slots.json"


# --- bot.py: cashing out happens outside the lock ------------------------------


class _Typing:
    def __init__(self, log: list, fail: bool = False) -> None:
        self.log, self.fail = log, fail

    async def __aenter__(self):
        if self.fail:
            raise RuntimeError("typing broke")
        self.log.append("enter")

    async def __aexit__(self, *exc):
        self.log.append("exit")


class _Channel:
    id = 4242

    def __init__(self, fail_typing: bool = False) -> None:
        self.typing_log: list[str] = []
        self.fail_typing = fail_typing

    def typing(self):
        return _Typing(self.typing_log, self.fail_typing)


class _Author:
    id = 111
    bot = False


class _Sent:
    def __init__(self, id_: int) -> None:
        self.id = id_


class _Inbound:
    def __init__(self, channel: _Channel) -> None:
        self.id = 77
        self.author = _Author()
        self.content = "hello"
        self.channel = channel
        self.guild = None
        self.reference = None
        self.mentions = []
        self.role_mentions = []
        self.attachments = []
        self.replies: list[str] = []

    async def reply(self, content: str, mention_author: bool = False):
        self.replies.append(content)
        return _Sent(900 + len(self.replies))


def _bot_with_jackpot(settings, log, brain, cash_out):
    client = BabbleClient(settings, log, brain=brain)
    local = Reply("local reply", reply_to="77", kind="generation", jackpot=REQUEST)
    brain.handle_message = lambda incoming: [local]
    brain.cash_out = cash_out
    return client, local


def test_on_message_cashes_out_outside_the_lock(settings, log, brain):
    seen = []

    def cash_out(reply):
        seen.append(client._lock.locked())
        return Reply("claude reply", reply_to=reply.reply_to, kind=reply.kind)

    client, _ = _bot_with_jackpot(settings, log, brain, cash_out)
    channel = _Channel()
    inbound = _Inbound(channel)

    asyncio.run(client.on_message(inbound))

    assert seen == [False]
    assert inbound.replies == ["claude reply"]
    assert channel.typing_log == ["enter", "exit"]


def test_on_message_falls_back_to_the_local_reply_when_cash_out_raises(settings, log, brain, read_log):
    def cash_out(reply):
        raise RuntimeError("api on fire")

    client, local = _bot_with_jackpot(settings, log, brain, cash_out)
    inbound = _Inbound(_Channel(fail_typing=True))

    asyncio.run(client.on_message(inbound))

    assert inbound.replies == ["local reply"]
    (event,) = read_log("bot.error")
    assert event["where"] == "cash_out"


def test_on_message_without_a_jackpot_never_cashes_out(settings, log, brain):
    client = BabbleClient(settings, log, brain=brain)
    brain.handle_message = lambda incoming: [Reply("plain", reply_to="77", kind="generation")]
    brain.cash_out = lambda reply: pytest.fail("cash_out called without a jackpot")
    channel = _Channel()
    inbound = _Inbound(channel)

    asyncio.run(client.on_message(inbound))

    assert inbound.replies == ["plain"]
    assert channel.typing_log == []
