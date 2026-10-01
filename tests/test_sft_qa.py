"""qa-v1 SFT data sources: pure helpers only (no network, no model)."""

import json
import random
import re
from argparse import Namespace
from itertools import islice
from types import SimpleNamespace

import pytest

from sft.sft_longform import (
    PERSONA_FILE,
    SFTRecord,
    _data_extras,
    _persona_key,
    _source_gate,
    _tokenize_records,
    arith_groups,
    arith_pair,
    arith_problem,
    clean_short_answer,
    dolly_prompt,
    identity_clash,
    load_persona,
    oasst_tree_groups,
    persona_groups,
    persona_variants,
    render_example,
    short_answer_pair,
    wiki_lead_pair,
)

_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty"
).split()


def _parse_problem(prompt: str) -> tuple[int, str, int]:
    """Recover (a, op, b) from any generated phrasing."""
    text = prompt.lower()
    for i, word in sorted(enumerate(_WORDS), key=lambda p: -len(p[1])):
        text = re.sub(rf"\b{word}\b", str(i), text)
    nums = [int(n) for n in re.findall(r"\d+", text)]
    assert len(nums) == 2, prompt
    a, b = nums
    if "subtract" in text:  # "subtract b from a"
        return b, "-", a
    if "+" in text or "plus" in text or "add" in text:
        return a, "+", b
    if re.search(r"\d\s*-\s*\d", text) or "minus" in text or "take away" in text:
        return a, "-", b
    if "/" in text or "÷" in text or "divide" in text:
        return a, "/", b
    assert re.search(r"\*|×|x|times|multipl", text), prompt
    return a, "*", b


def _apply(a: int, op: str, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else -1}[op]


def test_arith_problems_are_correct_and_division_exact():
    rng = random.Random(3)
    for _ in range(5000):
        a, op, b, result = arith_problem(rng)
        if op == "/":
            assert b != 0 and a % b == 0
        if op == "-":
            assert result >= 0
        assert result == _apply(a, op, b)


def test_every_arith_reply_states_the_right_answer_for_its_prompt():
    for group in islice(arith_groups(seed=11), 2000):
        assert group and len({r.group_id for r in group}) == 1
        for record in group:
            a, op, b = _parse_problem(record.current_user)
            expected = _apply(a, op, b)
            # the answer is the last number in the reply ("7 + 5 = 12", "that's 12")
            assert int(re.findall(r"\d+", record.response)[-1]) == expected, (record.current_user, record.response)


def test_arith_stream_is_deterministic_per_seed_and_problems_unique():
    take = lambda seed: [(r.current_user, r.response) for g in islice(arith_groups(seed), 300) for r in g]
    assert take(11) == take(11)
    assert take(11) != take(12)
    gids = [g[0].group_id for g in islice(arith_groups(11), 3000)]
    assert len(gids) == len(set(gids)), "one group per problem: phrasings never straddle the split"


def test_arith_phrasings_vary():
    rng = random.Random(0)
    prompts = {arith_pair(7, "+", 5, 12, rng)[0] for _ in range(200)}
    replies = {arith_pair(7, "+", 5, 12, rng)[1] for _ in range(200)}
    assert len(prompts) > 20 and len(replies) > 6


def test_persona_file_loads_and_stays_in_character():
    pairs = load_persona(PERSONA_FILE)
    assert len(pairs) >= 140
    for prompt, response in pairs:
        assert prompt and response
        # booper never claims another assistant's identity or an invented creator
        assert not re.search(r"\bi(?:'m| am) (?:chatgpt|gpt|claude|gemini|siri|alexa|open ?assistant)\b", response, re.I)
        assert not re.search(r"\b(?:made|built|created|trained) by (?:openai|google|anthropic|meta)\b", response, re.I)
    creators = [r for p, r in pairs if re.search(r"who (made|built|created)", p)]
    assert creators and all("server" in r for r in creators)


def test_persona_validation_rejects_bad_rows(tmp_path):
    bad = tmp_path / "p.jsonl"
    bad.write_text(json.dumps({"prompt": "hi", "response": ""}) + "\n")
    with pytest.raises(ValueError):
        load_persona(bad)
    bad.write_text(json.dumps({"prompt": "hi", "response": "a", "x": 1}) + "\n")
    with pytest.raises(ValueError):
        load_persona(bad)
    bad.write_text("\n".join(json.dumps({"prompt": p, "response": "a"}) for p in ("Hi", "hi")) + "\n")
    with pytest.raises(ValueError):
        load_persona(bad)


def test_persona_groups_never_split_one_question():
    groups = list(persona_groups(PERSONA_FILE, seed=11))
    owner: dict[str, str] = {}
    for group in groups:
        assert len({r.group_id for r in group}) == 1
        for record in group:
            key = _persona_key(record.current_user)
            assert owner.setdefault(key, group[0].group_id) == group[0].group_id, record.current_user
    variants = persona_variants("who made you", seed=1)
    assert variants[0] == "who made you" and len(variants) == len(set(variants)) >= 3


@pytest.mark.parametrize(
    "question,answers,expect",
    [
        ("what is the capital of france", ["Paris"], "Paris"),
        ("who wrote hamlet", ["William Shakespeare"], "William Shakespeare"),
        ("names of the members", ["A", "B"], "A and B"),
        ("three things", ["A", "B", "C"], "A, B and C"),
    ],
)
def test_short_answer_reply_contains_the_answer(question, answers, expect):
    prompt, reply = short_answer_pair(question, answers, seed=0)
    assert expect in reply
    assert question in prompt.lower()
    assert short_answer_pair(question, answers, seed=0) == (prompt, reply)


def test_short_answer_rejects_unusable_answers():
    assert clean_short_answer(["a", "b", "c", "d"]) is None
    assert clean_short_answer(["x" * 61]) is None
    assert clean_short_answer([]) is None
    assert clean_short_answer(["November\xa06, 1986."]) == "November 6, 1986"
    assert clean_short_answer(["Paris", "paris"]) == "Paris"
    assert short_answer_pair("", ["Paris"]) is None


def test_short_answer_templates_vary_and_stay_grammatical():
    replies = [short_answer_pair(f"what is thing number {i}", ["Paris"], seed=0)[1] for i in range(300)]
    shapes = {r.replace("Paris", "{a}") for r in replies}
    assert len(shapes) >= 8
    assert max(replies.count(r) for r in set(replies)) < 0.35 * len(replies)
    for i in range(200):
        _, reply = short_answer_pair(f"when was treaty {i} signed", ["November 6, 1986"], seed=0)
        assert "in November" not in reply
    years = {short_answer_pair(f"when did event {i} happen", ["1950"], seed=0)[1] for i in range(200)}
    assert "in 1950" in years


def test_dolly_context_rides_in_the_user_turn():
    assert dolly_prompt(" When was X born? ", "X was born in 1981.") == "When was X born?\n\nX was born in 1981."
    assert dolly_prompt("Name a fish", "") == "Name a fish"


def _msg(mid, parent, role, text, rank=None, lang="en", **kw):
    return {
        "message_id": mid, "parent_id": parent, "message_tree_id": "t1", "role": role, "text": text,
        "lang": lang, "rank": rank, "deleted": False, "review_result": True, "synthetic": False, **kw,
    }


def test_oasst_tree_takes_best_reply_and_keeps_true_history():
    rows = [
        _msg("p1", None, "prompter", "what is a cat"),
        _msg("a1", "p1", "assistant", "a small furry animal", rank=0),
        _msg("a2", "p1", "assistant", "worse answer", rank=1),
        _msg("p2", "a1", "prompter", "do they purr"),
        _msg("a3", "p2", "assistant", "yes, most do", rank=0),
        _msg("p3", "a2", "prompter", "follow-up under a lower-ranked reply"),
        _msg("a4", "p3", "assistant", "best reply there", rank=0),
        _msg("p5", "a5", "prompter", "follow-up under a filtered reply"),
        _msg("a7", "p5", "assistant", "never used", rank=0),
        _msg("p4", "a1", "prompter", "are they cute"),
        _msg("a5", "p4", "assistant", "As an AI language model I cannot say", rank=0),
        _msg("a6", "p4", "assistant", "deleted one", rank=1, deleted=True),
    ]
    (group,) = oasst_tree_groups(rows)
    assert len({r.group_id for r in group}) == 1
    pairs = {(r.current_user, r.response, tuple((t.user, t.assistant) for t in r.history)) for r in group}
    assert pairs == {
        ("what is a cat", "a small furry animal", ()),
        ("do they purr", "yes, most do", (("what is a cat", "a small furry animal"),)),
        # the rank-1 reply is history only, never a target
        ("follow-up under a lower-ranked reply", "best reply there", (("what is a cat", "worse answer"),)),
    }
    assert all(r.response not in ("worse answer", "never used") for r in group)


def test_oasst_skips_non_english_and_reviewed_out():
    rows = [
        _msg("p1", None, "prompter", "hola", lang="es"),
        _msg("a1", "p1", "assistant", "hola!", rank=0, lang="es"),
        _msg("p2", None, "prompter", "hi", review_result=False),
        _msg("a2", "p2", "assistant", "hello", rank=0),
    ]
    assert oasst_tree_groups(rows) == []


def test_identity_filter():
    assert identity_clash("I am Open Assistant, a chatbot")
    assert identity_clash("As an AI, I can't")
    assert identity_clash("ChatGPT was released in 2022")
    assert not identity_clash("Paris is the capital of France.")


def test_wiki_lead_pair_filters_and_trims():
    text = "Air is the Earth's atmosphere. Air is a mixture of gases. It is clear. It has mass.\n\nMore text."
    prompt, answer = wiki_lead_pair("Air", text, seed=0)
    assert "Air" in prompt
    assert answer.startswith("Air is the Earth's atmosphere.") and "More text" not in answer
    assert answer.count(". ") <= 2
    assert wiki_lead_pair("List of rivers", text) is None
    assert wiki_lead_pair("1999", text) is None
    assert wiki_lead_pair("Mercury", "Mercury may refer to: the planet, the element, and more things here.") is None


def test_rendered_target_is_response_only():
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    class CharTokenizer:
        specials = {"<bos>": 1000, "<sep>": 1001, "<eos>": 1002}

        def token_to_id(self, token):
            return self.specials[token]

        def encode(self, text, add_special_tokens=False):
            return Encoded([ord(c) for c in text])

        def decode(self, ids, skip_special_tokens=False):
            inv = {v: k for k, v in self.specials.items()}
            return "".join(inv.get(i, chr(i) if i < 1000 else "") for i in ids)

    args = SimpleNamespace(prompt_budget=256, history_turns=8, min_response=1, seq_len=512)
    (example,) = _tokenize_records(CharTokenizer(), [SFTRecord("arith", "g", "what's 7+5", "12")], args)
    prompt, target = render_example(CharTokenizer(), example)
    assert prompt == "<bos>user: what's 7+5<sep>"
    assert target == "12<eos>"


def _gate_fixture():
    baseline = {
        "discord": 1.0, "discord_multiturn": 1.2, "ultrachat": 1.5, "ultrachat_multiturn": 1.4,
        "oasst": 2.0, "oasst_multiturn": 2.1, "nq": 3.0, "nq_single": 3.0, "nq_legacy": 3.2,
    }
    candidate = {
        "discord": 1.03, "discord_multiturn": 1.21, "ultrachat": 1.52, "ultrachat_multiturn": 1.41,
        "oasst": 1.8, "oasst_multiturn": 1.9, "nq": 2.0, "nq_single": 2.0, "nq_legacy": 3.1,
    }
    return baseline, candidate


def test_gate_default_still_demands_every_multiturn_improve():
    baseline, candidate = _gate_fixture()
    assert _source_gate(candidate, baseline, 0.05)[0] is False


def test_gate_scoped_multiturn_and_guards():
    baseline, candidate = _gate_fixture()
    kw = dict(guard_sources=("discord", "ultrachat"), guard_limit=0.05, multiturn_must_improve=("oasst",))
    passed, regressions = _source_gate(candidate, baseline, 0.05, **kw)
    assert passed is True
    assert regressions["discord_guard"] == pytest.approx(0.03)
    assert regressions["discord_multiturn_guard"] == pytest.approx(0.01)
    # a guarded rehearsal source regressing past the ceiling fails
    assert _source_gate({**candidate, "discord_multiturn": 1.26}, baseline, 0.05, **kw)[0] is False
    assert _source_gate({**candidate, "ultrachat": 1.56}, baseline, 0.05, **kw)[0] is False
    # a tighter guard ceiling applies only to guarded sources
    assert _source_gate(candidate, baseline, 0.05, **{**kw, "guard_limit": 0.02})[0] is False
    # the QA source's multi-turn view must still strictly improve
    assert _source_gate({**candidate, "oasst_multiturn": 2.1}, baseline, 0.05, **kw)[0] is False
    # a guard missing from the eval fails closed instead of passing silently
    no_discord_base = {k: v for k, v in baseline.items() if not k.startswith("discord")}
    no_discord_cand = {k: v for k, v in candidate.items() if not k.startswith("discord")}
    passed, regressions = _source_gate(no_discord_cand, no_discord_base, 0.05, **kw)
    assert passed is False and regressions["discord_guard_missing"] == float("inf")


def test_older_presets_keep_their_data_signature():
    old = Namespace(mix_ultrachat=0.28, ultrachat_revision="r", smoltalk_multiturn=True, gif_tags=True,
                    gif_synth_rate=0.5, gif_synth_max_frac=0.025)
    assert _data_extras(old) == {
        "mix_ultrachat": 0.28, "ultrachat_revision": "r", "smoltalk_multiturn": True, "gif_tags": True,
        "gif_synth_rate": 0.5, "gif_synth_max_frac": 0.025,
    }
    qa = Namespace(**vars(old), mix_arith=0.05, mix_nq=0.1, nq_revision="n")
    extras = _data_extras(qa)
    assert extras["mix_arith"] == 0.05 and extras["nq_revision"] == "n" and "qa_data_version" in extras
