"""Scoring functions of the held-out QA eval (bench/qa). No model, no network."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from bench.qa.scoring import (
    aggregate,
    contains_alias,
    first_polarity,
    is_echo,
    is_empty,
    is_non_answer,
    is_repetitive,
    score_item,
    tokens,
)

QUESTIONS = Path(__file__).resolve().parent.parent / "bench" / "qa" / "questions.jsonl"


def test_tokens_fold_number_words_and_punctuation():
    assert tokens("Twenty-one!") == ["21"]
    assert tokens("It's TWELVE.") == ["it", "12"]
    assert tokens("1,000 apples") == ["1000", "apples"]
    assert tokens("I don't know") == ["i", "dont", "know"]
    assert tokens("Russia's capital") == ["russia", "capital"]
    assert tokens("one hundred") == ["100"]


@pytest.mark.parametrize(
    "response,alias,expected",
    [
        ("it's 12", "12", True),
        ("twelve!", "12", True),
        ("the answer is 12", "twelve", True),
        ("112", "12", False),
        ("12.5", "12", False),
        ("Paris, obviously", "paris", True),
        ("parisian food", "paris", False),
        ("I live in Washington DC", "washington", True),
        ("It was Leonardo da Vinci", "da vinci", True),
        ("da", "da vinci", False),
        ("seven", "7", True),
        ("twenty one", "21", True),
    ],
)
def test_contains_alias_whole_word_and_number_forms(response, alias, expected):
    assert contains_alias(response, alias) is expected


def test_keyword_prefix_only_for_long_keywords():
    assert contains_alias("it erupts with lava", "erupt", prefix=True)
    assert contains_alias("an eruption", "erupt", prefix=True)
    assert not contains_alias("ashes", "ash", prefix=True)  # < 4 letters: exact only
    assert not contains_alias("an eruption", "erupt")  # answers are exact


def test_first_polarity():
    assert first_polarity("Yes, of course") == "yes"
    assert first_polarity("nah, dogs can't fly") == "no"
    assert first_polarity("dogs can't fly, yes") == "no"
    assert first_polarity("of course") is None


def test_empty():
    assert is_empty("")
    assert is_empty("   ...!!  ")
    assert not is_empty("ok")


def test_echo_substring_and_overlap():
    q = "what is the capital of france?"
    assert is_echo("The capital of France", q)
    assert is_echo("france", q)
    assert is_echo("the capital is france", q)  # 3/4 tokens from the question
    assert not is_echo("paris", q)
    assert not is_echo("", q)
    assert is_echo("I am tired", "i'm so tired")  # contraction-aware


def test_non_answer():
    assert is_non_answer("idk")
    assert is_non_answer("I don't know lol")
    assert is_non_answer("why do you ask?")
    assert is_non_answer("What do you think?")
    assert is_non_answer("huh? what? why?")
    assert is_non_answer("why")
    assert not is_non_answer("good, how about you?")
    assert not is_non_answer("what a nice question, it's paris")
    assert not is_non_answer("paris")
    assert not is_non_answer("")


def test_repetitive():
    assert is_repetitive("boop boop boop boop boop boop boop boop")
    assert is_repetitive("i like it and i like it and i like it and more")
    assert not is_repetitive("the capital of france is paris, a big city in europe")
    assert not is_repetitive("no no no")  # too short to judge


def test_score_answer_item():
    item = {"id": "m", "category": "math", "question": "what's 7+5", "answers": ["12"]}
    s = score_item(item, "7 + 5 = twelve")
    assert s.correct is True and not s.echo and s.flags == []
    s = score_item(item, "what's 7+5")
    assert s.correct is False and s.echo and s.flags == ["echo"]
    s = score_item(item, "13")
    assert s.correct is False and s.flags == []


def test_score_polar_item_uses_first_polarity_word():
    item = {"id": "c", "category": "commonsense", "question": "can dogs fly?", "answers": ["no"]}
    assert score_item(item, "No, dogs can't fly").correct is True
    assert score_item(item, "nope").correct is True
    assert score_item(item, "yes they can, no joke").correct is False
    assert score_item(item, "dogs are cute").correct is False


def test_score_polar_falls_back_to_other_aliases_without_polarity_word():
    item = {"id": "b", "category": "about_bot", "question": "are you human?", "answers": ["no", "bot", "booper"],
            "must_not": ["i am human", "im human"]}
    assert score_item(item, "I'm a bot").correct is True
    assert score_item(item, "booper here").correct is True
    assert score_item(item, "yes i'm booper").correct is False  # first polarity word wins
    assert score_item(item, "I'm human").correct is False  # must_not


def test_alias_in_question_does_not_count_when_echoing():
    item = {"id": "b", "category": "about_bot", "question": "is this a bot?", "answers": ["yes", "bot"]}
    s = score_item(item, "is this a bot")
    assert s.correct is False and s.echo
    assert score_item(item, "yep").correct is True
    assert score_item(item, "i'm a bot called booper").correct is True


def test_must_not_overrides_a_match():
    item = {"id": "f", "category": "followup", "question": "what's my sister's name?", "answers": ["zoe"],
            "must_not": ["max"]}
    assert score_item(item, "zoe").correct is True
    assert score_item(item, "zoe and max").correct is False


def test_keywords_item():
    item = {"id": "d", "category": "definition", "question": "what is a volcano?",
            "keywords": ["erupt", "lava", "magma", "mountain", "ash"]}
    assert score_item(item, "a mountain that can erupt").correct is True
    assert score_item(item, "it's an eruption of hot rock").correct is True
    assert score_item(item, "a volcano is a volcano").correct is False


def test_chat_item_has_no_correctness():
    item = {"id": "x", "category": "chat", "question": "how's your day going?"}
    s = score_item(item, "pretty good, you?")
    assert s.correct is None and s.flags == []
    s = score_item(item, "how's your day going")
    assert s.correct is None and s.flags == ["echo"]
    assert score_item(item, "how is your day going?").flags == ["echo", "non_answer"]


def test_aggregate():
    def row(cat, correct, **flags):
        sc = {"correct": correct, "echo": False, "empty": False, "non_answer": False, "repetitive": False}
        sc.update(flags)
        return {"category": cat, "score": sc}

    agg = aggregate([
        row("math", True), row("math", False, echo=True),
        row("fact_common", True), row("fact_common", True),
        row("chat", None, empty=True),
    ])
    assert agg["n"] == 5
    assert agg["overall_accuracy"] == pytest.approx(3 / 4)
    assert agg["macro_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert agg["categories"]["chat"]["accuracy"] is None
    assert agg["echo_rate"] == pytest.approx(1 / 5)
    assert agg["empty_rate"] == pytest.approx(1 / 5)
    assert agg["clean_rate"] == pytest.approx(3 / 5)


def test_question_file_shape():
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    counts = Counter(r["category"] for r in rows)
    assert counts == {"fact_common": 70, "math": 40, "definition": 30, "about_bot": 25,
                      "commonsense": 35, "followup": 25, "chat": 25}
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["question"].strip()
        if r["category"] == "chat":
            assert "answers" not in r and "keywords" not in r
        elif r["category"] == "definition":
            assert len(r["keywords"]) >= 3
            assert all(k == k.lower() for k in r["keywords"])
        else:
            assert r["answers"] and all(a == a.lower() for a in r["answers"])
        if r["category"] == "followup":
            assert r["history"] and r["history"][-1]["role"] == "assistant"
            assert all(m["role"] in ("user", "assistant") for m in r["history"])


def test_fact_and_math_answers_are_not_given_away_by_the_question():
    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    for r in rows:
        if r["category"] in ("fact_common", "math", "definition"):
            q = tokens(r["question"])
            prefix = r["category"] == "definition"
            for alias in r.get("answers") or r.get("keywords"):
                assert not contains_alias(r["question"], alias, prefix=prefix), (r["id"], alias, q)
