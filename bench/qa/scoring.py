"""Scoring for the booper held-out QA eval. Pure functions, no model, no torch.

Every response is reduced to a list of canonical tokens (`tokens`): lowercase,
punctuation stripped, number words folded to digits ("twenty-one" -> "21",
"twelve" -> "12", "1,000" -> "1000"). Answer aliases go through the same
function, so "12" and "twelve" match each other in either direction.

Per item:

* ``correct`` -- some answer alias appears as a whole-token run in the
  response (definition items: some keyword appears; a keyword of 4+ letters
  also matches as a word prefix, so "erupt" matches "eruption"). An item
  with a single-word yes/no alias is polar: it is judged on the *first*
  yes/no word in the response ("nah, dogs can't fly" -> no); only when the
  response has no polarity word at all do its other aliases count.
  A ``must_not`` hit always makes the item wrong. An alias that also occurs in
  the question does not count when the response is an echo of the question
  (copying "is this a bot" back is not an answer).
* ``echo`` -- the response mostly copies the question: its canonical text is
  a substring of the question's, or >= 60% of its distinct tokens occur in the
  question. A correct response is never an echo (its answer is new
  information), so "7 + 5 = 12" to "what's 7+5" is correct, not echo.
* ``empty`` -- no letters or digits at all.
* ``non_answer`` -- a short "idk"-style dodge, or a reply made only of
  questions ("why do you ask?", "what do you think?").
* ``repetitive`` -- a long-enough reply whose distinct-token ratio is below
  0.5, or that repeats some 3-gram three or more times.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "ya", "yea", "ye", "sure", "definitely",
                       "absolutely", "ofc", "indeed", "correct", "true"})
NO_WORDS = frozenset({"no", "nope", "nah", "never", "not", "dont", "doesnt", "isnt", "cant",
                      "cannot", "wont", "arent", "false", "nothing"})

DODGES = (
    "idk", "i dont know", "i do not know", "dont know", "dunno", "no idea", "not sure",
    "im not sure", "i have no idea", "no clue", "i have no clue", "who knows", "i cant say",
    "i cant tell", "why do you ask", "i forgot", "i dont remember",
    "you tell me", "idc", "i dont care", "not telling",
)
# Dodges only when they are the entire reply ("what" opens real answers too).
EXACT_DODGES = ("why", "what", "huh", "hmm", "maybe")

QUESTION_OPENERS = frozenset({
    "what", "why", "how", "who", "where", "when", "which", "whose",
    "do", "does", "did", "dont", "doesnt", "didnt", "are", "arent", "is", "isnt", "am", "can", "cant",
    "could", "would", "will", "wont", "should", "shall", "have", "has", "had", "may", "might",
    "wanna", "u", "you", "ur", "and", "but", "so", "or", "huh", "hmm", "eh", "really", "wait",
    "lol", "well", "idk", "any", "anything", "got",
})
ECHO_OVERLAP = 0.6
REPEAT_MIN_TOKENS = 8


def _fold_numbers(words: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            value = _TENS[w]
            if i + 1 < len(words) and words[i + 1] in _UNITS and 0 < _UNITS[words[i + 1]] < 10:
                value += _UNITS[words[i + 1]]
                i += 1
            out.append(str(value))
        elif w in _UNITS:
            out.append(str(_UNITS[w]))
        elif w == "hundred" and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 100)
        elif w == "hundred":
            out.append("100")
        else:
            out.append(w)
        i += 1
    return out


def tokens(text: str) -> list[str]:
    """Canonical tokens: lowercase words/numbers, number words as digits."""
    s = str(text or "").lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,000 -> 1000
    s = re.sub(r"(?<=\d)\.(?=\d)", "p", s)  # 12.5 stays one token ("12p5"), never "12"
    s = re.sub(r"'s\b", "", s)  # russia's -> russia, what's -> what
    s = s.replace("'", "")  # don't -> dont, i'm -> im
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _fold_numbers(s.split())


def canon(text: str) -> str:
    return " ".join(tokens(text))


def _contains_run(hay: list[str], needle: list[str], *, prefix: bool = False) -> bool:
    if not needle:
        return False
    n = len(needle)
    for i in range(len(hay) - n + 1):
        ok = True
        for j, want in enumerate(needle):
            got = hay[i + j]
            if got == want:
                continue
            if prefix and j == n - 1 and len(want) >= 4 and got.startswith(want):
                continue
            ok = False
            break
        if ok:
            return True
    return False


def contains_alias(response: str, alias: str, *, prefix: bool = False) -> bool:
    """``alias`` occurs as whole tokens in ``response`` (digit/word aware)."""
    return _contains_run(tokens(response), tokens(alias), prefix=prefix)


def first_polarity(response: str) -> str | None:
    for tok in tokens(response):
        if tok in YES_WORDS:
            return "yes"
        if tok in NO_WORDS:
            return "no"
    return None


def is_empty(response: str) -> bool:
    return not tokens(response)


def is_echo(response: str, question: str) -> bool:
    r, q = tokens(response), tokens(question)
    if not r:
        return False
    if _contains_run(q, r):
        return True
    rs = set(r)
    return len(rs & set(q)) / len(rs) >= ECHO_OVERLAP


def is_non_answer(response: str) -> bool:
    text = str(response or "").strip()
    toks = tokens(text)
    if not toks:
        return False
    joined = " ".join(toks)
    if len(toks) <= 8:
        if joined in EXACT_DODGES:
            return True
        for d in DODGES:
            if joined == d or joined.startswith(d + " "):
                return True
    # Only questions: every sentence ends with "?" and every comma clause in
    # it opens like a question ("pretty good, you?" carries an answer).
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", text) if c.strip()]
    chunks = [c for c in chunks if tokens(c)]
    if not chunks or not all(c.rstrip(" \"')").endswith("?") for c in chunks):
        return False
    for chunk in chunks:
        for clause in re.split(r"[,;:]", chunk):
            ct = tokens(clause)
            if ct and ct[0] not in QUESTION_OPENERS:
                return False
    return True


def is_repetitive(response: str) -> bool:
    toks = tokens(response)
    if len(toks) < REPEAT_MIN_TOKENS:
        return False
    if len(set(toks)) / len(toks) < 0.5:
        return True
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(toks) - 2):
        g = tuple(toks[i : i + 3])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= 3:
            return True
    return False


@dataclass
class ItemScore:
    correct: bool | None  # None: no answer key (chat)
    echo: bool
    empty: bool
    non_answer: bool
    repetitive: bool
    must_not_hit: bool = False
    matched: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def score_item(item: dict, response: str) -> ItemScore:
    """Score one response against one questions.jsonl item."""
    question = item.get("question", "")
    echo_raw = is_echo(response, question)
    q_toks = tokens(question)
    must_not_hit = any(contains_alias(response, bad) for bad in item.get("must_not", []) or [])

    correct: bool | None = None
    matched: str | None = None
    keywords = item.get("keywords")
    answers = item.get("answers")
    if keywords:
        candidates, prefix = list(keywords), True
    elif answers:
        candidates, prefix = list(answers), False
    else:
        candidates, prefix = [], False

    if candidates:
        correct = False

        def usable(alias: str) -> bool:
            in_question = _contains_run(q_toks, tokens(alias), prefix=prefix)
            return not (in_question and echo_raw)

        polar = [a for a in candidates if len(tokens(a)) == 1 and tokens(a)[0] in (YES_WORDS | NO_WORDS)]
        others = [a for a in candidates if a not in polar]
        if polar and not keywords:
            want = {"yes" if tokens(a)[0] in YES_WORDS else "no" for a in polar}
            got = first_polarity(response)
            if got is not None:
                correct = got in want
                matched = got if correct else None
            else:
                for alias in others:
                    if usable(alias) and contains_alias(response, alias):
                        correct, matched = True, alias
                        break
        else:
            for alias in others:
                if usable(alias) and contains_alias(response, alias, prefix=prefix):
                    correct, matched = True, alias
                    break
        if must_not_hit:
            correct, matched = False, None

    s = ItemScore(
        correct=correct,
        echo=bool(echo_raw and not correct),
        empty=is_empty(response),
        non_answer=is_non_answer(response),
        repetitive=is_repetitive(response),
        must_not_hit=must_not_hit,
        matched=matched,
    )
    s.flags = [n for n in ("echo", "empty", "non_answer", "repetitive") if getattr(s, n)]
    return s


FLAGS = ("echo", "empty", "non_answer", "repetitive")


def aggregate(rows: list[dict]) -> dict:
    """Per-category accuracy and flag rates from scored rows.

    Each row needs ``category`` and ``score`` (an `ItemScore.to_dict()`).
    ``overall_accuracy`` is micro-averaged over every item with an answer key
    (chat has none); ``macro_accuracy`` averages the per-category accuracies;
    ``clean_rate`` is the share of *all* items with no failure flag.
    """
    cats: dict[str, dict] = {}
    for row in rows:
        c = cats.setdefault(row["category"], {"n": 0, "scored": 0, "correct": 0, **{f: 0 for f in FLAGS}, "clean": 0})
        sc = row["score"]
        c["n"] += 1
        if sc["correct"] is not None:
            c["scored"] += 1
            c["correct"] += int(bool(sc["correct"]))
        for f in FLAGS:
            c[f] += int(bool(sc[f]))
        c["clean"] += int(not any(sc[f] for f in FLAGS))
    out_cats = {}
    for name, c in cats.items():
        n = max(1, c["n"])
        out_cats[name] = {
            "n": c["n"],
            "accuracy": (c["correct"] / c["scored"]) if c["scored"] else None,
            **{f"{f}_rate": c[f] / n for f in FLAGS},
            "clean_rate": c["clean"] / n,
        }
    total = len(rows)
    scored = sum(c["scored"] for c in cats.values())
    correct = sum(c["correct"] for c in cats.values())
    accs = [v["accuracy"] for v in out_cats.values() if v["accuracy"] is not None]
    return {
        "n": total,
        "overall_accuracy": correct / scored if scored else None,
        "macro_accuracy": sum(accs) / len(accs) if accs else None,
        **{f"{f}_rate": (sum(c[f] for c in cats.values()) / total if total else 0.0) for f in FLAGS},
        "clean_rate": (sum(c["clean"] for c in cats.values()) / total if total else 0.0),
        "categories": out_cats,
    }
