"""Transparent "Discord vibe" heuristics for picking fun, conversational SFT data.

Pure and importable: no network, no model, no torch. Three things live here:

* ``vibe_score`` / ``vibe_features``: a hand-weighted score for one chat turn.
  It rewards casual Discord energy (lowercase, emoji and emoticons, elongated
  words, laughter, caps bursts, slang, exclamation bursts, playful markers
  such as ``*action*`` asterisks or keysmashes) and penalizes what booper
  should *not* sound like (assistant-speak, a bare "ok" as the whole reply,
  raw URLs, number/character spam, copypasta repetition, walls of text).
  Every weight is a named constant below, and ``vibe_features`` returns the
  per-feature contributions so any score can be explained line by line.

* ``conversation_verdict``: a conversation-level filter. It keeps
  conversations with several assistant targets whose replies engage with the
  turn before them, drops ones dominated by one-word turns or containing a
  blocklisted term, and requires a minimum mean vibe.

* ``local_snapshot_*``: reads the consented ``babble export`` snapshot's
  ``train.jsonl`` (approvals and corrections from booper's real users) into
  ``SFTRecord`` groups, keeping only corrections whose ``chosen`` text is an
  actual reply rather than an instruction about how booper should behave.

None of this loosens the project blocklist (``babble/blocklist.py``): it is
loaded exactly as the bot loads it and a hit rejects the whole conversation
or pair.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from babble.blocklist import Blocklist

# --------------------------------------------------------------- scoring ---
#
# Each positive feature saturates (``min(count, CAP) * WEIGHT``) so one
# message cannot win by spamming a single trick; the score is the plain sum
# of the contributions. Typical values: a dry, correct, capitalised sentence
# scores ~0; "LMAOOO no way 💀" scores ~3; assistant-speak scores < -2.

# lowercase: first letter lowercase and at least this fraction of cased letters lowercase.
LOWERCASE_MIN_FRAC = 0.9
LOWERCASE_WEIGHT = 0.5
# emoji (unicode pictographs), Discord custom emoji (`:name:`) and emoticons.
EMOJI_WEIGHT = 0.6
EMOJI_CAP = 2
# elongated words: a letter repeated 3+ times ("hiiii", "nooo", "brooo").
ELONGATED_WEIGHT = 0.6
ELONGATED_CAP = 2
# laughter tokens (lol, lmao, haha, 💀, 😭 ...).
LAUGH_WEIGHT = 0.8
LAUGH_CAP = 2
# caps burst: a 3+ letter ALL-CAPS word in a message that is not one long shout.
CAPS_WEIGHT = 0.5
CAPS_CAP = 2
# casual slang (fr, ngl, bruh, lowkey ...).
SLANG_WEIGHT = 0.4
SLANG_CAP = 3
# exclamation/interrobang bursts: "!!", "?!", "!?", "???".
BURST_WEIGHT = 0.4
BURST_CAP = 2
# playful markers: *action* asterisks, keysmash, uwu/rawr/owo, ~ tildes.
PLAYFUL_WEIGHT = 0.6
PLAYFUL_CAP = 2
# a [gif: ...] tag (the bot's gif contract) is a very Discord way to answer.
GIF_TAG_WEIGHT = 0.5

# assistant-speak ("as an AI", "I'd be happy to help", "Certainly!").
ASSISTANT_PENALTY = 3.0
# the whole reply is a low-effort filler ("ok", "k", "yeah").
FILLER_PENALTY = 1.0
# any raw URL left in the text.
URL_PENALTY = 2.0
# number spam (mostly digits) or one character/emoji repeated many times.
SPAM_PENALTY = 2.0
SPAM_DIGIT_FRAC = 0.5
SPAM_MIN_CHARS = 8
SPAM_RUN = 12
# copypasta: many words but few distinct ones, or a line repeated.
COPYPASTA_PENALTY = 1.5
COPYPASTA_MIN_WORDS = 10
COPYPASTA_DISTINCT_FRAC = 0.4
# walls of text: every WALL_STEP chars past WALL_CHARS costs WALL_PENALTY, capped.
WALL_CHARS = 400
WALL_STEP = 200
WALL_PENALTY = 0.5
WALL_CAP = 3.0

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF]"
)
_DISCORD_EMOJI_RE = re.compile(r"<a?:\w{2,32}:\d+>|(?<!\w):[a-z0-9_]{2,32}:(?!\w)", re.I)
_EMOTICON_RE = re.compile(
    r"(?<!\w)(?:[:;=xX8][-']?[)(DPpOo3/\\|\]\[]|<3|;-;|;_;|T[_.]T|>:[)(]|:3|\^\^|\^_\^|uwu|owo|qwq|xd)(?!\w)",
    re.I,
)
_ELONGATED_RE = re.compile(r"\b\w*([a-z])\1{2,}\w*\b", re.I)
_LAUGH_RE = re.compile(
    r"(?<!\w)(?:l+o+l+|l+m+f*a+o+|lu+l+|ro+fl|ha(?:ha)+h*|he(?:he)+|ja(?:ja)+|ke(?:ke)+k*|kek|xd+|"
    r"i'?m (?:dead|crying|weak)|dying)(?!\w)|[💀😭😂🤣☠]",
    re.I,
)
_SLANG = frozenset(
    "fr frfr ngl tbh bruh bro brah lowkey highkey deadass bestie ong sus based slay vibe vibes yeet rizz goated "
    "cap nocap fax smh idk idc wtf omg omfg istg imo imho tf af asf lmk nvm ty pls plz bc cuz finna gonna wanna "
    "gotta ain't aint yall y'all dawg dude homie fam bussin mid ratio cringe yikes oof welp yall bet sheesh "
    "pog poggers kekw sadge copium hopium gg ez ong sksk ate valid unhinged chaos chaotic gremlin goblin".split()
)
_BURST_RE = re.compile(r"[!?]{2,}")
_ACTION_RE = re.compile(r"(?<!\*)\*[a-z][^*\n]{1,40}\*(?!\*)", re.I)
_KEYSMASH_RE = re.compile(r"(?<!\w)[bcdfghjklmnpqrstvwxz]{6,}(?!\w)", re.I)
_PLAYFUL_WORD_RE = re.compile(r"(?<!\w)(?:uwu|owo|rawr+|nya+|hehe+|teehee|boop+|bonk|yippee+|wheee+)(?!\w)|~\s*$", re.I)
_GIF_TAG_RE = re.compile(r"\[gif: [a-z0-9]+(?: [a-z0-9]+){1,4}\]")
_URL_RE = re.compile(r"<?https?://[^\s<>]+>?|\bwww\.[^\s]+", re.I)
_ASSISTANT_RE = re.compile(
    r"\bas an ai\b|\bi(?:'m| am) (?:just )?an? (?:ai|language model|virtual assistant|ai assistant)\b|"
    r"\blanguage model\b|\bi(?:'?d| would) be (?:happy|glad|delighted) to (?:help|assist)\b|\bhow (?:can|may) i (?:help|assist) you\b|"
    r"^\s*(?:certainly|absolutely|of course)[!,.]|\bi hope this helps\b|\bfeel free to\b|\bplease note that\b|"
    r"\bi (?:cannot|can't|am unable to) (?:assist|help) with\b|\bgreat question\b|\bhere(?:'s| is) a (?:summary|breakdown|list)\b|"
    r"\bit'?s important to (?:note|remember)\b|\bin conclusion\b|\bi apologi[sz]e for (?:any|the) confusion\b",
    re.I,
)
FILLERS = frozenset(
    "ok okay k kk okk okie oki yeah yea ye ya yes yep yup no nope nah sure mhm mm hm hmm hmmm idk true same "
    "nice cool alright aight right fine thanks thx ty oh ah".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+", re.I)


def _strip_gif(text: str) -> tuple[str, bool]:
    stripped = _GIF_TAG_RE.sub("", text)
    return stripped.strip(), stripped != text


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _emoji_count(text: str) -> int:
    return len(_EMOJI_RE.findall(text)) + len(_DISCORD_EMOJI_RE.findall(text)) + len(_EMOTICON_RE.findall(text))


def _is_spam(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= SPAM_MIN_CHARS:
        digits = sum(ch.isdigit() for ch in compact)
        if digits / len(compact) > SPAM_DIGIT_FRAC:
            return True
    # One character (letter, digit, punctuation or emoji) repeated SPAM_RUN+ times.
    return bool(re.search(rf"(.)\1{{{SPAM_RUN - 1},}}", compact))


def _is_copypasta(text: str, words: list[str]) -> bool:
    if len(words) >= COPYPASTA_MIN_WORDS and len(set(words)) / len(words) < COPYPASTA_DISTINCT_FRAC:
        return True
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    return len(lines) >= 3 and len(set(lines)) <= len(lines) / 2


_REPEAT_FOLD = re.compile(r"(.)\1+")
_FILLERS_FOLDED = frozenset(_REPEAT_FOLD.sub(r"\1", f) for f in FILLERS)


def is_filler(text: str) -> bool:
    """The whole message is one low-effort filler word ("ok", "k", "yeah", "okkk")."""
    words = _words(_strip_gif(text)[0])
    return len(words) == 1 and (words[0] in FILLERS or _REPEAT_FOLD.sub(r"\1", words[0]) in _FILLERS_FOLDED)


def is_assistant_speak(text: str) -> bool:
    return bool(_ASSISTANT_RE.search(text))


def vibe_features(text: str) -> dict[str, float]:
    """Per-feature contributions to ``vibe_score`` (only non-zero entries)."""
    body, has_gif = _strip_gif(text)
    words = _words(body)
    out: dict[str, float] = {}

    def add(name: str, value: float) -> None:
        if value:
            out[name] = round(value, 4)

    letters = [ch for ch in body if ch.isalpha() and ch.lower() != ch.upper()]
    if letters:
        # Casual lowercase = the message does not open with a capital (no
        # sentence-case) and is mostly lowercase; caps bursts are scored apart.
        lower_frac = sum(ch.islower() for ch in letters) / len(letters)
        if letters[0].islower() and lower_frac >= LOWERCASE_MIN_FRAC:
            add("lowercase", LOWERCASE_WEIGHT)
    add("emoji", min(_emoji_count(body), EMOJI_CAP) * EMOJI_WEIGHT)
    elongated = [m.group(0) for m in _ELONGATED_RE.finditer(body) if not m.group(0).isdigit()]
    add("elongated", min(len(elongated), ELONGATED_CAP) * ELONGATED_WEIGHT)
    add("laughter", min(len(_LAUGH_RE.findall(body)), LAUGH_CAP) * LAUGH_WEIGHT)
    caps_words = [w for w in re.findall(r"\b[A-Z]{3,}\b", body)]
    shout_frac = len(caps_words) / max(len(words), 1)
    # A caps burst is emphasis inside a message; a long message entirely in
    # caps is shouting/copypasta and gets nothing here.
    if caps_words and not (len(words) > 8 and shout_frac > 0.8):
        add("caps_burst", min(len(caps_words), CAPS_CAP) * CAPS_WEIGHT)
    slang = [w for w in words if w in _SLANG or _REPEAT_FOLD.sub(r"\1", w) in _SLANG]
    add("slang", min(len(slang), SLANG_CAP) * SLANG_WEIGHT)
    add("burst", min(len(_BURST_RE.findall(body)), BURST_CAP) * BURST_WEIGHT)
    playful = len(_ACTION_RE.findall(body)) + len(_KEYSMASH_RE.findall(body)) + len(_PLAYFUL_WORD_RE.findall(body))
    add("playful", min(playful, PLAYFUL_CAP) * PLAYFUL_WEIGHT)
    if has_gif:
        add("gif_tag", GIF_TAG_WEIGHT)

    if is_assistant_speak(body):
        add("assistant_speak", -ASSISTANT_PENALTY)
    if is_filler(text) and not has_gif:
        add("filler", -FILLER_PENALTY)
    if _URL_RE.search(body):
        add("url", -URL_PENALTY)
    if _is_spam(body):
        add("spam", -SPAM_PENALTY)
    if _is_copypasta(body, words):
        add("copypasta", -COPYPASTA_PENALTY)
    if len(body) > WALL_CHARS:
        steps = 1 + (len(body) - WALL_CHARS) // WALL_STEP
        add("wall_of_text", -min(steps * WALL_PENALTY, WALL_CAP))
    return out


def vibe_score(text: str) -> float:
    """Sum of ``vibe_features`` contributions; higher = more Discord energy."""
    return round(sum(vibe_features(text).values()), 4)


# ------------------------------------------------------- conversation ---

# Words that carry no topic; overlap on these alone is not engagement.
STOPWORDS = frozenset(
    "a an the and or but if so to of in on at by for with from up down out over under is are was were be been "
    "being am do does did done have has had having i me my mine you your yours u ur we us our they them their "
    "he him his she her it its this that these those there here what which who whom whose when where why how "
    "not no yes yeah ya ye just like really very too also then than as can could would should will wont won't "
    "dont don't im i'm its it's thats that's lol lmao ok okay oh all any some one get got go going know think "
    "about more much now well good bad gonna wanna".split()
)
# A reply that opens like this is answering a question, even with no overlap.
_ANSWER_OPENER_RE = re.compile(
    r"^\s*(?:yes|yeah|yea|ye|yep|yup|no|nah|nope|maybe|probably|idk|i don'?t|i do|i did|i think|i guess|"
    r"because|cuz|bc|it'?s|its|it was|they|he|she|depends|not really|kinda|sort of|definitely|absolutely|"
    r"never|always|sometimes|\d+)\b",
    re.I,
)
_QUESTION_RE = re.compile(r"\?\s*$|^\s*(?:who|what|when|where|why|how|which|do|does|did|are|is|can|could|would|will|should|have|has)\b", re.I)
_SECOND_PERSON_RE = re.compile(r"\b(?:you|u|ur|your|yours|you're|youre|ya)\b", re.I)
_REACTION_RE = re.compile(
    r"^\W*(?:l+o+l+|l+m+f*a+o+|ha(?:ha)+h*|bru+h+|bro+|no+ wa+y+|wa+i+t+( what)?|wh?a+t+|huh+|real+|true+|fr+|"
    r"facts|based|nice+|w+|omg+|wtf+|oof+|rip|same+|mood|yikes|ew+|a+w+|damn+|dang+|sheesh+)\W*$",
    re.I,
)

# conversation_verdict defaults (overridable from the CLI).
DEFAULT_MIN_VIBE = 1.0
DEFAULT_MIN_TARGETS = 2
ONE_WORD_MAX_FRAC = 0.5  # more than this share of all turns being <= 1 word -> drop
MIN_ENGAGED_FRAC = 0.5  # at least this share of assistant targets must engage
TARGET_FLOOR = -2.0  # any single target at or below this (assistant-speak, URL, spam) -> drop


STEM_CHARS = 5  # words compare on this many leading chars ("studying" ~ "study", "creator" ~ "creation")
ANSWER_MAX_WORDS = 15  # a declarative reply this short right after a question counts as answering it
_ACK_RE = re.compile(r"^\W*(?:thanks|thank you|thx|ty|tysm|np|no worries|no problem|you too|u too|same to you|ily|love you)\b", re.I)


def content_words(text: str) -> set[str]:
    """Topic words (stopwords and 1-2 letter words dropped), crudely stemmed."""
    return {
        _REPEAT_FOLD.sub(r"\1\1", w)[:STEM_CHARS]
        for w in _words(_strip_gif(text)[0])
        if len(w) > 2 and w not in STOPWORDS and not w.isdigit()
    }


def is_question(text: str) -> bool:
    return bool(_QUESTION_RE.search(text.strip()))


def engagement_reason(prev: str, reply: str) -> str | None:
    """Why ``reply`` counts as responding to ``prev`` (None = non-sequitur).

    Transparent rules, first match wins: shared (stemmed) content words; a
    question answered with an answer-shaped opener or a short declarative
    reply; a short reaction (laughter, "bruh", "no way") which is how people
    actually reply on Discord; an acknowledgement ("thanks", "you too"); a
    reply that addresses the other person in a full clause (>= 3 words with
    "you/u").
    """
    if not prev.strip() or not reply.strip():
        return None
    if content_words(prev) & content_words(reply):
        return "overlap"
    body = _strip_gif(reply)[0]
    if is_question(prev) and (
        _ANSWER_OPENER_RE.search(body)
        or (body and "?" not in body and len(_words(body)) <= ANSWER_MAX_WORDS)
    ):
        return "answer"
    if (body and _REACTION_RE.match(body)) or (not body and _GIF_TAG_RE.search(reply)):
        return "reaction"
    if _ACK_RE.match(body):
        return "ack"
    if len(_words(body)) >= 3 and _SECOND_PERSON_RE.search(body):
        return "addresses"
    return None


def engages(prev: str, reply: str) -> bool:
    return engagement_reason(prev, reply) is not None


def trim_to_user_start(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop leading non-user turns.

    About 45% of Discord-Dialogues rows open with the `assistant` speaker (the
    two roles are just two people). `discord` discards those rows; the vibe
    filter keeps them by starting at the first user turn, as `_messages_group`
    does for HF chat data.
    """
    turns = list(turns)
    while turns and turns[0][0] != "user":
        turns.pop(0)
    return turns


@dataclass
class Verdict:
    ok: bool
    reason: str  # "ok" or the first failed check
    targets: int = 0
    mean_vibe: float = 0.0
    engaged_frac: float = 0.0
    one_word_frac: float = 0.0
    scores: list[float] = field(default_factory=list)


def conversation_verdict(
    turns: list[tuple[str, str]],
    *,
    min_vibe: float = DEFAULT_MIN_VIBE,
    min_targets: int = DEFAULT_MIN_TARGETS,
    blocklist: Blocklist | None = None,
) -> Verdict:
    """Judge one conversation of alternating ``(role, text)`` turns.

    Checks run in this order and the first failure is the reason:
    ``malformed`` (not user/assistant alternating from a user turn),
    ``blocklist`` (any turn hits the project blocklist), ``too_few_targets``,
    ``one_word`` (over ``ONE_WORD_MAX_FRAC`` of all turns are <= 1 word),
    ``bad_target`` (some target scores <= ``TARGET_FLOOR``),
    ``low_engagement`` (under ``MIN_ENGAGED_FRAC`` of targets engage the turn
    before them), ``low_vibe`` (mean target vibe < ``min_vibe``).
    """
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(turns) - 1, 2):
        (r1, u), (r2, a) = turns[i], turns[i + 1]
        if r1 != "user" or r2 != "assistant":
            return Verdict(False, "malformed")
        if u.strip() and a.strip():
            pairs.append((u, a))
    if not turns or turns[0][0] != "user":
        return Verdict(False, "malformed")
    if blocklist is not None and blocklist.matches(*(text for _, text in turns)):
        return Verdict(False, "blocklist", targets=len(pairs))
    if len(pairs) < max(1, min_targets):
        return Verdict(False, "too_few_targets", targets=len(pairs))
    one_word = sum(len(_words(_strip_gif(text)[0])) <= 1 for _, text in turns) / len(turns)
    scores = [vibe_score(a) for _, a in pairs]
    engaged = sum(engages(u, a) for u, a in pairs) / len(pairs)
    verdict = Verdict(
        False, "ok", targets=len(pairs), mean_vibe=statistics.fmean(scores), engaged_frac=engaged,
        one_word_frac=one_word, scores=scores,
    )
    if one_word > ONE_WORD_MAX_FRAC:
        verdict.reason = "one_word"
    elif min(scores) <= TARGET_FLOOR:
        verdict.reason = "bad_target"
    elif engaged < MIN_ENGAGED_FRAC:
        verdict.reason = "low_engagement"
    elif verdict.mean_vibe < min_vibe:
        verdict.reason = "low_vibe"
    else:
        verdict.ok = True
    return verdict


# ------------------------------------------------------ local snapshot ---
#
# `babble export` writes data/train.jsonl rows of
# {prompt, chosen, rejected, signal: correction|approval, weight, ...}.
# approval  = a user approved booper's own reply (`chosen`): always a reply.
# correction = a user typed what booper *should* have said. People sometimes
#   type an instruction about booper instead ("say something like X whenever
#   someone says Y", "when someone is asking you X, you'd say Y"). Trained as
#   a response, that teaches booper to answer with instructions, so such rows
#   are rejected by META_INSTRUCTION_RE below.

META_INSTRUCTION_RE = re.compile(
    # conditional rules about other people's messages
    r"\b(?:when|whenever|if|every ?time)\s+(?:someone|somebody|anyone|any ?one|people|ppl|they|a user|users|"
    r"someone's|somebody's|a person|he|she)\b|"
    # "you'd say", "you should reply", "you need to answer"
    r"\byou(?:'d| would| should| could| need to| have to| must| gotta|'re supposed to| are supposed to)\s+"
    r"(?:say|reply|respond|answer|tell|mention|write|type|send)\b|"
    # imperative coaching: "say something like", "reply with", "respond saying"
    r"^\s*(?:just\s+)?(?:say|reply|respond|answer|tell (?:them|him|her|people))\s+"
    r"(?:something like|stuff like|with|saying|that|like)\b|"
    # talking about the bot in the third person / correcting its behaviour
    r"\b(?:booper|the bot|this bot)\s+(?:should|must|needs to|has to|is supposed to)\b|"
    r"\binstead of saying\b",
    re.I,
)


def is_meta_instruction(chosen: str) -> bool:
    """True when a correction's ``chosen`` text instructs booper rather than replying."""
    return bool(META_INSTRUCTION_RE.search(chosen))


@dataclass
class LocalStats:
    rows: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    by_signal: Counter = field(default_factory=Counter)
    files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "kept": self.kept,
            "rejected": dict(self.rejected),
            "kept_by_signal": dict(self.by_signal),
        }


def snapshot_train_file(path: Path) -> Path:
    """Resolve a snapshot dir (``<snap>``, ``<snap>/data``) or file to its train.jsonl."""
    path = Path(path)
    for candidate in (path, path / "train.jsonl", path / "data" / "train.jsonl"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no train.jsonl under local snapshot path {path}")


def snapshot_corpus_file(path: Path) -> Path:
    """Resolve a snapshot dir or file to its corpus.jsonl."""
    path = Path(path)
    for candidate in (path, path / "corpus.jsonl", path / "data" / "corpus.jsonl"):
        if candidate.is_file() and candidate.name == "corpus.jsonl":
            return candidate
    raise FileNotFoundError(f"no corpus.jsonl under local snapshot path {path}")


def snapshot_sha256(paths: Iterable[Path]) -> str:
    """Hash of the snapshot's train.jsonl files (bytes + order), for run metadata.

    Like ``persona_sha256`` this records *which* inputs built the data without
    copying any of their text into metadata.
    """
    digest = hashlib.sha256()
    for path in paths:
        file = snapshot_train_file(path)
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def _norm_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w ]+", "", re.sub(r"\s+", " ", text)).strip()


def local_pair_verdict(row: dict, blocklist: Blocklist | None) -> str:
    """"ok" or the reason a snapshot train row is not used as a training pair."""
    prompt = str(row.get("prompt") or "").strip()
    chosen = str(row.get("chosen") or "").strip()
    signal = row.get("signal")
    if signal not in ("approval", "correction"):
        return "unknown_signal"
    if not prompt or not chosen:
        return "empty"
    if blocklist is not None and blocklist.matches(prompt, chosen):
        return "blocklist"
    if _URL_RE.search(chosen):
        return "url"
    if is_assistant_speak(chosen):
        return "assistant_speak"
    if signal == "correction" and is_meta_instruction(chosen):
        return "meta_instruction"
    return "ok"


def local_snapshot_records(
    paths: Iterable[Path] | Path,
    *,
    source: str = "booper_local",
    blocklist: Blocklist | None = None,
    stats: LocalStats | None = None,
):
    """Snapshot train.jsonl rows -> SFTRecords (approvals + direct-reply corrections).

    Raises FileNotFoundError when a given path has no train.jsonl and
    ValueError when no row across all paths survives the filters, so a run
    that asked for local data can never silently train without it. One record
    per row; records sharing a normalized prompt share a ``group_id`` so they
    stay on one side of the train/val split.
    """
    from sft.sft_longform import SFTRecord, _group_id

    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("local snapshot: no paths given")
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    stats = stats if stats is not None else LocalStats()
    records = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        file = snapshot_train_file(path)
        stats.files.append(str(file))
        for n, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{file}:{n}: not JSON ({e})") from e
            stats.rows += 1
            reason = local_pair_verdict(row, blocklist)
            prompt, chosen = str(row.get("prompt") or "").strip(), str(row.get("chosen") or "").strip()
            if reason == "ok" and (prompt, chosen) in seen:
                reason = "duplicate"
            if reason != "ok":
                stats.rejected[reason] += 1
                continue
            seen.add((prompt, chosen))
            stats.kept += 1
            stats.by_signal[row["signal"]] += 1
            records.append(
                SFTRecord(
                    source=source,
                    group_id=_group_id("booper_local", _norm_prompt(prompt)),
                    current_user=prompt,
                    response=chosen,
                    history=(),
                )
            )
    if not records:
        raise ValueError(
            f"local snapshot {[str(p) for p in paths]}: no usable rows "
            f"({stats.rows} rows, rejected {dict(stats.rejected)})"
        )
    return records


def local_snapshot_groups(paths, *, source: str = "booper_local", blocklist: Blocklist | None = None, stats: LocalStats | None = None):
    """``local_snapshot_records`` grouped by ``group_id`` (first-seen order)."""
    grouped: dict[str, list] = {}
    for record in local_snapshot_records(paths, source=source, blocklist=blocklist, stats=stats):
        grouped.setdefault(record.group_id, []).append(record)
    yield from grouped.values()


CORPUS_PROMPT_SOURCES = ("mention", "dm", "reply")


def corpus_prompts(paths: Iterable[Path] | Path, *, blocklist: Blocklist | None = None, sources=CORPUS_PROMPT_SOURCES) -> list[str]:
    """Distinct non-empty, non-blocklisted messages people sent booper directly."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    blocklist = blocklist if blocklist is not None else Blocklist.load()
    out: dict[str, None] = {}
    for path in paths:
        file = snapshot_corpus_file(Path(path))
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if row.get("source") in sources and text and not blocklist.matches(text):
                out.setdefault(text, None)
    return list(out)


# --------------------------------------------------------- coherence ---

ECHO_JACCARD = 0.8  # reply/prompt token Jaccard at or above this is an echo
DEGENERATE_DISTINCT2 = 0.5  # distinct-bigram ratio below this (>= 6 words) is a loop


def coherence_checks(prompt: str, reply: str) -> dict[str, bool]:
    """Cheap per-reply coherence proxy used by ``sft/vibe_eval.py``.

    ``coherent`` = non-empty, not an echo of the prompt, not a degenerate
    loop/spam, and engaging the prompt by ``engagement_reason``.
    """
    words = _words(_strip_gif(reply)[0])
    prompt_words = _words(prompt)
    empty = not reply.strip()
    union = set(words) | set(prompt_words)
    jaccard = len(set(words) & set(prompt_words)) / len(union) if union else 0.0
    echo = (not empty) and (_norm_prompt(reply) == _norm_prompt(prompt) or (len(words) >= 2 and jaccard >= ECHO_JACCARD))
    bigrams = list(zip(words, words[1:]))
    degenerate = (len(words) >= 6 and len(set(bigrams)) / len(bigrams) < DEGENERATE_DISTINCT2) or _is_spam(reply)
    engaged = engages(prompt, reply)
    return {
        "empty": empty,
        "echo": echo,
        "degenerate": degenerate,
        "engaged": engaged,
        "assistant_speak": is_assistant_speak(reply),
        "coherent": not empty and not echo and not degenerate and engaged,
    }
