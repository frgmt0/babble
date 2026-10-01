"""Long-form SFT for Booper-Big-Chat on a laptop (MPS / CPU / CUDA).

The live model (`ProCreations/Booper-Big-Chat-INT8`) was SFT'd only on
`mookiezi/Discord-Dialogues`, whose replies are one-liners, so after `<sep>`
its prior is "say a few tokens, emit <eos>". This script continues the SFT on
a mix of long-form pairs (TinyStories-Instruct stories, no_robots answers)
plus a Discord-Dialogues rehearsal slice so the chat voice survives, all in
the pair layout the bot serves: `<bos> prompt <sep> response <eos>`, loss on
the response only.

Outputs land in `runs/<name>/`:
  metrics.jsonl   one JSON line per log step (loss, lr, tok/s, val, samples)
  train.log       stdout of the run (what `monitor.sh` tails)
  ckpt/           latest bf16 safetensors + config + tokenizer (resumable)
  export/         INT8 pack in the exact layout `babble.hfserve._load_int8`
                  reads (`--export` or automatic at the end)

Usage (from the repo root, venv active):
  python sft/sft_longform.py --name story-v1 --tokens 30e6
  python sft/sft_longform.py --name story-v1 --resume
  python sft/sft_longform.py --name story-v1 --export   # only re-pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import subprocess
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from babble.conversation import (  # noqa: E402 - repo root is added above
    ConversationTurn,
    conversation_prompt_for_token_budget,
)

INT8_REPO = "ProCreations/Booper-Big-Chat-INT8"
PROMPT_METADATA_KEYS = (
    "babble_prompt_format",
    "babble_history_turns",
    "babble_prompt_budget",
)
SAMPLE_PROMPTS = [
    "write me a short story about a dragon who is afraid of fire",
    "write a short story about a detective in a city that never sleeps",
    "hey booper whats up",
    "LMAOOO did you see that",
]
STORY_TEMPLATES = [
    "write me a story about {summary}",
    "can you write a short story? {summary}",
    "tell me a story where {summary}",
    "write a story using the words {words}",
    "story time! something about {summary}",
    "write a short story with these words: {words}",
]


@dataclass(frozen=True)
class SFTRecord:
    """One supervised response and the conversation it belongs to.

    `group_id` is the split unit. Every assistant target derived from one
    Discord conversation therefore stays wholly on train or validation.
    """

    source: str
    group_id: str
    current_user: str
    response: str
    history: tuple[ConversationTurn, ...] = ()
    legacy_prompt: bool = False


def _group_id(*parts: str) -> str:
    """Content identity independent of source, preventing cross-source leaks."""
    body = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


# ------------------------------------------------------------- gif tags ---
#
# Contract shared with the bot side: the model says it wants to send a gif by
# emitting `[gif: <2-5 lowercase search words>]`, either as the whole reply or
# at the very end of one. The bot turns the words into a gif search. Never
# change the shape here without changing the bot.

GIF_TAG_RE = re.compile(r"\[gif: [a-z0-9]+(?: [a-z0-9]+){1,4}\]")
_URL_RE = re.compile(r"<?https?://[^\s<>]+>?", re.I)
_GIF_URL_RE = re.compile(r"(tenor\.com|giphy\.com|\.gif(\?|#|>|$))", re.I)
# A whole message (or its last line) that is just a Discord attachment
# filename such as `SipsBubble.gif` or `danny devito clapping.gif`.
_GIF_FILENAME_RE = re.compile(r"(?:^|\n)\s*([^\n/\\:*?\"<>|]{1,80}?)\.gif\s*$", re.I)
_GIF_DROP_WORDS = {"gif", "gifs", "view", "tenor", "giphy", "media", "animated", "download", "search"}


def _looks_like_id(token: str) -> bool:
    """Tenor/Giphy ids: all digits, or long mixed-case/digit noise."""
    if token.isdigit():
        return True
    has_digit = any(c.isdigit() for c in token)
    mixed = any(c.isupper() for c in token[1:]) and any(c.islower() for c in token)
    return len(token) >= 8 and (has_digit or (mixed and not token[0].isupper()))


def gif_slug_words(slug: str) -> list[str]:
    """Split a url slug / filename stem into lowercase search words."""
    from urllib.parse import unquote

    slug = unquote(slug)
    raw = [t for t in re.split(r"[-_+\s.]+", slug) if t]
    words: list[str] = []
    for token in raw:
        if _looks_like_id(token):
            continue
        # camelCase / PascalCase -> separate words ("SipsBubble" -> sips bubble)
        for part in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", token):
            part = part.lower()
            if part in _GIF_DROP_WORDS or part.isdigit():
                continue
            words.append(part)
    return words


def gif_tag_from_words(words: list[str]) -> str | None:
    """Render the contract tag, or None when 2-5 usable words are unavailable."""
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
    words = [w for w in words if w]
    if not 2 <= len(words) <= 5:
        return None
    return f"[gif: {' '.join(words)}]"


def gif_url_to_tag(url: str) -> str | None:
    """`https://tenor.com/view/cat-laughing-funny-gif-12345` -> `[gif: cat laughing funny]`.

    Returns None for gif URLs without a usable slug (short links, raw media
    ids); the caller drops those URLs rather than keeping them.
    """
    from urllib.parse import urlparse

    url = url.strip("<>")
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]
    slug = None
    if "tenor.com" in host:
        if "view" in parts and parts.index("view") + 1 < len(parts):
            slug = parts[parts.index("view") + 1]
        elif host.startswith("media") and parts and parts[-1].lower().endswith(".gif"):
            slug = parts[-1][:-4]
    elif "giphy.com" in host:
        if parts and parts[0] in ("gifs", "stickers") and len(parts) > 1:
            slug = parts[-1]
        # media.giphy.com/media/<id>/giphy.gif carries no words
    elif parts and parts[-1].lower().endswith(".gif"):
        slug = parts[-1][:-4]
    if not slug:
        return None
    return gif_tag_from_words(gif_slug_words(slug))


def rewrite_gifs(text: str) -> tuple[str, int, int]:
    """Replace gif sends with one trailing `[gif: ...]` tag.

    Returns (text, tags_made, gif_urls_dropped). Gif URLs anywhere in the
    message are removed; the first one with a usable slug becomes the tag,
    appended at the end. A message that is, or ends with, a Discord gif
    attachment filename (`SipsBubble.gif`) is handled the same way. Non-gif
    URLs are left for the caller to judge.
    """
    tag = None
    dropped = 0

    def sub(match: re.Match) -> str:
        nonlocal tag, dropped
        url = match.group(0)
        if not _GIF_URL_RE.search(url.strip("<>")):
            return url
        found = gif_url_to_tag(url)
        if found and tag is None:
            tag = found
        elif not found:
            dropped += 1
        return ""

    out = _URL_RE.sub(sub, text)
    if tag is None and not _URL_RE.search(out):
        m = _GIF_FILENAME_RE.search(out)
        if m:
            line = m.group(1).strip()
            head, _, last = line.rpartition(" ")
            # `name it jerma_cute.gif`: a separator-bearing final token is the
            # filename on its own; otherwise the whole short line is.
            found = gif_tag_from_words(gif_slug_words(last))
            keep = head
            if not found:
                found = gif_tag_from_words(gif_slug_words(line))
                keep = ""
            if found:
                tag = found
                out = out[: m.start()] + ("\n" if out[m.start() : m.start() + 1] == "\n" else "") + keep
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if tag:
        out = f"{out} {tag}".strip()
    return out, int(tag is not None), dropped


# Short Discord reactions that a person could just as well have answered with
# a gif. Only these are eligible for the small synthetic slice.
_REACTION_GIFS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"(l+m+f*a+o+|lo+l+|ha(ha)+h*|(i'?m )?dead|i'?m crying|[😂🤣💀☠️]+)"),
     ("dying laughing", "laughing so hard", "cat laughing", "crying laughing", "spongebob laughing",
      "man laughing hysterically", "monkey laughing", "laughing on floor")),
    (re.compile(r"(bru+h+|bro+|bruh moment|bro what)"),
     ("bruh moment", "disappointed stare", "blank stare", "bro really", "confused blink", "unimpressed face")),
    (re.compile(r"(no+ wa+y+|na+h+|wa+i+t+ what|wtf+|what+|huh+|\?{2,}|wh+a+t+ the)"),
     ("no way", "shocked face", "surprised pikachu", "wait what", "confused math lady", "jaw drop")),
    (re.compile(r"(true+|facts|real+|so true|fr+|fr fr|based|w+|yes+|ye+a+h*)"),
     ("so true", "nodding yes", "thumbs up", "facts nod", "you right", "agree nod")),
    (re.compile(r"(nice+|cool+|pog+|poggers|let'?s go+|lets go+|yay+|hype)"),
     ("lets go", "celebration dance", "hype dance", "nice thumbs up", "happy dance", "party time")),
    (re.compile(r"(sad+|rip|f|oof+|[😭😢]+|pain)"),
     ("sad cat", "crying sad", "press f", "sad violin", "rip funeral")),
    (re.compile(r"(e+w+|cringe|gross|yikes)"),
     ("cringe face", "disgusted face", "ew gross", "yikes face")),
    (re.compile(r"(a+w+|cute|adorable|🥺+)"),
     ("cute cat", "aww puppy", "heart eyes", "cute hamster")),
    (re.compile(r"(hi+|hello+|hey+|yo+|sup)"),
     ("cat waving hello", "wave hi", "hello there", "hey wave")),
    (re.compile(r"(bye+|gn|good ?night|cya)"),
     ("good night sleep", "bye wave", "sleepy cat", "see you later")),
]


def reaction_gif_words(text: str) -> tuple[str, ...] | None:
    """Candidate search phrases when `text` is a bare short reaction."""
    norm = re.sub(r"[.!,~*_]+", "", text.strip().lower()).strip()
    if not norm or len(norm) > 16 or "\n" in norm:
        return None
    for pattern, phrases in _REACTION_GIFS:
        if pattern.fullmatch(norm):
            return phrases
    return None


class GifStats:
    """Counts + a deterministic budgeted synthesizer for the gif slice.

    ``synth_frac`` caps synthetic tags as a fraction of the assistant targets
    seen so far (checked as they stream), so the slice can never exceed it.
    ``synth_rate`` is the chance an eligible short reaction is converted.
    """

    def __init__(self, synth_rate: float = 0.0, synth_frac: float = 0.0, seed: int = 0):
        self.synth_rate = synth_rate
        self.synth_frac = synth_frac
        self.seed = seed
        self.targets = 0
        self.real = 0
        self.synthetic = 0
        self.urls_dropped = 0
        self.url_targets_skipped = 0
        self.examples: list[str] = []
        self.synthetic_examples: list[str] = []

    def rewrite(self, text: str) -> str:
        out, made, dropped = rewrite_gifs(text)
        self.urls_dropped += dropped
        return out

    def count_target(self, response: str, *, synthetic: bool = False) -> None:
        self.targets += 1
        if GIF_TAG_RE.search(response):
            bucket = self.synthetic_examples if synthetic else self.examples
            if synthetic:
                self.synthetic += 1
            else:
                self.real += 1
            if len(bucket) < 40:
                bucket.append(response)

    def maybe_synthesize(self, key: str, text: str) -> str | None:
        """Turn a short reaction into a tag reply, deterministically by content."""
        if self.synth_rate <= 0 or GIF_TAG_RE.search(text):
            return None
        phrases = reaction_gif_words(text)
        if not phrases:
            return None
        if self.synthetic + 1 > self.synth_frac * (self.targets + 1):
            return None
        h = hashlib.sha256(f"{self.seed}\x1fgif\x1f{key}\x1f{text}".encode()).digest()
        if int.from_bytes(h[:4], "big") / 2**32 >= self.synth_rate:
            return None
        tag = f"[gif: {phrases[h[4] % len(phrases)]}]"
        # Mostly a bare gif; sometimes the words plus a gif, as people do.
        return tag if h[5] % 5 < 3 else f"{text.strip()} {tag}"

    def as_dict(self) -> dict:
        return {
            "targets": self.targets,
            "gif_real": self.real,
            "gif_synthetic": self.synthetic,
            "gif_urls_dropped": self.urls_dropped,
            "url_targets_skipped": self.url_targets_skipped,
        }


# ----------------------------------------------------------------- data ---


def _tinystories_records(split: str, seed: int, revision: str | None = None):
    """Yield (prompt, story) from TinyStoriesInstruct's line-per-row layout."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("roneneldan/TinyStoriesInstruct", split=split, streaming=True, revision=revision)
    fields: dict[str, str] = {}
    story: list[str] = []
    in_story = False
    for row in ds:
        line = row["text"]
        if line == "<|endoftext|>":
            text = "\n".join(story).strip()
            if text and (fields.get("Summary") or fields.get("Words")):
                summary = fields.get("Summary", "").strip().rstrip(".")
                words = fields.get("Words", "").strip()
                if summary:
                    summary = summary[0].lower() + summary[1:]
                tmpl = rng.choice(STORY_TEMPLATES)
                if "{words}" in tmpl and not words:
                    tmpl = STORY_TEMPLATES[0]
                if "{summary}" in tmpl and not summary:
                    tmpl = STORY_TEMPLATES[3]
                yield tmpl.format(summary=summary, words=words), text
            fields, story, in_story = {}, [], False
            continue
        if in_story:
            story.append(line)
        elif line.startswith("Story:"):
            in_story = True
        elif ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()


def _no_robots_records(split: str, revision: str | None = None):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/no_robots", split=split, revision=revision)
    for row in ds:
        msgs = row["messages"]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            yield msgs[0]["content"].strip(), msgs[1]["content"].strip()


_WP_FIXES = [
    (r"\s+([,.!?;:%])", r"\1"), (r"\s+'\s*(s|t|re|ve|ll|d|m)\b", r"'\1"), (r"\bn't\b", "n't"),
    (r"``\s*", '"'), (r"\s*''", '"'), (r"\(\s+", "("), (r"\s+\)", ")"), (r"\s+n't", "n't"),
    (r"([\"])\s+([^\"]*?)\s+([\"])", r"\1\2\3"), (r" {2,}", " "),
]


def _wp_clean(text: str) -> str:
    """Undo writingprompts' PTB-style tokenisation ("You 've", "`` quote '')."""
    import re

    text = re.sub(r"^\s*\[\s*[A-Z]{2,3}\s*\]\s*", "", text)  # [ WP ] / [ EU ] / [ TT ] tags
    for pat, rep in _WP_FIXES:
        text = re.sub(pat, rep, text)
    return text.strip()


def _writingprompts_records(split: str, max_chars: int = 3500, revision: str | None = None):
    """r/WritingPrompts prompt -> story, adult register. Long ones are skipped, not cut."""
    from datasets import load_dataset

    ds = load_dataset("euclaise/writingprompts", split=split, streaming=True, revision=revision)
    for row in ds:
        story = row["story"]
        if len(story) > max_chars or len(story) < 200:
            continue
        prompt = _wp_clean(row["prompt"])
        if not prompt:
            continue
        yield prompt, _wp_clean(story)


def _smoltalk_records(split: str, configs=("smol-magpie-ultra", "everyday-conversations"), revision: str | None = None):
    """First user->assistant turn of SmolTalk's general-assistant subsets."""
    from datasets import load_dataset

    for cfg in configs:
        ds = load_dataset("HuggingFaceTB/smoltalk", cfg, split=split, streaming=True, revision=revision)
        for row in ds:
            msgs = [m for m in row["messages"] if m["role"] in ("user", "assistant")]
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                yield msgs[0]["content"].strip(), msgs[1]["content"].strip()


def _chatml_turns(text: str) -> list[tuple[str, str]]:
    """Parse the concrete ChatML layout used by Discord-Dialogues.

    The dataset's actual `text` column is a sequence of
    `<|im_start|>role\nbody<|im_end|>` blocks, optionally followed by
    `<|end_of_text|>`. Unknown roles and empty bodies are ignored.
    """
    turns: list[tuple[str, str]] = []
    for chunk in text.split("<|im_start|>")[1:]:
        head, sep, rest = chunk.partition("\n")
        if not sep:
            continue
        role = head.strip()
        body = rest.split("<|im_end|>", 1)[0].strip()
        if role in ("user", "assistant") and body:
            turns.append((role, body))
    return turns


def _turn_records(
    source: str,
    group_id: str,
    turns: list[tuple[str, str]],
    gif: GifStats | None = None,
    synthesize: bool = False,
) -> list[SFTRecord]:
    """Alternating user/assistant turns -> one target per assistant turn.

    With ``gif``: gif URLs/filenames in every turn become `[gif: ...]` tags,
    a budgeted slice of short assistant reactions may become synthetic tags
    (``synthesize``), and assistant turns still carrying a raw URL are not
    used as targets (they stay in later turns' history).
    """
    history: list[ConversationTurn] = []
    records: list[SFTRecord] = []
    for i in range(0, len(turns) - 1, 2):
        user_role, user_text = turns[i]
        assistant_role, assistant_text = turns[i + 1]
        # Stay strict: a malformed turn must not silently assign one person's
        # words to the other role.
        if user_role != "user" or assistant_role != "assistant":
            return []
        synthetic = False
        if gif is not None:
            user_text = gif.rewrite(user_text)
            assistant_text = gif.rewrite(assistant_text)
            if synthesize:
                made = gif.maybe_synthesize(f"{group_id}\x1f{i}", assistant_text)
                if made:
                    assistant_text, synthetic = made, True
        if gif is not None and _URL_RE.search(assistant_text):
            gif.url_targets_skipped += 1
        elif user_text and assistant_text:
            if gif is not None:
                gif.count_target(assistant_text, synthetic=synthetic)
            records.append(
                SFTRecord(
                    source=source,
                    group_id=group_id,
                    current_user=user_text,
                    response=assistant_text,
                    history=tuple(history),
                )
            )
        history.append(ConversationTurn(user=user_text, assistant=assistant_text))
    return records


def _discord_group(raw: str, history_turns: int, gif: GifStats | None = None) -> list[SFTRecord]:
    """Turn one ChatML conversation into chronological assistant targets."""
    # Discord-Dialogues is documented as alternating two-author chains.
    return _turn_records("discord", _group_id(raw), _chatml_turns(raw), gif, synthesize=True)


def _discord_groups(split: str, history_turns: int, revision: str | None = None, gif: GifStats | None = None):
    """Yield all chronological assistant targets, grouped by conversation."""
    from datasets import load_dataset

    ds = load_dataset("mookiezi/Discord-Dialogues", split=split, streaming=True, revision=revision)
    for row in ds:
        records = _discord_group(row["text"], history_turns, gif)
        if records:
            yield records


def _messages_group(source: str, messages, gif: GifStats | None = None) -> list[SFTRecord]:
    """HF chat `messages` ([{role, content}]) -> per-assistant-turn targets.

    System turns are dropped; a leading assistant turn (no user before it) is
    skipped. Everything after must alternate or the conversation is dropped.
    """
    turns = [
        (m["role"], (m.get("content") or "").strip())
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
    while turns and turns[0][0] != "user":
        turns.pop(0)
    if len(turns) < 2:
        return []
    group_id = _group_id(*(f"{role}:{text}" for role, text in turns))
    return _turn_records(source, group_id, turns, gif)


def _ultrachat_groups(split: str = "train_sft", revision: str | None = None, gif: GifStats | None = None):
    """HuggingFaceH4/ultrachat_200k (MIT): multi-turn assistant Q&A, ~3.2 answers each."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True, revision=revision)
    for row in ds:
        records = _messages_group("ultrachat", row["messages"], gif)
        if records:
            yield records


def _smoltalk_multiturn_groups(
    split: str,
    configs=("everyday-conversations", "smol-magpie-ultra"),
    revision: str | None = None,
    gif: GifStats | None = None,
):
    """Every assistant turn of SmolTalk's multi-turn subsets, interleaved.

    everyday-conversations (~2.3k short chats) runs out quickly; interleaving
    keeps it from being all-or-nothing at the head of the stream.
    """
    from datasets import load_dataset

    streams = [
        iter(load_dataset("HuggingFaceTB/smoltalk", cfg, split=split, streaming=True, revision=revision))
        for cfg in configs
    ]
    while streams:
        for stream in list(streams):
            row = next(stream, None)
            if row is None:
                streams.remove(stream)
                continue
            records = _messages_group("smoltalk", row["messages"], gif)
            if records:
                yield records


def _single_record_groups(source: str, records):
    for prompt, response in records:
        yield [
            SFTRecord(
                source=source,
                group_id=_group_id(prompt, response),
                current_user=prompt,
                response=response,
            )
        ]


# ------------------------------------------------- Q&A / knowledge data ---
#
# Sources for the qa-v1 run (`configs/sft/qa-mac.json`). Every builder yields
# groups (lists of SFTRecord sharing a group_id) so the grouped val split keeps
# related targets on one side, exactly like the conversational sources above.

PERSONA_FILE = Path(__file__).resolve().parent / "data" / "booper_persona.jsonl"

# Replies that assert another assistant's identity contradict booper's persona
# ("I am Open Assistant", "as an AI language model"). Matched on the target
# only; a prompt may still ask "are you chatgpt?".
_IDENTITY_RE = re.compile(
    r"open ?assistant|\blaion\b|chat ?gpt|\bopenai\b|\bgpt-?[34]|\bas an ai\b|\bi(?:'m| am) an ai\b"
    r"|\blanguage model\b|\bai assistant\b",
    re.I,
)


def _hash_rng(*parts) -> random.Random:
    """Deterministic per-item RNG: same seed + same item -> same choices."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def identity_clash(text: str) -> bool:
    return bool(_IDENTITY_RE.search(text))


DOLLY_CATEGORIES = (
    "open_qa", "closed_qa", "general_qa", "brainstorming", "classification",
    "information_extraction", "summarization",
)


def dolly_prompt(instruction: str, context: str) -> str:
    """Dolly row -> user turn. Context-bearing rows keep their passage.

    closed_qa answers are drawn from the passage ("born on July 10, 1981"),
    so dropping it would teach confident recall of facts the 45M-active model
    cannot know. The passage therefore always rides along in the user turn.
    """
    instruction, context = instruction.strip(), context.strip()
    return f"{instruction}\n\n{context}" if context else instruction


def _dolly_records(split: str = "train", revision: str | None = None, max_context: int = 2000, max_response: int = 1500):
    """databricks/databricks-dolly-15k (CC-BY-SA-3.0), Q&A-shaped categories only."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split=split, streaming=True, revision=revision)
    for row in ds:
        if row["category"] not in DOLLY_CATEGORIES:
            continue
        response = (row["response"] or "").strip()
        context = row["context"] or ""
        if not response or len(response) > max_response or len(context) > max_context or identity_clash(response):
            continue
        yield dolly_prompt(row["instruction"] or "", context), response


def oasst_tree_groups(messages, source: str = "oasst", max_response: int = 2000) -> list[list[SFTRecord]]:
    """OASST message rows -> one group per English conversation tree.

    At every prompter node the best-ranked assistant reply (rank 0, or the
    only unranked reply) is the target, with the path from the root as its
    history. Follow-ups are explored under every reviewed reply, so a
    lower-ranked (but review-passed) reply can appear as *history* for a
    later best reply; it is never itself a target. Deleted, review-failed
    and synthetic messages are skipped, and nothing beneath a reply that
    fails the identity/length filter is used.
    """

    def ok(m) -> bool:
        return (
            m.get("lang") == "en"
            and not m.get("deleted")
            and m.get("review_result") is not False
            and not m.get("synthetic")
            and bool((m.get("text") or "").strip())
        )

    children: dict[str, list[dict]] = defaultdict(list)
    roots: list[dict] = []
    for m in messages:
        if not ok(m):
            continue
        if m.get("parent_id") is None:
            if m.get("role") == "prompter":
                roots.append(m)
        else:
            children[m["parent_id"]].append(m)

    def best_reply(prompter: dict) -> dict | None:
        replies = [c for c in children.get(prompter["message_id"], []) if c.get("role") == "assistant"]
        ranked = [c for c in replies if c.get("rank") is not None]
        if ranked:
            return min(ranked, key=lambda c: c["rank"])
        return replies[0] if len(replies) == 1 else None

    groups: list[list[SFTRecord]] = []
    for root in roots:
        gid = _group_id("oasst-tree", root["message_tree_id"])
        records: list[SFTRecord] = []
        stack: list[tuple[dict, tuple[ConversationTurn, ...]]] = [(root, ())]
        while stack:
            prompter, history = stack.pop()
            user = prompter["text"].strip()
            best = best_reply(prompter)
            replies = [c for c in children.get(prompter["message_id"], []) if c.get("role") == "assistant"]
            for reply in reversed(replies):
                answer = reply["text"].strip()
                if len(answer) > max_response or identity_clash(answer):
                    continue
                if reply is best:
                    records.append(
                        SFTRecord(source=source, group_id=gid, current_user=user, response=answer, history=history)
                    )
                turn = history + (ConversationTurn(user=user, assistant=answer),)
                follow = [c for c in children.get(reply["message_id"], []) if c.get("role") == "prompter"]
                stack.extend((c, turn) for c in reversed(follow))
        if records:
            groups.append(records)
    return groups


def _oasst_groups(split: str = "train", revision: str | None = None):
    """OpenAssistant/oasst2 (Apache-2.0): streamed whole (~64 MB), then rebuilt into trees."""
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst2", split=split, streaming=True, revision=revision)
    keep = ("message_id", "parent_id", "message_tree_id", "text", "role", "lang", "review_result", "deleted", "rank", "synthetic")
    rows = [{k: row.get(k) for k in keep} for row in ds if row.get("lang") == "en"]
    yield from oasst_tree_groups(rows)


# Last three of six train shards, read last-first. longctx-v1 only reached
# the head of shard 0, so these are disjoint from what the base has seen.
SMOL_SHORT_SHARDS = tuple(f"data/smol-magpie-ultra/train-0000{i}-of-00006.parquet" for i in (5, 4, 3))
SMOL_SHORT_SKIP_CATEGORIES = ("coding", "role-playing", "creative-writing", "editing", "data-analysis")


def _smol_short_records(revision: str | None = None, max_chars: int = 900):
    """Short first answers from SmolTalk smol-magpie-ultra (Apache-2.0).

    Reads only the LAST train shards. longctx-v1's `--smoltalk-multiturn`
    slice streamed the head of shard 0 (~20k conversations), so this is
    disjoint by construction rather than by a dedupe pass. Coding /
    role-play / editing / creative / data-analysis rows are skipped:
    booper's job here is answering, not writing code or reports.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceTB/smoltalk", data_files={"train": list(SMOL_SHORT_SHARDS)}, split="train", streaming=True, revision=revision
    )
    for row in ds:
        if row.get("category") in SMOL_SHORT_SKIP_CATEGORIES or row.get("quality") not in ("good", "excellent", "average"):
            continue
        msgs = [m for m in row["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        answer = msgs[1]["content"].strip()
        if not answer or len(answer) > max_chars or identity_clash(answer):
            continue
        yield msgs[0]["content"].strip(), answer


# Short-answer templating. A bare span ("Fernie Alpine Resort") is a fine
# answer but one fixed wrapper would become a tic, so each item draws from
# a pool; question-type pools keep the phrasing grammatical.
_QA_GENERIC = (
    "{a}", "{a}", "{a}!", "it's {a}", "that'd be {a}", "pretty sure it's {a}", "i think it's {a}",
    "{a} i think", "{a}, if i remember right", "should be {a}", "oh that's {a}", "the answer's {a}",
    "{a} afaik", "i believe it's {a}",
)
_QA_BY_TYPE = {
    "who": ("{a}", "{a}", "that'd be {a}", "it was {a}", "pretty sure it's {a}", "{a} i think", "i think it was {a}", "{a}!"),
    "when": ("{a}", "{a}", "in {a}", "that was {a}", "{a} i think", "pretty sure it was {a}", "i believe {a}", "{a}, if i remember right"),
    "where": ("{a}", "{a}", "in {a}", "it's in {a}", "{a} i think", "pretty sure it's {a}", "that'd be {a}"),
    "count": ("{a}", "{a}", "it's {a}", "{a} i think", "pretty sure it's {a}", "i believe {a}", "{a}!"),
}
_QA_PROMPTS = (
    "{q}", "{q}", "{q}?", "{q}?", "{Q}?", "hey booper {q}", "quick question, {q}?",
    "random q but {q}", "booper {q}?", "{q} ??",
)
# Only a wh-question reads right after "do you know" ("do you know who ...").
_QA_PROMPTS_WH = _QA_PROMPTS + ("do you know {q}?", "do you know {q}")
_WH_START = re.compile(r"(who|whom|whose|what|when|where|which|why|how)\b")


_YEARISH = re.compile(
    r"(?:(?:january|february|march|april|may|june|july|august|september|october|november|december) )?"
    r"\d{3,4}s?(?: (?:bc|bce|ad))?"
)


def _question_type(question: str) -> str | None:
    q = question.lower().lstrip()
    if q.startswith("who") or q.startswith("whom"):
        return "who"
    if q.startswith("when") or q.startswith("what year") or q.startswith("what date"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("how many") or q.startswith("how much"):
        return "count"
    return None


def clean_short_answer(answers) -> str | None:
    """1-3 short answer spans -> one phrase ("X", "X and Y", "X, Y and Z"); None if unusable."""
    spans: list[str] = []
    for a in answers:
        a = re.sub(r"\s+", " ", str(a).replace("\xa0", " ")).strip().rstrip(".")
        if a and a.lower() not in (s.lower() for s in spans):
            spans.append(a)
    if not spans or len(spans) > 3 or any(len(s) > 60 for s in spans):
        return None
    if len(spans) == 1:
        return spans[0]
    return ", ".join(spans[:-1]) + " and " + spans[-1]


def short_answer_pair(question: str, answers, seed: int = 0) -> tuple[str, str] | None:
    """NQ-style (question, answer spans) -> a casual (prompt, reply), deterministic per item."""
    question = re.sub(r"\s+", " ", question).strip().rstrip("?").strip()
    answer = clean_short_answer(answers)
    if not question or answer is None:
        return None
    rng = _hash_rng(seed, "qa", question, answer)
    pool = _QA_BY_TYPE.get(_question_type(question), _QA_GENERIC)
    # "in {a}" only reads right for a bare year/decade/month-year (and never
    # before an answer that already starts with a preposition).
    if pool is _QA_BY_TYPE["when"] and not _YEARISH.fullmatch(answer.lower()):
        pool = tuple(t for t in pool if not t.startswith("in "))
    if re.match(r"(in|on|at|the year) ", answer.lower()):
        pool = tuple(t for t in pool if "in {a}" not in t)
    template = rng.choice(pool)
    if not template.startswith("{a}") and re.match(r"(The|A|An) [a-z]", answer):
        answer = answer[0].lower() + answer[1:]  # "should be the asteroid belt"
    reply = template.format(a=answer)
    prompts = _QA_PROMPTS_WH if _WH_START.match(question.lower()) else _QA_PROMPTS
    prompt = rng.choice(prompts).format(q=question, Q=question[:1].upper() + question[1:])
    return prompt, reply


def _nq_records(split: str = "train", revision: str | None = None, seed: int = 0):
    """google-research-datasets/nq_open (CC-BY-SA-3.0), TRAIN split only, templated into chat replies."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/nq_open", split=split, streaming=True, revision=revision)
    for row in ds:
        pair = short_answer_pair(row["question"], row["answer"], seed)
        if pair:
            yield pair


# Synthetic arithmetic: generated, so always correct; problems are the split
# unit, so a held-out problem never appears in train under another phrasing.
_NUM_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty"
).split()
_ARITH_OPS = {
    "+": ((" + ", "+", " plus "), ("add {a} and {b}", "{a} and {b} added together")),
    "-": ((" - ", "-", " minus "), ("subtract {b} from {a}", "{a} take away {b}")),
    "*": ((" * ", "*", " x ", "x", " times ", " multiplied by ", " × "), ("multiply {a} by {b}", "{a} times {b}")),
    "/": ((" / ", "/", " divided by ", " ÷ "), ("divide {a} by {b}", "{a} divided by {b}")),
}
_ARITH_SYMBOL = {"+": ("+",), "-": ("-",), "*": ("*", "x", "×"), "/": ("/", "÷")}
_ARITH_PROMPTS = (
    "what's {e}", "what is {e}", "whats {e}", "{e}?", "{e}", "how much is {e}", "what does {e} equal",
    "{e} = ?", "quick math: {e}", "can you do {e}", "booper what's {e}", "hey what is {e}?", "solve {e}",
    "{e} is what", "what's {e}?", "What is {e}?", "whats {e} lol", "do you know what {e} is",
)
_ARITH_REPLIES = (
    "{r}", "{r}", "{r}", "{r}!", "that's {r}", "it's {r}", "{eq}", "{eq}", "easy, {r}", "{a} {s} {b} is {r}",
    "that'd be {r}", "{r} :)", "pretty sure it's {r}",
)


def arith_problem(rng: random.Random) -> tuple[int, str, int, int]:
    """(a, op, b, result). Small numbers mostly; division is always exact."""
    op = rng.choice("+-*/")
    if op == "+":
        hi = 20 if rng.random() < 0.7 else 100
        a, b = rng.randint(0, hi), rng.randint(0, hi)
        return a, op, b, a + b
    if op == "-":
        hi = 20 if rng.random() < 0.7 else 100
        a = rng.randint(0, hi)
        b = rng.randint(0, a)
        return a, op, b, a - b
    if op == "*":
        if rng.random() < 0.8:
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        else:
            a, b = rng.randint(2, 20), rng.randint(2, 9)
        return a, op, b, a * b
    b, q = rng.randint(1, 12), rng.randint(0, 12)
    return b * q, op, b, q


def arith_pair(a: int, op: str, b: int, result: int, rng: random.Random) -> tuple[str, str]:
    """One phrasing of a problem and one casual (always correct) reply."""

    def num(n: int) -> str:
        return _NUM_WORDS[n] if n <= 20 and rng.random() < 0.1 else str(n)

    infix, verbal = _ARITH_OPS[op]
    if rng.random() < 0.15:
        prompt = rng.choice(verbal).format(a=num(a), b=num(b))
        if rng.random() < 0.5:
            prompt += "?"
    else:
        left, right = num(a), num(b)
        ops = infix
        if not (left.isdigit() and right.isdigit()):
            ops = tuple(o for o in infix if o.startswith(" "))  # "onexten" is not a question
        expr = f"{left}{rng.choice(ops)}{right}"
        prompt = rng.choice(_ARITH_PROMPTS).format(e=expr)
    sym = rng.choice(_ARITH_SYMBOL[op])
    eq = f"{a} {sym} {b} = {result}" if rng.random() < 0.7 else f"{a}{sym}{b}={result}"
    reply = rng.choice(_ARITH_REPLIES).format(r=result, eq=eq, a=a, b=b, s=sym)
    return prompt, reply


def arith_groups(seed: int, source: str = "arith", variants: int = 3):
    """Endless deterministic stream of problems, each a group of distinct phrasings."""
    rng = random.Random(f"{seed}\x1farith")
    seen: set[tuple[int, str, int]] = set()
    while True:
        a, op, b, result = arith_problem(rng)
        if (a, op, b) in seen:
            continue
        seen.add((a, op, b))
        gid = _group_id("arith", str(a), op, str(b))
        pairs: dict[tuple[str, str], None] = {}
        for _ in range(variants * 3):
            pairs.setdefault(arith_pair(a, op, b, result, rng), None)
            if len(pairs) >= variants:
                break
        yield [SFTRecord(source=source, group_id=gid, current_user=p, response=r) for p, r in pairs]


def load_persona(path: Path = PERSONA_FILE) -> list[tuple[str, str]]:
    """Read and validate the hand-written persona pairs (raises on a bad file)."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"prompt", "response"}:
            raise ValueError(f"{path}:{n}: expected exactly prompt/response keys, got {sorted(row)}")
        prompt, response = (str(row[k]).strip() for k in ("prompt", "response"))
        if not prompt or not response:
            raise ValueError(f"{path}:{n}: empty prompt or response")
        if prompt.lower() in seen:
            raise ValueError(f"{path}:{n}: duplicate prompt {prompt!r}")
        seen.add(prompt.lower())
        pairs.append((prompt, response))
    if not pairs:
        raise ValueError(f"{path}: no persona pairs")
    return pairs


_PERSONA_PREFIXES = ("hey booper ", "booper ", "yo booper ", "hey ", "@booper ")


def persona_variants(prompt: str, seed: int = 0) -> list[str]:
    """The authored prompt plus surface variants people actually type."""
    rng = _hash_rng(seed, "persona", prompt)
    bare = prompt.rstrip("?!. ").lower()
    out = [prompt, bare + ("" if prompt.endswith("?") else "?"), rng.choice(_PERSONA_PREFIXES) + bare]
    if not bare.startswith(("hey", "hi", "yo", "hello", "good", "booper")):
        out.append(bare[:1].upper() + bare[1:] + "?")
    return list(dict.fromkeys(out))


def _persona_key(prompt: str) -> str:
    """Normalized question: case, punctuation and the address prefixes that
    `persona_variants` adds ("hey booper", "@booper") do not make it new."""
    key = prompt.lower().strip()
    for prefix in sorted(_PERSONA_PREFIXES, key=len, reverse=True):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return re.sub(r"[^a-z0-9 ]+", "", key).strip()


def persona_groups(path: Path = PERSONA_FILE, seed: int = 0, source: str = "persona"):
    """One group per normalized question, so "who are you" and "who are you?"
    (and their surface variants) can never straddle the train/val split."""
    grouped: dict[str, list[SFTRecord]] = {}
    for prompt, response in load_persona(path):
        key = _persona_key(prompt)
        gid = _group_id("persona", key)
        group = grouped.setdefault(key, [])
        have = {(r.current_user, r.response) for r in group}
        for v in persona_variants(prompt, seed):
            if (v, response) not in have:
                group.append(SFTRecord(source=source, group_id=gid, current_user=v, response=response))
    yield from grouped.values()


_WIKI_PROMPTS = (
    "tell me about {t}", "what do you know about {t}?", "do you know anything about {t}", "can you tell me about {t}?",
    "explain {t} to me", "{t}?", "what's the deal with {t}", "tell me something about {t}", "who or what is {t}?",
    "hey booper tell me about {t}", "give me a quick rundown on {t}",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def wiki_lead_pair(title: str, text: str, seed: int = 0, max_chars: int = 450) -> tuple[str, str] | None:
    """Article -> ("tell me about X", first 1-3 sentences of the lead), or None if it is not a topic page."""
    title = title.strip()
    if (
        not title
        or title.lower().startswith(("list of", "lists of"))
        or re.fullmatch(r"[\d\s\-–/]+(bc|bce|ad)?", title.lower())
        or "(disambiguation)" in title
    ):
        return None
    lead = re.sub(r"[ \t]{2,}", " ", text.strip().split("\n", 1)[0]).strip()
    if len(lead) < 60 or "may refer to" in lead or lead.endswith(":"):
        return None
    sentences = _SENTENCE_END.split(lead)
    rng = _hash_rng(seed, "wiki", title)
    keep = sentences[: rng.randint(1, 3)]
    while len(keep) > 1 and len(" ".join(keep)) > max_chars:
        keep.pop()
    answer = " ".join(keep).strip()
    if len(answer) > max_chars or len(answer) < 40 or not answer.endswith((".", "!", "?")):
        return None
    return rng.choice(_WIKI_PROMPTS).format(t=title), answer


def _wiki_records(revision: str | None = None, config: str = "20231101.simple", seed: int = 0):
    """wikimedia/wikipedia Simple English (CC-BY-SA-3.0 / GFDL) lead sentences, streamed."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True, revision=revision)
    for row in ds:
        pair = wiki_lead_pair(row["title"], row["text"], seed)
        if pair:
            yield pair


def _collect_groups(groups, want: int) -> list[SFTRecord]:
    """Take whole groups until at least `want` examples have been collected.

    Always at least two groups (when available): with one group the split
    would put everything in val and leave the source with no train data
    (seen at smoke scale, where persona's first group alone exceeds `want`).
    """
    out: list[SFTRecord] = []
    n_groups = 0
    for group in groups:
        out.extend(group)
        n_groups += 1
        if len(out) >= want and n_groups >= 2:
            break
    return out


def _split_grouped(
    records: list[SFTRecord], val_examples: int, *, seed: int
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Stable group split; related targets can never cross the boundary."""
    grouped: dict[str, list[SFTRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    ranked = sorted(
        grouped,
        key=lambda gid: hashlib.sha256(f"{seed}\x1f{gid}".encode()).digest(),
    )
    held: set[str] = set()
    n_val = 0
    for gid in ranked:
        if n_val >= val_examples:
            break
        held.add(gid)
        n_val += len(grouped[gid])
    return (
        [record for record in records if record.group_id not in held],
        [record for record in records if record.group_id in held],
    )


def _dedupe_groups(
    records: list[SFTRecord], seen_content: set[str]
) -> tuple[list[SFTRecord], int]:
    """Cull a whole group if any of its targets duplicates an earlier group."""
    grouped: dict[str, list[SFTRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    kept: list[SFTRecord] = []
    dropped = 0
    for group in grouped.values():
        content_ids = {_group_id(r.current_user, r.response) for r in group}
        if content_ids & seen_content:
            dropped += 1
            continue
        kept.extend(group)
        seen_content.update(content_ids)
    return kept, dropped


MIN_FITTED_PROMPT = 32  # below this a long response is skipped, not given a stub prompt


def _tokenize_records(tok, records: list[SFTRecord], args):
    """Records -> (uint16 token array, n_prompt) examples.

    The role transcript is fitted to ``min(prompt_budget, room left after the
    response)``, so a long answer drops its oldest history turns instead of
    being thrown away. Arrays are uint16 (vocab 16384): a multi-day run holds
    hundreds of thousands of long examples, which as Python int lists would
    cost several GB of the laptop's 16.
    """
    import numpy as np

    bos, sep, eos = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>"))
    examples: list[tuple[np.ndarray, int]] = []
    for record in records:
        r = tok.encode(record.response, add_special_tokens=False).ids
        if len(r) < args.min_response:
            continue
        if record.legacy_prompt:
            prompt = record.current_user
        else:
            budget = min(args.prompt_budget, args.seq_len - 3 - len(r))
            if budget < min(MIN_FITTED_PROMPT, args.prompt_budget):
                continue
            prompt = conversation_prompt_for_token_budget(
                record.history,
                record.current_user,
                max_turns=args.history_turns,
                max_chars=0,
                max_tokens=budget,
                token_count=lambda text: len(tok.encode(text, add_special_tokens=False).ids),
            )
        p = tok.encode(prompt, add_special_tokens=False).ids
        if len(p) + len(r) + 3 > args.seq_len:
            continue
        examples.append((np.asarray([bos, *p, sep, *r, eos], dtype=np.uint16), len(p) + 2))
    return examples


def _source_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    limit: float,
    *,
    guard_sources: tuple[str, ...] | list[str] = (),
    guard_limit: float | None = None,
    multiturn_must_improve: tuple[str, ...] | list[str] | None = None,
):
    """Gate role-formatted candidates against the legacy base and role baseline.

    ``*_single`` and ``*_legacy`` contain identical targets, so their
    cross-format delta measures the actual migration the user will experience.
    History-bearing views must improve, since learning follow-ups is the
    objective of this run rather than an optional side effect of rehearsal.

    ``multiturn_must_improve`` narrows that last rule to the named sources
    (None = every source, the multi-turn/longctx behaviour); the other
    ``*_multiturn`` views are then held to the ordinary ``limit``. A follow-on
    run whose objective is not multi-turn (qa-v1) cannot be expected to
    strictly improve conversation views the base was already trained on.

    ``guard_sources`` are rehearsal sources the run must not break: each must
    be present in the baseline (a missing guard fails the gate instead of
    silently passing it) and its role and multi-turn views must stay within
    ``guard_limit`` (default ``limit``).
    """
    passed, regressions = _source_gate_core(candidate, baseline, limit, multiturn_must_improve)
    if limit < 0 or not guard_sources:
        return passed, regressions
    g_limit = limit if guard_limit is None else guard_limit
    for name in guard_sources:
        if name not in baseline or name not in candidate:
            regressions[f"{name}_guard_missing"] = float("inf")
            passed = False
            continue
        for view in (name, f"{name}_multiturn"):
            if view in baseline:
                delta = candidate.get(view, float("inf")) - baseline[view]
                regressions[f"{view}_guard"] = delta
                if not delta <= g_limit:
                    passed = False
    return passed, regressions


def _source_gate_core(candidate, baseline, limit, multiturn_must_improve=None):
    complete = candidate.keys() == baseline.keys()
    finite = all(math.isfinite(value) for value in (*candidate.values(), *baseline.values()))
    primary = {
        name: candidate.get(name, float("inf")) - value
        for name, value in baseline.items()
        if not name.endswith(("_single", "_legacy", "_multiturn"))
    }
    retention = {
        name.removesuffix("_single"): candidate.get(name, float("inf"))
        - baseline.get(name.removesuffix("_single") + "_legacy", float("-inf"))
        for name in baseline
        if name.endswith("_single")
    }
    multiturn = {
        name.removesuffix("_multiturn"): candidate.get(name, float("inf")) - value
        for name, value in baseline.items()
        if name.endswith("_multiturn")
    }
    regressions = {
        **{f"{name}_role": delta for name, delta in primary.items()},
        **{f"{name}_migration": delta for name, delta in retention.items()},
        **{f"{name}_multiturn": delta for name, delta in multiturn.items()},
    }
    passed = complete and finite and (
        limit < 0
        or (
            all(delta <= limit for delta in primary.values())
            and all(delta <= limit for delta in retention.values())
            and all(
                delta < 0 if multiturn_must_improve is None or name in multiturn_must_improve else delta <= limit
                for name, delta in multiturn.items()
            )
        )
    )
    return passed, regressions


def _get(args, name, default=0):
    return getattr(args, name, default)


def _mix(args, name) -> float:
    return float(_get(args, name, 0.0) or 0.0)


def render_example(tok, example) -> tuple[str, str]:
    """Decode a tokenized example into (model input, supervised target) text.

    The target is exactly the span that receives loss in `batches`
    (``toks[n_prompt:]``: the response plus <eos>), so logging it proves the
    targets are response-only.
    """
    toks, n_prompt = example
    ids = [int(t) for t in toks]
    return tok.decode(ids[:n_prompt], skip_special_tokens=False), tok.decode(ids[n_prompt:], skip_special_tokens=False)


def build_examples(tok, args, log):
    """Build a mixed train set and source-specific, group-held-out validation."""
    sources = [
        ("tinystories", args.mix_story, 1, lambda: _single_record_groups("tinystories", _tinystories_records("train", args.seed, args.tinystories_revision))),
        ("writingprompts", args.mix_wp, 1, lambda: _single_record_groups("writingprompts", _writingprompts_records("train", revision=args.writingprompts_revision))),
        ("no_robots", args.mix_norobots, max(1, args.repeat_norobots), lambda: _single_record_groups("no_robots", _no_robots_records("train", args.no_robots_revision))),
        (
            "smoltalk",
            args.mix_smoltalk,
            1,
            (lambda: _smoltalk_multiturn_groups("train", revision=args.smoltalk_revision, gif=gif_stats["smoltalk"]))
            if args.smoltalk_multiturn
            else (lambda: _single_record_groups("smoltalk", _smoltalk_records("train", revision=args.smoltalk_revision))),
        ),
        ("discord", args.mix_discord, 1, lambda: _discord_groups("train", args.history_turns, args.discord_revision, gif=gif_stats["discord"])),
        # Appended last so earlier sources keep their split seeds (seed + index).
        ("ultrachat", args.mix_ultrachat, 1, lambda: _ultrachat_groups("train_sft", args.ultrachat_revision, gif=gif_stats["ultrachat"])),
        # Q&A / knowledge sources (qa-v1), again appended so older presets keep their seeds.
        ("dolly", _mix(args, "mix_dolly"), max(1, _get(args, "repeat_dolly", 1)), lambda: _single_record_groups("dolly", _dolly_records("train", args.dolly_revision))),
        ("oasst", _mix(args, "mix_oasst"), max(1, _get(args, "repeat_oasst", 1)), lambda: _oasst_groups("train", args.oasst_revision)),
        ("smol_short", _mix(args, "mix_smol_short"), 1, lambda: _single_record_groups("smol_short", _smol_short_records(args.smoltalk_revision, args.smol_short_max_chars))),
        ("nq", _mix(args, "mix_nq"), 1, lambda: _single_record_groups("nq", _nq_records("train", args.nq_revision, args.seed))),
        ("arith", _mix(args, "mix_arith"), 1, lambda: arith_groups(args.seed)),
        ("persona", _mix(args, "mix_persona"), max(1, _get(args, "repeat_persona", 1)), lambda: persona_groups(Path(args.persona_file), args.seed)),
        ("wiki", _mix(args, "mix_wiki"), 1, lambda: _single_record_groups("wiki", _wiki_records(args.wiki_revision, args.wiki_config, args.seed))),
    ]
    gif_on = bool(getattr(args, "gif_tags", False))
    gif_stats: dict[str, GifStats | None] = {
        "smoltalk": GifStats(seed=args.seed) if gif_on else None,
        "discord": GifStats(args.gif_synth_rate, args.gif_synth_max_frac, args.seed) if gif_on else None,
        "ultrachat": GifStats(seed=args.seed) if gif_on else None,
    }
    total = sum(w for _, w, _, _ in sources)
    train_records: list[SFTRecord] = []
    val_records: dict[str, list[SFTRecord]] = {}
    counts: dict[str, dict[str, int]] = {}
    seen_content: set[str] = set()
    for source_i, (name, weight, repeat_train, groups) in enumerate(sources):
        if weight <= 0:
            continue
        want = int(args.examples * weight / total)
        want_val = max(1, round(args.val_examples * weight / total))
        # Repetition is applied only after group splitting. In particular,
        # no_robots' second pass can no longer leak exact duplicates into val.
        unique_want = math.ceil(want / repeat_train) + want_val
        unique = _collect_groups(groups(), unique_want)
        unique, duplicate_groups = _dedupe_groups(unique, seen_content)
        source_train, source_val = _split_grouped(
            unique, want_val, seed=args.seed + source_i
        )
        repeated_train = (source_train * repeat_train)[:want]
        train_records.extend(repeated_train)
        val_records[name] = source_val
        # Measure compatibility with the raw-prompt behavior Story-v2 serves
        # outside the opt-in conversation mode. This is evaluation only: new
        # training inputs all use the shared role transcript.
        single = [
            SFTRecord(
                source=f"{name}_single",
                group_id=record.group_id,
                current_user=record.current_user,
                response=record.response,
            )
            for record in source_val
            if not record.history
        ]
        legacy = [
            SFTRecord(
                source=f"{name}_legacy",
                group_id=record.group_id,
                current_user=record.current_user,
                response=record.response,
                legacy_prompt=True,
            )
            for record in source_val
            if not record.history
        ]
        multiturn = [
            SFTRecord(
                source=f"{name}_multiturn",
                group_id=record.group_id,
                current_user=record.current_user,
                response=record.response,
                history=record.history,
            )
            for record in source_val
            if record.history
        ]
        if single:
            val_records[f"{name}_single"] = single
        if legacy:
            val_records[f"{name}_legacy"] = legacy
        if multiturn:
            val_records[f"{name}_multiturn"] = multiturn
        counts[name] = {
            "train": len(repeated_train),
            "val": len(source_val),
            "groups": len({r.group_id for r in unique}),
            "duplicate_groups_dropped": duplicate_groups,
        }
        if gif_stats.get(name) is not None:
            stats = gif_stats[name]
            counts[name].update(stats.as_dict())
            counts[name]["gif_tagged_train"] = sum(bool(GIF_TAG_RE.search(r.response)) for r in repeated_train)
            counts[name]["history_train"] = sum(bool(r.history) for r in repeated_train)
            log(f"gif: {name} {stats.as_dict()} tagged train targets {counts[name]['gif_tagged_train']}")
            for kind, examples in (("real", stats.examples), ("synthetic", stats.synthetic_examples)):
                for example in examples[:6]:
                    log(f"gif example ({name}, {kind}): {example[-160:]!r}")
        log(
            f"data: {name} -> {len(repeated_train)} train / "
            f"{len(source_val)} val examples in {counts[name]['groups']} groups"
        )
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    train_by_source = {
        name: _tokenize_records(tok, [r for r in train_records if r.source == name], args)
        for name, weight, _, _ in sources
        if weight > 0
    }
    for name, examples in train_by_source.items():
        # Logged, not stored in `counts`: counts feed the resume signature of
        # older presets, which must stay byte-identical.
        tokens = sum(len(e[0]) for e in examples)
        log(f"data: {name} tokenized {len(examples)} train examples, mean len {tokens / max(len(examples), 1):.0f}")
        for example in examples[: max(0, int(_get(args, "log_examples", 0)))]:
            prompt, target = render_example(tok, example)
            log(f"data example ({name}): input={prompt[-300:]!r} target={target[:300]!r}")
    empty_train = [name for name, examples in train_by_source.items() if not examples]
    if empty_train:
        raise RuntimeError(f"active sources produced no train examples after tokenization: {empty_train}")
    train = [example for examples in train_by_source.values() for example in examples]
    rng.shuffle(train)
    val = {
        name: _tokenize_records(tok, records, args)
        for name, records in val_records.items()
        if not name.endswith(("_single", "_legacy"))
    }
    # Keep the migration views exactly paired after tokenization. A long role
    # prefix can make an example fail the sequence budget even when its raw
    # legacy prompt fits; including only one side would invalidate the loss
    # comparison.
    for name, weight, _, _ in sources:
        if weight <= 0:
            continue
        single_records = val_records.get(f"{name}_single", [])
        legacy_records = val_records.get(f"{name}_legacy", [])
        paired_single: list[tuple[list[int], int]] = []
        paired_legacy: list[tuple[list[int], int]] = []
        for single_record, legacy_record in zip(single_records, legacy_records, strict=True):
            single_example = _tokenize_records(tok, [single_record], args)
            legacy_example = _tokenize_records(tok, [legacy_record], args)
            if single_example and legacy_example:
                paired_single.extend(single_example)
                paired_legacy.extend(legacy_example)
        val[f"{name}_single"] = paired_single
        val[f"{name}_legacy"] = paired_legacy
    empty_val = [name for name, examples in val.items() if not examples]
    if empty_val:
        raise RuntimeError(f"active validation sources produced no examples after tokenization: {empty_val}")
    return train, val, counts


def batch_shape(n_rows: int, max_len: int, tokens_per_batch: int, pad_multiple: int = 1, fixed_rows: bool = False):
    """(rows, width) of a padded batch; see `batches`."""
    width = -(-max_len // pad_multiple) * pad_multiple
    rows = max(n_rows, tokens_per_batch // width) if fixed_rows else n_rows
    return rows, width


def batches(examples, tokens_per_batch, pad_id, shuffle_seed=None, skip=0, pad_multiple=1, fixed_rows=False):
    """Length-bucketed batches under a token budget (padding counted).

    ``skip`` drops the first N batches without materialising them (resume).
    ``pad_multiple`` rounds each batch width up and ``fixed_rows`` fills the
    batch to ``tokens_per_batch // width`` rows: on MPS every new tensor shape
    compiles and caches another graph, and the cache never shrinks. With free
    shapes a seq-2048 run grew to a 15-19 GB footprint on a 16 GB Mac (mostly
    swapped-out graph objects). Filler rows are a lone BOS (so attention never
    sees a fully-masked row) with no loss.
    """
    import numpy as np

    groups = _batch_groups(examples, tokens_per_batch, shuffle_seed, pad_multiple)
    for g in groups[skip:]:
        rows, width = batch_shape(
            len(g), max(len(examples[i][0]) for i in g), tokens_per_batch, pad_multiple, fixed_rows
        )
        ids = torch.full((rows, width), pad_id, dtype=torch.long)
        labels = torch.full((rows, width), -100, dtype=torch.long)
        ids[len(g) :, 0] = int(examples[g[0]][0][0])
        for r, i in enumerate(g):
            toks, n_prompt = examples[i]
            row = torch.from_numpy(np.asarray(toks, dtype=np.int64))
            ids[r, : len(toks)] = row
            labels[r, n_prompt : len(toks)] = row[n_prompt:]
        yield ids, labels


def _batch_groups(examples, tokens_per_batch, shuffle_seed=None, pad_multiple=1) -> list[list[int]]:
    """Example indices per batch; deterministic for a given shuffle seed.

    Lengths are rounded up to ``pad_multiple`` so the padded batch still fits
    ``tokens_per_batch`` (which makes fixed-row shapes exact).
    """

    def size(i):
        return -(-len(examples[i][0]) // pad_multiple) * pad_multiple

    order = list(range(len(examples)))
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(order)
        # Sort within wide windows so padding stays small but order stays random.
        window = 256
        chunks = [order[i : i + window] for i in range(0, len(order), window)]
        order = [i for c in chunks for i in sorted(c, key=lambda j: len(examples[j][0]))]
        groups = []
        cur, cur_max = [], 0
        for i in order:
            n = size(i)
            if cur and max(cur_max, n) * (len(cur) + 1) > tokens_per_batch:
                groups.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, n)
        if cur:
            groups.append(cur)
        rng.shuffle(groups)
    else:
        order.sort(key=lambda j: len(examples[j][0]))
        groups, cur, cur_max = [], [], 0
        for i in order:
            n = size(i)
            if cur and max(cur_max, n) * (len(cur) + 1) > tokens_per_batch:
                groups.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, n)
        if cur:
            groups.append(cur)
    return groups


# ---------------------------------------------------------------- model ---


def bucketed_experts_forward(self, hidden_states, top_k_index, top_k_weights, bucket=128):
    """Mixtral experts with a bounded set of MPS shapes.

    The eager HF loop runs each expert on exactly its routed tokens, so the
    matmul shapes change with every batch's routing. MPS compiles and keeps a
    graph per shape (grouped_mm with data-dependent offsets does the same),
    which leaked ~35 MB of CPU heap per micro-batch here: a 15 GB footprint on
    a 16 GB Mac within minutes, measured with vmmap. Each expert's token count
    is padded up to a multiple of ``bucket`` instead (padding rows gather a
    real token, get weight 0 and land in a dummy output row), so only
    ``seq/bucket`` shapes ever exist. Mathematically identical to the eager
    loop; costs up to ``bucket - 1`` wasted rows per active expert.
    """
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    pairs = num_tokens * num_top_k
    expert_ids = top_k_index.reshape(-1)
    weights = top_k_weights.reshape(-1)
    expert_sorted, perm = torch.sort(expert_ids, stable=True)
    counts = torch.histc(expert_sorted.float(), bins=self.num_experts, min=0, max=self.num_experts - 1)
    out = torch.zeros(pairs + 1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
    start = 0
    for expert, n in enumerate(int(c) for c in counts.tolist()):  # one sync per layer, like eager
        if n == 0:
            continue
        padded = -(-n // bucket) * bucket
        pos = torch.arange(padded, device=hidden_states.device)
        valid = pos < n
        pair = perm[start + pos.clamp(max=n - 1)]
        x = hidden_states[pair // num_top_k]
        gate, up = F.linear(x, self.gate_up_proj[expert]).chunk(2, dim=-1)
        y = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
        y = y * (weights[pair] * valid).unsqueeze(-1).to(y.dtype)
        out.index_add_(0, torch.where(valid, pair, pairs), y.to(out.dtype))
        start += n
    return out[:pairs].view(num_tokens, num_top_k, hidden_dim).sum(dim=1)


def use_bucketed_experts(model, bucket=128) -> int:
    """Bind `bucketed_experts_forward` onto every fused Mixtral experts module."""
    import types

    patched = 0
    for module in model.modules():
        if type(module).__name__ == "MixtralExperts" and hasattr(module, "gate_up_proj"):
            module.forward = types.MethodType(
                lambda self, h, idx, w, _b=bucket: bucketed_experts_forward(self, h, idx, w, bucket=_b), module
            )
            patched += 1
    return patched


def load_base(model_dir: Path, device, log):
    from babble.hfserve import _load_int8

    model, config = _load_int8(model_dir)
    model.train()
    log(f"model: {sum(p.numel() for p in model.parameters()):,} params from {model_dir}")
    return model.to(device), config


def fetch_base(cache: Path, log) -> Path:
    from huggingface_hub import snapshot_download

    if (cache / "model-int8.safetensors").exists():
        return cache
    log(f"fetching {INT8_REPO} -> {cache}")
    return Path(snapshot_download(INT8_REPO, local_dir=str(cache)))


def save_ckpt(model, config, tok_path: Path, out: Path, step: int, tokens: int, opt=None):
    from safetensors.torch import save_file

    tmp = out.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    # NOTE: `.to("cpu", torch.bfloat16)` in ONE call corrupts the source fp32
    # tensor on MPS (torch 2.13) -- measured: val 2.33 -> 4.12 from that line
    # alone. Copy to CPU first, cast second. Same in export_int8.
    state = {k: v.detach().to("cpu").to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    if "lm_head.weight" in state and getattr(config, "tie_word_embeddings", False):
        del state["lm_head.weight"]  # tied; re-tied on load
    save_file(state, str(tmp / "model.safetensors"))
    config.save_pretrained(tmp)
    shutil.copy(tok_path, tmp / "tokenizer.json")
    (tmp / "state.json").write_text(json.dumps({"step": step, "tokens": tokens}))
    if opt is not None:
        torch.save(opt.state_dict(), tmp / "optim.pt")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def load_ckpt(model, ckpt: Path, device, opt=None):
    from safetensors.torch import load_file

    state = load_file(str(ckpt / "model.safetensors"))
    missing, unexpected = model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=False)
    assert not unexpected and all(m == "lm_head.weight" for m in missing), (missing, unexpected)
    model.tie_weights()
    meta = json.loads((ckpt / "state.json").read_text())
    if opt is not None and (ckpt / "optim.pt").exists():
        opt.load_state_dict(torch.load(ckpt / "optim.pt", map_location=device))
    return meta["step"], meta["tokens"]


def _restore_prompt_metadata(config, ckpt: Path):
    """Make exported prompt metadata match the saved weights exactly."""
    saved = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    for key in PROMPT_METADATA_KEYS:
        if key in saved:
            setattr(config, key, saved[key])
        elif hasattr(config, key):
            delattr(config, key)


def export_int8(model, config, tok_path: Path, src_dir: Path, out: Path, log):
    """Re-pack into the per-output-channel symmetric INT8 layout of the live snapshot."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to("cpu").to(torch.float32) for k, v in model.state_dict().items()}
    fused = any(".mlp.experts.gate_up_proj" in k for k in sd)
    packed: dict[str, torch.Tensor] = {}
    report: dict[str, dict] = {}

    def q(name: str, w: torch.Tensor):
        if w.dim() == 1:
            packed[name] = w.to(torch.bfloat16)
            return
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        packed[name] = torch.round(w / scale).clamp(-127, 127).to(torch.int8).contiguous()
        packed[name + ".scale"] = scale.to(torch.bfloat16).contiguous()
        report[name] = {"axis": 1, "dtype": "int8", "scale": name + ".scale"}

    for name, w in sd.items():
        if name == "lm_head.weight" and "model.embed_tokens.weight" in sd:
            q(name, w)  # stored explicitly in the original artifact too
            continue
        if ".mlp." in name and fused:
            continue
        if ".block_sparse_moe." in name:
            q(name, w)
            continue
        q(name, w)
    if "lm_head.weight" not in sd:
        q("lm_head.weight", sd["model.embed_tokens.weight"])
    if fused:
        inter = config.intermediate_size
        for layer in range(config.num_hidden_layers):
            new = f"model.layers.{layer}.mlp"
            old = f"model.layers.{layer}.block_sparse_moe"
            q(old + ".gate.weight", sd[new + ".gate.weight"])
            gate_up = sd[new + ".experts.gate_up_proj"]
            down = sd[new + ".experts.down_proj"]
            for e in range(config.num_local_experts):
                q(f"{old}.experts.{e}.w1.weight", gate_up[e, :inter])
                q(f"{old}.experts.{e}.w3.weight", gate_up[e, inter:])
                q(f"{old}.experts.{e}.w2.weight", down[e])
    save_file(packed, str(out / "model-int8.safetensors"))
    config.save_pretrained(out)
    shutil.copy(tok_path, out / "tokenizer.json")
    for extra in ("tokenizer_config.json", "load_int8.py"):
        if (src_dir / extra).exists():
            shutil.copy(src_dir / extra, out / extra)
    (out / "quantization_config.json").write_text(
        json.dumps(
            {
                "activations": "bfloat16",
                "granularity": "per-output-channel",
                "quant_method": "booper_symmetric_int8",
                "source": "sft/sft_longform.py",
                "tensors": report,
            },
            indent=2,
        )
    )
    log(f"export: {out / 'model-int8.safetensors'} ({(out / 'model-int8.safetensors').stat().st_size / 1e6:.1f} MB)")
    # Round-trip through the serving loader so a bad pack fails here, not live.
    from babble.hfserve import _load_int8

    _load_int8(out)
    log("export: round-trip load through babble.hfserve._load_int8 OK")


def push_export(run_dir: Path, repo: str, log):
    """Upload runs/<name>/export (plus a model card built from metrics.jsonl) to the Hub."""
    from huggingface_hub import HfApi

    export = run_dir / "export"
    recs = [json.loads(l) for l in (run_dir / "metrics.jsonl").open()] if (run_dir / "metrics.jsonl").exists() else []
    start = next((r for r in recs if r.get("event") == "start"), {})
    evals = [r for r in recs if "val" in r]
    trains = [r for r in recs if "loss" in r]
    samples = evals[-1].get("samples", []) if evals else []
    card = [
        "---", "license: apache-2.0", "language: [en]", "library_name: transformers", "pipeline_tag: text-generation",
        f"base_model: {INT8_REPO}", "tags: [moe, booper, int8, sft, long-form]", "---", "",
        f"# {repo.split('/')[-1]}", "",
        f"Long-form SFT of `{INT8_REPO}` (Mixtral MoE, 150M total / ~50M active, vocab 16384) so booper answers",
        "story/long-answer requests instead of one-liners. Trained with `sft/sft_longform.py` from",
        "https://github.com/frgmt0/babble in the pair layout `<bos> prompt <sep> response <eos>` (loss on the response).", "",
        "## Data mix", "",
        *(f"- {k}: {v} examples" for k, v in (start.get("counts") or {}).items()),
        "", "## Training", "",
        f"- steps: {trains[-1]['step'] if trains else '?'}, tokens: {trains[-1]['tokens']:,}" if trains else "- (no train records)",
        f"- val loss: {evals[0]['val']:.4f} -> {evals[-1]['val']:.4f}" if len(evals) > 1 else "",
        f"- device: {start.get('device', '?')}, lr {start.get('args', {}).get('lr')}, seq len {start.get('args', {}).get('seq_len')}",
        "", "## Samples (temperature 0.8)", "",
        *(f"**{s['prompt']}**\n\n> {s['reply'].replace(chr(10), chr(10) + '> ')}\n" for s in samples),
        "", "## Loading", "",
        "Same INT8 layout as the base: `load_int8.py` in this repo, or `babble.hfserve` with `BABBLE_HF_MODEL_DIR` pointed at a snapshot.",
    ]
    (export / "README.md").write_text("\n".join(c for c in card if c is not None))
    shutil.copy(run_dir / "metrics.jsonl", export / "metrics.jsonl")
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    log(f"push: uploading {export} -> https://huggingface.co/{repo}")
    api.upload_folder(folder_path=str(export), repo_id=repo, commit_message=f"SFT run {run_dir.name}")
    log(f"push: done https://huggingface.co/{repo}")


# ------------------------------------------------------------ training ---


@torch.no_grad()
def free_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def duty_sleep_seconds(compute_s: float, duty: float) -> float:
    """Sleep that makes compute / (compute + sleep) == duty.

    duty 1.0 never sleeps; 0.5 sleeps exactly as long as it computed; 0.25
    sleeps three times as long.
    """
    if not 0.0 < duty <= 1.0:
        raise ValueError(f"duty cycle must be in (0, 1], got {duty}")
    return max(0.0, compute_s) * (1.0 - duty) / duty


def on_battery(pmset_output: str) -> bool:
    """Parse `pmset -g batt`: first line is "Now drawing from 'AC Power'" or 'Battery Power'."""
    return "'Battery Power'" in pmset_output


def pmset_on_battery() -> bool:
    """False when not on macOS or pmset fails: never pause on a guess."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return on_battery(out)


def rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)


def sample(model, tok, device, max_new=120, draws=2, prompt_budget=512):
    bos, sep, eos, pad = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>", "<pad>"))
    model.eval()
    outs = []
    for p in SAMPLE_PROMPTS:
        rendered = conversation_prompt_for_token_budget(
            (),
            p,
            max_turns=0,
            max_chars=0,
            max_tokens=prompt_budget,
            token_count=lambda text: len(tok.encode(text, add_special_tokens=False).ids),
        )
        ids = torch.tensor([[bos, *tok.encode(rendered, add_special_tokens=False).ids, sep]], device=device)
        # Two draws at a cooler temperature than serving (0.5 vs 0.8) so a
        # sample says something about the weights, not the dice; no_repeat_ngram
        # matches what live serves.
        gen = model.generate(
            ids, do_sample=True, temperature=0.5, top_p=0.95, max_new_tokens=max_new, num_return_sequences=draws,
            eos_token_id=eos, pad_token_id=pad, repetition_penalty=1.1, no_repeat_ngram_size=4,
        )[:, ids.shape[1] :]
        for row in gen:
            keep = [int(t) for t in row if int(t) not in (pad, eos)]
            outs.append({"prompt": p, "reply": tok.decode(keep, skip_special_tokens=True).strip(), "tokens": len(keep)})
        del gen
        free_cache(device)
    model.train()
    return outs


@torch.no_grad()
def evaluate(model, val, device, pad, dtype, pad_multiple=1, fixed_rows=False):
    model.eval()
    tot, n = 0.0, 0
    for ids, labels in batches(val, 2048, pad, pad_multiple=pad_multiple, fixed_rows=fixed_rows):
        ids, labels = ids.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(input_ids=ids, attention_mask=(ids != pad), use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        tot += float(loss)
        n += int((labels[:, 1:] != -100).sum())
        del logits, loss
    free_cache(device)
    model.train()
    return tot / max(n, 1)


def evaluate_sources(model, val_by_source, device, pad, dtype, pad_multiple=1, fixed_rows=False):
    """Return aggregate and per-source loss over independent holdouts."""
    per_source = {
        name: evaluate(model, examples, device, pad, dtype, pad_multiple, fixed_rows)
        for name, examples in val_by_source.items()
        if examples
    }
    # Aggregate each target once. Suffixed views are diagnostic/gating slices
    # of these same primary source holdouts.
    combined = [
        example
        for name, examples in val_by_source.items()
        if not name.endswith(("_single", "_legacy", "_multiturn"))
        for example in examples
    ]
    return evaluate(model, combined, device, pad, dtype, pad_multiple, fixed_rows), per_source


class Reporter:
    """Append to metrics.jsonl and optionally POST each record to the /runs endpoint."""

    def __init__(self, run_dir: Path, name: str):
        self.path = run_dir / "metrics.jsonl"
        self.name = name
        self.url = os.environ.get("BABBLE_RUNS_URL")
        self.token = os.environ.get("BABBLE_RUNS_TOKEN")

    def __call__(self, rec: dict):
        rec = {"run": self.name, "t": time.time(), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if self.url and self.token:
            try:
                req = urllib.request.Request(
                    f"{self.url.rstrip('/')}/api/runs/{self.name}",
                    data=json.dumps(rec, default=str).encode(),
                    # Cloudflare's bot rules 403 the default "Python-urllib" agent.
                    headers={"content-type": "application/json", "authorization": f"Bearer {self.token}", "user-agent": "babble-sft/1.0"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as e:  # never let the dashboard kill a run
                print(f"reporter: {e}", flush=True)


DATA_CACHE_VERSION = 1
# Bump when a qa source's filtering/templating changes, so caches and resume
# signatures of qa presets invalidate (older presets never include it).
QA_DATA_VERSION = 1
# mix flag -> other build inputs of that source (recorded only when active).
QA_SOURCE_SETTINGS = {
    "mix_dolly": ("dolly_revision", "repeat_dolly"),
    "mix_oasst": ("oasst_revision", "repeat_oasst"),
    "mix_smol_short": ("smoltalk_revision", "smol_short_max_chars"),
    "mix_nq": ("nq_revision",),
    "mix_arith": (),
    "mix_persona": ("persona_file", "repeat_persona"),
    "mix_wiki": ("wiki_revision", "wiki_config"),
}


def _data_extras(args) -> dict:
    """Build-relevant settings added after multiturn-v1 (only when in use)."""
    extras = {}
    if getattr(args, "mix_ultrachat", 0) > 0:
        extras["mix_ultrachat"] = args.mix_ultrachat
        extras["ultrachat_revision"] = args.ultrachat_revision
    if getattr(args, "smoltalk_multiturn", False):
        extras["smoltalk_multiturn"] = True
    if getattr(args, "gif_tags", False):
        extras["gif_tags"] = True
        extras["gif_synth_rate"] = args.gif_synth_rate
        extras["gif_synth_max_frac"] = args.gif_synth_max_frac
    # qa-v1 sources: recorded only when active, so pre-qa signatures are unchanged.
    for mix, settings in QA_SOURCE_SETTINGS.items():
        if _mix(args, mix) > 0:
            extras[mix] = _mix(args, mix)
            for name in settings:
                extras[name] = _get(args, name, None)
    if extras.get("mix_persona"):
        extras["persona_sha256"] = hashlib.sha256(Path(args.persona_file).read_bytes()).hexdigest()
    if any(name.startswith("mix_") and name in QA_SOURCE_SETTINGS for name in extras):
        extras["qa_data_version"] = QA_DATA_VERSION
    return extras


def _cached_build(tok, tok_path: Path, args, run_dir: Path, log):
    """build_examples, memoised in runs/<name>/data-cache.pkl.

    Streaming and tokenizing a few hundred thousand long conversations takes a
    while; a multi-day laptop run gets stopped and resumed (reboots, battery),
    so resumes reuse the identical tokenized split instead of rebuilding it.
    The key covers every build input, so a changed preset rebuilds.
    """
    import pickle

    keyed = {
        name: getattr(args, name)
        for name in (
            "examples", "val_examples", "seed", "seq_len", "prompt_budget", "history_turns", "min_response",
            "mix_story", "mix_wp", "mix_norobots", "repeat_norobots", "mix_smoltalk", "mix_discord",
            "tinystories_revision", "writingprompts_revision", "no_robots_revision", "smoltalk_revision",
            "discord_revision",
        )
    }
    keyed.update(_data_extras(args))
    keyed["version"] = DATA_CACHE_VERSION
    keyed["tokenizer"] = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(keyed, sort_keys=True, default=str).encode()).hexdigest()
    path = run_dir / "data-cache.pkl"
    if path.exists():
        try:
            with path.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == key:
                log(f"data: reusing tokenized cache {path.name}")
                return cached["train"], cached["val"], cached["counts"]
            log("data: cache key changed; rebuilding")
        except Exception as e:  # a torn cache must never block a resume
            log(f"data: unreadable cache ({e}); rebuilding")
    train, val, counts = build_examples(tok, args, log)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"key": key, "train": train, "val": val, "counts": counts}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return train, val, counts


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, help="JSON preset; explicit CLI flags override it")
    known, _ = pre.parse_known_args()
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=float, default=30e6, help="training-token budget (input tokens incl. prompt)")
    ap.add_argument("--examples", type=int, default=120_000, help="examples to tokenize across the mix")
    ap.add_argument("--mix-story", type=float, default=0.45, help="roneneldan/TinyStoriesInstruct")
    ap.add_argument("--mix-wp", type=float, default=0.0, help="euclaise/writingprompts (adult-register fiction)")
    ap.add_argument("--mix-norobots", type=float, default=0.15, help="HuggingFaceH4/no_robots")
    ap.add_argument("--repeat-norobots", type=int, default=1, help="no_robots is only ~8.4k pairs; epochs to upsample it")
    ap.add_argument("--mix-smoltalk", type=float, default=0.0, help="HuggingFaceTB/smoltalk general subsets")
    ap.add_argument("--mix-discord", type=float, default=0.40, help="mookiezi/Discord-Dialogues rehearsal")
    ap.add_argument("--tinystories-revision", default=None)
    ap.add_argument("--writingprompts-revision", default=None)
    ap.add_argument("--no-robots-revision", default=None)
    ap.add_argument("--smoltalk-revision", default=None)
    ap.add_argument("--discord-revision", default=None)
    ap.add_argument("--mix-ultrachat", type=float, default=0.0, help="HuggingFaceH4/ultrachat_200k multi-turn Q&A (MIT)")
    ap.add_argument("--ultrachat-revision", default=None)
    ap.add_argument("--smoltalk-multiturn", action="store_true", help="train every assistant turn of SmolTalk's multi-turn subsets")
    # Q&A / knowledge sources (qa-v1). Licences: every one permits redistribution of derived models.
    ap.add_argument("--mix-dolly", type=float, default=0.0, help="databricks/databricks-dolly-15k Q&A categories (CC-BY-SA-3.0); passage kept in the user turn")
    ap.add_argument("--dolly-revision", default=None)
    ap.add_argument("--repeat-dolly", type=int, default=1, help="dolly has ~12k usable rows; epochs to upsample it (keep <= 3)")
    ap.add_argument("--mix-oasst", type=float, default=0.0, help="OpenAssistant/oasst2 English, best-ranked replies as multi-turn trees (Apache-2.0)")
    ap.add_argument("--oasst-revision", default=None)
    ap.add_argument("--repeat-oasst", type=int, default=1, help="oasst2 English has ~15k usable targets; epochs to upsample (keep <= 3)")
    ap.add_argument("--mix-smol-short", type=float, default=0.0, help="HuggingFaceTB/smoltalk smol-magpie-ultra short first answers from the last train shard (Apache-2.0); uses --smoltalk-revision")
    ap.add_argument("--smol-short-max-chars", type=int, default=900, help="longest smol-magpie-ultra answer kept by --mix-smol-short")
    ap.add_argument("--mix-nq", type=float, default=0.0, help="google-research-datasets/nq_open train questions, answers templated into casual replies (CC-BY-SA-3.0)")
    ap.add_argument("--nq-revision", default=None)
    ap.add_argument("--mix-arith", type=float, default=0.0, help="synthetic + - x / arithmetic generated in-script from --seed (no licence; always correct)")
    ap.add_argument("--mix-persona", type=float, default=0.0, help="hand-written booper identity pairs from --persona-file (project-authored)")
    ap.add_argument("--persona-file", default=str(PERSONA_FILE))
    ap.add_argument("--repeat-persona", type=int, default=1, help="persona is ~150 pairs (x3-4 surface variants); epochs to upsample (keep <= 3)")
    ap.add_argument("--mix-wiki", type=float, default=0.0, help="wikimedia/wikipedia Simple English lead sentences as 'tell me about X' (CC-BY-SA-3.0/GFDL)")
    ap.add_argument("--wiki-revision", default=None)
    ap.add_argument("--wiki-config", default="20231101.simple")
    ap.add_argument("--log-examples", type=int, default=0, help="log this many rendered train examples (input + loss-bearing target) per source at data build")
    ap.add_argument("--guard-sources", nargs="*", default=[], help="rehearsal sources that must be present in val and stay within --guard-max-regression (role + multiturn views)")
    ap.add_argument("--guard-max-regression", type=float, default=None, help="regression ceiling for --guard-sources (default: --max-source-val-regression)")
    ap.add_argument("--multiturn-must-improve", nargs="*", default=None, help="only these sources' *_multiturn views must strictly improve (others: regression ceiling); default all")
    ap.add_argument("--gif-tags", action="store_true", help="rewrite gif URLs/filenames into [gif: words] tags")
    ap.add_argument("--gif-synth-rate", type=float, default=0.0, help="chance a short Discord reaction becomes a synthetic gif tag")
    ap.add_argument("--gif-synth-max-frac", type=float, default=0.03, help="hard cap: synthetic gif targets / Discord targets")
    ap.add_argument("--duty-cycle", type=float, default=1.0, help="fraction of wall time spent computing; 0.5 sleeps as long as each optimizer step took")
    ap.add_argument("--expert-bucket", type=int, default=0, help="pad each MoE expert's token count to this multiple (bounds MPS graph-cache growth); 0 = HF eager loop")
    ap.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing: much less MPS memory for ~1/3 more compute")
    ap.add_argument("--fixed-rows", action="store_true", help="fill every batch to tokens_per_batch // width rows (bounded set of MPS shapes)")
    ap.add_argument("--pad-multiple", type=int, default=1, help="round batch widths up to this (bounds MPS shape caches)")
    ap.add_argument("--mps-high-watermark", type=float, default=0.7, help="PYTORCH_MPS_HIGH_WATERMARK_RATIO (hard cap, fraction of RAM)")
    ap.add_argument("--mps-low-watermark", type=float, default=0.5, help="PYTORCH_MPS_LOW_WATERMARK_RATIO (allocator frees cached blocks above this)")
    ap.add_argument("--prepare-data", action="store_true", help="build runs/<name>/data-cache.pkl and exit (a fresh build leaves a fat RSS; train from the cache)")
    ap.add_argument("--pause-on-battery", action="store_true", help="macOS: sleep while `pmset -g batt` reports battery power")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--prompt-budget", type=int, default=256)
    ap.add_argument("--history-turns", type=int, default=3, help="completed exchanges retained before the current user turn")
    ap.add_argument("--min-response", type=int, default=1)
    ap.add_argument("--val-examples", type=int, default=400)
    ap.add_argument("--tokens-per-batch", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", default=None, help="dir with model-int8.safetensors (default: fetch to artifacts/)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export", action="store_true", help="only re-pack runs/<name>/ckpt to INT8")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 600 examples, 12 steps")
    ap.add_argument("--push", default=None, metavar="NAMESPACE/REPO", help="after export, upload runs/<name>/export to this HF repo (uses the cached HF login)")
    ap.add_argument(
        "--max-source-val-regression",
        type=float,
        default=0.05,
        help="maximum allowed loss increase for every source; negative disables the gate",
    )
    if known.config:
        preset = json.loads(known.config.read_text(encoding="utf-8"))
        valid = {action.dest for action in ap._actions}
        unknown = sorted(set(preset) - valid)
        if unknown:
            ap.error(f"unknown config keys in {known.config}: {', '.join(unknown)}")
        ap.set_defaults(**preset)
    args = ap.parse_args()
    if args.smoke:
        args.examples, args.tokens, args.val_examples = 600, 12 * args.tokens_per_batch * args.accum, 50
        args.log_every, args.eval_every, args.ckpt_every = 2, 6, 6
        args.max_source_val_regression = -1
    sample_kwargs = {"prompt_budget": args.prompt_budget}
    if args.smoke:
        sample_kwargs.update({"max_new": 48, "draws": 1})

    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = (run_dir / "train.log").open("a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    # Bound the MPS allocator to a fraction of unified memory so a leak fails
    # loudly instead of paging the whole machine (default 1.0 = "everything").
    # Both must be set: PyTorch's default LOW is 1.4 and it rejects HIGH < LOW.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(args.mps_high_watermark))
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(args.mps_low_watermark))
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    log(f"device={device} autocast={dtype}")

    base = Path(args.base) if args.base else fetch_base(ROOT / "artifacts" / "hf-booper-big-chat-int8", log)
    from tokenizers import Tokenizer

    tok_path = base / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    pad = tok.token_to_id("<pad>")
    model, config = load_base(base, device, log)
    if args.expert_bucket > 0:
        log(f"model: bucketed MoE experts (x{args.expert_bucket}) on {use_bucketed_experts(model, args.expert_bucket)} layers")
    if args.grad_checkpoint and not args.export:
        # Recompute activations in backward: at seq 2048 they are most of the
        # MPS footprint. `use_cache=False` is passed per call instead of being
        # written into config, which is saved with the export and serves with
        # the KV cache on.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("model: gradient checkpointing on")
    ckpt_dir = run_dir / "ckpt"

    if args.export:
        quality_path = run_dir / "quality.json"
        export_ckpt = ckpt_dir
        if quality_path.exists():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if int(quality.get("best_step", 0)) <= 0 or not (run_dir / "best").exists():
                raise RuntimeError("quality gate has no passing checkpoint to export")
            export_ckpt = run_dir / "best"
        _restore_prompt_metadata(config, export_ckpt)
        step, tokens = load_ckpt(model, export_ckpt, device)
        log(f"export from passing step {step} ({tokens:,} tokens)")
        export_int8(model, config, tok_path, base, run_dir / "export", log)
        if args.push:
            push_export(run_dir, args.push, log)
        return

    # Persist the input contract beside newly trained weights so promotion
    # tooling can distinguish them from raw-prompt Story-v2 checkpoints.
    config.babble_prompt_format = "role_transcript_v1"
    config.babble_history_turns = args.history_turns
    config.babble_prompt_budget = args.prompt_budget

    if not 0.0 < args.duty_cycle <= 1.0:
        ap.error("--duty-cycle must be in (0, 1]")
    train, val_by_source, counts = _cached_build(tok, tok_path, args, run_dir, log)
    if args.prepare_data:
        log(f"data: cache ready in {run_dir / 'data-cache.pkl'}; --prepare-data exits here")
        return
    n_val = sum(len(examples) for examples in val_by_source.values())
    log(f"data: {len(train)} train / {n_val} val examples, mean len {sum(len(e[0]) for e in train)/max(len(train),1):.0f}")
    data_provenance = {
        "counts": counts,
        "seed": args.seed,
        "examples": args.examples,
        "mix": {name: getattr(args, name) for name in ("mix_story", "mix_wp", "mix_norobots", "mix_smoltalk", "mix_discord")},
        "revisions": {name: getattr(args, name) for name in ("tinystories_revision", "writingprompts_revision", "no_robots_revision", "smoltalk_revision", "discord_revision")},
        "prompt_format": "role_transcript_v1",
        "history_turns": args.history_turns,
        "prompt_budget": args.prompt_budget,
        "seq_len": args.seq_len,
    }
    extras = _data_extras(args)
    if extras:  # absent for older presets, so their signatures stay resumable
        data_provenance["extras"] = extras
    data_signature = hashlib.sha256(
        json.dumps(data_provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    steps_total = max(1, int(args.tokens // (args.tokens_per_batch * args.accum)))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    step, tokens_seen = 0, 0
    if args.resume and ckpt_dir.exists():
        step, tokens_seen = load_ckpt(model, ckpt_dir, device, opt)
        log(f"resumed at step {step}, {tokens_seen:,} tokens")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = min(1.0, (s - args.warmup) / max(1, steps_total - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    report = Reporter(run_dir, args.name)
    report({"event": "start", "duty_cycle": args.duty_cycle, "steps_total": steps_total, "device": str(device), "counts": counts, "args": vars(args), "resumed_step": step})
    quality_path = run_dir / "quality.json"
    best_dir = run_dir / "best"
    resuming_quality = args.resume and quality_path.exists()
    # A resume already has its baseline in quality.json; re-running the full
    # long-context eval (many minutes on a throttled laptop) buys nothing.
    v, source_val = (None, None) if resuming_quality else evaluate_sources(model, val_by_source, device, pad, dtype, args.pad_multiple, args.fixed_rows)
    if resuming_quality:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality.get("data_signature") != data_signature:
            raise RuntimeError("refusing to resume: dataset revisions, split, or prompt format changed")
        baseline_val = float(quality["baseline_val"])
        baseline_source_val = {k: float(v) for k, v in quality["baseline_source_val"].items()}
        best_val = float(quality["best_val"])
        best_step = int(quality["best_step"])
    else:
        baseline_val, baseline_source_val = v, source_val
        best_val = float("inf") if args.max_source_val_regression < 0 else baseline_val
        best_step = -1 if args.max_source_val_regression < 0 else 0

    def write_quality():
        quality_path.write_text(
            json.dumps(
                {
                    "baseline_val": baseline_val,
                    "baseline_source_val": baseline_source_val,
                    "best_val": best_val,
                    "best_step": best_step,
                    "max_source_val_regression": args.max_source_val_regression,
                    "data_signature": data_signature,
                    "data_provenance": data_provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def consider_candidate(candidate_val, candidate_sources):
        nonlocal best_val, best_step
        source_ok, regressions = _source_gate(
            candidate_sources,
            baseline_source_val,
            args.max_source_val_regression,
            guard_sources=tuple(args.guard_sources or ()),
            guard_limit=args.guard_max_regression,
            multiturn_must_improve=args.multiturn_must_improve,
        )
        improved = candidate_val < best_val
        if source_ok and improved and step > 0:
            best_val, best_step = candidate_val, step
            save_ckpt(model, config, tok_path, best_dir, step, tokens_seen)
            write_quality()
            log(f"best: step {step} val {candidate_val:.4f} source gate passed")
        return source_ok, regressions

    write_quality()
    if v is not None:
        log(f"step {step} val {v:.4f} sources={source_val} rss {rss_gb():.1f}G")
        report({"step": step, "val": v, "source_val": source_val, "samples": sample(model, tok, device, **sample_kwargs)})
        log(f"samples done rss {rss_gb():.1f}G")

    epoch = 0
    t_log, tok_log, loss_acc, loss_n = time.perf_counter(), 0, 0.0, 0
    micro = 0
    # Throttle ("passive, non-impeding"): after each optimizer step, sleep so
    # compute / wall == duty_cycle. Sleep and battery pauses are excluded from
    # tok_s_active so the dashboard shows both the real and the raw rate.
    t_active = time.perf_counter()
    idle_log = 0.0
    last_batt_check = 0.0
    # Resume continues where the data order left off instead of replaying the
    # head of epoch 0: skip the micro-batches the checkpoint already consumed.
    to_skip = step * args.accum
    while step < steps_total:
        if to_skip:
            per_epoch = len(_batch_groups(train, args.tokens_per_batch, args.seed + epoch, args.pad_multiple))
            if to_skip >= per_epoch:
                to_skip -= per_epoch
                epoch += 1
                continue
            log(f"resume: skipping {to_skip} of {per_epoch} batches in epoch {epoch}")
        skip, to_skip = to_skip, 0
        for ids, labels in batches(train, args.tokens_per_batch, pad, shuffle_seed=args.seed + epoch, skip=skip, pad_multiple=args.pad_multiple, fixed_rows=args.fixed_rows):
            ids, labels = ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(input_ids=ids, attention_mask=(ids != pad), use_cache=False).logits
            n_tgt = int((labels[:, 1:] != -100).sum())
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / args.accum).backward()
            loss_acc += float(loss.detach())
            del logits, loss
            loss_n += 1
            n_real = int((ids != pad).sum())  # filler rows / padding are not training tokens
            tokens_seen += n_real
            tok_log += n_real
            micro += 1
            if micro % args.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            nap = duty_sleep_seconds(time.perf_counter() - t_active, args.duty_cycle)
            if nap > 0:
                time.sleep(nap)
                idle_log += nap
            if args.pause_on_battery and time.perf_counter() - last_batt_check > 60:
                last_batt_check = time.perf_counter()
                if pmset_on_battery():
                    t_pause = time.perf_counter()
                    log("battery: on battery power; pausing until AC")
                    report({"event": "pause", "reason": "battery", "step": step})
                    while pmset_on_battery():
                        time.sleep(60)
                    paused = time.perf_counter() - t_pause
                    idle_log += paused
                    log(f"battery: back on AC after {paused/60:.0f}m; resuming")
                    report({"event": "resume", "reason": "battery", "step": step, "paused_s": paused})
            t_active = time.perf_counter()
            if step % args.log_every == 0:
                dt = time.perf_counter() - t_log
                rec = {
                    "step": step, "loss": loss_acc / loss_n, "lr": lr_at(step - 1), "grad_norm": float(gn),
                    "tok_s": tok_log / dt, "tok_s_active": tok_log / max(dt - idle_log, 1e-9),
                    "duty_cycle": args.duty_cycle, "idle_s": idle_log,
                    "tokens": tokens_seen, "steps_total": steps_total,
                    "eta_s": (steps_total - step) * dt / args.log_every, "rss_gb": rss_gb(),
                }
                log(f"step {step}/{steps_total} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['tok_s']:.0f} tok/s ({rec['tok_s_active']:.0f} active, duty {args.duty_cycle:g}) rss {rec['rss_gb']:.1f}G eta {rec['eta_s']/60:.0f}m")
                report(rec)
                t_log, tok_log, loss_acc, loss_n = time.perf_counter(), 0, 0.0, 0
                idle_log = 0.0
            if step % args.eval_every == 0:
                free_cache(device)
                v, source_val = evaluate_sources(model, val_by_source, device, pad, dtype, args.pad_multiple, args.fixed_rows)
                source_ok, regressions = consider_candidate(v, source_val)
                s = sample(model, tok, device, **sample_kwargs)
                log(f"step {step} val {v:.4f} | sample: {s[0]['reply'][:160]!r} ({s[0]['tokens']} tok)")
                report({"step": step, "val": v, "source_val": source_val, "source_regression": regressions, "source_gate": source_ok, "samples": s})
            if step % args.ckpt_every == 0:
                save_ckpt(model, config, tok_path, ckpt_dir, step, tokens_seen, opt)
                log(f"ckpt saved at step {step}")
            if step >= steps_total:
                break
        epoch += 1
    save_ckpt(model, config, tok_path, ckpt_dir, step, tokens_seen, opt)
    v, source_val = evaluate_sources(model, val_by_source, device, pad, dtype, args.pad_multiple, args.fixed_rows)
    source_ok, regressions = consider_candidate(v, source_val)
    s = sample(model, tok, device, **sample_kwargs)
    gated = best_step <= 0
    report({"step": step, "val": v, "source_val": source_val, "source_regression": regressions, "source_gate": source_ok, "best_step": best_step, "best_val": best_val, "gated": gated, "samples": s, "event": "done"})
    log(f"done: step {step} val {v:.4f}; best step {best_step} val {best_val:.4f}")
    if gated:
        log("export gated: no post-base checkpoint improved aggregate val while passing every source gate")
        return
    load_ckpt(model, best_dir, device)
    export_int8(model, config, tok_path, base, run_dir / "export", log)
    if args.push:
        push_export(run_dir, args.push, log)


if __name__ == "__main__":
    main()
