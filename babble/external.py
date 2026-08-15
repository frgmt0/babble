"""The external base-stage corpus: dictionary words and simple short stories.

This is the text stage 1 (`babble base-pretrain`) learns word shape and grammar
from *before* the human voice pass ever runs. None of it is anybody's Discord
message, so **none of it goes through the consent path** -- that gate is for the
human corpus in `corpus.py`, and the two never mix. A word list has no author; a
public story dataset has no consenting Discord user behind each line.

Two sources, deliberately kept apart from the consented rows:

* **A word list** -- for word *shape* and spelling. `/usr/share/dict/cracklib-small`
  ships on this box; point `BABBLE_WORDLIST_PATH` at a larger list for more. A
  word list on its own cannot teach grammar, because it contains no sentences.
* **Simple short stories** -- TinyStories (`roneneldan/TinyStories`), downloaded
  once and cached under `external/`. This is the part that actually teaches
  sentence structure.

`prepare_base_corpus` writes a single `base_corpus.jsonl` (one `{"text": ...}`
per line) that stage 1 reads back with `read_base_rows`. It **fails loudly** --
raises `EmptyCorpusError` -- rather than let a base pretrain silently train on
nothing when a download or a word list is missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

# TinyStories separates stories with this literal token. Splitting on it turns
# one giant file into individual stories, each of which becomes a corpus row.
STORY_SEPARATOR = "<|endoftext|>"

# A "usable" word: plain lowercase letters, optionally an internal apostrophe,
# 2-20 long. This throws away the cracklib junk ("007bond", "063dyjuy") and keeps
# the real English words, which are the only thing that teaches spelling.
_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")


class EmptyCorpusError(RuntimeError):
    """Raised when the prepared base corpus would be empty. Fail loud, never
    train on nothing."""


@dataclass
class BasePrepareResult:
    path: Path
    words: int
    stories: int
    story_chars: int
    total_rows: int
    total_chars: int

    def summary(self) -> str:
        return (
            f"base corpus: {self.total_rows} rows, {self.total_chars} chars "
            f"({self.words} words + {self.stories} stories / {self.story_chars} story chars) "
            f"-> {self.path}"
        )


# --- word list -----------------------------------------------------------


def load_words(source: Path, *, limit: int = 0) -> list[str]:
    """Usable, de-duplicated, lowercased words from a word-list file.

    `limit` of 0 keeps every usable word; a positive limit keeps the first that
    many, in file order.
    """
    if not source.exists():
        raise EmptyCorpusError(f"word list not found: {source}")
    seen: set[str] = set()
    words: list[str] = []
    for raw in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        word = raw.strip().lower()
        if not _WORD_RE.match(word) or word in seen:
            continue
        seen.add(word)
        words.append(word)
        if limit and len(words) >= limit:
            break
    return words


# --- stories -------------------------------------------------------------


def download_stories(settings: Settings) -> Path:
    """Fetch the stories file once and cache it under `external/`. Returns the
    local path. Raises loudly if the download is unavailable in this environment.
    """
    cache_dir = settings.external_dir / "hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - huggingface_hub is a hard dep
        raise EmptyCorpusError(f"huggingface_hub unavailable: {exc}") from exc
    try:
        local = hf_hub_download(
            repo_id=settings.stories_repo,
            filename=settings.stories_file,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    except Exception as exc:  # network down, gated repo, offline -- fail loud
        raise EmptyCorpusError(
            f"could not download stories {settings.stories_repo}/{settings.stories_file}: "
            f"{type(exc).__name__}: {exc}. Point BABBLE_STORIES_FILE at a local file, "
            f"or run with a fixture."
        ) from exc
    return Path(local)


def load_stories(source: Path, *, char_budget: int = 0) -> list[str]:
    """Individual, whole stories from a TinyStories-style file.

    `char_budget` caps the total characters kept, but only ever on a story
    boundary -- a partial story is never kept, and at least one story always is.
    0 keeps every story.
    """
    if not source.exists():
        raise EmptyCorpusError(f"stories file not found: {source}")
    text = source.read_text(encoding="utf-8", errors="ignore")
    stories = [s for s in (part.strip() for part in text.split(STORY_SEPARATOR)) if s]
    if not char_budget:
        return stories
    kept: list[str] = []
    total = 0
    for story in stories:
        if kept and total + len(story) > char_budget:
            break  # stop on a boundary rather than truncate this story
        kept.append(story)
        total += len(story)
        if total >= char_budget:
            break
    return kept


# --- the combined corpus -------------------------------------------------


def prepare_base_corpus(
    settings: Settings,
    *,
    wordlist_path: Path | None = None,
    stories_path: Path | None = None,
    word_limit: int | None = None,
    story_chars: int | None = None,
) -> BasePrepareResult:
    """Build `base_corpus.jsonl` from the word list and the stories.

    Pass `wordlist_path` / `stories_path` to use local files (tests, or a
    pre-downloaded story split); leave `stories_path` unset to download the
    configured HF file once and cache it. Raises `EmptyCorpusError` if the
    result would be empty -- a base pretrain must never silently see no data.
    """
    wordlist_path = wordlist_path or settings.wordlist_path
    word_limit = settings.base_word_limit if word_limit is None else word_limit
    story_chars = settings.base_story_chars if story_chars is None else story_chars

    words = load_words(wordlist_path, limit=word_limit)
    stories_src = stories_path or download_stories(settings)
    stories = load_stories(stories_src, char_budget=story_chars)

    if not words and not stories:
        raise EmptyCorpusError(
            f"no base corpus produced from {wordlist_path} + {stories_src}"
        )

    settings.base_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    story_chars_written = 0
    with open(settings.base_corpus_path, "w", encoding="utf-8") as fh:
        # Words first, then stories -- order does not matter, batches are sampled
        # with replacement, but it keeps the file readable.
        for word in words:
            fh.write(json.dumps({"text": word}, ensure_ascii=False) + "\n")
        for story in stories:
            story_chars_written += len(story)
            fh.write(json.dumps({"text": story}, ensure_ascii=False) + "\n")

    total_rows = len(words) + len(stories)
    total_chars = sum(len(w) for w in words) + story_chars_written
    return BasePrepareResult(
        path=settings.base_corpus_path,
        words=len(words),
        stories=len(stories),
        story_chars=story_chars_written,
        total_rows=total_rows,
        total_chars=total_chars,
    )


def read_base_rows(settings: Settings) -> list[str]:
    """The prepared base corpus as a list of text rows. Raises loudly if the
    corpus has not been prepared or is empty -- stage 1 must not train on nothing.
    """
    path = settings.base_corpus_path
    if not path.exists():
        raise EmptyCorpusError(
            f"no base corpus at {path}; run `babble prepare-base` first"
        )
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line)["text"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if text:
            rows.append(text)
    if not rows:
        raise EmptyCorpusError(f"base corpus at {path} is empty")
    return rows
