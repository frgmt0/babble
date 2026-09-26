"""Turn a model-emitted `[gif: <query>]` marker into a real GIF URL.

The SFT run teaches the model to end a reply with `[gif: 2-5 lowercase search
words]` (or make that the whole reply) when a GIF fits better than words.
This module is the other half of that contract: find the marker, ask a
provider for candidate GIFs, and hand back a direct media URL Discord can
embed on its own -- nothing here ever uploads or re-hosts an image.

Three things kept deliberately separate, same discipline as the rest of the
bot:

* **Parsing** (`extract_gif_tag`) is pure text surgery -- no network, no
  provider, always available, always tested without a token.
* **Providers** (`GifProvider` implementations) know how to turn a query into
  a list of candidate URLs. Swappable by name (`BABBLE_GIF_PROVIDER`), so a
  provider going away (as Tenor's own API is, in 2026) is a config change,
  not a rewrite.
* **`GifResolver`** is the thing `core.py` actually holds: it wraps a
  provider with a cache and a "never raise, never hang" contract, and is
  `None`-provider (`enabled = False`) when GIFs are off or misconfigured, so
  every caller can skip the `if` and just call `.resolve()`.

Every provider call is best-effort: a bad network, a timeout, a malformed
response, an empty result -- all of it comes back as `None`/`[]`, never an
exception. A GIF lookup must never be allowed to break a reply that would
otherwise have gone out fine as plain text.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

#: How long any single provider call is allowed to take. Discord replies are
#: on the clock; a slow GIF lookup must lose, not stall, the reply.
TIMEOUT_SECONDS = 1.5

#: How long a query's candidate list stays cached before a fresh lookup.
CACHE_TTL_SECONDS = 600.0

#: How many distinct queries the cache remembers before evicting the oldest.
CACHE_MAX_ENTRIES = 256

#: Random pick among the top-N results, not always the first -- so the same
#: query doesn't always post the identical GIF.
TOP_N_RESULTS = 6

#: Cloudflare (and friends) 403 the stdlib's default "Python-urllib/3.x"
#: user agent before a request ever reaches the page. A browser-shaped UA is
#: the documented workaround; this is the one `curl -A "Mozilla/5.0"` used to
#: verify the scrape path works at all.
USER_AGENT = "Mozilla/5.0 (compatible; babble-bot/1.0; +https://github.com/kowo-co/babble)"

GIFS_ENV = "BABBLE_GIFS"
PROVIDER_ENV = "BABBLE_GIF_PROVIDER"
API_KEY_ENV = "BABBLE_GIF_API_KEY"

DEFAULT_PROVIDER = "tenor-scrape"


# --- tag parsing -----------------------------------------------------------

_OPEN_RE = re.compile(r"\[\s*gif\s*:\s*", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalise_query(raw: str) -> str:
    """Fold a raw tag body down to lowercase, space-joined search words.

    Strips punctuation the model might emit around the words ("cats!",
    "cats, dogs", trailing periods) without touching the words themselves.
    """
    text = raw.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    words = _SPACE_RE.split(text.strip())
    return " ".join(w for w in words if w)


def extract_gif_tag(text: str) -> tuple[str, str | None]:
    """Pull a `[gif: query]` marker out of `text`.

    Returns `(remaining_text, query)`. `query` is `None` when there is no
    tag, or when what looked like one didn't qualify:

    * **Whole reply or trailing only.** The contract is "the whole reply, or
      at the end" -- a tag found mid-sentence with real content after it is
      not a marker, it's the model quoting the syntax, and is left alone.
    * **Unterminated at the end.** A generation cut off mid-tag by
      `max_new_tokens` (`"...that reminds me of [gif: sunse"`) has no `]` to
      find. There is no query to search for, so the whole dangling fragment
      is dropped and the rest of the reply stands on its own.
    * **Case and whitespace** inside `[gif: ...]` are irrelevant --
      `[GIF:cats]`, `[ gif :  cats ]` and `[gif: cats]` all parse the same.

    Only the *last* occurrence of the opening marker is considered, so a
    reply that merely mentions `[gif:` earlier is not corrupted by treating
    that mention as the boundary.
    """
    opens = list(_OPEN_RE.finditer(text))
    if not opens:
        return text, None
    last_open = opens[-1]
    tail = text[last_open.end() :]
    close = tail.find("]")
    before = text[: last_open.start()].rstrip()
    if close == -1:
        # Unterminated -- a truncated generation cut off mid-tag. Strip it.
        return before, None
    after = tail[close + 1 :]
    if after.strip():
        # Something real follows the closing bracket -- not a recognised
        # marker position (whole reply, or the end of it). Leave untouched.
        return text, None
    query = normalise_query(tail[:close])
    return before, (query or None)


# --- providers --------------------------------------------------------------


class GifProvider(Protocol):
    """Turns a query into candidate direct-media GIF URLs, most-relevant first.

    Implementations may raise on any failure (network, parsing, HTTP status)
    -- `GifResolver` is the layer that catches and swallows, so a provider is
    free to be as strict as it likes about what counts as success.
    """

    def search(self, query: str) -> list[str]: ...


_TENOR_MEDIA_RE = re.compile(r"https://media\.tenor\.com/[^\s\"'\\<>]+?\.gif")


def extract_tenor_media_urls(html: str) -> list[str]:
    """Pull direct `media.tenor.com/....gif` URLs out of a tenor.com search page.

    Verified (2026-09-26): without any API key, `https://tenor.com/search/
    <query-words-joined-by-dash>-gifs` fetched with a browser-shaped
    User-Agent returns an HTML page whose embedded JSON/attributes contain
    these URLs directly, each of which serves `200 image/gif` and embeds
    fine in Discord on its own. This is screen-scraping a page Tenor did not
    publish as an API and offers no stability guarantee -- Tenor's own GIF
    *API* is being retired in 2026 (new key issuance already closed), which
    is exactly why this fragile page-scrape is the *keyless* fallback here
    and not the recommended path; see the keyed provider below for that.
    """
    seen: dict[str, None] = {}
    for match in _TENOR_MEDIA_RE.finditer(html):
        seen.setdefault(match.group(0), None)
    return list(seen)


@dataclass
class TenorScrapeProvider:
    """Keyless default: scrapes a tenor.com search results page for direct GIF URLs.

    No key, no signup, no quota -- and no safe-search parameter either.
    Tenor's *site* (as opposed to its API) does not document a content-rating
    query param for this path, and none was found while verifying it; this
    provider is therefore unmoderated beyond the query-string blocklist check
    `core.py` runs before ever calling it. Treat it as a fallback for a dev
    box, not as the moderated path for a public server -- use a keyed
    provider (below) there.
    """

    timeout: float = TIMEOUT_SECONDS

    def search(self, query: str) -> list[str]:
        slug = "-".join(query.split())
        path = urllib.parse.quote(f"{slug}-gifs")
        url = f"https://tenor.com/search/{path}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
        return extract_tenor_media_urls(html)


@dataclass
class GiphyProvider:
    """GIPHY's search API -- the keyed provider, chosen over the alternatives:

    * **Tenor's own v2 API** is being shut down by Google in 2026; new key
      issuance was already closed as of this writing, so it is not a viable
      choice for anyone setting this up now even though it is well-known.
    * **Klipy** (built by ex-Tenor staff, near-identical API, free tier with
      no hard cap) is the natural next migration once Tenor's API is fully
      gone, but its current endpoint/response shape could not be pinned down
      precisely enough during this build to implement with confidence --
      revisit `docs.klipy.com` and swap in a `KlipyProvider` here when that
      migration is actually needed.
    * **GIPHY** no longer has an unlimited free tier (beta keys are capped
      at 100 requests/hour), but the API itself is stable, extremely well
      documented, and a beta key is still free to obtain -- the safest bet
      to implement correctly today.

    `rating="g"` is GIPHY's safest content-rating value (`g` < `pg` < `pg-13`
    < `r`) and is the default here; override via the constructor if a looser
    rating is genuinely wanted.
    """

    api_key: str
    timeout: float = TIMEOUT_SECONDS
    rating: str = "g"
    limit: int = TOP_N_RESULTS

    def search(self, query: str) -> list[str]:
        params = urllib.parse.urlencode(
            {
                "api_key": self.api_key,
                "q": query,
                "limit": self.limit,
                "rating": self.rating,
                "lang": "en",
            }
        )
        url = f"https://api.giphy.com/v1/gifs/search?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        urls: list[str] = []
        for item in payload.get("data", []):
            images = item.get("images", {})
            rendition = images.get("original") or images.get("fixed_height") or {}
            candidate = rendition.get("url")
            if candidate:
                urls.append(candidate)
        return urls


def provider_from_name(name: str, api_key: str | None) -> GifProvider | None:
    """Build the named provider, or `None` if the name is unknown or a
    required key is missing. Never raises -- an unresolvable provider name
    or a missing key is a "GIFs are off" outcome, not a crash."""
    key = (name or "").strip().lower()
    if key in ("", DEFAULT_PROVIDER, "tenor_scrape", "tenor-scrape", "scrape"):
        return TenorScrapeProvider()
    if key == "giphy":
        if not api_key:
            return None
        return GiphyProvider(api_key=api_key)
    return None


# --- cache + resolver --------------------------------------------------------


class _TTLCache:
    """A small, thread-safe LRU cache with a per-entry TTL.

    Caches the whole candidate list per query, not a single URL, so
    `GifResolver.resolve()` can pick a different one from the top few on
    each call without a network round trip every time.
    """

    def __init__(self, maxsize: int = CACHE_MAX_ENTRIES, ttl: float = CACHE_TTL_SECONDS) -> None:
        self._data: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> list[str] | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            stamp, value = entry
            if time.monotonic() - stamp > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: list[str]) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


@dataclass
class GifResolver:
    """Query -> URL, with caching and a "never raise, never hang" contract.

    `provider is None` means GIFs are off (the feature flag was never set,
    or the configured provider/key didn't resolve to anything) -- every
    method degrades to a no-op rather than requiring callers to check first,
    but `.enabled` is there for callers (and tests) that want to short-circuit
    explicitly.
    """

    provider: GifProvider | None
    top_n: int = TOP_N_RESULTS
    _cache: _TTLCache = field(default_factory=_TTLCache)
    _random: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._random is None:
            import random

            self._random = random.Random()

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    def resolve(self, query: str) -> str | None:
        """One GIF URL for `query`, or `None` if disabled, unavailable, or failed.

        Every provider failure -- timeout, HTTP error, malformed response --
        is caught here and treated exactly like "no results": the caller
        falls back to plain text, it never sees an exception.
        """
        if self.provider is None or not query:
            return None
        candidates = self._cache.get(query)
        if candidates is None:
            try:
                candidates = self.provider.search(query)
            except Exception:
                candidates = []
            self._cache.set(query, candidates)
        if not candidates:
            return None
        pool = candidates[: self.top_n] or candidates
        return self._random.choice(pool)

    @classmethod
    def from_env_bool(cls, enabled: bool, provider_name: str, api_key: str | None) -> "GifResolver":
        """Build a resolver from already-parsed settings values.

        Kept free of any direct `os.environ` reads so it stays trivial to
        construct in tests with arbitrary combinations; `core.py` is the one
        place that reads `Settings` and calls this.
        """
        if not enabled:
            return cls(provider=None)
        return cls(provider=provider_from_name(provider_name, api_key))


# --- reply assembly ----------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def compose_with_gif(remaining_text: str, gif_url: str, limit: int) -> str:
    """The text before the tag, plus the resolved GIF URL, within `limit` chars.

    The URL always survives intact -- it is what makes Discord embed the
    image -- so if the combination doesn't fit, the *text* is what gets
    truncated (down to nothing, if it must). A URL on its own line matches
    "as the whole reply or at the end" from the contract: Discord's
    auto-embed doesn't care whether the URL is alone or after prose.
    """
    remaining = _CONTROL_RE.sub("", remaining_text).strip()
    if not remaining:
        return gif_url
    combined = f"{remaining}\n{gif_url}"
    if len(combined) <= limit:
        return combined
    budget = limit - len(gif_url) - 1  # 1 for the newline
    if budget <= 0:
        return gif_url
    return f"{remaining[:budget]}\n{gif_url}"
