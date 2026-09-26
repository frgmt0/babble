"""`babble/gifs.py` and its hook into `core.Babble._respond` -- no network.

Layout mirrors `gifs.py`'s own sections: tag parsing, HTML extraction, the
provider protocol, the cache/resolver, and finally the Discord send path
through the fake gateway, exercised with a fake provider so the whole
"[gif: query] -> URL Discord embeds" contract is covered without a socket.
"""

from __future__ import annotations

import time

import pytest

from babble.blocklist import Blocklist
from babble.core import EMPTY_REPLY, Babble, Generation, IncomingMessage
from babble.gifs import (
    CACHE_TTL_SECONDS,
    GifResolver,
    TenorScrapeProvider,
    compose_with_gif,
    extract_gif_tag,
    extract_tenor_media_urls,
    normalise_query,
    provider_from_name,
)
from conftest import FakeDiscord

ALICE = "111111111111111111"


# --- tag parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_remaining, expected_query",
    [
        # Whole reply.
        ("[gif: cats dogs]", "", "cats dogs"),
        # Trailing, with real text before it.
        ("that reminds me of cats [gif: cats]", "that reminds me of cats", "cats"),
        # Case and whitespace inside the tag are irrelevant.
        ("hi [GIF:cats]", "hi", "cats"),
        ("hi [ gif :  cats dogs  ]", "hi", "cats dogs"),
        # Stray punctuation in the query is stripped, words untouched.
        ("hi [gif: cats!, dogs.]", "hi", "cats dogs"),
        # Only the *last* opening marker matters -- an earlier mention of
        # "[gif:" in the model's own text must not be treated as the tag.
        ("i can do [gif: things] you know [gif: cats]", "i can do [gif: things] you know", "cats"),
        # No tag at all.
        ("just a normal reply", "just a normal reply", None),
    ],
)
def test_extract_gif_tag_parses(text, expected_remaining, expected_query):
    remaining, query = extract_gif_tag(text)
    assert remaining == expected_remaining
    assert query == expected_query


def test_extract_gif_tag_strips_unterminated_trailing_fragment():
    """`max_new_tokens` can cut a generation off mid-tag -- no closing `]`,
    nothing to search for, so the dangling fragment is dropped entirely and
    the rest of the reply is left to stand on its own."""
    remaining, query = extract_gif_tag("that reminds me of [gif: sunse")
    assert remaining == "that reminds me of"
    assert query is None


def test_extract_gif_tag_leaves_midsentence_mention_alone():
    """Real content survives after the closing bracket -- that's the model
    quoting the syntax, not placing the marker, so nothing is touched."""
    text = "[gif: cats] wow that's a good one"
    remaining, query = extract_gif_tag(text)
    assert (remaining, query) == (text, None)


def test_extract_gif_tag_empty_query_is_no_tag():
    remaining, query = extract_gif_tag("ok [gif: !!!]")
    assert query is None
    assert remaining == "ok"


def test_normalise_query_folds_case_and_punctuation():
    assert normalise_query("  CATS!,  Dogs.  ") == "cats dogs"
    assert normalise_query("self-driving cars") == "self-driving cars"


# --- HTML extraction --------------------------------------------------------

# A trimmed, representative slice of what tenor.com/search/<slug>-gifs
# actually returns: the media URL shows up more than once (thumbnail +
# full-size + a JSON blob), and there's noise (a non-tenor domain, a
# non-.gif tenor asset) that must not be picked up.
TENOR_FIXTURE_HTML = r"""
<html><body>
<div class="Gif" data-src="https://media.tenor.com/abc123XYZ/happy-cats.gif">
  <img src="https://media.tenor.com/abc123XYZ/happy-cats.gif" />
</div>
<script type="application/json">
{"results": [{"url": "https://media.tenor.com/abc123XYZ/happy-cats.gif",
              "itemurl": "https://tenor.com/view/happy-cats-gif-123"}]}
</script>
<div><img src="https://media.tenor.com/def456ABC/sad-dogs.gif"></div>
<img src="https://media.tenor.com/def456ABC/preview.webp">
<img src="https://cdn.example.com/not-tenor/fake.gif">
</body></html>
"""


def test_extract_tenor_media_urls_from_fixture():
    urls = extract_tenor_media_urls(TENOR_FIXTURE_HTML)
    assert urls == [
        "https://media.tenor.com/abc123XYZ/happy-cats.gif",
        "https://media.tenor.com/def456ABC/sad-dogs.gif",
    ]
    # Every returned URL is the direct media host, is a .gif, and duplicates
    # (the same URL appeared three times above) collapse to one entry.
    assert all(u.startswith("https://media.tenor.com/") and u.endswith(".gif") for u in urls)


def test_extract_tenor_media_urls_empty_on_no_match():
    assert extract_tenor_media_urls("<html>nothing here</html>") == []


# --- provider selection ------------------------------------------------------


def test_provider_from_name_defaults_to_tenor_scrape():
    assert isinstance(provider_from_name("", None), TenorScrapeProvider)
    assert isinstance(provider_from_name("tenor-scrape", None), TenorScrapeProvider)
    assert isinstance(provider_from_name("TENOR_SCRAPE", None), TenorScrapeProvider)


def test_provider_from_name_giphy_requires_key():
    assert provider_from_name("giphy", None) is None
    provider = provider_from_name("giphy", "key123")
    assert provider is not None
    assert provider.api_key == "key123"


def test_provider_from_name_unknown_is_none():
    assert provider_from_name("not-a-real-provider", "key") is None


# --- cache + resolver --------------------------------------------------------


class FakeProvider:
    """A provider stand-in: records calls, returns a canned candidate list."""

    def __init__(self, urls=None, *, raises: Exception | None = None):
        self.urls = list(urls) if urls is not None else []
        self.raises = raises
        self.calls: list[str] = []

    def search(self, query: str) -> list[str]:
        self.calls.append(query)
        if self.raises is not None:
            raise self.raises
        return list(self.urls)


def test_resolver_disabled_without_provider():
    resolver = GifResolver(provider=None)
    assert resolver.enabled is False
    assert resolver.resolve("cats") is None


def test_resolver_returns_none_for_empty_query():
    resolver = GifResolver(provider=FakeProvider(["https://x/1.gif"]))
    assert resolver.resolve("") is None


def test_resolver_picks_among_top_n_results():
    urls = [f"https://media.tenor.com/{i}/x.gif" for i in range(10)]
    provider = FakeProvider(urls)
    resolver = GifResolver(provider=provider, top_n=3)
    seen = {resolver.resolve("cats") for _ in range(50)}
    # Only ever picks from the top 3, never reaches into the tail.
    assert seen <= set(urls[:3])
    # With 50 draws over 3 options, more than one distinct URL must show up
    # (this is not a strict guarantee, but the odds of a false failure here
    # are astronomically small and it is what "random pick" means).
    assert len(seen) > 1


def test_resolver_caches_so_provider_is_called_once_per_query():
    provider = FakeProvider(["https://x/1.gif"])
    resolver = GifResolver(provider=provider)
    for _ in range(5):
        resolver.resolve("cats")
    assert provider.calls == ["cats"]


def test_resolver_cache_respects_ttl():
    provider = FakeProvider(["https://x/1.gif"])
    resolver = GifResolver(provider=provider)
    resolver._cache._ttl = 0.01
    resolver.resolve("cats")
    time.sleep(0.05)
    resolver.resolve("cats")
    assert provider.calls == ["cats", "cats"]


def test_resolver_cache_evicts_oldest_over_maxsize():
    provider = FakeProvider(["https://x/1.gif"])
    resolver = GifResolver(provider=provider)
    resolver._cache._maxsize = 2
    resolver.resolve("a")
    resolver.resolve("b")
    resolver.resolve("c")  # evicts "a"
    assert len(resolver._cache) == 2
    resolver.resolve("a")
    assert provider.calls == ["a", "b", "c", "a"]  # "a" was a miss again


def test_resolver_swallows_provider_exceptions():
    provider = FakeProvider(raises=TimeoutError("timed out"))
    resolver = GifResolver(provider=provider)
    assert resolver.resolve("cats") is None
    # The failure is cached too (as "no results"), so a bad query doesn't
    # hammer a struggling provider every single message.
    resolver.resolve("cats")
    assert provider.calls == ["cats"]


def test_resolver_no_results_returns_none():
    resolver = GifResolver(provider=FakeProvider([]))
    assert resolver.resolve("cats") is None


def test_from_env_bool_off_has_no_provider():
    resolver = GifResolver.from_env_bool(False, "giphy", "key")
    assert resolver.enabled is False


def test_from_env_bool_on_with_unresolvable_provider_is_disabled():
    resolver = GifResolver.from_env_bool(True, "giphy", None)  # no key
    assert resolver.enabled is False


def test_from_env_bool_on_with_default_provider_is_enabled():
    resolver = GifResolver.from_env_bool(True, "tenor-scrape", None)
    assert resolver.enabled is True


# --- reply assembly ----------------------------------------------------------


def test_compose_with_gif_puts_url_on_its_own_line():
    assert compose_with_gif("look at this", "https://x/1.gif", 2000) == "look at this\nhttps://x/1.gif"


def test_compose_with_gif_url_alone_when_nothing_remains():
    assert compose_with_gif("", "https://x/1.gif", 2000) == "https://x/1.gif"
    assert compose_with_gif("   ", "https://x/1.gif", 2000) == "https://x/1.gif"


def test_compose_with_gif_truncates_text_not_url_when_over_limit():
    url = "https://media.tenor.com/abc/x.gif"
    text = "a" * 50
    limit = len(url) + 10  # room for a little text and the newline, not all 50 a's
    result = compose_with_gif(text, url, limit=limit)
    assert result.endswith(f"\n{url}")
    assert len(result) <= limit
    assert url in result


def test_compose_with_gif_url_wins_even_if_it_alone_exceeds_limit():
    url = "https://media.tenor.com/" + "a" * 100 + "/x.gif"
    assert compose_with_gif("some text", url, limit=10) == url


# --- the live, keyless lookup (network -- see test_gifs_live.py-style guard) --


def test_tenor_scrape_provider_builds_expected_url(monkeypatch):
    """No network: just pins down the URL shape the live check (run manually,
    see the report) depends on -- '<query words joined by dash>-gifs'."""
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return TENOR_FIXTURE_HTML.encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("babble.gifs.urllib.request.urlopen", fake_urlopen)
    provider = TenorScrapeProvider()
    urls = provider.search("happy cats")
    assert captured["url"] == "https://tenor.com/search/happy-cats-gifs"
    assert any(h.lower() == "user-agent" for h in captured["headers"])
    assert urls[0].startswith("https://media.tenor.com/")


# --- the Discord send path (through core.Babble, fake gateway, fake provider) -


@pytest.fixture
def fake_provider():
    return FakeProvider(["https://media.tenor.com/aaa/cats.gif", "https://media.tenor.com/bbb/cats2.gif"])


@pytest.fixture
def gif_brain(settings, generator, log, fake_provider):
    resolver = GifResolver(provider=fake_provider, top_n=2)
    return Babble(settings, generator=generator, log=log, gifs=resolver, bot_user_id="bot-9999")


def test_gif_tag_becomes_a_real_url_in_the_sent_message(settings, log, fake_provider):
    generator = lambda prompt: Generation(text="[gif: happy cats]")  # noqa: E731
    resolver = GifResolver(provider=fake_provider, top_n=2)
    brain = Babble(settings, generator=generator, log=log, gifs=resolver, bot_user_id="bot-9999")
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    generation_reply = [p for p in posted if p.kind == "generation"][0]

    assert generation_reply.content in fake_provider.urls
    assert fake_provider.calls == ["happy cats"]
    # The stored/remembered exchange is exactly what was sent -- a later
    # correction reacts to the GIF the person actually saw, not the tag.
    exchange = brain.exchanges.get(generation_reply.id)
    assert exchange.response == generation_reply.content


def test_gif_tag_with_text_before_it_puts_url_on_its_own_line(settings, log, fake_provider):
    generator = lambda prompt: Generation(text="here you go [gif: happy cats]")  # noqa: E731
    resolver = GifResolver(provider=fake_provider, top_n=2)
    brain = Babble(settings, generator=generator, log=log, gifs=resolver, bot_user_id="bot-9999")
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    content = [p for p in posted if p.kind == "generation"][0].content

    lines = content.split("\n")
    assert lines[0] == "here you go"
    assert lines[1] in fake_provider.urls


def test_gif_feature_off_strips_tag_and_sends_remaining_text(settings, log):
    """No `gifs=` resolver at all -- the default -- is exactly "off": the
    tag is stripped (it's not something a person should see verbatim) but
    no lookup is ever attempted."""
    generator = lambda prompt: Generation(text="here you go [gif: happy cats]")  # noqa: E731
    brain = Babble(settings, generator=generator, log=log, bot_user_id="bot-9999")
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    content = [p for p in posted if p.kind == "generation"][0].content
    assert content == "here you go"


def test_gif_lookup_failure_falls_back_to_remaining_text(settings, log):
    generator = lambda prompt: Generation(text="here you go [gif: happy cats]")  # noqa: E731
    resolver = GifResolver(provider=FakeProvider([]))  # no results
    brain = Babble(settings, generator=generator, log=log, gifs=resolver, bot_user_id="bot-9999")
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    content = [p for p in posted if p.kind == "generation"][0].content
    assert content == "here you go"


def test_gif_lookup_timeout_falls_back_to_remaining_text(settings, log):
    generator = lambda prompt: Generation(text="here you go [gif: happy cats]")  # noqa: E731
    resolver = GifResolver(provider=FakeProvider(raises=TimeoutError("slow")))
    brain = Babble(settings, generator=generator, log=log, gifs=resolver, bot_user_id="bot-9999")
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    content = [p for p in posted if p.kind == "generation"][0].content
    assert content == "here you go"


def test_gif_only_reply_with_nothing_remaining_and_no_gif_matches_empty_reply_fallback(settings, log):
    """Whole reply is the tag, and resolution fails: nothing remains, so this
    must match the bot's existing empty-reply behaviour, never an empty
    Discord message."""
    generator = lambda prompt: Generation(text="[gif: happy cats]")  # noqa: E731
    resolver = GifResolver(provider=FakeProvider([]))
    brain = Babble(settings, generator=generator, log=log, gifs=resolver, bot_user_id="bot-9999")
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    content = [p for p in posted if p.kind == "generation"][0].content
    assert content == EMPTY_REPLY
    assert content != ""


def test_full_reply_with_blocked_word_never_reaches_gif_resolution(settings, log):
    """`badterm` inside `[gif: badterm]` also trips the existing whole-body
    blocklist check in `_respond`, which runs *before* gif resolution and
    replaces the entire reply -- so the provider must never see it either."""
    generator = lambda prompt: Generation(text="here: [gif: badterm]")  # noqa: E731
    provider = FakeProvider(["https://media.tenor.com/x/x.gif"])
    resolver = GifResolver(provider=provider)
    blocklist = Blocklist(terms=frozenset({"badterm"}))
    brain = Babble(
        settings,
        generator=generator,
        log=log,
        gifs=resolver,
        blocklist=blocklist,
        bot_user_id="bot-9999",
    )
    fake = FakeDiscord(brain)
    fake.onboard(ALICE)

    posted = fake.ping(ALICE, "show me something")
    content = [p for p in posted if p.kind == "generation"][0].content

    assert "badterm" not in content
    assert provider.calls == []  # the provider was never called at all


def test_resolve_gif_tag_checks_the_isolated_query_before_any_lookup(settings, log, read_log):
    """`_resolve_gif_tag` is the safety net for a query that reaches it
    without having already cleared a whole-body blocklist check (defense in
    depth: the query is what a provider -- a network call, or a page scrape
    -- actually receives). Exercised directly so it's covered regardless of
    what `_respond` already filtered upstream."""
    provider = FakeProvider(["https://media.tenor.com/x/x.gif"])
    resolver = GifResolver(provider=provider)
    blocklist = Blocklist(terms=frozenset({"badterm"}))
    brain = Babble(
        settings,
        generator=lambda prompt: Generation(text=""),  # noqa: E731 -- unused here
        log=log,
        gifs=resolver,
        blocklist=blocklist,
        bot_user_id="bot-9999",
    )
    msg = IncomingMessage(message_id="1", author_id=ALICE, content="hi")

    result = brain._resolve_gif_tag("here: [gif: badterm]", msg)

    assert result == "here:"
    assert provider.calls == []  # never handed to the provider
    assert read_log("capture.blocked")
