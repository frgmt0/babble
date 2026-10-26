"""A hidden slot machine on every ordinary reply.

Every time somebody who is eligible (see `Babble._slots_eligible`) pings booper
normally, three invisible reels spin. Three of a kind is a *jackpot*: that one
reply is written by Claude instead of the local model, thinking very hard about
a ridiculous "obsession" picked by the winning symbol (`babble/oracle.py` makes
the call, `Babble.cash_out` swaps the text in). Anything else is a miss and
nothing about the reply changes. There is no command and nothing is shown --
the reels exist only here.

Everything in this module is pure arithmetic over a sha256 digest, except for
`SlotMachine`'s small JSON state file. That split is deliberate: the odds, the
pity timer, the cooldown and the daily cap are all testable to the exact roll
with an injected secret and an injected clock, without a network, a model or
an event loop.

Why a digest rather than `random`:

* the outcome depends on the user's actual message, so it feels like *their*
  spin, and two pings in a row are independent;
* the per-process `secret` is mixed in, so nobody can precompute which message
  wins -- without it, "sha256 of my prompt" would be a cheat code;
* a fixed secret makes every roll reproducible in a test.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .util import atomic_write_text

#: The one weighted strip every reel shares: symbol -> number of stops out of
#: 100. Jackpot odds are the sum of (w/100)^3, which for these weights is
#: 0.027 + 0.015625 + 0.008 + 0.001728 + 0.000512 + 0.000125 = ~5.3%.
WEIGHTS: tuple[tuple[str, int], ...] = (
    ("🍒", 30),
    ("🍋", 25),
    ("🔔", 20),
    ("🍇", 12),
    ("💎", 8),
    ("🐸", 5),
)
STRIP: tuple[str, ...] = tuple(symbol for symbol, weight in WEIGHTS for _ in range(weight))
assert len(STRIP) == 100, "the strip must have exactly 100 stops"

#: P(three of a kind) for the strip above, for tests and the README.
JACKPOT_ODDS = sum((weight / len(STRIP)) ** 3 for _, weight in WEIGHTS)

#: What Claude is asked to obsess over, by winning symbol. Rarer symbols get
#: sillier obsessions. All harmless, all PG, none about real people.
OBSESSIONS: dict[str, tuple[str, ...]] = {
    "🍒": (
        "the cherry tomato conspiracy",
        "whether a cherry pit is legally a tiny house",
        "the exact moment a fruit salad becomes a soup",
        "why cherries come in pairs and who is keeping score",
    ),
    "🍋": (
        "the lemon's lifelong grudge against paper cuts",
        "whether lemonade is just lemons in witness protection",
        "the structural engineering of the lemon wedge",
        "who decided sour was a flavour and not a warning",
    ),
    "🔔": (
        "the secret union negotiations of doorbells",
        "whether a bell that nobody hears has technically rung",
        "the emotional inner life of the microwave beep",
        "why bicycle bells are so polite about it",
    ),
    "🍇": (
        "the grape-to-raisin pipeline and its quiet tragedy",
        "whether a grape counts as a single-serving ball pit",
        "the social hierarchy inside a bunch of grapes",
        "the unsolved mystery of the seedless grape's family tree",
    ),
    "💎": (
        "whether soup is a beverage, as a 4-hour treatise",
        "the fair market value of a really good rock",
        "whether a sandwich cut diagonally tastes legally better",
        "a formal taxonomy of every kind of cereal-milk ratio",
    ),
    "🐸": (
        "frog-based macroeconomics",
        "the frog interest rate and what it means for lily pads",
        "a constitutional convention held entirely by frogs",
        "whether a toad is just a frog that went to law school",
    ),
}

#: Reasons a three-of-a-kind does not pay out.
VOID_COOLDOWN = "cooldown"
VOID_DAILY_CAP = "daily_cap"


@dataclass(frozen=True)
class SpinResult:
    """One roll of the reels, and whether it paid."""

    symbols: tuple[str, str, str]
    jackpot: bool
    forced_by_pity: bool = False
    #: Why a jackpot did not pay (`VOID_COOLDOWN` / `VOID_DAILY_CAP`), else None.
    voided: str | None = None
    #: The obsession the winning symbol picked. None on a miss.
    obsession: str | None = None

    @property
    def paid(self) -> bool:
        """A jackpot that actually gets cashed out: three of a kind, not voided."""
        return self.jackpot and self.voided is None


@dataclass(frozen=True)
class JackpotRequest:
    """Everything a paid jackpot needs to become a Claude-written reply.

    Deliberately small and deliberately text-only: the user's own message, the
    obsession, and up to N of booper's *own* earlier replies as voice examples.
    No author ids, no usernames, no other people's prompts, no conversation
    history -- this is the complete payload that leaves the box.
    """

    prompt: str
    symbol: str
    obsession: str
    symbols: tuple[str, str, str]
    voice_examples: tuple[str, ...] = ()


def digest_for(secret: str, user: str, spin_count: int, prompt: str) -> bytes:
    """The 32 bytes every decision in a roll is read from."""
    return hashlib.sha256(f"{secret}|{user}|{spin_count}|{prompt}".encode("utf-8")).digest()


def reels_from_digest(digest: bytes) -> tuple[str, str, str]:
    """Three stops from three disjoint 8-byte slices, each mapped onto the strip."""
    stops = [int.from_bytes(digest[i : i + 8], "big") % len(STRIP) for i in (0, 8, 16)]
    return (STRIP[stops[0]], STRIP[stops[1]], STRIP[stops[2]])


def obsession_for(symbol: str, digest: bytes) -> str:
    """Which of the symbol's obsessions, from bytes no reel used."""
    options = OBSESSIONS[symbol]
    return options[int.from_bytes(digest[24:32], "big") % len(options)]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _UserState:
    spins: int = 0
    miss_streak: int = 0
    last_jackpot_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "spins": self.spins,
            "miss_streak": self.miss_streak,
            "last_jackpot_at": self.last_jackpot_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "_UserState":
        last = raw.get("last_jackpot_at")
        return cls(
            spins=max(0, int(raw.get("spins", 0))),
            miss_streak=max(0, int(raw.get("miss_streak", 0))),
            last_jackpot_at=str(last) if last else None,
        )


class SlotMachine:
    """The reels plus the little bit of memory they need.

    State lives in `slots.json` beside the other data files, keyed by the
    salted pseudonym (`Pseudonymiser.user`) -- never a raw Discord id -- so the
    file is no more identifying than the corpus is. Only `Babble._respond` and
    `!babble forget` touch it, both inside the bot's lock, so there is exactly
    one writer at a time.

    Knobs (all from `Settings`, see `from_settings`):

    * `pity` -- after `pity - 1` misses in a row, the next roll is forced into a
      jackpot. 0 turns the pity timer off.
    * `cooldown_minutes` -- after a paid jackpot, that person's jackpots are
      void for this long. A voided roll does not count as a miss.
    * `daily_cap` -- at most this many paid jackpots per UTC day across
      everybody. 0 means none pay at all (a cost kill switch that keeps the
      feature otherwise wired). A capped roll does not count as a miss either.
    """

    def __init__(
        self,
        path: Path,
        *,
        pity: int = 15,
        daily_cap: int = 20,
        cooldown_minutes: float = 30,
        secret: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = path
        self.pity = max(0, int(pity))
        self.daily_cap = int(daily_cap)
        self.cooldown = timedelta(minutes=max(0.0, float(cooldown_minutes)))
        # Per process on purpose: a restart reshuffles which messages win, and
        # nothing about the secret is ever written down or logged.
        self._secret = secret if secret is not None else secrets.token_hex(16)
        self._clock = clock or _utcnow
        self._users: dict[str, _UserState] = {}
        self._day: str = ""
        self._payouts_today: int = 0
        self._load()

    @classmethod
    def from_settings(cls, settings, **kwargs) -> "SlotMachine":
        return cls(
            settings.slots_path,
            pity=settings.slots_pity,
            daily_cap=settings.slots_daily_cap,
            cooldown_minutes=settings.slots_cooldown_minutes,
            **kwargs,
        )

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Game state, not consent: an unreadable file just means everyone
            # starts from zero spins. Nothing here can widen what is collected.
            return
        if not isinstance(raw, dict):
            return
        users = raw.get("users")
        if isinstance(users, dict):
            for pseudonym, entry in users.items():
                if isinstance(entry, dict):
                    try:
                        self._users[str(pseudonym)] = _UserState.from_dict(entry)
                    except (TypeError, ValueError):
                        continue
        self._day = str(raw.get("day") or "")
        try:
            self._payouts_today = max(0, int(raw.get("payouts_today", 0)))
        except (TypeError, ValueError):
            self._payouts_today = 0

    def _save(self) -> None:
        payload = {
            "day": self._day,
            "payouts_today": self._payouts_today,
            "users": {p: s.to_dict() for p, s in sorted(self._users.items())},
        }
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # --- the game ------------------------------------------------------

    def roll(self, user: str, prompt: str) -> SpinResult:
        """Spin once for `user` (a pseudonym) on `prompt`, and remember it."""
        now = self._clock()
        state = self._users.setdefault(user, _UserState())
        digest = digest_for(self._secret, user, state.spins, prompt)
        symbols = reels_from_digest(digest)
        natural = symbols[0] == symbols[1] == symbols[2]
        forced = not natural and self.pity > 0 and state.miss_streak >= self.pity - 1
        if forced:
            # Weighted pick from the same digest: reel one's stop already lands
            # on the strip with exactly the symbol weights.
            symbols = (symbols[0],) * 3
        jackpot = natural or forced
        state.spins += 1

        voided: str | None = None
        obsession: str | None = None
        if jackpot:
            obsession = obsession_for(symbols[0], digest)
            voided = self._void_reason(state, now)
            if voided is None:
                self._payouts_today += 1
                state.last_jackpot_at = now.isoformat(timespec="seconds")
                state.miss_streak = 0
        else:
            state.miss_streak += 1

        self._save()
        return SpinResult(
            symbols=symbols,
            jackpot=jackpot,
            forced_by_pity=forced,
            voided=voided,
            obsession=obsession,
        )

    def _void_reason(self, state: _UserState, now: datetime) -> str | None:
        if state.last_jackpot_at and self.cooldown > timedelta(0):
            try:
                last = datetime.fromisoformat(state.last_jackpot_at)
            except ValueError:
                last = None
            if last is not None and now - last < self.cooldown:
                return VOID_COOLDOWN
        today = now.astimezone(timezone.utc).date().isoformat()
        if self._day != today:
            self._day = today
            self._payouts_today = 0
        if self._payouts_today >= self.daily_cap:
            return VOID_DAILY_CAP
        return None

    # --- housekeeping --------------------------------------------------

    def forget(self, user: str) -> bool:
        """Drop everything stored about one pseudonym. True if there was any."""
        if self._users.pop(user, None) is None:
            return False
        self._save()
        return True

    def knows(self, user: str) -> bool:
        return user in self._users

    def miss_streak(self, user: str) -> int:
        state = self._users.get(user)
        return state.miss_streak if state else 0

    def spins(self, user: str) -> int:
        state = self._users.get(user)
        return state.spins if state else 0

    @property
    def payouts_today(self) -> int:
        today = self._clock().astimezone(timezone.utc).date().isoformat()
        return self._payouts_today if self._day == today else 0
