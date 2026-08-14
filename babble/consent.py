"""Who has agreed to be in the dataset, and to exactly what.

This is the only file in the project besides `exchanges.json` that holds raw
Discord ids, and neither is ever exported. The rule the rest of the code relies
on is simple and absolute: **capture requires GRANTED**. Silence is not consent,
a missing file is not consent, and an unreadable file is not consent.

There are two separate grants, because there are two separate collections:

* **`corrections`** — the original deal. Someone teaches the bot by replying with
  what it should have said, and that pair is kept. This is what everybody who
  used babble before the corpus existed agreed to.
* **`corpus`** — the messages you address to the bot, kept as plain text to train
  on and to publish. This is strictly broader than `corrections`, so a grant made
  under the old notice **does not** carry over to it. A legacy file loads with
  `corrections` granted and `corpus` unknown, and that person gets asked again.

On top of the `corpus` grant sits one widening, per person *and* per channel:
`wide_channels`. Normally only messages you address to the bot are collected. If
you run `!babble all` in a channel, everything **you** say **in that channel** is
collected too. It never widens anything for anybody else, it never follows you to
another channel, and `!babble pings` takes it straight back off again. Channel
ids are stored raw because scope cannot be enforced without them, and a channel
id identifies a room, not a person.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .util import atomic_write_text, utcnow_iso

UNKNOWN = "unknown"  # never been asked
PENDING = "pending"  # asked, has not answered
GRANTED = "granted"
DECLINED = "declined"
WITHDRAWN = "withdrawn"  # said yes once, changed their mind

#: The only state that permits storing anything. Deliberately a one-element set.
CAPTURE_OK = frozenset({GRANTED})

#: What a grant can be *for*. `corrections` is the narrower, older one.
SCOPE_CORRECTIONS = "corrections"
SCOPE_CORPUS = "corpus"
SCOPES = (SCOPE_CORRECTIONS, SCOPE_CORPUS)

#: A "no" needs no re-asking to stay a no, so these carry across every scope
#: when a legacy single-grant file is loaded. A "yes" does not.
NEGATIVE = frozenset({DECLINED, WITHDRAWN})

#: Bump when the notice text materially changes, to spot stale agreements.
#: 1 was the corrections-only notice; 2 is the one that describes the corpus.
NOTICE_VERSION = 2
LEGACY_NOTICE_VERSION = 1


@dataclass(frozen=True)
class ConsentRecord:
    """One person's answer, for one scope."""

    decision: str
    updated_at: str
    notice_version: int = NOTICE_VERSION


@dataclass
class _Person:
    """Everything stored against one Discord id. Internal to this module."""

    grants: dict[str, ConsentRecord] = field(default_factory=dict)
    wide_channels: list[str] = field(default_factory=list)


class ConsentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._people: dict[str, _Person] = {}
        self._load()

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Fail closed: a corrupt consent file means nobody has consented.
            self._people = {}
            return
        for user_id, entry in (raw or {}).items():
            if not isinstance(entry, dict):
                continue
            person = _Person(grants=self._read_grants(entry))
            channels = entry.get("wide_channels")
            if isinstance(channels, list):
                person.wide_channels = [str(c) for c in channels if str(c)]
            if person.grants or person.wide_channels:
                self._people[str(user_id)] = person

    @staticmethod
    def _read_grants(entry: dict) -> dict[str, ConsentRecord]:
        """Per-scope grants out of one stored entry, old shape or new.

        A file written before the corpus existed has a single top-level
        `decision` and no `scopes`. That answer was given to a notice that only
        ever mentioned corrections, so it is loaded as a `corrections` grant and
        nothing else -- which is precisely what makes the person get asked again
        before a word of theirs goes into the corpus.
        """
        scopes = entry.get("scopes")
        if isinstance(scopes, dict):
            grants = {}
            for scope, sub in scopes.items():
                if isinstance(sub, dict) and sub.get("decision"):
                    grants[str(scope)] = ConsentRecord(
                        decision=str(sub["decision"]),
                        updated_at=str(sub.get("updated_at", "")),
                        notice_version=int(sub.get("notice_version", NOTICE_VERSION)),
                    )
            return grants

        if not entry.get("decision"):
            return {}
        legacy = ConsentRecord(
            decision=str(entry["decision"]),
            updated_at=str(entry.get("updated_at", "")),
            notice_version=int(entry.get("notice_version", LEGACY_NOTICE_VERSION)),
        )
        grants = {SCOPE_CORRECTIONS: legacy}
        if legacy.decision in NEGATIVE:
            grants[SCOPE_CORPUS] = legacy
        return grants

    def _save(self) -> None:
        payload = {}
        for user_id, person in sorted(self._people.items()):
            entry: dict = {
                "scopes": {
                    scope: {
                        "decision": rec.decision,
                        "updated_at": rec.updated_at,
                        "notice_version": rec.notice_version,
                    }
                    for scope, rec in sorted(person.grants.items())
                }
            }
            # The narrower grant is also mirrored at the top level, in the shape
            # a pre-corpus babble would read. If this code is ever rolled back,
            # corrections consent survives and corpus consent silently reverts to
            # "never asked" -- which is the safe direction for it to fail in.
            narrow = person.grants.get(SCOPE_CORRECTIONS)
            if narrow:
                entry["decision"] = narrow.decision
                entry["updated_at"] = narrow.updated_at
                entry["notice_version"] = narrow.notice_version
            if person.wide_channels:
                entry["wide_channels"] = list(person.wide_channels)
            payload[user_id] = entry
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _person(self, user_id: object) -> _Person:
        return self._people.setdefault(str(user_id), _Person())

    def _set(self, user_id: object, decision: str, scopes: tuple[str, ...]) -> ConsentRecord:
        record = ConsentRecord(decision=decision, updated_at=utcnow_iso())
        person = self._person(user_id)
        for scope in scopes:
            person.grants[scope] = record
        self._save()
        return record

    # --- queries -------------------------------------------------------

    def decision(self, user_id: object, scope: str = SCOPE_CORRECTIONS) -> str:
        """This person's answer for one scope. Defaults to the narrower grant."""
        person = self._people.get(str(user_id))
        record = person.grants.get(scope) if person else None
        return record.decision if record else UNKNOWN

    def may_capture(self, *user_ids: object, scope: str = SCOPE_CORRECTIONS) -> bool:
        """True only if *every* person involved has actively granted this scope."""
        return all(self.decision(uid, scope) in CAPTURE_OK for uid in user_ids)

    def may_capture_channel(self, user_id: object, channel_id: object) -> bool:
        """True if *everything* this person says in this channel is collectable.

        Two conditions, both required: they consented to corpus collection at
        all, and they ran the widening command in this exact channel. Revoking
        either one stops the collection on the very next message, because this is
        read fresh every time rather than cached.
        """
        return (
            self.decision(user_id, SCOPE_CORPUS) in CAPTURE_OK
            and str(channel_id) in self.wide_channels(user_id)
        )

    def wide_channels(self, user_id: object) -> list[str]:
        person = self._people.get(str(user_id))
        return list(person.wide_channels) if person else []

    def has_been_asked(self, user_id: object, scope: str = SCOPE_CORRECTIONS) -> bool:
        return self.decision(user_id, scope) != UNKNOWN

    def notice_version(self, user_id: object, scope: str = SCOPE_CORRECTIONS) -> int | None:
        person = self._people.get(str(user_id))
        record = person.grants.get(scope) if person else None
        return record.notice_version if record else None

    def granted_ids(self, scope: str = SCOPE_CORRECTIONS) -> list[str]:
        return [uid for uid in self._people if self.decision(uid, scope) in CAPTURE_OK]

    def known_ids(self) -> list[str]:
        return list(self._people)

    def counts(self, scope: str = SCOPE_CORRECTIONS) -> dict[str, int]:
        tally: dict[str, int] = {}
        for user_id in self._people:
            decision = self.decision(user_id, scope)
            if decision != UNKNOWN:
                tally[decision] = tally.get(decision, 0) + 1
        return tally

    # --- transitions ---------------------------------------------------

    def mark_prompted(self, user_id: object, scope: str = SCOPE_CORRECTIONS) -> bool:
        """Remember that we showed the notice, so we only show it once.

        Returns True if this was the first time, which is how the caller knows
        whether to actually post the wall of text.
        """
        if self.has_been_asked(user_id, scope):
            return False
        self._set(user_id, PENDING, (scope,))
        return True

    def grant(self, user_id: object, *scopes: str) -> ConsentRecord:
        """Say yes. With no scopes named, yes to everything the notice covers."""
        return self._set(user_id, GRANTED, scopes or SCOPES)

    def decline(self, user_id: object, *scopes: str) -> ConsentRecord:
        record = self._set(user_id, DECLINED, scopes or SCOPES)
        self.clear_wide(user_id)
        return record

    def withdraw(self, user_id: object, *scopes: str) -> ConsentRecord:
        record = self._set(user_id, WITHDRAWN, scopes or SCOPES)
        self.clear_wide(user_id)
        return record

    def widen(self, user_id: object, channel_id: object) -> bool:
        """Collect everything this person says in this one channel.

        Returns False if it was already on. Callers must check the `corpus` grant
        first: this only records the channel, it does not grant anything.
        """
        channel = str(channel_id)
        person = self._person(user_id)
        if channel in person.wide_channels:
            return False
        person.wide_channels.append(channel)
        self._save()
        return True

    def narrow(self, user_id: object, channel_id: object) -> bool:
        """Back to collecting only what this person addresses to the bot here."""
        channel = str(channel_id)
        person = self._people.get(str(user_id))
        if not person or channel not in person.wide_channels:
            return False
        person.wide_channels.remove(channel)
        self._save()
        return True

    def clear_wide(self, user_id: object) -> int:
        """Drop every widened channel for this person. Returns how many went."""
        person = self._people.get(str(user_id))
        if not person or not person.wide_channels:
            return 0
        dropped = len(person.wide_channels)
        person.wide_channels = []
        self._save()
        return dropped
