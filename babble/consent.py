"""Who has agreed to be in the dataset.

This is the only file in the project besides `exchanges.json` that holds raw
Discord ids, and neither is ever exported. The rule the rest of the code relies
on is simple and absolute: **capture requires GRANTED**. Silence is not consent,
a missing file is not consent, and an unreadable file is not consent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .util import atomic_write_text, utcnow_iso

UNKNOWN = "unknown"  # never been asked
PENDING = "pending"  # asked, has not answered
GRANTED = "granted"
DECLINED = "declined"
WITHDRAWN = "withdrawn"  # said yes once, changed their mind

#: The only state that permits storing anything. Deliberately a one-element set.
CAPTURE_OK = frozenset({GRANTED})

#: Bump when the notice text materially changes, to spot stale agreements.
NOTICE_VERSION = 1


@dataclass(frozen=True)
class ConsentRecord:
    decision: str
    updated_at: str
    notice_version: int = NOTICE_VERSION


class ConsentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, ConsentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Fail closed: a corrupt consent file means nobody has consented.
            self._records = {}
            return
        for user_id, entry in (raw or {}).items():
            if isinstance(entry, dict) and entry.get("decision"):
                self._records[str(user_id)] = ConsentRecord(
                    decision=str(entry["decision"]),
                    updated_at=str(entry.get("updated_at", "")),
                    notice_version=int(entry.get("notice_version", NOTICE_VERSION)),
                )

    def _save(self) -> None:
        payload = {
            user_id: {
                "decision": rec.decision,
                "updated_at": rec.updated_at,
                "notice_version": rec.notice_version,
            }
            for user_id, rec in sorted(self._records.items())
        }
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _set(self, user_id: object, decision: str) -> ConsentRecord:
        record = ConsentRecord(decision=decision, updated_at=utcnow_iso())
        self._records[str(user_id)] = record
        self._save()
        return record

    # --- queries -------------------------------------------------------

    def decision(self, user_id: object) -> str:
        record = self._records.get(str(user_id))
        return record.decision if record else UNKNOWN

    def may_capture(self, *user_ids: object) -> bool:
        """True only if *every* person involved has actively granted consent."""
        return all(self.decision(uid) in CAPTURE_OK for uid in user_ids)

    def has_been_asked(self, user_id: object) -> bool:
        return self.decision(user_id) != UNKNOWN

    def granted_ids(self) -> list[str]:
        return [uid for uid, rec in self._records.items() if rec.decision in CAPTURE_OK]

    def known_ids(self) -> list[str]:
        return list(self._records)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for record in self._records.values():
            tally[record.decision] = tally.get(record.decision, 0) + 1
        return tally

    # --- transitions ---------------------------------------------------

    def mark_prompted(self, user_id: object) -> None:
        """Remember that we showed the notice, so we only show it once."""
        if not self.has_been_asked(user_id):
            self._set(user_id, PENDING)

    def grant(self, user_id: object) -> ConsentRecord:
        return self._set(user_id, GRANTED)

    def decline(self, user_id: object) -> ConsentRecord:
        return self._set(user_id, DECLINED)

    def withdraw(self, user_id: object) -> ConsentRecord:
        return self._set(user_id, WITHDRAWN)
