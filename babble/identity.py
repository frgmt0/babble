"""Turning Discord ids into stable pseudonyms.

This is the single chokepoint for identity in the whole project. Raw Discord ids
exist in exactly two local operational files (consent.json, exchanges.json) and
nowhere else: the interaction store, the logs and the exported dataset only ever
see the hashes produced here.
"""

from __future__ import annotations

import hashlib
import secrets

from .config import Settings


class Pseudonymiser:
    """Salted one-way hashes for user and channel ids.

    The salt is the reason these cannot be reversed by someone who downloads the
    dataset and hashes every Discord id they can think of. It also means the salt
    must never change: `!babble forget` finds a user's rows by re-deriving their
    hash, so a rotated salt would orphan the very rows it needs to delete.
    """

    def __init__(self, salt: str) -> None:
        if not salt:
            raise ValueError("refusing to pseudonymise with an empty salt")
        self._salt = salt.encode("utf-8")

    def user(self, user_id: object) -> str:
        return "u_" + self._digest(f"user:{user_id}")[:16]

    def channel(self, channel_id: object) -> str:
        return "c_" + self._digest(f"channel:{channel_id}")[:12]

    def _digest(self, value: str) -> str:
        return hashlib.sha256(self._salt + b"|" + value.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, settings: Settings) -> "Pseudonymiser":
        """Env salt if set, else the one persisted on first ever run."""
        if settings.salt:
            return cls(settings.salt)

        path = settings.salt_path
        if path.exists():
            salt = path.read_text(encoding="utf-8").strip()
            if salt:
                return cls(salt)

        salt = secrets.token_hex(16)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(salt + "\n", encoding="utf-8")
        path.chmod(0o600)
        return cls(salt)
