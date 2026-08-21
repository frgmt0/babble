"""A minimal LLM call path, for the one job in this repo that genuinely needs
a language model rather than a markov chain: paraphrasing a correction pair
while keeping its meaning intact (`babble/pairaugment.py`).

There is no API key setting anywhere in this repo and no LLM SDK in
`pyproject.toml` -- by design, everything else here is corpus-internal. But
this box already has the `claude` CLI installed (it is what runs this very
agent), and its non-interactive print mode (`claude -p --output-format
json`) is a complete, scriptable LLM call path with nothing new to
provision: no key to generate, no dependency to add, no network config to
reason about beyond what the CLI already handles. That is "the cheapest LLM
call path already available on this box", chosen over adding an API-key
setting and an SDK dependency for a single call site.

`LLMClient` is the seam: `ClaudeCLIClient` is the only implementation, but
every caller takes the protocol, not the class, so a test can hand in a fake
that never shells out, and a future deployment without the CLI on PATH can
swap in something else without touching `pairaugment.py`.

Two failure classes, kept separate on purpose:

- **`LLMError`** (this module) -- the call itself is broken: the binary is
  missing, the process errored or timed out, or the response has no text to
  read at all. There is nothing downstream can do with this but give up.
- **Content that came back but doesn't parse as a valid variant** is a
  different problem, specific to what the caller asked for -- that is
  `pairaugment.ParaphraseError`, not this module's concern.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Protocol

from .config import Settings


class LLMError(Exception):
    """The call path itself failed -- binary missing, non-zero exit, timeout,
    or a response with no usable text."""


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return the model's raw text reply to `prompt`, or raise `LLMError`."""
        ...


@dataclass
class ClaudeCLIClient:
    """Shells out to `claude -p --output-format json` for one turn, one
    reply, no tools, no session left behind.

    Each call is a fresh process -- no conversation state carries between
    variants, which is what makes every paraphrase call independent of every
    other one (and safe to run concurrently, see `pairaugment.generate_
    augmented_pairs`'s `max_workers`).
    """

    binary: str = "claude"
    model: str = "haiku"
    timeout_seconds: float = 60.0

    def complete(self, prompt: str) -> str:
        cmd = [
            self.binary,
            "-p",
            "--model", self.model,
            "--output-format", "json",
            prompt,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{self.binary!r} timed out after {self.timeout_seconds}s") from exc
        except OSError as exc:
            raise LLMError(f"failed to launch {self.binary!r}: {exc}") from exc

        if proc.returncode != 0:
            raise LLMError(
                f"{self.binary!r} exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{self.binary!r} did not return a JSON envelope: {proc.stdout[:200]!r}"
            ) from exc
        if payload.get("is_error"):
            raise LLMError(f"{self.binary!r} reported an error: {payload.get('result')!r}")
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise LLMError(f"{self.binary!r} returned no text: {payload!r}")
        return result


def client_from_settings(settings: Settings) -> LLMClient:
    """The configured client -- `BABBLE_PARAPHRASE_MODEL` /
    `BABBLE_PARAPHRASE_TIMEOUT` / `BABBLE_PARAPHRASE_BIN` -- so swapping the
    model tier or pointing at a different binary never touches call sites."""
    return ClaudeCLIClient(
        binary=settings.paraphrase_bin,
        model=settings.paraphrase_model,
        timeout_seconds=settings.paraphrase_timeout_seconds,
    )
