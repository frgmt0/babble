"""Sampling, and the bit that keeps the bot's weights in sync with the trainer."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .config import Settings
from .core import Generation
from .logs import EventLog, NullLog
from .model import Babbler, ModelConfig, config_from_settings
from .tokenizer import BOS_ID, EOS_ID, PAD_ID, SEP_ID, decode, prompt_context

# The structural tokens are inputs, not outputs. Banning them from sampling means
# a random model cannot emit a stray <sep> into the middle of a Discord message.
_BANNED = [PAD_ID, BOS_ID, SEP_ID]


@torch.no_grad()
def sample(
    model: Babbler,
    prompt: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> str:
    """Continue `<bos> prompt <sep>` one byte at a time until <eos>.

    An untrained model almost never emits <eos>, so `max_new_tokens` is what
    actually ends most early generations. That is expected.
    """
    was_training = model.training
    model.eval()
    try:
        context = prompt_context(prompt, model.config.block_size)
        produced: list[int] = []
        for _ in range(max_new_tokens):
            window = context[-model.config.block_size :]
            logits = model(torch.tensor([window], dtype=torch.long))[0, -1]
            logits[_BANNED] = float("-inf")

            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k and 0 < top_k < logits.numel():
                    cutoff = torch.topk(logits, top_k).values[-1]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1, generator=generator))

            if nxt == EOS_ID:
                break
            produced.append(nxt)
            context.append(nxt)
        return decode(produced)
    finally:
        model.train(was_training)


def load_model(settings: Settings) -> tuple[Babbler, int]:
    """The latest checkpoint, or a brand new random model if there isn't one.

    The fallback is a feature: with no checkpoint on disk the bot answers with
    pure noise, which is the honest state of a model that has learned nothing.
    """
    path = settings.latest_checkpoint
    if not path.exists():
        return Babbler(config_from_settings(settings)), 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, int(payload.get("step", 0))


class CheckpointGenerator:
    """Callable the bot hands to `Babble`. Reloads weights as training advances.

    The trainer writes `latest.pt` atomically every checkpoint; this notices the
    new mtime before the next generation and picks it up, so the bot gets smarter
    (or at least different) without a restart.
    """

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        self.settings = settings
        self.log = log or NullLog()
        self._model: Babbler | None = None
        self._step = 0
        self._mtime: float | None = None

    @property
    def step(self) -> int:
        self._ensure_current()
        return self._step

    def _ensure_current(self) -> None:
        path = self.settings.latest_checkpoint
        mtime = path.stat().st_mtime if path.exists() else None
        if self._model is not None and mtime == self._mtime:
            return

        previous = self._step
        self._model, self._step = load_model(self.settings)
        self._mtime = mtime
        self.log.event(
            "model.load",
            source="checkpoint" if mtime else "random_init",
            step=self._step,
            previous_step=previous if previous != self._step else None,
            params=self._model.num_params(),
        )

    def __call__(self, prompt: str) -> Generation:
        self._ensure_current()
        assert self._model is not None
        started = time.perf_counter()
        text = sample(
            self._model,
            prompt,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k,
        )
        return Generation(
            text=text,
            step=self._step,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k,
            max_new_tokens=self.settings.max_new_tokens,
            ms=(time.perf_counter() - started) * 1000,
        )
