"""Sampling, and the bit that keeps the bot's weights in sync with the trainer."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .config import Settings
from .core import Generation
from .logs import EventLog, NullLog
from .model import Babbler, ModelConfig, config_from_settings, per_token_loss
from .tokenizer import BOS_ID, EOS_ID, PAD_ID, SEP_ID, build_example, decode, prompt_context

# The structural tokens are inputs, not outputs. Banning them from sampling means
# a random model cannot emit a stray <sep> into the middle of a Discord message.
_BANNED = [PAD_ID, BOS_ID, SEP_ID]


def _next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """One sampling step over a `(batch, vocab)` block of logits.

    Shared by the single and batched samplers so there is exactly one place
    where temperature, top_k and the banned structural tokens are applied.
    """
    logits = logits.clone()
    logits[:, _BANNED] = float("-inf")
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k and 0 < top_k < logits.size(-1):
        cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


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
            logits = model(torch.tensor([window], dtype=torch.long))[:, -1]
            nxt = int(_next_token(logits, temperature, top_k, generator)[0])

            if nxt == EOS_ID:
                break
            produced.append(nxt)
            context.append(nxt)
        return decode(produced)
    finally:
        model.train(was_training)


@torch.no_grad()
def sample_many(
    model: Babbler,
    prompt: str,
    n: int,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> list[str]:
    """`n` independent continuations of the same prompt, in one batched pass.

    Drawing candidates one at a time is wasteful: they share a prompt and are
    the same length at every step, so they can go through the model together.
    On the shipped 3.3M model at two threads, four 96-byte candidates cost about
    1.5s batched against 2.1s drawn in sequence -- sub-linear in `n`, but not
    free, because a wider batch is still more work for the same two threads.

    That 1.5s is the *worst* case: it is measured on a randomly initialised
    model, which almost never emits <eos> and so always runs the full
    `max_new_tokens`. A trained model stops early and is considerably faster.
    """
    was_training = model.training
    model.eval()
    try:
        context = prompt_context(prompt, model.config.block_size)
        tokens = torch.tensor([context], dtype=torch.long).repeat(n, 1)
        produced: list[list[int]] = [[] for _ in range(n)]
        finished = torch.zeros(n, dtype=torch.bool)

        for _ in range(max_new_tokens):
            window = tokens[:, -model.config.block_size :]
            nxt = _next_token(model(window)[:, -1], temperature, top_k, generator)
            # A row that has already emitted <eos> keeps being fed -- rows are
            # independent inside the batch, so its continuation is ignored
            # rather than removed.
            for i in range(n):
                if not finished[i] and int(nxt[i]) != EOS_ID:
                    produced[i].append(int(nxt[i]))
            finished |= nxt == EOS_ID
            if bool(finished.all()):
                break
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        return [decode(row) for row in produced]
    finally:
        model.train(was_training)


@torch.no_grad()
def score_many(model: Babbler, prompt: str, responses: list[str]) -> list[float]:
    """Mean per-byte loss for each response after `prompt`, in one forward pass.

    Lower is better. This is the same quantity the trainer optimises, over the
    same `<bos> prompt <sep> response <eos>` layout, so "the candidate the model
    likes best" means exactly what it sounds like.

    Right-padded to the longest candidate, with the pad excluded from the mean
    exactly the way the trainer excludes it -- a candidate must not be rewarded
    or punished for how long the *other* candidates happened to be.
    """
    if not responses:
        return []
    was_training = model.training
    model.eval()
    try:
        examples = [build_example(prompt, r, model.config.block_size) for r in responses]
        width = max(len(e) for e in examples)
        tokens = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(examples), width), dtype=torch.float32)
        for i, example in enumerate(examples):
            tokens[i, : len(example)] = torch.tensor(example.tokens, dtype=torch.long)
            mask[i, : len(example)] = torch.tensor(example.mask, dtype=torch.float32)
        per_token = per_token_loss(model, tokens)
        scale = mask[:, 1:]
        totals = (per_token * scale).sum(dim=1) / scale.sum(dim=1).clamp(min=1e-8)
        return [float(t) for t in totals]
    finally:
        model.train(was_training)


def score(model: Babbler, prompt: str, response: str) -> float:
    """Mean per-byte loss the model assigns to `response` after `prompt`."""
    return score_many(model, prompt, [response])[0]


def best_of(
    model: Babbler,
    prompt: str,
    *,
    n: int = 4,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> str:
    """Draw `n` candidates and keep the one the model itself scores best.

    This is the cheap, honest version of "reward the good answer": the reward
    model is the model, which at this data scale is the only scorer that exists.
    It does not need a second network, a preference head or a training loop --
    it just stops the reply being whichever sample happened to come out first.

    `n <= 1` is a plain `sample()`, and an empty candidate never wins over a
    non-empty one: scoring an empty response is scoring nothing at all.
    """
    if n <= 1:
        return sample(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

    candidates = sample_many(
        model,
        prompt,
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    real = [c for c in candidates if c]
    if not real:
        return candidates[0]
    return min(zip(score_many(model, prompt, real), real))[1]


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
        text = best_of(
            self._model,
            prompt,
            n=max(1, self.settings.best_of),
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
