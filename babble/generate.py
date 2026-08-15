"""Sampling, and the bit that keeps the bot's weights in sync with the trainer.

Two layouts live here, sharing one decoder and one scorer:

* **Continuation** (`continue_text`, `best_continuation`) -- `<bos> text`, keep
  going. This is what the model is trained on now, so it is what the bot uses
  and what the trainer probes with. A model that only ever saw plain text does
  not answer questions; it continues them, and pretending otherwise would be a
  lie told in the shape of an API.
* **Pair** (`sample`, `best_of`, `score`) -- `<bos> prompt <sep> response`. No
  longer the training objective, but still the honest way to ask "how likely
  does the model think this answer is, given this prompt", which is what
  comparing correction pairs needs.

Decode uses a preallocated KV cache so each new byte is a single-token forward
on CPU rather than a full-prefix attention redo — that is the difference
between a multi-second `best_of` reply and a snappy one on this 3.3M model.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .config import Settings
from .core import Generation
from .cpu_runtime import configure_cpu, force_cpu_device
from .logs import EventLog, NullLog
from .model import Babbler, KVCache, ModelConfig, config_from_settings, per_token_loss
from .tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    Example,
    build_continuation_example,
    build_example,
    decode,
    prompt_context,
    text_context,
)

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
    Vocab is only 260, so a full-softmax over the top-k-masked row stays cheap;
    keeping the classic mask-then-softmax path preserves the RNG behaviour the
    tests (and the live bot) already depend on.
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


def _can_cache(model: Babbler, context_len: int, max_new_tokens: int) -> bool:
    """KV cache fits when the whole reply stays inside `block_size`."""
    return context_len + max_new_tokens <= model.config.block_size


# --- the decoder ---------------------------------------------------------


@torch.inference_mode()
def _decode_from(
    model: Babbler,
    context: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None,
) -> str:
    """Continue `context` one byte at a time until <eos>.

    An untrained model almost never emits <eos>, so `max_new_tokens` is what
    actually ends most early generations. That is expected.

    Prefills the prompt once into a KV cache, then feeds a single token per
    step. Falls back to the old windowed full-forward path only when the reply
    would overflow `block_size` (rare for Discord prompts).
    """
    was_training = model.training
    model.eval()
    try:
        context = list(context)
        produced: list[int] = []
        use_cache = _can_cache(model, len(context), max_new_tokens) and hasattr(
            model, "new_cache"
        )
        if use_cache:
            cache: KVCache | None = model.new_cache(1)
            prompt = torch.tensor([context], dtype=torch.long)
            logits = model(prompt, cache=cache)[:, -1]
            for _ in range(max_new_tokens):
                nxt = int(_next_token(logits, temperature, top_k, generator)[0])
                if nxt == EOS_ID:
                    break
                produced.append(nxt)
                context.append(nxt)
                logits = model(
                    torch.tensor([[nxt]], dtype=torch.long), cache=cache
                )[:, -1]
            return decode(produced)

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


@torch.inference_mode()
def _decode_many_from(
    model: Babbler,
    context: list[int],
    n: int,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None,
) -> list[str]:
    """`n` independent continuations of the same context, in one batched pass.

    Drawing candidates one at a time is wasteful: they share a context and are
    the same length at every step, so they can go through the model together.
    With a shared-shape KV cache the per-step cost stays near one-token-forward
    rather than growing with the prefix — that is what keeps `best_of` under a
    couple of seconds on two CPU threads.

    That worst case is measured on a randomly initialised model, which almost
    never emits <eos> and so always runs the full `max_new_tokens`. A trained
    model stops early and is considerably faster.
    """
    was_training = model.training
    model.eval()
    try:
        produced: list[list[int]] = [[] for _ in range(n)]
        finished = torch.zeros(n, dtype=torch.bool)
        use_cache = _can_cache(model, len(context), max_new_tokens) and hasattr(
            model, "new_cache"
        )

        if use_cache:
            cache: KVCache | None = model.new_cache(n)
            prompt = torch.tensor([list(context)], dtype=torch.long).repeat(n, 1)
            logits = model(prompt, cache=cache)[:, -1]
            for _ in range(max_new_tokens):
                nxt = _next_token(logits, temperature, top_k, generator)
                for i in range(n):
                    if not finished[i] and int(nxt[i]) != EOS_ID:
                        produced[i].append(int(nxt[i]))
                finished |= nxt == EOS_ID
                if bool(finished.all()):
                    break
                logits = model(nxt[:, None], cache=cache)[:, -1]
            return [decode(row) for row in produced]

        tokens = torch.tensor([list(context)], dtype=torch.long).repeat(n, 1)
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


@torch.inference_mode()
def _score_examples(model: Babbler, examples: list[Example]) -> list[float]:
    """Mean per-byte loss over each example's masked tokens. Lower is better.

    Right-padded to the longest, with the pad excluded from the mean exactly the
    way the trainer excludes it -- a candidate must not be rewarded or punished
    for how long the *other* candidates happened to be.
    """
    if not examples:
        return []
    was_training = model.training
    model.eval()
    try:
        width = max(len(e) for e in examples)
        tokens = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(examples), width), dtype=torch.float32)
        for i, example in enumerate(examples):
            tokens[i, : len(example)] = torch.as_tensor(example.tokens, dtype=torch.long)
            mask[i, : len(example)] = torch.as_tensor(example.mask, dtype=torch.float32)
        per_token = per_token_loss(model, tokens)
        scale = mask[:, 1:]
        totals = (per_token * scale).sum(dim=1) / scale.sum(dim=1).clamp(min=1e-8)
        return [float(t) for t in totals]
    finally:
        model.train(was_training)


def _pick_best(candidates: list[str], scores: list[float]) -> str:
    """The lowest-scoring candidate. `min` on the pair, so ties break on text."""
    return min(zip(scores, candidates))[1]


# --- the corpus layout: continue what was said ---------------------------


def continue_text(
    model: Babbler,
    prefix: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> str:
    """Keep writing after `prefix`, in the layout the corpus objective trains."""
    return _decode_from(
        model,
        text_context(prefix, model.config.block_size),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )


def continue_many(
    model: Babbler,
    prefix: str,
    n: int,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> list[str]:
    """`n` independent continuations of `prefix`, in one batched pass."""
    return _decode_many_from(
        model,
        text_context(prefix, model.config.block_size),
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )


def score_continuations(model: Babbler, prefix: str, continuations: list[str]) -> list[float]:
    """Mean per-byte loss of each continuation after `prefix`. Lower is better."""
    return _score_examples(
        model,
        [build_continuation_example(prefix, c, model.config.block_size) for c in continuations],
    )


def best_continuation(
    model: Babbler,
    prefix: str,
    *,
    n: int = 4,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> str:
    """Draw `n` continuations of `prefix` and keep the one the model likes best.

    Same bargain as `best_of`, in the layout the model is actually trained on:
    the reward model is the model, because at this data scale it is the only
    scorer that exists.
    """
    if n <= 1:
        return continue_text(
            model,
            prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
    candidates = continue_many(
        model,
        prefix,
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    real = [c for c in candidates if c]
    if not real:
        return candidates[0]
    return _pick_best(real, score_continuations(model, prefix, real))


# --- the pair layout: score an answer against a prompt -------------------


def sample(
    model: Babbler,
    prompt: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    top_k: int = 40,
    generator: torch.Generator | None = None,
) -> str:
    """Continue `<bos> prompt <sep>` one byte at a time until <eos>."""
    return _decode_from(
        model,
        prompt_context(prompt, model.config.block_size),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )


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
    """`n` independent responses to the same prompt, in one batched pass."""
    return _decode_many_from(
        model,
        prompt_context(prompt, model.config.block_size),
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )


def score_many(model: Babbler, prompt: str, responses: list[str]) -> list[float]:
    """Mean per-byte loss for each response after `prompt`, in one forward pass.

    Lower is better, over the same `<bos> prompt <sep> response <eos>` layout a
    correction pair is stored in, so "the answer the model likes best" means
    exactly what it sounds like.
    """
    return _score_examples(
        model, [build_example(prompt, r, model.config.block_size) for r in responses]
    )


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
    """Draw `n` answers to `prompt` and keep the one the model scores best.

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
    return _pick_best(real, score_many(model, prompt, real))


# --- loading -------------------------------------------------------------


def load_model(settings: Settings) -> tuple[Babbler, int]:
    """The latest checkpoint, or a brand new random model if there isn't one.

    The fallback is a feature: with no checkpoint on disk the bot answers with
    pure noise, which is the honest state of a model that has learned nothing.
    Always lands on CPU — babble never offloads inference to a GPU.
    """
    configure_cpu(getattr(settings, "train_threads", None))
    device = force_cpu_device()
    path = settings.latest_checkpoint
    if not path.exists():
        model = Babbler(config_from_settings(settings)).to(device)
        model.eval()
        return model, 0
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, int(payload.get("step", 0))


class CheckpointGenerator:
    """Callable the bot hands to `Babble`. Reloads weights as training advances.

    The trainer writes `latest.pt` atomically every checkpoint; this notices the
    new mtime before the next generation and picks it up, so the bot gets smarter
    (or at least different) without a restart.

    What it produces is a **continuation** of what was said to it, not an answer
    to it, because that is the only thing the corpus objective ever taught it to
    do. Feeding it `<bos> prompt <sep>` instead would put a token in front of it
    that never once appeared in training.
    """

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        self.settings = settings
        self.log = log or NullLog()
        self._model: Babbler | None = None
        self._step = 0
        self._mtime: float | None = None
        # Bot replies share the machine with everything else; keep inference on
        # the same capped thread count the trainer uses so we never stampede.
        configure_cpu(settings.train_threads)

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
            device="cpu",
        )

    def __call__(self, prompt: str) -> Generation:
        self._ensure_current()
        assert self._model is not None
        started = time.perf_counter()
        text = best_continuation(
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
