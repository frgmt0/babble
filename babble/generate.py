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
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Settings
from .core import Generation
from .cpu_runtime import configure_cpu, force_cpu_device, prepare_for_cpu_infer
from .logs import EventLog, NullLog
from .model import Babbler, KVCache, ModelConfig, config_from_settings, per_token_loss
from .subword import BPETokenizer, ByteTokenizer, Tokenizer
from .subword import build_continuation_example as tok_continuation_example
from .subword import text_context as tok_text_context
from .tokenizer import (
    BOS_ID,
    PAD_ID,
    SEP_ID,
    VOCAB_SIZE,
    Example,
    build_example,
    prompt_context,
)

# The structural tokens are inputs, not outputs. Banning them from sampling means
# a random model cannot emit a stray <sep> into the middle of a Discord message.
_BANNED = [PAD_ID, BOS_ID, SEP_ID]


def tokenizer_for_checkpoint(checkpoint_path, vocab_size: int) -> Tokenizer:
    """The tokenizer that belongs with this checkpoint file.

    Looks for `tokenizer.json` beside the `.pt`. Present → load that BPE
    (and refuse if its vocab_size disagrees with the weights). Absent →
    the historical byte tokenizer, but only when the checkpoint itself is
    a 260-id byte model. A larger vocab with no json is a broken promotion,
    not a silent fallback to bytes.
    """
    path = Path(checkpoint_path)
    json_path = path.parent / "tokenizer.json"
    if json_path.is_file():
        tok = BPETokenizer.from_json(json_path)
        if tok.vocab_size != vocab_size:
            raise ValueError(
                f"checkpoint vocab_size {vocab_size} does not match tokenizer "
                f"vocab_size {tok.vocab_size} ({json_path}) -- wrong tokenizer.json "
                f"for this checkpoint"
            )
        return tok
    if vocab_size == VOCAB_SIZE:
        return ByteTokenizer()
    raise ValueError(
        f"checkpoint at {path} has vocab_size {vocab_size} but no tokenizer.json "
        f"beside it -- cannot serve a non-byte checkpoint without the tokenizer "
        f"that shipped with it"
    )


def _serving_tokenizer(model: Babbler, tok: Tokenizer | None = None) -> Tokenizer:
    if tok is not None:
        return tok
    attached = getattr(model, "tokenizer", None)
    if attached is not None:
        return attached
    return ByteTokenizer()


def _banned_ids(tok: Tokenizer) -> list[int]:
    return [tok.specials.pad, tok.specials.bos, tok.specials.sep]


def _apply_repetition_penalty(
    logits: torch.Tensor, seen: torch.Tensor | None, penalty: float
) -> torch.Tensor:
    """HuggingFace-style penalty on tokens already in the prompt or completion.

    Positive logits are divided by `penalty`, negative logits are multiplied,
    so already-seen tokens become less likely without flipping their sign.
    `penalty == 1` is a no-op.
    """
    if penalty == 1.0 or seen is None or seen.numel() == 0:
        return logits
    if seen.dim() == 1:
        seen = seen.unsqueeze(0)
    logits = logits.clone()
    for i in range(logits.size(0)):
        uniq = seen[i].unique()
        scores = logits[i].index_select(0, uniq)
        logits[i].index_copy_(
            0, uniq, torch.where(scores < 0, scores * penalty, scores / penalty)
        )
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus mask: keep the smallest prefix of tokens whose mass >= `top_p`."""
    if not (0.0 < top_p < 1.0):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    # Always keep the highest-probability token, then drop the tail that
    # pushed the running mass over the nucleus.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def _next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None,
    banned: list[int] | None = None,
    *,
    seen: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """One sampling step over a `(batch, vocab)` block of logits.

    Shared by the single and batched samplers so there is exactly one place
    where temperature, top_k, top_p, repetition penalty and the banned
    structural tokens are applied. Vocab is only 260 on the byte path, so a
    full-softmax over the top-k-masked row stays cheap; keeping the classic
    mask-then-softmax path preserves the RNG behaviour the tests (and the
    live bot) already depend on when top_p is 1 and the penalty is 1.
    """
    logits = logits.clone()
    logits[:, banned if banned is not None else _BANNED] = float("-inf")
    logits = _apply_repetition_penalty(logits, seen, repetition_penalty)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k and 0 < top_k < logits.size(-1):
        cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    logits = _apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _can_cache(model: Babbler, context_len: int, max_new_tokens: int) -> bool:
    """Use the KV cache whenever the prompt itself still fits in `block_size`.

    Older logic required `context + max_new_tokens <= block_size`, which
    silently fell back to full-prefix attention for any long Discord prompt
    even when hundreds of cached slots were still free. Decode now caches
    up to `block_size`, then continues on the sliding-window path if the
    window fills so `max_new_tokens` is not silently truncated.
    """
    del max_new_tokens
    return context_len < model.config.block_size and hasattr(model, "new_cache")


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
    tok: Tokenizer | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """Continue `context` one token at a time until <eos>.

    An untrained model almost never emits <eos>, so `max_new_tokens` is what
    actually ends most early generations. That is expected.

    Prefills the prompt once into a KV cache, then feeds a single token per
    step. Falls back to the old windowed full-forward path only when the reply
    would overflow `block_size` (rare for Discord prompts).
    """
    tok = _serving_tokenizer(model, tok)
    banned = _banned_ids(tok)
    eos_id = tok.specials.eos
    was_training = model.training
    model.eval()
    try:
        context = list(context)
        produced: list[int] = []
        seen = torch.tensor([context], dtype=torch.long)
        use_cache = _can_cache(model, len(context), max_new_tokens)
        remaining = max_new_tokens
        if use_cache:
            cache: KVCache | None = model.new_cache(
                1, max_len=min(model.config.block_size, len(context) + max_new_tokens)
            )
            logits = model(seen, cache=cache)[:, -1]
            step = torch.empty((1, 1), dtype=torch.long)
            for _ in range(remaining):
                nxt = int(
                    _next_token(
                        logits,
                        temperature,
                        top_k,
                        generator,
                        banned,
                        seen=seen,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                    )[0]
                )
                remaining -= 1
                if nxt == eos_id:
                    return tok.decode(produced)
                produced.append(nxt)
                context.append(nxt)
                seen = torch.cat([seen, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
                if cache.length >= cache.max_len:
                    break
                step[0, 0] = nxt
                logits = model(step, cache=cache)[:, -1]
            else:
                return tok.decode(produced)

        window_t = torch.empty((1, model.config.block_size), dtype=torch.long)
        while remaining > 0:
            window = context[-model.config.block_size :]
            nwin = len(window)
            window_t[0, :nwin].copy_(torch.as_tensor(window, dtype=torch.long))
            logits = model(window_t[:, :nwin])[:, -1]
            nxt = int(
                _next_token(
                    logits,
                    temperature,
                    top_k,
                    generator,
                    banned,
                    seen=seen,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                )[0]
            )
            remaining -= 1
            if nxt == eos_id:
                break
            produced.append(nxt)
            context.append(nxt)
            seen = torch.cat([seen, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        return tok.decode(produced)
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
    tok: Tokenizer | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
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
    tok = _serving_tokenizer(model, tok)
    banned = _banned_ids(tok)
    eos_id = tok.specials.eos
    was_training = model.training
    model.eval()
    try:
        produced: list[list[int]] = [[] for _ in range(n)]
        finished = torch.zeros(n, dtype=torch.bool)
        tokens = torch.tensor([list(context)], dtype=torch.long).repeat(n, 1)
        use_cache = _can_cache(model, len(context), max_new_tokens)
        remaining = max_new_tokens

        if use_cache:
            cache: KVCache | None = model.new_cache(
                n, max_len=min(model.config.block_size, len(context) + max_new_tokens)
            )
            logits = model(tokens, cache=cache)[:, -1]
            step = torch.empty((n, 1), dtype=torch.long)
            for _ in range(remaining):
                nxt = _next_token(
                    logits,
                    temperature,
                    top_k,
                    generator,
                    banned,
                    seen=tokens,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                )
                remaining -= 1
                ids = nxt.tolist()
                for i, nid in enumerate(ids):
                    if not finished[i] and nid != eos_id:
                        produced[i].append(nid)
                finished |= nxt == eos_id
                tokens = torch.cat([tokens, nxt[:, None]], dim=1)
                if bool(finished.all()):
                    return [tok.decode(row) for row in produced]
                if cache.length >= cache.max_len:
                    break
                step[:, 0] = nxt
                logits = model(step, cache=cache)[:, -1]
            else:
                return [tok.decode(row) for row in produced]

        for _ in range(remaining):
            window = tokens[:, -model.config.block_size :]
            nxt = _next_token(
                model(window)[:, -1],
                temperature,
                top_k,
                generator,
                banned,
                seen=tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
            )
            # A row that has already emitted <eos> keeps being fed -- rows are
            # independent inside the batch, so its continuation is ignored
            # rather than removed.
            for i in range(n):
                if not finished[i] and int(nxt[i]) != eos_id:
                    produced[i].append(int(nxt[i]))
            finished |= nxt == eos_id
            if bool(finished.all()):
                break
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        return [tok.decode(row) for row in produced]
    finally:
        model.train(was_training)


@torch.inference_mode()
def _score_examples(
    model: Babbler, examples: list[Example], *, pad_id: int = PAD_ID
) -> list[float]:
    """Mean per-token loss over each example's masked tokens. Lower is better.

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
        tokens = torch.full((len(examples), width), pad_id, dtype=torch.long)
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
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> str:
    """Keep writing after `prefix`, in the layout the corpus objective trains."""
    tok = _serving_tokenizer(model)
    return _decode_from(
        model,
        tok_text_context(tok, prefix, model.config.block_size),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
        tok=tok,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def continue_many(
    model: Babbler,
    prefix: str,
    n: int,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[str]:
    """`n` independent continuations of `prefix`, in one batched pass."""
    tok = _serving_tokenizer(model)
    return _decode_many_from(
        model,
        tok_text_context(tok, prefix, model.config.block_size),
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
        tok=tok,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def score_continuations(model: Babbler, prefix: str, continuations: list[str]) -> list[float]:
    """Mean per-token loss of each continuation after `prefix`. Lower is better."""
    tok = _serving_tokenizer(model)
    return _score_examples(
        model,
        [tok_continuation_example(tok, prefix, c, model.config.block_size) for c in continuations],
        pad_id=tok.specials.pad,
    )


def best_continuation(
    model: Babbler,
    prefix: str,
    *,
    n: int = 4,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
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
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
    candidates = continue_many(
        model,
        prefix,
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generator=generator,
    )
    real = [c for c in candidates if c]
    if not real:
        return candidates[0]
    return _pick_best(real, score_continuations(model, prefix, real))


# --- the pair layout: score an answer against a prompt -------------------


def _pair_context(tok: Tokenizer, prompt: str, block_size: int) -> list[int]:
    if isinstance(tok, ByteTokenizer):
        return prompt_context(prompt, block_size)
    reserved = max(1, block_size // 4)
    budget = max(1, block_size - 3 - reserved)
    return [tok.specials.bos, *tok.encode(prompt)[-budget:], tok.specials.sep]


def _pair_example(tok: Tokenizer, prompt: str, response: str, block_size: int) -> Example:
    if isinstance(tok, ByteTokenizer):
        return build_example(prompt, response, block_size)
    reserved = max(1, block_size // 4)
    pbudget = max(1, block_size - 3 - reserved)
    budget = block_size - 3
    prompt_ids = tok.encode(prompt)[-pbudget:]
    response_ids = tok.encode(response)[: budget - len(prompt_ids)]
    tokens = [tok.specials.bos, *prompt_ids, tok.specials.sep, *response_ids, tok.specials.eos]
    mask = [0] * (len(prompt_ids) + 2) + [1] * (len(response_ids) + 1)
    return Example(tokens=tokens, mask=mask)


def sample(
    model: Babbler,
    prompt: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> str:
    """Continue `<bos> prompt <sep>` one token at a time until <eos>."""
    tok = _serving_tokenizer(model)
    return _decode_from(
        model,
        _pair_context(tok, prompt, model.config.block_size),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
        tok=tok,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def sample_many(
    model: Babbler,
    prompt: str,
    n: int,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[str]:
    """`n` independent responses to the same prompt, in one batched pass."""
    tok = _serving_tokenizer(model)
    return _decode_many_from(
        model,
        _pair_context(tok, prompt, model.config.block_size),
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
        tok=tok,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def score_many(model: Babbler, prompt: str, responses: list[str]) -> list[float]:
    """Mean per-byte loss for each response after `prompt`, in one forward pass.

    Lower is better, over the same `<bos> prompt <sep> response <eos>` layout a
    correction pair is stored in, so "the answer the model likes best" means
    exactly what it sounds like.
    """
    tok = _serving_tokenizer(model)
    return _score_examples(
        model,
        [_pair_example(tok, prompt, r, model.config.block_size) for r in responses],
        pad_id=tok.specials.pad,
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
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
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
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
    candidates = sample_many(
        model,
        prompt,
        n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
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
    configure_cpu(getattr(settings, "infer_threads", None) or getattr(settings, "train_threads", None))
    device = force_cpu_device()
    path = settings.latest_checkpoint
    if not path.exists():
        model = Babbler(config_from_settings(settings)).to(device)
        model.tokenizer = ByteTokenizer()
        model.eval()
        model = prepare_for_cpu_infer(model)
        return model, 0
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig.from_dict(payload["config"])
    tok = tokenizer_for_checkpoint(path, cfg.vocab_size)
    model = Babbler(cfg)
    model.load_state_dict(payload["model"])
    model.tokenizer = tok
    model.to(device)
    model.eval()
    model = prepare_for_cpu_infer(model)
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
        self._mtime: tuple[float | None, float | None] | None = None
        # Bot replies share the machine with everything else; keep inference on
        # the same capped thread count the trainer uses so we never stampede.
        configure_cpu(getattr(settings, "infer_threads", None) or settings.train_threads)

    @property
    def step(self) -> int:
        self._ensure_current()
        return self._step

    def _ensure_current(self) -> None:
        path = self.settings.latest_checkpoint
        tok_path = self.settings.tokenizer_path
        mtime = path.stat().st_mtime if path.exists() else None
        tok_mtime = tok_path.stat().st_mtime if tok_path.exists() else None
        signature = (mtime, tok_mtime)
        if self._model is not None and signature == self._mtime:
            return

        previous = self._step
        self._model, self._step = load_model(self.settings)
        self._mtime = signature
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
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
        )
        return Generation(
            text=text,
            step=self._step,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            max_new_tokens=self.settings.max_new_tokens,
            ms=(time.perf_counter() - started) * 1000,
        )
