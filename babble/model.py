"""A small decoder-only transformer, initialised from noise.

There is no pretrained anything in this file and no code path anywhere in the
project that downloads weights. `Babbler(ModelConfig())` is ~3.3M random
parameters, which is about 13MB of float32 -- roughly 55MB once AdamW's two
moment buffers exist, and it trains fine on a couple of CPU threads.

CPU-first choices baked into this architecture:

* bias-free Linear projections (fewer ops per matmul)
* tied input/output embeddings (one table to keep hot in cache)
* Dropout elided to Identity when dropout is 0 (the shipped default)
* tanh-approximate GELU (faster on CPU, same shape)
* preallocated KV cache for autoregressive decode (avoids redoing attention
  over the whole prefix on every new byte — the big Discord-reply win)
* oneDNN-friendly contiguous layouts; everything stays on CPU
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        return cls(**known)


def config_from_settings(settings) -> ModelConfig:
    """Model shape from `Settings`, so `config.py` need never import torch."""
    return ModelConfig(
        block_size=settings.block_size,
        n_layer=settings.n_layer,
        n_head=settings.n_head,
        n_embd=settings.n_embd,
        dropout=getattr(settings, "dropout", 0.0),
    )


def _drop(p: float) -> nn.Module:
    """Dropout is a pure no-op at p=0; Identity skips the Python/dispatch tax."""
    return nn.Dropout(p) if p > 0.0 else nn.Identity()


class KVCache:
    """Preallocated per-layer K/V buffers for CPU autoregressive decode.

    Recomputing full-sequence attention for every new byte is the dominant cost
    of Discord replies (`best_of` × `max_new_tokens`). Writing each new key/value
    into a fixed buffer and attending only up to `length` turns that into a
    cheap single-token forward. Callers should size `max_len` to
    `len(context) + max_new_tokens` (capped at `block_size`) rather than always
    allocating the full context window.
    """

    __slots__ = ("k", "v", "length", "max_len")

    def __init__(
        self,
        n_layer: int,
        batch: int,
        n_head: int,
        head_dim: int,
        max_len: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_len = max_len
        self.length = 0
        self.k = [
            torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layer)
        ]
        self.v = [
            torch.zeros(batch, n_head, max_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layer)
        ]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must divide evenly into n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        drop = self.dropout if self.training else 0.0
        if cache is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=drop
            )
        else:
            start = cache.length
            end = start + T
            if end > cache.max_len:
                raise ValueError(
                    f"KV cache overflow: need {end} slots, have {cache.max_len}"
                )
            cache.k[self.layer_idx][:, :, start:end, :] = k
            cache.v[self.layer_idx][:, :, start:end, :] = v
            k_full = cache.k[self.layer_idx][:, :, :end, :]
            v_full = cache.v[self.layer_idx][:, :, :end, :]
            if start == 0:
                # Fresh prefill: standard causal attention over the prompt.
                out = F.scaled_dot_product_attention(
                    q, k_full, v_full, is_causal=True, dropout_p=drop
                )
            elif T == 1:
                # Single new query may see every cached key.
                out = F.scaled_dot_product_attention(
                    q, k_full, v_full, is_causal=False, dropout_p=drop
                )
            else:
                # Rare multi-token append with a warm cache: build an absolute
                # causal mask. SDPA's is_causal bottom-right alignment is not
                # reliable enough here across torch builds.
                rows = torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(end, device=q.device)[None, :]
                allow = cols <= rows
                out = F.scaled_dot_product_attention(
                    q, k_full, v_full, attn_mask=allow, dropout_p=drop
                )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # tanh-approx GELU: faster on CPU than erf. Stateless, so checkpoints
        # still load — but the forward is not bit-identical to erf-GELU weights.
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            _drop(config.dropout),
        )

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache=cache)
        return x + self.mlp(self.ln2(x))


class Babbler(nn.Module):
    """The whole model. Pre-norm transformer, learned positions, tied output."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = _drop(c.dropout)
        self.blocks = nn.ModuleList(Block(c, layer_idx=i) for i in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        # Tie input and output embeddings: fewer parameters, and the model only
        # ever has to learn one representation per byte.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down what feeds each residual stream, so a 4-layer stack does not
        # start out with exploding activations.
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self, batch_size: int = 1, max_len: int | None = None) -> KVCache:
        """Fresh preallocated KV cache on CPU.

        `max_len` defaults to `block_size`. Prefer passing the tight bound
        `len(context) + max_new_tokens` so Discord replies do not zero a full
        512-step buffer when they only need a fraction of it.
        """
        c = self.config
        head_dim = c.n_embd // c.n_head
        weight = self.tok_emb.weight
        slots = c.block_size if max_len is None else min(int(max_len), c.block_size)
        if slots < 1:
            raise ValueError("max_len must be at least 1")
        return KVCache(
            c.n_layer,
            batch_size,
            c.n_head,
            head_dim,
            slots,
            device=weight.device,
            dtype=weight.dtype,
        )

    def forward(
        self, idx: torch.Tensor, cache: KVCache | None = None
    ) -> torch.Tensor:
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence of {T} exceeds block_size {self.config.block_size}"
            )
        if cache is None:
            pos = torch.arange(T, device=idx.device)
        else:
            start = cache.length
            if start + T > self.config.block_size:
                raise ValueError(
                    f"sequence of {start + T} exceeds block_size {self.config.block_size}"
                )
            pos = torch.arange(start, start + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, cache=cache)
        if cache is not None:
            cache.length += T
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        """Tied weights are counted once, which is what you want to report."""
        seen = {id(p): p.numel() for p in self.parameters()}
        return sum(seen.values())


def per_token_loss(model: Babbler, tokens: torch.Tensor) -> torch.Tensor:
    """Next-byte cross-entropy at every position, unreduced and unmasked.

    Shaped like `tokens[:, 1:]`: entry `[i, j]` is the loss of predicting
    `tokens[i, j + 1]`. Everything that reduces a loss in this file starts here,
    so there is exactly one place where the off-by-one between inputs and
    targets is decided.
    """
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view_as(targets)


def sequence_loss(
    model: Babbler,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Masked, per-example-weighted next-byte cross-entropy.

    The corpus training path passes a bos-only mask and uniform `weights` of 1.0,
    so in practice it is plain next-token loss over every real token -- there is
    no response to isolate and no row to upweight any more. `mask` and `weights`
    survive because this same function scores correction *pairs* in `generate.py`,
    where masking the shared prefix and weighting a candidate still mean something.
    Dividing by the same masked, weighted count keeps the number comparable
    between batches, so the loss curve means something.
    """
    per_token = per_token_loss(model, tokens)
    scale = mask[:, 1:].to(per_token.dtype) * weights[:, None]
    return (per_token * scale).sum() / scale.sum().clamp(min=1e-8)
