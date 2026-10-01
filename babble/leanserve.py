"""Lean serving runtime for the `hf` backend's Mixtral snapshot.

`BABBLE_HF_RUNTIME=lean` swaps `hfserve.HFGenerator` (MixtralForCausalLM +
``generate()``) for this module: a hand-written forward pass and decode loop
over the very same ``model-int8.safetensors``. Nothing here imports
transformers; the tokenizer is the ``tokenizers`` file the HF path uses too.

What it does differently from the transformers path, and why it is faster on
this CPU-only, memory-bandwidth-bound box:

* **int8 weights at decode** (``BABBLE_LEAN_PRECISION=int8``, the default).
  Decode-sized matmuls (<= 8 rows) run ``aten._weight_int8pack_mm`` on the
  on-disk int8 weights and bf16 scales with bf16 activations, reading a quarter
  of the bytes an fp32 GEMV reads. Prefill-sized matmuls use an fp32 copy
  (``BABBLE_LEAN_PREFILL_FP32``), so prompt processing stays exact.
* **Exact candidate logits.** The tied lm_head scores the whole vocab through
  the int8 kernel, then every token that survives top-k is recomputed exactly
  in fp32 before penalties/top-p/sampling/scoring see it. What a reply can
  contain, and the probabilities it is sampled with, are fp32 values; only
  *which* tokens make the top-k cut is decided on bf16-accurate scores.
* **Static KV cache, one batch-1 prefill** broadcast to the best-of rows.
* **Routed-only MoE**: top-1 routing touches only the experts rows chose.
* **Row compaction**: a candidate that emits ``<eos>`` leaves the batch, so
  finished rows stop costing bandwidth (transformers keeps decoding pads).
* **Prefix KV cache** across turns (`PrefixKVCache`).

``BABBLE_LEAN_PRECISION=fp32`` keeps every matrix as the same dequantized
fp32 tensor transformers uses: the lossless fallback.

Sampling semantics match `hfserve` exactly, in its order: repetition penalty
-> no-repeat-ngram -> frequency/presence (gated like the HF path) ->
temperature -> top-k -> top-p, sampling from the post-warp distribution, and
best-of picks the candidate with the highest mean post-warp log-probability of
its tokens (``<eos>`` included, pads excluded). ``tests/test_leanserve.py``
checks the processors against transformers' own on random logits and the
forward pass against MixtralForCausalLM.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Settings
from .core import Generation
from .cpu_runtime import configure_cpu, force_cpu_device
from .hfserve import HFGenerationStats, HFServeError
from .logs import EventLog, NullLog

# Matmuls with at most this many rows use the int8 decode kernel; larger ones
# (prefill) use the fp32 path. Best-of is <= 8 in practice.
DECODE_MAX_ROWS = 8
# In int8 mode, top-k up to this size gets exact fp32 candidate rescoring;
# a larger (or disabled) top-k computes the full fp32 head instead.
RESCORE_MAX_K = 256
# `full_logits` (the parity/NLL harness) rescored this many top tokens exactly.
CHECK_RESCORE_K = 64

PRECISIONS = ("int8", "fp32")


def int8_kernel_available() -> bool:
    """Whether this torch build has a working weight-only int8 CPU matmul."""
    try:
        q = torch.ones(8, 32, dtype=torch.int8)
        s = torch.ones(8, dtype=torch.bfloat16)
        y = torch.ops.aten._weight_int8pack_mm(torch.ones(1, 32, dtype=torch.bfloat16), q, s)
    except (AttributeError, RuntimeError, NotImplementedError):
        return False
    return bool(torch.allclose(y.float(), torch.full((1, 8), 32.0)))


def _release_free_heap() -> None:
    """Hand freed load-time scratch back to the OS (glibc only; else no-op)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def _dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exactly `hfserve._unpack`: int8 -> fp32, times fp32(scale)."""
    return (q.to(torch.float32) * scale.to(torch.float32)).contiguous()


class _Fp32Linear:
    __slots__ = ("w",)

    def __init__(self, w: torch.Tensor) -> None:
        self.w = w

    def __call__(self, x: torch.Tensor, dense: bool = True) -> torch.Tensor:
        return F.linear(x, self.w)


class _Int8Linear:
    """int8 weight x bf16 activation for decode rows; fp32 for prefill rows."""

    __slots__ = ("q", "s", "out", "wf")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, *, prefill_fp32: bool) -> None:
        self.q = q.contiguous()
        self.s = scale.to(torch.bfloat16).reshape(-1).contiguous()
        self.out = int(q.shape[0])
        self.wf = _dequant(q, scale) if prefill_fp32 else None

    def dense(self) -> torch.Tensor:
        return self.wf if self.wf is not None else _dequant(self.q, self.s[:, None])

    def __call__(self, x: torch.Tensor, dense: bool | None = None) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if dense if dense is not None else x2.shape[0] > DECODE_MAX_ROWS:
            return F.linear(x, self.dense())
        y = torch.ops.aten._weight_int8pack_mm(x2.to(torch.bfloat16), self.q, self.s)
        return y.to(torch.float32).reshape(*shape[:-1], self.out)


@dataclass
class _Layer:
    ln1: torch.Tensor
    qkv: object
    o: object
    ln2: torch.Tensor
    router: torch.Tensor
    w13: list
    w2: list


@dataclass(frozen=True)
class LeanConfig:
    hidden: int
    n_heads: int
    n_kv: int
    head_dim: int
    n_layers: int
    n_experts: int
    inter: int
    vocab: int
    eps: float
    max_pos: int
    rope_theta: float
    tied: bool

    @classmethod
    def from_dict(cls, cfg: dict) -> "LeanConfig":
        """Validate that this runtime implements exactly what `cfg` describes."""

        def refuse(why: str):
            raise HFServeError(f"lean runtime does not support this snapshot ({why}); use BABBLE_HF_RUNTIME=transformers")

        if cfg.get("model_type") != "mixtral":
            refuse(f"model_type={cfg.get('model_type')!r}")
        if int(cfg.get("num_experts_per_tok", 2)) != 1:
            refuse(f"num_experts_per_tok={cfg.get('num_experts_per_tok')}")
        if cfg.get("hidden_act", "silu") != "silu":
            refuse(f"hidden_act={cfg.get('hidden_act')!r}")
        max_pos = int(cfg["max_position_embeddings"])
        window = cfg.get("sliding_window")
        if window is not None and int(window) < max_pos:
            refuse(f"sliding_window={window}")
        rope = cfg.get("rope_parameters") or {}
        if cfg.get("rope_scaling") or rope.get("rope_type", "default") != "default":
            refuse("rope scaling")
        theta = rope.get("rope_theta", cfg.get("rope_theta", 1e6))
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        return cls(
            hidden=hidden,
            n_heads=heads,
            n_kv=int(cfg.get("num_key_value_heads") or heads),
            head_dim=int(cfg.get("head_dim") or hidden // heads),
            n_layers=int(cfg["num_hidden_layers"]),
            n_experts=int(cfg["num_local_experts"]),
            inter=int(cfg["intermediate_size"]),
            vocab=int(cfg["vocab_size"]),
            eps=float(cfg["rms_norm_eps"]),
            max_pos=max_pos,
            rope_theta=float(theta),
            tied=bool(cfg.get("tie_word_embeddings", False)),
        )


class KVCache:
    """Static per-layer ``[rows, kv_heads, tmax, head_dim]`` K/V buffers.

    Only positions below the current length are ever read, so the buffers are
    allocated uninitialised. The live batch is always the leading rows.
    """

    def __init__(self, cfg: LeanConfig, rows: int, tmax: int) -> None:
        shape = (rows, cfg.n_kv, tmax, cfg.head_dim)
        self.k = [torch.empty(shape) for _ in range(cfg.n_layers)]
        self.v = [torch.empty(shape) for _ in range(cfg.n_layers)]
        self.tmax = tmax

    def broadcast_row0(self, rows: int, upto: int) -> None:
        for t in (*self.k, *self.v):
            t[1:rows, :, :upto] = t[:1, :, :upto]

    def compact(self, keep: torch.Tensor, upto: int) -> None:
        """Move rows ``keep`` (ascending) to the front, first ``upto`` positions."""
        n = int(keep.numel())
        for t in (*self.k, *self.v):
            t[:n, :, :upto] = t[:, :, :upto].index_select(0, keep)


class LeanModel:
    """Mixtral (top-1 MoE) forward over ``model-int8.safetensors``."""

    def __init__(self, model_dir: Path, *, precision: str = "int8", prefill_fp32: bool = True) -> None:
        from safetensors import safe_open

        if precision not in PRECISIONS:
            raise HFServeError(f"BABBLE_LEAN_PRECISION={precision!r} -- expected one of {PRECISIONS}")
        model_dir = Path(model_dir)
        weights = model_dir / "model-int8.safetensors"
        if not weights.exists():
            raise HFServeError(f"no model-int8.safetensors under {model_dir}")
        self.cfg = c = LeanConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
        self.precision = precision
        self.prefill_fp32 = prefill_fp32 if precision == "int8" else True
        # Tensors are read one at a time rather than as one dict, so the load
        # peak is the resident model plus one matrix, not the file twice over.
        packed = safe_open(str(weights), framework="pt")
        names = set(packed.keys())

        def get(name: str) -> torch.Tensor:
            if name not in names:
                raise HFServeError(f"snapshot is missing tensor {name}")
            return packed.get_tensor(name)

        def qs(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            q = get(name)
            if q.dtype != torch.int8:
                # A floating matrix: represent it exactly as fp32 with unit scale
                # only in fp32 mode; int8 mode needs genuine int8 weights.
                if precision == "int8":
                    raise HFServeError(f"{name} is {q.dtype}, not int8; use BABBLE_LEAN_PRECISION=fp32")
                return q.to(torch.float32), torch.ones(q.shape[0], 1)
            return q, get(name + ".scale")

        def dense(name: str) -> torch.Tensor:
            value = get(name)
            if value.dtype == torch.int8:
                return _dequant(value, get(name + ".scale"))
            return value.to(torch.float32).contiguous()

        def linear(q: torch.Tensor, s: torch.Tensor):
            if precision == "fp32":
                return _Fp32Linear(_dequant(q, s))
            return _Int8Linear(q, s, prefill_fp32=self.prefill_fp32)

        self.layers: list[_Layer] = []
        for i in range(c.n_layers):
            a = f"model.layers.{i}.self_attn."
            m = f"model.layers.{i}.block_sparse_moe."
            parts = [qs(a + f"{n}_proj.weight") for n in "qkv"]
            w13, w2 = [], []
            for e in range(c.n_experts):
                ex = f"{m}experts.{e}."
                (q1, s1), (q3, s3) = qs(ex + "w1.weight"), qs(ex + "w3.weight")
                w13.append(linear(torch.cat([q1, q3], 0), torch.cat([s1, s3], 0)))
                w2.append(linear(*qs(ex + "w2.weight")))
            self.layers.append(
                _Layer(
                    ln1=dense(f"model.layers.{i}.input_layernorm.weight"),
                    qkv=linear(torch.cat([p[0] for p in parts], 0), torch.cat([p[1] for p in parts], 0)),
                    o=linear(*qs(a + "o_proj.weight")),
                    ln2=dense(f"model.layers.{i}.post_attention_layernorm.weight"),
                    # The router stays fp32 in every mode: it is tiny, and a
                    # flipped expert choice is a far bigger error than rounding.
                    router=dense(m + "gate.weight"),
                    w13=w13,
                    w2=w2,
                )
            )
        self.norm = dense("model.norm.weight")
        self.embed = dense("model.embed_tokens.weight")
        # Transformers ties lm_head to the embedding when the config says so,
        # overwriting any stored lm_head; mirror that.
        head_name = "model.embed_tokens.weight" if c.tied or "lm_head.weight" not in names else "lm_head.weight"
        self.head_w = self.embed if head_name == "model.embed_tokens.weight" else dense(head_name)
        self.head_q: _Int8Linear | None = None
        if precision == "int8":
            self.head_q = _Int8Linear(*qs(head_name), prefill_fp32=False)
        self.param_count = sum(
            math.prod(packed.get_slice(name).get_shape())
            for name in names
            if not name.endswith(".scale") and not (name == "lm_head.weight" and head_name != name)
        )
        del packed
        _release_free_heap()

        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.int64).float() / c.head_dim))
        freqs = torch.outer(torch.arange(c.max_pos, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().contiguous()
        self.sin = emb.sin().contiguous()
        self.attn_scale = c.head_dim**-0.5

    # ---- pieces ------------------------------------------------------------

    @property
    def approx_head(self) -> bool:
        return self.head_q is not None

    def new_cache(self, rows: int, tmax: int) -> KVCache:
        return KVCache(self.cfg, rows, tmax)

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.eps))

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = self.cfg.head_dim // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rot * sin

    def _moe(self, layer: _Layer, h: torch.Tensor, dense: bool) -> torch.Tensor:
        # Top-1 routing: softmax then renormalise over one expert == weight 1.0.
        choice = F.linear(h, layer.router).argmax(-1)
        chosen = choice.tolist()
        experts = sorted(set(chosen))
        if len(experts) == 1:
            e = experts[0]
            g, u = layer.w13[e](h, dense).chunk(2, dim=-1)
            return layer.w2[e](F.silu(g) * u, dense)
        out = torch.empty_like(h)
        for e in experts:
            idx = (choice == e).nonzero().squeeze(1)
            g, u = layer.w13[e](h.index_select(0, idx), dense).chunk(2, dim=-1)
            out.index_copy_(0, idx, layer.w2[e](F.silu(g) * u, dense))
        return out

    def forward(self, ids: torch.Tensor, cache: KVCache, start: int, *, all_positions: bool = False) -> torch.Tensor:
        """Final-normed hidden states for ``ids [B, T]`` at ``start..start+T-1``.

        Writes K/V into the first ``B`` rows of ``cache``. Returns ``[B, hidden]``
        for the last position, or ``[B, T, hidden]`` with ``all_positions``.
        """
        c = self.cfg
        B, T = ids.shape
        end = start + T
        if end > cache.tmax or end > c.max_pos:
            raise HFServeError(f"sequence of {end} tokens exceeds the cache/context")
        x = self.embed[ids]
        cos, sin = self.cos[start:end], self.sin[start:end]
        mask, causal = None, False
        if T > 1 and start == 0:
            causal = True
        elif T > 1:
            kpos = torch.arange(end)
            mask = kpos[None, :] <= torch.arange(start, end)[:, None]
        qd, kd = c.n_heads * c.head_dim, c.n_kv * c.head_dim
        gqa = {"enable_gqa": True} if c.n_kv != c.n_heads else {}
        # One precision decision per call, by total rows: a prefill runs every
        # matmul (experts included, however few tokens each receives) on the
        # fp32 path; a decode step runs all of them through the int8 kernel.
        dense = B * T > DECODE_MAX_ROWS
        for li, layer in enumerate(self.layers):
            h = self._rms(x, layer.ln1)
            qkv = layer.qkv(h, dense)
            q = qkv[..., :qd].view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
            k = qkv[..., qd : qd + kd].view(B, T, c.n_kv, c.head_dim).transpose(1, 2)
            v = qkv[..., qd + kd :].view(B, T, c.n_kv, c.head_dim).transpose(1, 2)
            kc, vc = cache.k[li][:B], cache.v[li][:B]
            kc[:, :, start:end] = self._rope(k, cos, sin)
            vc[:, :, start:end] = v
            attn = F.scaled_dot_product_attention(
                self._rope(q, cos, sin),
                kc[:, :, :end],
                vc[:, :, :end],
                attn_mask=mask,
                is_causal=causal,
                scale=self.attn_scale,
                **gqa,
            )
            x = x + layer.o(attn.transpose(1, 2).reshape(B, T, qd), dense)
            h = self._rms(x, layer.ln2)
            x = x + self._moe(layer, h.reshape(B * T, -1), dense).view(B, T, -1)
        if not all_positions:
            x = x[:, -1]
        return self._rms(x, self.norm)

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Whole-vocab logits: bf16-accurate in int8 mode, exact in fp32 mode."""
        if self.head_q is not None and h.reshape(-1, h.shape[-1]).shape[0] <= DECODE_MAX_ROWS:
            return self.head_q(h)
        return F.linear(h, self.head_w)

    def exact_logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.head_w)

    def exact_logits_at(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Exact fp32 logits of tokens ``idx [B, k]`` for hidden ``h [B, H]``."""
        rows = self.head_w.index_select(0, idx.reshape(-1)).view(*idx.shape, -1)
        return torch.bmm(rows, h.unsqueeze(-1)).squeeze(-1)

    def rescored_logits(self, h: torch.Tensor, k: int = CHECK_RESCORE_K) -> torch.Tensor:
        """Whole-vocab logits with the top ``k`` exact (int8) / all exact (fp32)."""
        if self.head_q is None:
            return self.exact_logits(h)
        h2 = h.reshape(-1, h.shape[-1])
        out = []
        for chunk in h2.split(DECODE_MAX_ROWS):
            logits = self.head_q(chunk)
            idx = logits.topk(k, dim=-1).indices
            logits.scatter_(1, idx, self.exact_logits_at(chunk, idx))
            out.append(logits)
        return torch.cat(out).reshape(*h.shape[:-1], -1)

    @torch.inference_mode()
    def full_logits(self, ids: list[int]) -> torch.Tensor:
        """``[T, vocab]`` logits in one prefill (parity harness)."""
        cache = self.new_cache(1, len(ids))
        h = self.forward(torch.tensor([ids]), cache, 0, all_positions=True)[0]
        return self.rescored_logits(h)

    @torch.inference_mode()
    def decode_logits(self, ids: list[int]) -> torch.Tensor:
        """``[T, vocab]`` logits fed one token at a time (exercises decode)."""
        cache = self.new_cache(1, len(ids))
        rows = [self.rescored_logits(self.forward(torch.tensor([[t]]), cache, i)) for i, t in enumerate(ids)]
        return torch.cat(rows, 0)


# --------------------------------------------------------------------------
# sampling (transformers-equivalent processors, transformers' order)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def keep_k(self, vocab: int) -> int:
        return min(max(int(self.top_k), 1), vocab) if self.top_k and self.top_k > 0 else vocab


def ngram_bans(hist: torch.Tensor, n: int, vocab: int) -> torch.Tensor | None:
    """``[B, V]`` mask of tokens that would repeat an n-gram of ``hist [B, L]``.

    Same rule as transformers' NoRepeatNGramLogitsProcessor: ban every token
    that followed an earlier occurrence of the last ``n-1`` tokens.
    """
    cur = hist.shape[1]
    if not n or n < 1 or cur + 1 < n:
        return None
    if n == 1:
        ban = torch.zeros(hist.shape[0], vocab, dtype=torch.bool)
        return ban.scatter_(1, hist, True)
    if cur < n:
        return None
    windows = hist.unfold(1, n, 1)  # [B, W, n]
    match = (windows[..., :-1] == hist[:, cur - n + 1 :].unsqueeze(1)).all(-1)
    if not bool(match.any()):
        return None
    ban = torch.zeros(hist.shape[0], vocab + 1, dtype=torch.bool)
    ban.scatter_(1, torch.where(match, windows[..., -1], vocab), True)
    return ban[:, :vocab]


def penalize(
    scores: torch.Tensor,
    counts: torch.Tensor,
    ban: torch.Tensor | None,
    cfg: SamplingConfig,
) -> torch.Tensor:
    """Elementwise processors up to temperature, on any gathered column subset.

    ``counts`` holds how often each column's token occurs in the sequence so
    far (prompt included), ``ban`` the no-repeat-ngram mask, same shape.
    """
    seen = counts > 0
    s = scores
    if cfg.repetition_penalty != 1.0:
        p = cfg.repetition_penalty
        s = torch.where(seen, torch.where(s < 0, s * p, s / p), s)
    if ban is not None:
        s = s.masked_fill(ban, -math.inf)
    if cfg.frequency_penalty or cfg.presence_penalty:
        s = s - counts * cfg.frequency_penalty
        if cfg.presence_penalty:
            s = s - seen.to(s.dtype) * cfg.presence_penalty
    if cfg.temperature != 1.0:
        s = s / cfg.temperature
    return s


def top_p_mask(vals: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p on descending-sorted ``vals`` exactly as TopPLogitsWarper does."""
    if top_p >= 1.0:
        return vals
    asc = vals.flip(-1)
    remove = asc.softmax(-1).cumsum(-1) <= (1 - top_p)
    remove[..., -1:] = False
    return vals.masked_fill(remove.flip(-1), -math.inf)


def warp(scores: torch.Tensor, counts: torch.Tensor, ban: torch.Tensor | None, cfg: SamplingConfig):
    """Full-vocab processors -> ``(vals [B, K] descending, idx [B, K])``.

    Every token outside ``idx`` has probability zero; ``vals`` may hold -inf
    for candidates removed by top-p. Identical to transformers' TopK->TopP
    masking except that exact floating-point ties at the k-th value keep
    exactly k tokens here (transformers keeps all tied ones).
    """
    s = penalize(scores, counts, ban, cfg)
    vals, idx = torch.topk(s, cfg.keep_k(s.shape[-1]), dim=-1)
    return top_p_mask(vals, cfg.top_p), idx


# --------------------------------------------------------------------------
# prefix KV cache
# --------------------------------------------------------------------------


@dataclass
class _PrefixEntry:
    ids: torch.Tensor  # [P] int64
    k: list  # per layer [1, kv, P, D]
    v: list
    nbytes: int


def _common_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0
    diff = (a[:n] != b[:n]).nonzero()
    return n if diff.numel() == 0 else int(diff[0])


class PrefixKVCache:
    """Bounded LRU of batch-1 prompt K/V, looked up by longest token prefix.

    The generator seam only ever sees the serialized transcript, never a
    channel id, and a transcript *is* its conversation: turn N's prompt is
    turn N-1's prompt (minus the trailing ``<sep>``) plus the reply and the
    new message. So entries are content-addressed: `lookup` scans every entry
    for the longest common token prefix. That hits across turns of any
    number of concurrent conversations and can never serve K/V computed for
    different tokens. An entry made redundant by its own continuation (its
    ids minus the last are a prefix of the new prompt) is replaced by it.

    Bounded by entry count and bytes; not thread-safe on its own (the
    generator holds its lock around every use).
    """

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[int, _PrefixEntry] = OrderedDict()
        self._next = 0
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.reused_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, ids: torch.Tensor) -> tuple[_PrefixEntry | None, int]:
        best_key, best, best_len = None, None, 0
        for key, entry in self._entries.items():
            n = _common_prefix(entry.ids, ids)
            if n > best_len:
                best_key, best, best_len = key, entry, n
        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best, best_len

    def record(self, reused: int) -> None:
        if reused > 0:
            self.hits += 1
            self.reused_tokens += reused
        else:
            self.misses += 1

    def store(self, ids: torch.Tensor, k: list, v: list) -> None:
        if not self.enabled:
            return
        nbytes = sum(t.numel() * t.element_size() for t in (*k, *v))
        if nbytes > self.max_bytes:
            return
        for key in [
            key
            for key, entry in self._entries.items()
            if _common_prefix(entry.ids, ids) >= entry.ids.numel() - 1
        ]:
            self._drop(key)
        self._entries[self._next] = _PrefixEntry(ids=ids.clone(), k=k, v=v, nbytes=nbytes)
        self._next += 1
        self.bytes += nbytes
        while len(self._entries) > self.max_entries or self.bytes > self.max_bytes:
            self._drop(next(iter(self._entries)))
            self.evictions += 1

    def _drop(self, key: int) -> None:
        entry = self._entries.pop(key)
        self.bytes -= entry.nbytes

    def clear(self) -> None:
        self._entries.clear()
        self.bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "bytes": self.bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "reused_tokens": self.reused_tokens,
        }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


@dataclass
class _Result:
    tokens: list  # per candidate, as sampled (eos included)
    mean_logprob: list
    best: int
    stats: HFGenerationStats
    prefix_reused: int


def generate(
    model: LeanModel,
    ids: list[int],
    *,
    n: int,
    max_new: int,
    cfg: SamplingConfig,
    pad_id: int,
    eos_id: int,
    prefix_cache: PrefixKVCache | None = None,
    started: float | None = None,
) -> _Result:
    """Best-of-``n`` sampling with transformers' generate() semantics."""
    started = time.perf_counter() if started is None else started
    V = model.cfg.vocab
    P = len(ids)
    n = max(1, int(n))
    max_new = max(1, int(max_new))
    tmax = min(P + max_new, model.cfg.max_pos)
    ids_t = torch.tensor(ids, dtype=torch.long)

    # --- prefill (batch 1, optionally resuming a cached prefix), then broadcast
    cache = model.new_cache(n, tmax)
    reused = 0
    if prefix_cache is not None and prefix_cache.enabled:
        entry, common = prefix_cache.lookup(ids_t)
        reused = min(common, P - 1)
        if entry is not None and reused > 0:
            for dst, src in zip((*cache.k, *cache.v), (*entry.k, *entry.v)):
                dst[:1, :, :reused] = src[:, :, :reused]
        prefix_cache.record(reused)
    h = model.forward(ids_t[reused:].unsqueeze(0), cache, reused)
    if prefix_cache is not None and prefix_cache.enabled:
        prefix_cache.store(
            ids_t,
            [t[:1, :, :P].clone() for t in cache.k],
            [t[:1, :, :P].clone() for t in cache.v],
        )
    if n > 1:
        cache.broadcast_row0(n, P)
        h = h.expand(n, -1)

    hist = torch.empty(n, tmax, dtype=torch.long)
    hist[:, :P] = ids_t
    counts = torch.zeros(n, V)
    counts.scatter_add_(1, hist[:, :P], torch.ones(n, P))
    use_rescore = model.approx_head and 0 < cfg.keep_k(V) <= RESCORE_MAX_K

    active = list(range(n))
    tokens: list[list[int]] = [[] for _ in range(n)]
    lp_sum = [0.0] * n
    n_tok = [0] * n
    first_s = last_s = None
    first_step_tokens = 0
    for t in range(max_new):
        cur = P + t
        ban = ngram_bans(hist[:, :cur], cfg.no_repeat_ngram_size, V)
        if use_rescore:
            s = penalize(model.head_logits(h), counts, ban, cfg)
            _, idx = torch.topk(s, cfg.keep_k(V), dim=-1)
            vals = penalize(
                model.exact_logits_at(h, idx),
                counts.gather(1, idx),
                None if ban is None else ban.gather(1, idx),
                cfg,
            )
            vals, order = vals.sort(dim=-1, descending=True)
            idx = idx.gather(1, order)
            vals = top_p_mask(vals, cfg.top_p)
        else:
            vals, idx = warp(model.exact_logits(h), counts, ban, cfg)
        logprobs = vals.log_softmax(-1)
        pick = torch.multinomial(logprobs.exp(), 1)
        nxt = idx.gather(1, pick).squeeze(1)
        chosen_lp = logprobs.gather(1, pick).squeeze(1).tolist()

        now = time.perf_counter()
        keep_rows = []
        for row, (orig, tok, lp) in enumerate(zip(active, nxt.tolist(), chosen_lp)):
            tokens[orig].append(tok)
            if tok != pad_id:
                lp_sum[orig] += lp
                n_tok[orig] += 1
            if tok != eos_id:
                keep_rows.append(row)
        if first_s is None:
            first_s = now
            first_step_tokens = sum(1 for tok in nxt.tolist() if tok != pad_id)
        last_s = now
        if t == max_new - 1 or not keep_rows or cur + 1 >= tmax:
            break

        hist[:, cur] = nxt
        counts.scatter_add_(1, nxt[:, None], torch.ones(len(active), 1))
        if len(keep_rows) < len(active):
            # Row compaction: finished candidates leave the batch for good.
            keep = torch.tensor(keep_rows, dtype=torch.long)
            cache.compact(keep, cur)
            hist = hist.index_select(0, keep)
            counts = counts.index_select(0, keep)
            nxt = nxt.index_select(0, keep)
            active = [active[r] for r in keep_rows]
        h = model.forward(nxt[:, None], cache, cur)

    means = [lp_sum[i] / n_tok[i] if n_tok[i] else -math.inf for i in range(n)]
    best = max(range(n), key=lambda i: (means[i], -i))
    end_s = time.perf_counter()
    first_s = first_s or end_s
    last_s = last_s or first_s
    keep_tokens = [tok for tok in tokens[best] if tok not in (pad_id, eos_id)]
    stats = HFGenerationStats(
        ttft_s=max(0.0, first_s - started),
        total_s=max(0.0, end_s - started),
        steady_s=max(0.0, last_s - first_s),
        candidate_token_counts=tuple(n_tok),
        first_step_tokens=first_step_tokens,
        selected_index=best,
        selected_content_tokens=len(keep_tokens),
    )
    return _Result(tokens=tokens, mean_logprob=means, best=best, stats=stats, prefix_reused=reused)


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class LeanGenerator:
    """Drop-in for `hfserve.HFGenerator` backed by `LeanModel`."""

    runtime = "lean"

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        from tokenizers import Tokenizer

        self.settings = settings
        self.log = log or NullLog()
        if settings.hf_model_dir is None:
            raise HFServeError("BABBLE_SERVE_BACKEND=hf but BABBLE_HF_MODEL_DIR is not set")
        model_dir = Path(settings.hf_model_dir)
        if not model_dir.is_dir():
            raise HFServeError(f"BABBLE_HF_MODEL_DIR={model_dir} is not a directory")
        configure_cpu(getattr(settings, "infer_threads", None) or settings.train_threads)
        self.device = force_cpu_device()

        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise HFServeError(f"no tokenizer.json under {model_dir}")
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        ids = {name: self.tokenizer.token_to_id(name) for name in ("<pad>", "<bos>", "<sep>", "<eos>")}
        absent = [name for name, tid in ids.items() if tid is None]
        if absent:
            raise HFServeError(f"tokenizer at {tok_path} lacks special tokens: {absent}")
        self.pad_id, self.bos_id, self.sep_id, self.eos_id = (ids[k] for k in ("<pad>", "<bos>", "<sep>", "<eos>"))

        precision = str(getattr(settings, "lean_precision", "int8") or "int8").lower()
        if precision not in PRECISIONS:
            raise HFServeError(f"BABBLE_LEAN_PRECISION={precision!r} -- expected one of {PRECISIONS}")
        self.precision_fallback = None
        if precision == "int8" and not int8_kernel_available():
            self.precision_fallback = "int8 kernel unavailable in this torch build"
            precision = "fp32"
        self.model = LeanModel(
            model_dir,
            precision=precision,
            prefill_fp32=bool(getattr(settings, "lean_prefill_fp32", True)),
        )
        self.precision = precision
        self.model_id = model_dir.name
        self.param_count = self.model.param_count
        self.max_position_embeddings = self.model.cfg.max_pos
        self.prefix_cache = PrefixKVCache(
            max_entries=int(getattr(settings, "lean_prefix_cache_entries", 32)),
            max_bytes=int(getattr(settings, "lean_prefix_cache_mb", 512)) * 1024 * 1024,
        )
        self._lock = threading.Lock()
        self._extra_penalties = _flag("BABBLE_HF_FREQUENCY_PENALTIES")
        self.step = 0
        self.log.event(
            "model.load",
            source="hf",
            runtime="lean",
            precision=self.precision,
            precision_fallback=self.precision_fallback,
            prefill_fp32=self.model.prefill_fp32,
            prefix_cache_mb=self.prefix_cache.max_bytes // (1024 * 1024),
            prefix_cache_entries=self.prefix_cache.max_entries,
            model_dir=str(model_dir),
            step=self.step,
            params=self.param_count,
            device="cpu",
            frequency_presence_penalties=self._extra_penalties,
        )

    # ---- prompt handling (identical to HFGenerator) ------------------------

    def _prompt_budget(self) -> int:
        return self.max_position_embeddings - self.settings.max_new_tokens - 2

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        budget = self._prompt_budget()
        if budget <= 0:
            raise HFServeError(
                "max_new_tokens leaves no room for an HF prompt: "
                f"context={self.max_position_embeddings}, max_new_tokens={self.settings.max_new_tokens}"
            )
        if len(tokens) > budget:
            tokens = tokens[-budget:]
        return torch.tensor([[self.bos_id, *tokens, self.sep_id]], dtype=torch.long)

    def conversation_prompt(
        self, history, current_user: str, *, max_turns: int, max_tokens: int, max_chars: int, overflow_keep: float = 1.0
    ) -> str:
        from .conversation import conversation_prompt_for_token_budget

        return conversation_prompt_for_token_budget(
            history,
            current_user,
            max_turns=max_turns,
            max_chars=max_chars,
            max_tokens=min(max(1, self._prompt_budget()), max(1, int(max_tokens))),
            token_count=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False).ids),
            overflow_keep=overflow_keep,
        )

    # ---- generation ----------------------------------------------------------

    def _sampling(self) -> SamplingConfig:
        s = self.settings
        return SamplingConfig(
            temperature=float(s.temperature),
            top_k=int(s.top_k),
            top_p=float(s.top_p),
            repetition_penalty=float(s.repetition_penalty),
            no_repeat_ngram_size=int(s.no_repeat_ngram_size or 0),
            frequency_penalty=float(s.frequency_penalty) if self._extra_penalties else 0.0,
            presence_penalty=float(s.presence_penalty) if self._extra_penalties else 0.0,
        )

    def _generate(self, prompt: str, *, max_new_tokens: int, best_of: int, use_prefix_cache: bool = True):
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            ids = self._encode_prompt(prompt)[0].tolist()
            result = generate(
                self.model,
                ids,
                n=best_of,
                max_new=max_new_tokens,
                cfg=self._sampling(),
                pad_id=self.pad_id,
                eos_id=self.eos_id,
                prefix_cache=self.prefix_cache if use_prefix_cache else None,
                started=started,
            )
        keep = [tok for tok in result.tokens[result.best] if tok not in (self.pad_id, self.eos_id)]
        return self.tokenizer.decode(keep, skip_special_tokens=True).strip(), result

    def __call__(self, prompt: str) -> Generation:
        s = self.settings
        started = time.perf_counter()
        text, _result = self._generate(prompt, max_new_tokens=s.max_new_tokens, best_of=s.best_of)
        return Generation(
            text=text,
            step=self.step,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            repetition_penalty=s.repetition_penalty,
            frequency_penalty=s.frequency_penalty if self._extra_penalties else 0.0,
            presence_penalty=s.presence_penalty if self._extra_penalties else 0.0,
            no_repeat_ngram_size=s.no_repeat_ngram_size,
            max_new_tokens=s.max_new_tokens,
            ms=(time.perf_counter() - started) * 1000,
        )

    def benchmark_sample(self, prompt: str, *, max_new_tokens: int, best_of: int) -> HFGenerationStats:
        """Bounded sample for `/bench`. Bypasses the prefix cache so repeated
        benchmark runs measure a cold prefill, comparable with transformers."""
        _text, result = self._generate(
            prompt, max_new_tokens=max_new_tokens, best_of=best_of, use_prefix_cache=False
        )
        return result.stats

    def benchmark_metadata(self) -> dict[str, object]:
        opts = [
            "lean runtime (no transformers)",
            "int8 weights x bf16 activations for decode" if self.precision == "int8" else "fp32 weights",
        ]
        if self.precision == "int8":
            opts.append("exact fp32 candidate logits")
            opts.append("fp32 prefill" if self.model.prefill_fp32 else "transient-dequant prefill")
        opts += ["static KV cache", "row compaction"]
        if self.prefix_cache.enabled:
            opts.append(f"prefix KV cache ({self.prefix_cache.max_bytes // (1024 * 1024)} MB)")
        opts.append("frequency/presence penalties on" if self._extra_penalties else "frequency/presence penalties off")
        return {
            "model": self.model_id,
            "params": self.param_count,
            "dtype": "int8/bf16" if self.precision == "int8" else "float32",
            "backend": "hf-lean",
            "runtime": "lean",
            "optimizations": tuple(opts),
        }


__all__ = [
    "KVCache",
    "LeanConfig",
    "LeanGenerator",
    "LeanModel",
    "PrefixKVCache",
    "SamplingConfig",
    "generate",
    "int8_kernel_available",
    "ngram_bans",
    "penalize",
    "top_p_mask",
    "warp",
]
