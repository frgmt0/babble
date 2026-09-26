"""Lean pure-PyTorch runtime for booper's Mixtral (no transformers at runtime).

Replaces ``MixtralForCausalLM`` + ``generate()`` with plain tensors and a
hand-written decode loop:

* weights pulled straight out of ``model-int8.safetensors`` (dequantized the
  same way ``babble.hfserve._load_int8`` does, so fp32 mode is the same model);
* fused QKV projection, fused expert gate/up (w1|w3) projection;
* RoPE from precomputed cos/sin tables;
* preallocated static KV cache ``[B, H, Tmax, D]`` per layer + SDPA;
* top-1 MoE dispatch that only touches experts that are actually routed to;
* one batch-1 prefill, then the KV cache is expanded to the best-of batch;
* a vectorized sampler whose semantics match HF's processors in HF's order
  (repetition penalty -> no-repeat-ngram -> temperature -> top-k -> top-p).

Weight modes (``LeanModel(mode=...)``):

* ``fp32``  -- int8 * scale expanded to fp32 (what serves live). Lossless.
* ``dyn``   -- on-disk int8 per-channel weights kept as int8, fbgemm dynamic
  (per-token activation quant) linear. Lossy: activations are quantized.
* ``w8``    -- ``aten._weight_int8pack_mm`` (weight-only int8). Lossless-ish
  but on AVX2 the kernel is scalar-ish and ~6x slower than fp32; kept only so
  the measurement is reproducible.

The embedding / tied lm_head can independently be ``fp32`` or ``dyn``
(``head=...``), because it is the single biggest matrix (16384 x 896).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "artifacts" / "hf-booper-multiturn-v1"


# --------------------------------------------------------------------------
# Linear backends
# --------------------------------------------------------------------------


class Fp32Linear:
    __slots__ = ("w",)

    def __init__(self, q: torch.Tensor, scale: torch.Tensor):
        # Identical dequant to hfserve._unpack: int8 -> fp32, times fp32(scale).
        self.w = (q.to(torch.float32) * scale.to(torch.float32)).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w)


class Bf16ActLinear(Fp32Linear):
    """Numerics emulation of an int8-weight x bf16-activation kernel (fp32 accum)."""

    __slots__ = ()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(torch.bfloat16).to(torch.float32), self.w)


class DynInt8Linear:
    """fbgemm dynamic-quant linear on the exact on-disk int8 values."""

    __slots__ = ("packed", "out")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor):
        import warnings

        s = scale.to(torch.float64).reshape(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qw = torch._make_per_channel_quantized_tensor(
                q.contiguous(), s, torch.zeros(q.shape[0], dtype=torch.long), 0
            )
            self.packed = torch.ops.quantized.linear_prepack(qw, None)
        self.out = q.shape[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = torch.ops.quantized.linear_dynamic(x.reshape(-1, shape[-1]), self.packed, DYN_REDUCE_RANGE)
        return y.reshape(*shape[:-1], self.out)


class W8PackLinear:
    __slots__ = ("q", "s", "out")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor):
        self.q = q.contiguous()
        self.s = scale.to(torch.float32).reshape(-1).contiguous()
        self.out = q.shape[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = torch.ops.aten._weight_int8pack_mm(x.reshape(-1, shape[-1]).contiguous(), self.q, self.s)
        return y.reshape(*shape[:-1], self.out)


class W8BfLinear:
    """Weight-only int8 via ``aten._weight_int8pack_mm`` with bf16 activations.

    The on-disk int8 weights and bf16 scales are used exactly; activations are
    rounded to bf16 going in and the product comes back as bf16. On AVX2 the
    bf16-activation variant of this kernel is vectorized (the fp32-activation
    one is not), which is what makes it ~2x faster than fp32 GEMV here.
    For large M (prefill), an fp32 dequantized copy is optionally used instead
    (``W8BF_PREFILL_FP32``), trading memory for prefill speed.
    """

    __slots__ = ("q", "s", "out", "wf")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor):
        self.q = q.contiguous()
        self.s = scale.to(torch.bfloat16).reshape(-1).contiguous()
        self.out = q.shape[0]
        self.wf = (
            (q.to(torch.float32) * scale.to(torch.float32)).contiguous() if W8BF_PREFILL_FP32 else None
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.shape[0] > W8BF_MAX_M:
            if self.wf is not None:
                return F.linear(x, self.wf)
            if W8BF_PREFILL_DEQUANT:
                # Transient fp32 weight: one extra pass over the matrix, but the
                # large-M GEMM then runs on MKL instead of the int8 kernel.
                return F.linear(x, self.q.to(torch.float32) * self.s.to(torch.float32)[:, None])
        y = torch.ops.aten._weight_int8pack_mm(x2.to(torch.bfloat16), self.q, self.s)
        return y.to(torch.float32).reshape(*shape[:-1], self.out)


class RescoreHead:
    """Tied lm_head: int8 (bf16-act) scores for the whole vocab, then the top
    ``k`` candidates are recomputed exactly in fp32 from the fp32 embedding
    rows (which are resident anyway for the input lookup).

    Every token that can survive top-k(40)/top-p therefore carries its exact
    fp32 logit; the rest of the vocab carries a bf16-accurate approximation,
    which only feeds the softmax normaliser for NLL (and is ~0 mass there).
    """

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, embed_fp32: torch.Tensor, k: int = 64):
        self.approx = W8BfLinear.__new__(W8BfLinear)
        self.approx.q = q.contiguous()
        self.approx.s = scale.to(torch.bfloat16).reshape(-1).contiguous()
        self.approx.out = q.shape[0]
        self.approx.wf = None
        self.embed = embed_fp32
        self.k = k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.shape[0] > W8BF_MAX_M:
            # Full-sequence logits (parity check / long prefill): chunk it.
            return torch.cat([self(c) for c in x2.split(W8BF_MAX_M)], 0).reshape(*shape[:-1], -1)
        logits = self.approx(x2)
        idx = logits.topk(self.k, dim=-1).indices  # [M, k]
        rows = self.embed.index_select(0, idx.reshape(-1)).view(*idx.shape, -1)  # [M,k,H]
        exact = torch.bmm(rows, x2.unsqueeze(-1)).squeeze(-1)
        logits.scatter_(1, idx, exact)
        return logits.reshape(*shape[:-1], -1)


W8BF_PREFILL_FP32 = False
W8BF_PREFILL_DEQUANT = False
W8BF_MAX_M = 8

DYN_REDUCE_RANGE = True

BACKENDS = {"w8bf": W8BfLinear, "fp32": Fp32Linear, "bf16act": Bf16ActLinear, "dyn": DynInt8Linear, "w8": W8PackLinear}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass
class Layer:
    ln1: torch.Tensor
    qkv: object
    o: object
    ln2: torch.Tensor
    router: torch.Tensor  # [E, hidden] fp32
    w13: list  # per expert, fused [2*I, hidden]
    w2: list  # per expert [hidden, I]


class LeanModel:
    def __init__(self, model_dir: Path = MODEL_DIR, mode: str = "fp32", head: str | None = None):
        import json

        from safetensors.torch import load_file

        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        assert self.n_kv == self.n_heads, "MHA only"
        self.head_dim = cfg.get("head_dim") or self.hidden // self.n_heads
        self.n_layers = cfg["num_hidden_layers"]
        self.n_experts = cfg["num_local_experts"]
        assert cfg["num_experts_per_tok"] == 1, "top-1 routing only"
        self.inter = cfg["intermediate_size"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]
        self.max_pos = cfg["max_position_embeddings"]
        theta = cfg["rope_parameters"]["rope_theta"]
        self.mode = mode
        self.head_mode = head or ("fp32" if mode == "w8" else mode)
        Lin = BACKENDS[mode]

        p = load_file(str(Path(model_dir) / "model-int8.safetensors"))

        def q(name):
            return p[name], p[name + ".scale"]

        def fp(name):
            return p[name].to(torch.float32).contiguous()

        def deq(name):
            w, s = q(name)
            return w.to(torch.float32) * s.to(torch.float32)

        self.layers: list[Layer] = []
        for i in range(self.n_layers):
            a = f"model.layers.{i}.self_attn."
            m = f"model.layers.{i}.block_sparse_moe."
            qkv_q = torch.cat([p[a + n + "_proj.weight"] for n in "qkv"], 0)
            qkv_s = torch.cat([p[a + n + "_proj.weight.scale"] for n in "qkv"], 0)
            w13, w2 = [], []
            for e in range(self.n_experts):
                ex = f"{m}experts.{e}."
                w13.append(
                    Lin(
                        torch.cat([p[ex + "w1.weight"], p[ex + "w3.weight"]], 0),
                        torch.cat([p[ex + "w1.weight.scale"], p[ex + "w3.weight.scale"]], 0),
                    )
                )
                w2.append(Lin(*q(ex + "w2.weight")))
            self.layers.append(
                Layer(
                    ln1=fp(f"model.layers.{i}.input_layernorm.weight"),
                    qkv=Lin(qkv_q, qkv_s),
                    o=Lin(*q(a + "o_proj.weight")),
                    ln2=fp(f"model.layers.{i}.post_attention_layernorm.weight"),
                    # Router stays fp32 in every mode: it's tiny and a flipped
                    # expert choice would be a much bigger error than a matmul.
                    router=deq(m + "gate.weight").contiguous(),
                    w13=w13,
                    w2=w2,
                )
            )
        self.norm = fp("model.norm.weight")
        self.embed = deq("model.embed_tokens.weight").contiguous()  # tied lm_head
        self.lm_head = Fp32Linear.__new__(Fp32Linear)
        if self.head_mode == "fp32":
            self.lm_head.w = self.embed
        elif self.head_mode == "rescore":
            self.lm_head = RescoreHead(*q("model.embed_tokens.weight"), self.embed)
        else:
            self.lm_head = BACKENDS[self.head_mode](*q("model.embed_tokens.weight"))
        del p

        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        pos = torch.arange(self.max_pos, dtype=torch.float)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().contiguous()  # [max_pos, D]
        self.sin = emb.sin().contiguous()
        self.scale = self.head_dim**-0.5

    # ---- cache -----------------------------------------------------------

    def new_cache(self, batch: int, tmax: int) -> "KVCache":
        shape = (batch, self.n_heads, tmax, self.head_dim)
        return KVCache(
            k=[torch.zeros(shape) for _ in range(self.n_layers)],
            v=[torch.zeros(shape) for _ in range(self.n_layers)],
            tmax=tmax,
        )

    # ---- blocks ----------------------------------------------------------

    def _rms(self, x, w):
        var = x.pow(2).mean(-1, keepdim=True)
        return w * (x * torch.rsqrt(var + self.eps))

    def _rope(self, x, cos, sin):
        # x: [B, H, T, D]; cos/sin: [T, D]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def _moe(self, L: Layer, h: torch.Tensor) -> torch.Tensor:
        # h: [N, hidden]. Top-1 with renormalised weight == exactly 1.0.
        return self._moe_routed(L, h, F.linear(h, L.router).argmax(-1))

    def _moe_routed(self, L: Layer, h: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        if n == 1:
            e = int(choice)
            g, u = L.w13[e](h).chunk(2, dim=-1)
            return L.w2[e](F.silu(g) * u)
        ch = choice.tolist()
        experts = sorted(set(ch))
        if len(experts) == 1:
            e = experts[0]
            g, u = L.w13[e](h).chunk(2, dim=-1)
            return L.w2[e](F.silu(g) * u)
        out = torch.empty_like(h)
        for e in experts:
            idx = torch.tensor([i for i, c in enumerate(ch) if c == e], dtype=torch.long)
            x = h.index_select(0, idx)
            g, u = L.w13[e](x).chunk(2, dim=-1)
            out.index_copy_(0, idx, L.w2[e](F.silu(g) * u))
        return out

    def forward(
        self,
        ids: torch.Tensor,
        cache: "KVCache",
        start: int,
        *,
        all_logits: bool = False,
    ) -> torch.Tensor:
        """ids [B, T] at positions start..start+T-1 -> logits ([B,V] or [B,T,V])."""
        B, T = ids.shape
        H, D = self.n_heads, self.head_dim
        end = start + T
        x = self.embed[ids]  # [B, T, hidden]
        cos = self.cos[start:end]
        sin = self.sin[start:end]
        if T == 1 or start == 0:
            mask, causal = None, T > 1
        else:
            # Prefix-cached prefill: query i (abs pos start+i) sees keys <= start+i.
            kpos = torch.arange(end)
            qpos = torch.arange(start, end)
            mask, causal = kpos[None, :] <= qpos[:, None], False
        for li, L in enumerate(self.layers):
            h = self._rms(x, L.ln1)
            qkv = L.qkv(h).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)  # [3,B,H,T,D]
            q = self._rope(qkv[0], cos, sin)
            k = self._rope(qkv[1], cos, sin)
            kc, vc = cache.k[li], cache.v[li]
            kc[:, :, start:end] = k
            vc[:, :, start:end] = qkv[2]
            a = F.scaled_dot_product_attention(
                q, kc[:, :, :end], vc[:, :, :end], attn_mask=mask, is_causal=causal, scale=self.scale
            )
            x = x + L.o(a.transpose(1, 2).reshape(B, T, H * D))
            h = self._rms(x, L.ln2)
            x = x + self._moe(L, h.reshape(B * T, -1)).view(B, T, -1)
        if not all_logits:
            x = x[:, -1]
        x = self._rms(x, self.norm)
        return self.lm_head(x)


@dataclass
class KVCache:
    k: list
    v: list
    tmax: int

    def expand(self, batch: int, upto: int, tmax: int | None = None) -> "KVCache":
        """Batch-1 cache -> batch-N cache (copies only the first `upto` positions)."""
        tmax = tmax or self.tmax
        out_k, out_v = [], []
        for k, v in zip(self.k, self.v):
            _, H, _, D = k.shape
            # zeros, not empty: the compiled step attends over the whole
            # static cache with a mask, and masked garbage could be NaN.
            nk = torch.zeros(batch, H, tmax, D)
            nv = torch.zeros(batch, H, tmax, D)
            nk[:, :, :upto] = k[:1, :, :upto]
            nv[:, :, :upto] = v[:1, :, :upto]
            out_k.append(nk)
            out_v.append(nv)
        return KVCache(out_k, out_v, tmax)


# --------------------------------------------------------------------------
# Sampler (HF-equivalent processors, HF order)
# --------------------------------------------------------------------------


@dataclass
class SampleCfg:
    temperature: float = 0.5
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4


def process_logits(scores: torch.Tensor, hist: torch.Tensor, cfg: SampleCfg):
    """HF-order processors on [B, V] scores given history ids [B, L].

    Returns (candidate_scores [B,K], candidate_ids [B,K]) where every token
    outside the candidates has -inf after the warpers; candidate scores may
    contain -inf (removed by top-p). Equivalent to HF's full-vocab
    TopK -> TopP masking except for exact fp ties at the k-th value.
    """
    if cfg.repetition_penalty != 1.0:
        s = scores.gather(1, hist)
        s = torch.where(s < 0, s * cfg.repetition_penalty, s / cfg.repetition_penalty)
        scores = scores.scatter(1, hist, s)
    n = cfg.no_repeat_ngram_size
    cur = hist.shape[1]
    if n and cur >= n:
        prefix = hist[:, cur + 1 - n :]
        win = hist.unfold(1, n, 1)
        match = (win[..., :-1] == prefix.unsqueeze(1)).all(-1)
        if bool(match.any()):
            V = scores.shape[1]
            ban = scores.new_zeros((scores.shape[0], V + 1), dtype=torch.bool)
            ban.scatter_(1, torch.where(match, win[..., -1], V), True)
            scores = scores.masked_fill(ban[:, :V], -math.inf)
    if cfg.temperature != 1.0:
        scores = scores / cfg.temperature
    k = min(cfg.top_k or scores.shape[1], scores.shape[1])
    vals, idx = torch.topk(scores, k, dim=-1)  # sorted descending
    if cfg.top_p < 1.0:
        asc = vals.flip(-1)
        cum = asc.softmax(-1).cumsum(-1)
        remove = cum <= (1 - cfg.top_p)
        remove[..., -1:] = False
        vals = vals.masked_fill(remove.flip(-1), -math.inf)
    return vals, idx


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class GenResult:
    tokens: torch.Tensor  # [B, n] (pad after eos)
    mean_logprob: torch.Tensor  # [B]
    counts: list
    best: int
    ttft_s: float
    total_s: float
    prefill_s: float = 0.0
    extra: dict = field(default_factory=dict)


class LeanGenerator:
    def __init__(self, model: LeanModel, *, pad_id=16380, eos_id=16383, step_fn=None):
        self.m = model
        self.pad_id = pad_id
        self.eos_id = eos_id
        # Optional compiled single-step decode replacement.
        self.step_fn = step_fn

    @torch.inference_mode()
    def prefill(self, ids: list[int], tmax: int, cache: KVCache | None = None, cached_len: int = 0):
        """Batch-1 prefill; with `cache`/`cached_len`, only ids[cached_len:] are run."""
        if cache is None or cache.tmax < tmax:
            cache, cached_len = self.m.new_cache(1, tmax), 0
        x = torch.tensor([ids[cached_len:]], dtype=torch.long)
        logits = self.m.forward(x, cache, cached_len)
        return logits, cache

    @torch.inference_mode()
    def generate(
        self,
        ids: list[int],
        *,
        n: int = 4,
        max_new: int = 64,
        cfg: SampleCfg = SampleCfg(),
        greedy: bool = False,
        ignore_eos: bool = False,
        prefix_cache: tuple[KVCache, int] | None = None,
    ) -> GenResult:
        t0 = time.perf_counter()
        P = len(ids)
        tmax = P + max_new
        bucket = getattr(self.step_fn, "tmax_bucket", 0)
        if bucket:
            tmax = -(-tmax // bucket) * bucket
        cache1, cached = (prefix_cache if prefix_cache else (None, 0))
        logits, cache1 = self.prefill(ids, tmax, cache1, cached)
        t_pre = time.perf_counter()
        cache = cache1.expand(n, P, tmax) if n > 1 else cache1
        logits = logits.expand(n, -1)
        hist = torch.empty(n, tmax, dtype=torch.long)
        hist[:, :P] = torch.tensor(ids)
        out = torch.full((n, max_new), self.pad_id, dtype=torch.long)
        logp_sum = torch.zeros(n)
        counts = torch.zeros(n, dtype=torch.long)
        live = torch.ones(n, dtype=torch.bool)
        ttft = None
        step = self.step_fn
        for t in range(max_new):
            cur = P + t
            if greedy:
                nxt = logits.argmax(-1)
                lp = torch.zeros(n)
            else:
                vals, idx = process_logits(logits, hist[:, :cur], cfg)
                logprobs = vals.log_softmax(-1)
                j = torch.multinomial(logprobs.exp(), 1)
                nxt = idx.gather(1, j).squeeze(1)
                lp = logprobs.gather(1, j).squeeze(1)
            if not ignore_eos:
                nxt = torch.where(live, nxt, torch.full_like(nxt, self.pad_id))
                logp_sum += lp * live
                counts += live
                live &= nxt != self.eos_id
            else:
                logp_sum += lp
                counts += 1
            out[:, t] = nxt
            hist[:, cur] = nxt
            if ttft is None:
                ttft = time.perf_counter() - t0
            if t == max_new - 1 or (not ignore_eos and not bool(live.any())):
                break
            if step is not None:
                logits = step(nxt, cur, cache)
            else:
                logits = self.m.forward(nxt[:, None], cache, cur)
        mean = (logp_sum / counts.clamp_min(1)).masked_fill(counts == 0, -math.inf)
        return GenResult(
            tokens=out,
            mean_logprob=mean,
            counts=counts.tolist(),
            best=int(mean.argmax()),
            ttft_s=ttft,
            total_s=time.perf_counter() - t0,
            prefill_s=t_pre - t0,
        )


# --------------------------------------------------------------------------
# reference.py check entry points: fn(ids) -> [T, vocab]
# --------------------------------------------------------------------------

_MODELS: dict = {}


def _full_logits(mode: str, head: str | None = None):
    key = (mode, head)
    if key not in _MODELS:
        torch.set_num_threads(4)
        _MODELS[key] = LeanModel(mode=mode, head=head)
    m = _MODELS[key]

    def fn(ids):
        with torch.inference_mode():
            cache = m.new_cache(1, len(ids))
            return m.forward(torch.tensor([ids]), cache, 0, all_logits=True)[0]

    return fn


def _decode_logits(mode: str, head: str | None = None, prefix: int = 0):
    """Logits produced token-by-token through the KV cache (exercises decode path).

    With prefix>0, the first `prefix` tokens are prefilled in one shot then the
    rest are fed via a second chunked prefill (exercises the prefix-reuse mask),
    then decoding one token at a time is not needed.
    """
    full = _full_logits(mode, head)  # ensures model loaded
    m = _MODELS[(mode, head)]

    def fn(ids):
        with torch.inference_mode():
            cache = m.new_cache(1, len(ids))
            rows = []
            if prefix:
                cut = max(1, len(ids) - 30)
                rows.append(m.forward(torch.tensor([ids[:cut]]), cache, 0, all_logits=True)[0])
                rows.append(m.forward(torch.tensor([ids[cut:]]), cache, cut, all_logits=True)[0])
            else:
                for i, t in enumerate(ids):
                    rows.append(m.forward(torch.tensor([[t]]), cache, i))
            return torch.cat(rows, 0)

    del full
    return fn


def fp32(ids):
    return _full_logits("fp32")(ids)


def fp32_decode(ids):
    return _decode_logits("fp32")(ids)


def fp32_prefix(ids):
    return _decode_logits("fp32", prefix=1)(ids)


def dyn(ids):
    return _full_logits("dyn")(ids)


def dyn_decode(ids):
    return _decode_logits("dyn")(ids)


def dyn_body_fp32_head(ids):
    return _full_logits("dyn", "fp32")(ids)


def w8(ids):
    return _full_logits("w8", "fp32")(ids)


def bf16act(ids):
    return _full_logits("bf16act")(ids)


def bf16act_body(ids):
    return _full_logits("bf16act", "fp32")(ids)


def w8bf(ids):
    return _full_logits("w8bf")(ids)


def w8bf_decode(ids):
    return _decode_logits("w8bf")(ids)


def w8bf_body(ids):
    return _full_logits("w8bf", "fp32")(ids)


def w8bf_body_decode(ids):
    return _decode_logits("w8bf", "fp32")(ids)


def w8bf_rescore(ids):
    return _full_logits("w8bf", "rescore")(ids)


def w8bf_rescore_decode(ids):
    return _decode_logits("w8bf", "rescore")(ids)


# Recommended serving config: int8 (bf16-act) decode, exact-rescored int8 head,
# fp32 copy for prefill-sized matmuls (M > W8BF_MAX_M).
def recommended(ids):
    global W8BF_PREFILL_FP32
    W8BF_PREFILL_FP32 = True
    return _full_logits("w8bf", "rescore")(ids)


def recommended_decode(ids):
    global W8BF_PREFILL_FP32
    W8BF_PREFILL_FP32 = True
    return _decode_logits("w8bf", "rescore")(ids)


def recommended_prefix(ids):
    global W8BF_PREFILL_FP32
    W8BF_PREFILL_FP32 = True
    return _decode_logits("w8bf", "rescore", prefix=1)(ids)
