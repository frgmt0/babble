"""Native C++ serving runtime for the `hf` backend (`BABBLE_HF_RUNTIME=native`).

Same snapshot, same prompts, same sampling semantics as the lean and
transformers runtimes; the forward pass, decode loop and sampler run in
``babble/_native/engine.cpp`` (int8 weights dequantized in AVX2 registers,
fp32 activations and accumulation -- lossless against the fp32 reference, see
docs/reports/NATIVE_RUNTIME_2026-09-26.md), one parallel region per reply.

`NativeGenerator` subclasses `leanserve.LeanGenerator` for everything that is
not the model: tokenizer and special ids, prompt encoding and budget,
conversation formatting, the Settings -> `SamplingConfig` mapping (including
the `BABBLE_HF_FREQUENCY_PENALTIES` gate), `__call__` and the benchmark hook.
The prefix KV cache is lean's `PrefixKVCache` holding engine snapshots.

What the engine does, in hfserve's order: repetition penalty ->
no-repeat-ngram -> frequency/presence (when enabled) -> temperature -> top-k
(ties kept, like transformers) -> top-p -> multinomial draw; best-of picks the
highest mean post-warp log-probability (``<eos>`` included). A candidate that
emits ``<eos>`` leaves the batch (row compaction); generation stops when all
have. The engine draws from its own RNG, seeded per call from torch's global
generator, so `torch.manual_seed` still makes a run reproducible.

The K/V cache (and so every prefix snapshot) is 16-bit by default:
``BABBLE_NATIVE_KV=q16`` stores K as int16 with one fp32 scale per (position,
head) and V as fp16, which halves the bandwidth of long-context attention and
the size of a snapshot while staying within ~4e-3 logits of fp32 at 2048
tokens. ``fp32`` is the original full-precision cache; ``fp16`` (K and V IEEE
half) is kept for comparison only -- its K rounding moves some long-context
logits by ~1 (see docs/reports/NATIVE_KV_2026-09-30.md).

Speculative decoding (opt-in, ``BABBLE_NATIVE_SPEC=1``): `NgramDrafter`
proposes up to ``BABBLE_NATIVE_SPEC_K`` tokens per stream from a static n-gram
table (``BABBLE_NATIVE_SPEC_TABLE``, default ``<model dir>/spec-ngram.pt``;
build it with ``bench/extreme/spec_ngram.py``); the engine verifies all
streams' drafts in one forward and accepts them by exact speculative sampling
against the warped distribution, warping each position with the history it
would have had (so repetition / no-repeat-ngram stay exact). The output
distribution is unchanged and greedy output is identical; only the RNG
consumption differs, so a fixed seed gives different samples with spec on.

Unusable here (CPU without AVX2/FMA/F16C, no compiler, failed build, a
snapshot shape the kernels do not implement) raises `NativeUnavailable`, which
`hfserve.make_generator` logs and answers with the lean runtime.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from . import _native
from ._native import NativeUnavailable, SampleParams
from .config import Settings
from .cpu_runtime import configure_cpu, force_cpu_device
from .hfserve import HFGenerationStats, HFServeError
from .leanserve import LeanConfig, LeanGenerator, PrefixKVCache, SamplingConfig
from .logs import EventLog, NullLog


# BABBLE_NATIVE_W4 scopes -> eng_set_matrix kinds (0-2 qkv, 3 o, 4-6 experts, 7 tied head)
W4_SCOPES = {
    "all": frozenset(range(8)),
    "noh": frozenset(range(7)),
    "exp": frozenset((4, 5, 6)),
    "attn": frozenset((0, 1, 2, 3)),
}


def parse_w4(spec: str) -> tuple[frozenset, int] | None:
    """`BABBLE_NATIVE_W4` = "" / "0" / "off" (default) or "<scope>[:<group>]".

    Opt-in QUALITY TRADE: decode GEMMs (<= 4 rows) read an int4 group-wise copy
    of the scoped matrices (round-to-nearest from the int8 snapshot, symmetric,
    fp16 group scales); prefill keeps the int8 panels. Scopes: all | noh (all
    but the tied lm_head) | exp (experts) | attn. Group defaults to 64.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none", "false"):
        return None
    scope, _, group = spec.partition(":")
    if scope not in W4_SCOPES:
        raise NativeUnavailable(f"BABBLE_NATIVE_W4={spec!r}: scope must be one of {sorted(W4_SCOPES)}")
    try:
        g = int(group or 64)
    except ValueError as exc:
        raise NativeUnavailable(f"BABBLE_NATIVE_W4={spec!r}: bad group") from exc
    if g < 4 or g % 4:
        raise NativeUnavailable(f"BABBLE_NATIVE_W4={spec!r}: group must be a multiple of 4")
    return W4_SCOPES[scope], g


def parse_head2(spec: str) -> tuple[int, float, int] | None:
    """`BABBLE_NATIVE_HEAD2` = "" / "0" / "off" (default) or "<N>[:<delta>[:<group>]]".

    Opt-in two-stage lm_head for sampled rows: an int4 (group-wise, default
    g64) screen of the tied head is penalized like the sampler does
    (repetition, frequency/presence, no-repeat-ngram), and the N..2N best
    tokens get exact int8 logits. If the k-th best exact (penalized) candidate
    is not at least `delta` (default 0.5 logits) above the best screen score
    left outside, the row falls back to the full exact head. Reads about half
    the head bytes per step. Sampling is exact whenever no outside token's
    screen error exceeds the margin: on 120 real prompts x best-of-4 every
    token matched the exact head, with 0 fallbacks (track w4 NOTES). Greedy
    uses k = 1; top_k > N, or top-k off, always takes the exact head.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none", "false"):
        return None
    parts = spec.split(":")
    try:
        n = int(parts[0])
        delta = float(parts[1]) if len(parts) > 1 and parts[1] else 0.5
        group = int(parts[2]) if len(parts) > 2 and parts[2] else 64
    except ValueError as exc:
        raise NativeUnavailable(f"BABBLE_NATIVE_HEAD2={spec!r}: expected N[:delta[:group]]") from exc
    if n < 1 or group < 4 or group % 4 or not math.isfinite(delta):
        raise NativeUnavailable(f"BABBLE_NATIVE_HEAD2={spec!r}: bad value")
    return n, delta, group


def quantize_q4(q: torch.Tensor, scale: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 rows (q * per-row scale) -> symmetric int4 [-8, 7] + fp32 group scales (fp16-exact)."""
    w = q.to(torch.float32) * scale.reshape(-1, 1)
    rows, cols = w.shape
    wg = w.reshape(rows, cols // group, group)
    # fp16-exact scales (what the engine stores); the floor keeps all-zero groups finite
    s = (wg.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-4).half().float()
    q4 = torch.clamp(torch.round(wg / s), -8, 7).to(torch.int8).reshape(rows, cols).contiguous()
    return q4, s.reshape(rows, cols // group).contiguous()


def _ptr(t: torch.Tensor | None) -> int | None:
    if t is None:
        return None
    if not t.is_contiguous():
        raise ValueError("native engine needs contiguous tensors")
    return t.data_ptr()


@dataclass
class NativeOutput:
    tokens: list  # per candidate, as sampled (eos included when it stopped)
    counts: list
    mean_logprob: list
    best: int
    prefill_s: float
    ttft_s: float  # from engine entry
    total_s: float
    last_s: float


KV_TYPES = {"fp32": 0, "fp16": 1, "q16": 2}
DEFAULT_KV = "q16"


def kv_type_from_env() -> str:
    """``BABBLE_NATIVE_KV`` (q16 | fp16 | fp32); unset or empty means the default (q16)."""
    import os

    raw = os.environ.get("BABBLE_NATIVE_KV", "").strip().lower()
    if not raw:
        return DEFAULT_KV
    if raw not in KV_TYPES:
        raise HFServeError(f"BABBLE_NATIVE_KV={raw!r}; expected one of {sorted(KV_TYPES)}")
    return raw


class NativeEngine:
    """One loaded model in the C++ engine. Not thread-safe; serialize calls."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        threads: int = 4,
        kv: str | None = None,
        w4: str | None = None,
        head2: str | None = None,
    ) -> None:
        from safetensors import safe_open

        model_dir = Path(model_dir)
        weights = model_dir / "model-int8.safetensors"
        if not weights.exists():
            raise HFServeError(f"no model-int8.safetensors under {model_dir}")
        raw = json.loads((model_dir / "config.json").read_text())
        try:
            c = LeanConfig.from_dict(raw)
        except HFServeError as exc:
            raise NativeUnavailable(str(exc)) from exc
        if c.n_kv != c.n_heads:
            raise NativeUnavailable(f"native engine has no GQA (num_key_value_heads={c.n_kv} != {c.n_heads})")
        if c.n_heads * c.head_dim != c.hidden:
            raise NativeUnavailable(f"head_dim {c.head_dim} x {c.n_heads} heads != hidden {c.hidden}")
        bad = {k: v for k, v in dict(hidden=c.hidden, head_dim=c.head_dim, inter=c.inter, vocab=c.vocab).items() if v % 16}
        if bad:
            raise NativeUnavailable(f"native kernels need multiples of 16, got {bad}")
        if c.n_experts > 64:
            raise NativeUnavailable(f"num_local_experts={c.n_experts} > 64")
        if raw.get("attention_bias") or raw.get("mlp_bias"):
            raise NativeUnavailable("attention/mlp bias is not implemented")
        self.cfg = c
        self.w4 = parse_w4(os.environ.get("BABBLE_NATIVE_W4", "") if w4 is None else w4)
        if self.w4 is not None and (c.hidden % self.w4[1] or c.inter % self.w4[1]):
            raise NativeUnavailable(f"BABBLE_NATIVE_W4 group {self.w4[1]} does not divide hidden/inter")
        self.head2 = parse_head2(os.environ.get("BABBLE_NATIVE_HEAD2", "") if head2 is None else head2)
        if self.head2 is not None and c.hidden % self.head2[2]:
            raise NativeUnavailable(f"BABBLE_NATIVE_HEAD2 group {self.head2[2]} does not divide hidden")
        self.lib, self.build = _native.load()
        self.threads = max(1, int(threads))

        packed = safe_open(str(weights), framework="pt")
        names = set(packed.keys())
        if not c.tied and "lm_head.weight" in names:
            raise NativeUnavailable("untied lm_head is not implemented")

        def get(name: str) -> torch.Tensor:
            if name not in names:
                raise HFServeError(f"snapshot is missing tensor {name}")
            return packed.get_tensor(name)

        h = self.lib.eng_create(self.threads, c.hidden, c.n_heads, c.n_layers, c.n_experts, c.inter, c.vocab, c.max_pos, c.eps)
        if not h:
            raise NativeUnavailable(f"native engine rejected the geometry {c}")
        self._h = h
        self.kv = kv_type_from_env() if kv is None else str(kv).strip().lower()
        try:
            if self.kv not in KV_TYPES:
                raise HFServeError(f"unknown native KV type {self.kv!r}; expected one of {sorted(KV_TYPES)}")
            if self.lib.eng_set_kv_type(h, KV_TYPES[self.kv]) != KV_TYPES[self.kv]:
                raise NativeUnavailable(f"native engine rejected KV type {self.kv}")
            self._load(get, names)
        except BaseException:
            self.close()
            raise
        del packed
        self.param_count = self._params

    def _load(self, get, names) -> None:
        c = self.cfg
        lib, h = self.lib, self._h
        params = 0

        def mat(kind: int, layer: int, expert: int, name: str) -> None:
            nonlocal params
            q = get(name)
            if q.dtype != torch.int8:
                raise NativeUnavailable(f"{name} is {q.dtype}; the native engine needs int8 matrices")
            q = q.contiguous()
            s = get(name + ".scale").to(torch.float32).reshape(-1).contiguous()
            if s.numel() != q.shape[0]:
                raise NativeUnavailable(f"{name}.scale is not per-output-channel")
            rc = lib.eng_set_matrix(h, kind, layer, expert, _ptr(q), _ptr(s), q.shape[0], q.shape[1])
            if rc != 0:
                raise NativeUnavailable(f"{name} {tuple(q.shape)} does not fit the engine geometry (rc={rc})")
            params += q.numel()
            if self.w4 is not None and kind in self.w4[0]:
                q4, s4 = quantize_q4(q, s, self.w4[1])
                rc = lib.eng_set_matrix_q4(h, kind, layer, expert, _ptr(q4), _ptr(s4), q.shape[0], q.shape[1], self.w4[1])
                if rc != 0:
                    raise NativeUnavailable(f"{name}: int4 copy rejected (rc={rc})")

        def dense(name: str) -> torch.Tensor:
            v = get(name)
            if v.dtype == torch.int8:  # exactly hfserve._unpack
                v = v.to(torch.float32) * get(name + ".scale").to(torch.float32)
            return v.to(torch.float32).contiguous()

        def vec(kind: int, layer: int, v: torch.Tensor, n: int) -> None:
            nonlocal params
            if v.numel() != n:
                raise NativeUnavailable(f"vector kind {kind} layer {layer}: {v.numel()} != {n}")
            if lib.eng_set_vector(h, kind, layer, _ptr(v)) != 0:
                raise NativeUnavailable("eng_set_vector failed")
            params += v.numel()

        # tied lm_head: the embedding panel serves both
        mat(7, -1, -1, "model.embed_tokens.weight")
        if self.head2 is not None:
            n, delta, group = self.head2
            emb = get("model.embed_tokens.weight").contiguous()
            q4, s4 = quantize_q4(emb, get("model.embed_tokens.weight.scale").to(torch.float32).reshape(-1), group)
            rc = lib.eng_set_matrix_q4(h, 8, -1, -1, _ptr(q4), _ptr(s4), emb.shape[0], emb.shape[1], group)
            if rc != 0 or lib.eng_set_head2(h, n, delta) != 0:
                raise NativeUnavailable(f"two-stage head rejected (rc={rc})")
        vec(3, 0, dense("model.norm.weight"), c.hidden)
        for layer in range(c.n_layers):
            p = f"model.layers.{layer}"
            for kind, n in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
                mat(kind, layer, -1, f"{p}.self_attn.{n}.weight")
            vec(0, layer, dense(f"{p}.input_layernorm.weight"), c.hidden)
            vec(1, layer, dense(f"{p}.post_attention_layernorm.weight"), c.hidden)
            vec(2, layer, dense(f"{p}.block_sparse_moe.gate.weight"), c.n_experts * c.hidden)
            for e in range(c.n_experts):
                ep = f"{p}.block_sparse_moe.experts.{e}"
                mat(4, layer, e, f"{ep}.w1.weight")
                mat(6, layer, e, f"{ep}.w3.weight")
                mat(5, layer, e, f"{ep}.w2.weight")
        # RoPE tables with the torch ops HF's rotary embedding uses
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.int64).float() / c.head_dim))
        freqs = torch.outer(torch.arange(c.max_pos, dtype=torch.float32), inv_freq)
        cos, sin = freqs.cos().contiguous(), freqs.sin().contiguous()
        lib.eng_set_rope(h, _ptr(cos), _ptr(sin))
        self._params = params

    def head2_stats(self) -> dict[str, int]:
        """Cumulative two-stage head counters: rows fixed, full-head fallbacks, candidates rescored."""
        out = (ctypes.c_longlong * 3)()
        self.lib.eng_head2_stats(self._h, out)
        return {"rows": out[0], "fallbacks": out[1], "candidates": out[2]}

    def close(self) -> None:
        h, self._h = getattr(self, "_h", None), None
        if h:
            self.lib.eng_destroy(h)

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass

    # ---- prefix snapshots ------------------------------------------------------

    def kv_bytes(self, positions: int) -> int:
        """Size of a prefix snapshot of ``positions`` tokens (fp16 KV: half of fp32)."""
        return int(self.lib.eng_kv_bytes(self._h, int(positions)))

    def new_snapshot(self, positions: int) -> torch.Tensor:
        """An uninitialized snapshot buffer (raw bytes; layout is engine-internal)."""
        return torch.empty(self.kv_bytes(positions), dtype=torch.uint8)

    # ---- correctness paths -------------------------------------------------

    def forward(
        self,
        ids: list[int],
        *,
        start: int = 0,
        kv_in: torch.Tensor | None = None,
        kv_in_len: int = 0,
        export: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Logits ``[T-start, vocab]`` for ``ids[start:]``, prefix from ``kv_in``."""
        t = torch.tensor(ids, dtype=torch.int32)
        T = len(ids)
        out = torch.empty(T - start, self.cfg.vocab)
        kv_out = self.new_snapshot(T) if export else None
        rc = self.lib.eng_forward(self._h, _ptr(t), T, start, _ptr(kv_in), int(kv_in_len), _ptr(kv_out), _ptr(out))
        if rc != 0:
            raise ValueError(f"eng_forward rejected T={T} start={start} (rc={rc})")
        return out, kv_out

    def full_logits(self, ids: list[int]) -> torch.Tensor:
        return self.forward(ids)[0]

    def decode_logits(self, ids: list[int], prefill: int = 1) -> torch.Tensor:
        """``[T, vocab]``: ``prefill`` tokens batched, the rest one at a time."""
        t = torch.tensor(ids, dtype=torch.int32)
        out = torch.empty(len(ids), self.cfg.vocab)
        if self.lib.eng_forward_incremental(self._h, _ptr(t), len(ids), int(prefill), _ptr(out)) != 0:
            raise ValueError("eng_forward_incremental rejected its arguments")
        return out

    def verify_logits(self, ids: list[int], prefill: int = 1, chunk: int = 4, junk: bool = True) -> torch.Tensor:
        """``[T, vocab]`` through the spec verify forward: ``prefill`` tokens
        batched, the rest as ``chunk``-row verify steps of one stream (each
        first fed as rejected junk and rolled back when ``junk``)."""
        t = torch.tensor(ids, dtype=torch.int32)
        out = torch.empty(len(ids), self.cfg.vocab)
        rc = self.lib.eng_forward_verify(self._h, _ptr(t), len(ids), int(prefill), int(chunk), int(junk), _ptr(out))
        if rc != 0:
            raise ValueError(f"eng_forward_verify rejected its arguments (rc={rc})")
        return out

    # ---- generation ------------------------------------------------------------

    def generate(
        self,
        ids: list[int],
        *,
        n: int,
        max_new: int,
        sampling: SamplingConfig,
        eos_id: int,
        seed: int,
        greedy: bool = False,
        stop_at_eos: bool = True,
        start: int = 0,
        kv_in: torch.Tensor | None = None,
        kv_in_len: int = 0,
        kv_out: torch.Tensor | None = None,
    ) -> NativeOutput:
        T = len(ids)
        n = max(1, int(n))
        max_new = max(1, int(max_new))
        sp = SampleParams(
            int(greedy),
            float(sampling.temperature),
            int(sampling.top_k or 0),
            float(sampling.top_p),
            float(sampling.repetition_penalty),
            int(sampling.no_repeat_ngram_size or 0),
            int(eos_id),
            int(stop_at_eos),
            float(sampling.frequency_penalty),
            float(sampling.presence_penalty),
        )
        if sp.temperature <= 0 and not greedy:
            raise HFServeError(f"temperature must be > 0, got {sp.temperature}")
        prompt = torch.tensor(ids, dtype=torch.int32)
        toks = torch.empty(n, max_new, dtype=torch.int32)
        counts = torch.empty(n, dtype=torch.int32)
        lp = torch.empty(n, dtype=torch.float64)
        timing = torch.zeros(4, dtype=torch.float64)
        rc = self.lib.eng_generate(
            self._h, _ptr(prompt), T, int(start), _ptr(kv_in), int(kv_in_len), _ptr(kv_out),
            n, max_new, ctypes.byref(sp), ctypes.c_uint64(seed & (2**64 - 1)),
            _ptr(toks), _ptr(counts), _ptr(lp), _ptr(timing),
        )
        if rc < 0:
            raise HFServeError(f"native engine rejected the request (rc={rc}, T={T}, max_new={max_new})")
        counts_l = counts.tolist()
        sums = lp.tolist()
        means = [sums[i] / c if c else -math.inf for i, c in enumerate(counts_l)]
        best = max(range(n), key=lambda i: (means[i], -i))
        rows = toks.tolist()
        return NativeOutput(
            tokens=[rows[i][: counts_l[i]] for i in range(n)],
            counts=counts_l,
            mean_logprob=means,
            best=best,
            prefill_s=float(timing[0]),
            ttft_s=float(timing[1]),
            total_s=float(timing[2]),
            last_s=float(timing[3]),
        )

    def spec_generate(
        self,
        ids: list[int],
        *,
        n: int,
        max_new: int,
        sampling: SamplingConfig,
        eos_id: int,
        seed: int,
        drafter: "NgramDrafter",
        k: int,
        greedy: bool = False,
        stop_at_eos: bool = True,
        start: int = 0,
        kv_in: torch.Tensor | None = None,
        kv_in_len: int = 0,
        kv_out: torch.Tensor | None = None,
    ) -> NativeOutput:
        """`generate` with speculative decoding: `drafter` proposes up to `k`
        tokens per stream per step; the engine verifies them in one forward
        and accepts by exact speculative sampling (same output distribution as
        `generate`; greedy output identical). Same return value."""
        t_enter = time.perf_counter()
        T = len(ids)
        n = max(1, int(n))
        max_new = max(1, int(max_new))
        k = max(0, int(k))
        if T + max_new + k + 1 > self.cfg.max_pos:  # no room for draft rows at the context edge
            return self.generate(
                ids, n=n, max_new=max_new, sampling=sampling, eos_id=eos_id, seed=seed, greedy=greedy,
                stop_at_eos=stop_at_eos, start=start, kv_in=kv_in, kv_in_len=kv_in_len, kv_out=kv_out,
            )
        sp = SampleParams(
            int(greedy),
            float(sampling.temperature),
            int(sampling.top_k or 0),
            float(sampling.top_p),
            float(sampling.repetition_penalty),
            int(sampling.no_repeat_ngram_size or 0),
            int(eos_id),
            int(stop_at_eos),
            float(sampling.frequency_penalty),
            float(sampling.presence_penalty),
        )
        if sp.temperature <= 0 and not greedy:
            raise HFServeError(f"temperature must be > 0, got {sp.temperature}")
        lib = self.lib
        prompt = torch.tensor(ids, dtype=torch.int32)
        first = torch.empty(n, dtype=torch.int32)
        timing = torch.zeros(2, dtype=torch.float64)
        job = lib.eng_spec_begin(
            self._h, _ptr(prompt), T, int(start), _ptr(kv_in), int(kv_in_len), _ptr(kv_out),
            n, max_new, k, ctypes.byref(sp), ctypes.c_uint64(seed & (2**64 - 1)), _ptr(first), _ptr(timing),
        )
        if not job:
            raise HFServeError(f"native engine rejected the spec request (T={T}, max_new={max_new}, k={k})")
        counts = torch.empty(n, dtype=torch.int32)
        lp = torch.empty(n, dtype=torch.float64)
        try:
            toks = [[t] for t in first.tolist()]
            done = [(stop_at_eos and t[0] == eos_id) or max_new <= 1 for t in toks]
            state = drafter.begin(ids, n)
            kk = max(1, k)
            # plain ctypes buffers: per-step marshalling is on the critical path
            drafts = (ctypes.c_int32 * (n * kk))()
            nd = (ctypes.c_int32 * n)()
            out = (ctypes.c_int32 * (n * (kk + 1)))()
            nout = (ctypes.c_int32 * n)()
            for s in range(n):
                drafter.push(state, s, toks[s])
            t_last = time.perf_counter()
            live = [s for s in range(n) if not done[s]]
            draft = drafter.draft
            while live:
                for s in live:
                    d = draft(state, s, min(k, max_new - len(toks[s]) - 1)) if k else ()
                    nd[s] = len(d)
                    if d:
                        drafts[s * kk : s * kk + len(d)] = d
                remaining = lib.eng_spec_step(job, drafts, nd, out, nout)
                if remaining < 0:
                    raise HFServeError(f"native engine rejected a spec step (rc={remaining})")
                t_last = time.perf_counter()
                for s in live:
                    m = nout[s]
                    new = out[s * (kk + 1) : s * (kk + 1) + m]
                    toks[s].extend(new)
                    drafter.push(state, s, new)
                    if (stop_at_eos and new[-1] == eos_id) or len(toks[s]) >= max_new:
                        done[s] = True
                live = [s for s in live if not done[s]]
                if len(live) != remaining:
                    raise HFServeError(f"spec bookkeeping diverged from the engine ({len(live)} != {remaining})")
        finally:
            lib.eng_spec_end(job, _ptr(counts), _ptr(lp))
        t_end = time.perf_counter()
        counts_l = counts.tolist()
        sums = lp.tolist()
        means = [sums[i] / c if c else -math.inf for i, c in enumerate(counts_l)]
        best = max(range(n), key=lambda i: (means[i], -i))
        return NativeOutput(
            tokens=toks,
            counts=counts_l,
            mean_logprob=means,
            best=best,
            prefill_s=float(timing[0]),
            ttft_s=float(timing[1]),
            total_s=t_end - t_enter,
            last_s=t_last - t_enter,
        )

    def warp_probs(self, logits: torch.Tensor, hist: list[int], sampling: SamplingConfig) -> torch.Tensor:
        """The engine sampler's post-warp distribution (test probe)."""
        V = int(logits.numel())
        lg = logits.to(torch.float32).contiguous()
        h = torch.tensor(hist, dtype=torch.int32)
        out = torch.empty(V)
        sp = SampleParams(
            0, float(sampling.temperature), int(sampling.top_k or 0), float(sampling.top_p),
            float(sampling.repetition_penalty), int(sampling.no_repeat_ngram_size or 0), 0, 1,
            float(sampling.frequency_penalty), float(sampling.presence_penalty),
        )
        self.lib.eng_warp_probs(_ptr(lg), V, _ptr(h), len(hist), ctypes.byref(sp), _ptr(out))
        return out


class NgramDrafter:
    """Static n-gram draft source for `NativeEngine.spec_generate`.

    ``tables[n]`` maps the last ``n`` tokens (packed as an int, see `pack`) to
    the most likely next token, or to -1 when that context is too uncertain to
    draft from (drafting stops there instead of backing off). Lookups go from
    the longest order down. A draft token that the no-repeat-ngram rule would
    ban is never proposed (the target gives it probability 0), so drafting
    stops there too. Build tables with `build_ngram_tables`.
    """

    def __init__(self, tables: dict[int, dict[int, int]], *, vocab: int, no_repeat_ngram: int = 0) -> None:
        self.tables = {int(n): t for n, t in tables.items() if t}
        self.orders = sorted(self.tables, reverse=True)
        self.V = int(vocab)
        self.nrn = int(no_repeat_ngram or 0)

    def pack(self, ctx) -> int:
        key = 0
        for t in ctx:
            key = key * self.V + int(t)
        return key

    # per-reply state: (histories, shared prompt ban index, per-stream ban overlays)
    def begin(self, prompt: list[int], n: int):
        bans: dict[tuple, set] = {}
        m = self.nrn
        if m > 1:
            for i in range(len(prompt) - m + 1):
                bans.setdefault(tuple(prompt[i : i + m - 1]), set()).add(prompt[i + m - 1])
        return ([list(prompt) for _ in range(n)], bans, [dict() for _ in range(n)])

    def push(self, state, s: int, toks: list[int]) -> None:
        hist, _bans, own = state
        h = hist[s]
        m = self.nrn
        for t in toks:
            h.append(t)
            if m > 1 and len(h) >= m:
                own[s].setdefault(tuple(h[-m:-1]), set()).add(t)

    def draft(self, state, s: int, k: int) -> list[int]:
        """Up to `k` tokens continuing stream `s` (fewer, or none, when the
        table has no confident continuation or the next one would be banned)."""
        if k <= 0 or not self.orders:
            return []
        hist, bans, own = state
        hs, own = hist[s], own[s]
        m, V = self.nrn, self.V
        ctx = hs[-max(self.orders[0], m - 1, 1):]
        local: set = set()  # n-grams completed inside this draft
        out: list[int] = []
        for _ in range(k):
            tok = -1
            L = len(ctx)
            for n in self.orders:
                if L < n:
                    continue
                key = 0
                for x in ctx[L - n:]:
                    key = key * V + x
                v = self.tables[n].get(key)
                if v is not None:
                    tok = v
                    break
            if tok < 0:
                break
            if m > 1 and L >= m - 1:
                g = tuple(ctx[L - m + 1:])
                b = bans.get(g)
                if b is not None and tok in b:
                    break
                b = own.get(g)
                if b is not None and tok in b:
                    break
                g = g + (tok,)
                if g in local:
                    break
                local.add(g)
            elif m == 1 and (tok in hs or tok in out):
                break
            out.append(tok)
            ctx.append(tok)
        return out


def build_ngram_tables(seqs, *, order: int = 3, min_conf: float = 0.5, min_count: int = 1, vocab: int) -> dict[int, dict[int, int]]:
    """Most-frequent-next-token tables for orders 1..`order` from token sequences.

    A context whose top continuation has relative frequency < `min_conf` (or
    fewer than `min_count` observations) maps to -1: draft nothing there.
    """
    from collections import Counter, defaultdict

    counts = [defaultdict(Counter) for _ in range(order + 1)]
    for seq in seqs:
        for i in range(1, len(seq)):
            for n in range(1, order + 1):
                if i - n < 0:
                    break
                counts[n][tuple(seq[i - n : i])][seq[i]] += 1
    tables: dict[int, dict[int, int]] = {}
    for n in range(1, order + 1):
        t: dict[int, int] = {}
        for ctx, c in counts[n].items():
            tok, top = c.most_common(1)[0]
            tot = sum(c.values())
            key = 0
            for x in ctx:
                key = key * vocab + int(x)
            t[key] = int(tok) if (top / tot >= min_conf and tot >= min_count) else -1
        tables[n] = t
    return tables


def save_ngram_tables(tables: dict[int, dict[int, int]], path: Path | str, *, vocab: int, meta: dict | None = None) -> None:
    blob = {"vocab": int(vocab), "meta": dict(meta or {})}
    for n, t in tables.items():
        blob[f"keys{n}"] = torch.tensor(list(t.keys()), dtype=torch.int64)
        blob[f"vals{n}"] = torch.tensor(list(t.values()), dtype=torch.int32)
    torch.save(blob, str(path))


def load_ngram_tables(path: Path | str) -> tuple[dict[int, dict[int, int]], int]:
    blob = torch.load(str(path), weights_only=True)
    tables = {}
    for name in blob:
        if name.startswith("keys"):
            n = int(name[4:])
            tables[n] = dict(zip(blob[name].tolist(), blob[f"vals{n}"].tolist()))
    return tables, int(blob["vocab"])


@dataclass
class _Result:
    tokens: list
    mean_logprob: list
    best: int
    stats: HFGenerationStats
    prefix_reused: int


class NativeGenerator(LeanGenerator):
    """Drop-in for `hfserve.HFGenerator` / `LeanGenerator` backed by `NativeEngine`."""

    runtime = "native"

    def __init__(self, settings: Settings, log: EventLog | None = None) -> None:
        from tokenizers import Tokenizer

        self.settings = settings
        self.log = log or NullLog()
        if settings.hf_model_dir is None:
            raise HFServeError("BABBLE_SERVE_BACKEND=hf but BABBLE_HF_MODEL_DIR is not set")
        model_dir = Path(settings.hf_model_dir)
        if not model_dir.is_dir():
            raise HFServeError(f"BABBLE_HF_MODEL_DIR={model_dir} is not a directory")
        threads = int(getattr(settings, "infer_threads", None) or settings.train_threads or 4)
        configure_cpu(threads)
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

        started = time.perf_counter()
        self.engine = NativeEngine(model_dir, threads=threads)
        load_s = time.perf_counter() - started
        self.precision = "int8"
        self.model_id = model_dir.name
        self.param_count = self.engine.param_count
        self.max_position_embeddings = self.engine.cfg.max_pos
        self.prefix_cache = PrefixKVCache(
            max_entries=int(getattr(settings, "lean_prefix_cache_entries", 32)),
            max_bytes=int(getattr(settings, "lean_prefix_cache_mb", 512)) * 1024 * 1024,
        )
        self._lock = threading.Lock()
        # Real generations waiting for `_lock`. `prewarm` works in chunks and
        # gives the engine up between them whenever this is non-zero, so a
        # background pre-prefill delays a reply by at most one chunk.
        self._waiting = 0
        self._waiting_lock = threading.Lock()
        self._prewarm = dict(calls=0, tokens=0, chunks=0, yielded=0, incomplete=0, skipped=0, seconds=0.0)
        from .leanserve import _flag

        self._extra_penalties = _flag("BABBLE_HF_FREQUENCY_PENALTIES")
        self.drafter, self.spec_k, spec_note = self._load_spec(model_dir)
        self.step = 0
        build = self.engine.build
        self.log.event(
            "model.load",
            source="hf",
            runtime="native",
            threads=threads,
            native_build="compiled" if build.built else "cached",
            native_build_s=round(build.build_s, 2),
            native_lib=str(build.path),
            native_kv=self.engine.kv,
            native_w4=os.environ.get("BABBLE_NATIVE_W4", "") if self.engine.w4 else "",
            native_head2=os.environ.get("BABBLE_NATIVE_HEAD2", "") if self.engine.head2 else "",
            load_s=round(load_s, 2),
            prefix_cache_mb=self.prefix_cache.max_bytes // (1024 * 1024),
            prefix_cache_entries=self.prefix_cache.max_entries,
            # how many full-budget conversation snapshots fit (engine's own KV size)
            prefix_cache_full_snapshots=self.prefix_cache.max_bytes // max(1, self.engine.kv_bytes(self._full_prompt_tokens())),
            model_dir=str(model_dir),
            step=self.step,
            params=self.param_count,
            device="cpu",
            frequency_presence_penalties=self._extra_penalties,
            native_spec=spec_note,
        )

    def _full_prompt_tokens(self) -> int:
        """Positions in a snapshot of a prompt at the conversation token budget."""
        cap = self._prompt_budget()
        if getattr(self.settings, "conversation_context", False):
            cap = min(cap, int(getattr(self.settings, "conversation_max_tokens", cap)))
        return max(1, cap + 2)  # <bos> ... <sep>

    @contextlib.contextmanager
    def _priority_lock(self):
        """`_lock`, announced first so a running `prewarm` yields to us."""
        with self._waiting_lock:
            self._waiting += 1
        try:
            self._lock.acquire()
        finally:
            with self._waiting_lock:
                self._waiting -= 1
        try:
            yield
        finally:
            self._lock.release()

    #: Tokens prefilled per `prewarm` step; the engine is released between
    #: steps. ~128 tokens is ~40-60 ms at a 1.5k-token history on 4 Haswell
    #: cores: the longest a real reply can wait behind a background warm.
    PREWARM_CHUNK = 128

    #: A warm still not done after this long (a backlog of real replies) is dropped.
    PREWARM_DEADLINE_S = 30.0

    def prewarm(self, text: str, *, deadline_s: float | None = None) -> dict:
        """Prefill ``<bos> text`` into the prefix cache, off the hot path.

        ``text`` is the start of a prompt the bot expects to serve soon (the
        next turn's transcript up to and including ``user: ``). Whatever part
        of it the cache already holds is reused; the rest is prefilled in
        `PREWARM_CHUNK`-token steps, each stored as a snapshot. Between steps
        it steps aside for any real generation waiting on the engine, then
        resumes from its stored progress once that one is done (another
        channel's reply must not cancel this channel's warm). Never raises:
        this is an optimization, and a failure only costs the next reply a
        longer prefill.
        """
        info = dict(tokens=0, reused=0, prefilled=0, chunks=0, yielded=0, done=False, busy_ms=0.0, ms=0.0)
        cache = self.prefix_cache
        if not cache.enabled:
            return info
        started = time.perf_counter()
        self._prewarm["calls"] += 1
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
            if len(tokens) > self._prompt_budget() - 1:
                # The real prompt would be left-truncated; its prefix is unknowable.
                self._prewarm["skipped"] += 1
                return info
            ids = [self.bos_id, *tokens]
            L = len(ids)
            info["tokens"] = L
            if self.engine.kv_bytes(L) > cache.max_bytes:
                self._prewarm["skipped"] += 1
                return info
            ids_t = torch.tensor(ids, dtype=torch.long)
            limit = started + (self.PREWARM_DEADLINE_S if deadline_s is None else float(deadline_s))
            first = True
            while True:
                if self._waiting:
                    # A real reply is queued: let it have the engine, and wait
                    # until it holds the lock (waiting drops to 0 on acquire);
                    # `with self._lock` below then waits for it to finish.
                    info["yielded"] += 1
                    while self._waiting and time.perf_counter() < limit:
                        time.sleep(0.002)
                if time.perf_counter() >= limit:
                    break
                with self._lock:
                    if self._waiting:
                        continue
                    entry, common = cache.lookup(ids_t)
                    if first:
                        info["reused"] = min(common, L)
                        first = False
                    if common >= L:
                        info["done"] = True
                        break
                    busy = time.perf_counter()
                    reused = common if entry is not None else 0
                    end = min(L, reused + self.PREWARM_CHUNK)
                    kv_in, stored = (entry.k[0], int(entry.ids.numel())) if reused > 0 else (None, 0)
                    kv_out = self.engine.new_snapshot(end)
                    self.engine.generate(
                        ids[:end], n=1, max_new=1, sampling=self._sampling(), eos_id=self.eos_id,
                        seed=0, greedy=True, start=reused, kv_in=kv_in, kv_in_len=stored, kv_out=kv_out,
                    )
                    cache.store(ids_t[:end], [kv_out], [])
                    info["prefilled"] += end - reused
                    info["chunks"] += 1
                    info["busy_ms"] += (time.perf_counter() - busy) * 1000
        except Exception as exc:  # pragma: no cover - never let warming hurt serving
            self.log.event("bot.error", where="prewarm", error=f"{type(exc).__name__}: {exc}")
        info["ms"] = round((time.perf_counter() - started) * 1000, 2)
        info["busy_ms"] = round(info["busy_ms"], 2)
        p = self._prewarm
        p["tokens"] += info["prefilled"]
        p["chunks"] += info["chunks"]
        p["yielded"] += info["yielded"]
        p["incomplete"] += int(not info["done"])
        p["seconds"] += info["busy_ms"] / 1000
        return info

    def prewarm_stats(self) -> dict:
        return dict(self._prewarm)

    def _generate(self, prompt: str, *, max_new_tokens: int, best_of: int, use_prefix_cache: bool = True):
        started = time.perf_counter()
        with self._priority_lock():
            ids = self._encode_prompt(prompt)[0].tolist()
            P = len(ids)
            max_new = min(max(1, int(max_new_tokens)), self.max_position_embeddings - P)
            if max_new < 1:
                raise HFServeError(f"prompt of {P} tokens leaves no room to generate")
            cache = self.prefix_cache if use_prefix_cache and self.prefix_cache.enabled else None
            kv_in, stored, reused, kv_out = None, 0, 0, None
            ids_t = None
            if cache is not None:
                ids_t = torch.tensor(ids, dtype=torch.long)
                entry, common = cache.lookup(ids_t)
                reused = min(common, P - 1)
                if entry is not None and reused > 0:
                    kv_in, stored = entry.k[0], int(entry.ids.numel())
                else:
                    reused = 0
                cache.record(reused)
                if self.engine.kv_bytes(P) <= cache.max_bytes:
                    kv_out = self.engine.new_snapshot(P)
            seed = int(torch.randint(0, 2**62, (1,)).item())
            entered = time.perf_counter()
            gen = self.engine.generate
            extra = {}
            if self.drafter is not None:
                gen = self.engine.spec_generate
                extra = dict(drafter=self.drafter, k=self.spec_k)
            out = gen(
                **extra,
                ids=ids,
                n=best_of,
                max_new=max_new,
                sampling=self._sampling(),
                eos_id=self.eos_id,
                seed=seed,
                start=reused,
                kv_in=kv_in,
                kv_in_len=stored,
                kv_out=kv_out,
            )
            if cache is not None and kv_out is not None:
                cache.store(ids_t, [kv_out], [])
        keep = [tok for tok in out.tokens[out.best] if tok not in (self.pad_id, self.eos_id)]
        text = self.tokenizer.decode(keep, skip_special_tokens=True).strip()
        end = time.perf_counter()
        base = entered - started
        stats = HFGenerationStats(
            ttft_s=max(0.0, base + out.ttft_s),
            total_s=max(0.0, end - started),
            steady_s=max(0.0, out.last_s - out.ttft_s),
            candidate_token_counts=tuple(out.counts),
            first_step_tokens=len(out.counts),
            selected_index=out.best,
            selected_content_tokens=len(keep),
        )
        return text, _Result(tokens=out.tokens, mean_logprob=out.mean_logprob, best=out.best, stats=stats, prefix_reused=reused)

    def _load_spec(self, model_dir: Path):
        """`BABBLE_NATIVE_SPEC=1`: n-gram speculative decoding. The table comes
        from `BABBLE_NATIVE_SPEC_TABLE` or `<model dir>/spec-ngram.pt`; without
        one, spec stays off (logged). `BABBLE_NATIVE_SPEC_K` drafts per step."""
        import os

        from .leanserve import _flag

        if not _flag("BABBLE_NATIVE_SPEC"):
            return None, 0, "off"
        try:
            k = max(1, min(16, int(os.environ.get("BABBLE_NATIVE_SPEC_K", "1") or 1)))
        except ValueError:
            k = 1
        path = Path(os.environ.get("BABBLE_NATIVE_SPEC_TABLE", "").strip() or (model_dir / "spec-ngram.pt"))
        if not path.is_file():
            return None, 0, f"off (no n-gram table at {path})"
        try:
            tables, vocab = load_ngram_tables(path)
        except Exception as exc:  # a corrupt table must not take serving down
            return None, 0, f"off (unreadable n-gram table {path}: {exc})"
        if vocab != self.engine.cfg.vocab:
            return None, 0, f"off (n-gram table vocab {vocab} != model vocab {self.engine.cfg.vocab})"
        nrn = int(self.settings.no_repeat_ngram_size or 0)
        return NgramDrafter(tables, vocab=vocab, no_repeat_ngram=nrn), k, f"on k={k} table={path}"

    def benchmark_metadata(self) -> dict[str, object]:
        opts = [
            "native C++ engine (AVX2/FMA, no transformers)",
            "int8 weights dequantized in-register x fp32 activations",
            f"{self.engine.threads} engine threads",
            "shared prompt prefill",
            "grouped-expert batched decode",
            "row compaction",
            f"{self.engine.kv} KV cache, flash attention (split-K decode)",
        ]
        if self.prefix_cache.enabled:
            opts.append(f"prefix KV cache ({self.prefix_cache.max_bytes // (1024 * 1024)} MB)")
        opts.append("frequency/presence penalties on" if self._extra_penalties else "frequency/presence penalties off")
        if self.engine.head2:
            opts.append(f"two-stage lm_head (N={self.engine.head2[0]})")
        dtype = "int8/fp32"
        if self.engine.w4:
            dtype = f"int8/fp32 + int4 g{self.engine.w4[1]} decode (QUALITY TRADE)"
            opts.append("int4 decode weights (BABBLE_NATIVE_W4, lossy)")
        if self.drafter is not None:
            opts.append(f"speculative decoding (n-gram drafts, k={self.spec_k})")
        return {
            "model": self.model_id,
            "params": self.param_count,
            "dtype": dtype,
            "backend": "hf-native",
            "runtime": "native",
            "optimizations": tuple(opts),
        }


__all__ = ["NativeEngine", "NativeGenerator", "NativeOutput", "NativeUnavailable"]
