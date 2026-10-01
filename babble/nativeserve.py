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

Unusable here (CPU without AVX2/FMA/F16C, no compiler, failed build, a
snapshot shape the kernels do not implement) raises `NativeUnavailable`, which
`hfserve.make_generator` logs and answers with the lean runtime.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
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


class NativeEngine:
    """One loaded model in the C++ engine. Not thread-safe; serialize calls."""

    def __init__(self, model_dir: Path | str, *, threads: int = 4) -> None:
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
        try:
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

    def kv_floats(self, positions: int) -> int:
        return int(self.lib.eng_kv_floats(self._h, int(positions)))

    def kv_bytes(self, positions: int) -> int:
        return 4 * self.kv_floats(positions)

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
        kv_out = torch.empty(self.kv_floats(T)) if export else None
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
        self._prewarm = dict(calls=0, tokens=0, chunks=0, yielded=0, skipped=0, seconds=0.0)
        from .leanserve import _flag

        self._extra_penalties = _flag("BABBLE_HF_FREQUENCY_PENALTIES")
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

    def prewarm(self, text: str) -> dict:
        """Prefill ``<bos> text`` into the prefix cache, off the hot path.

        ``text`` is the start of a prompt the bot expects to serve soon (the
        next turn's transcript up to and including ``user: ``). Whatever part
        of it the cache already holds is reused; the rest is prefilled in
        `PREWARM_CHUNK`-token steps, each stored as a snapshot, so the work is
        never lost even when a real request preempts it. Never raises: this
        is an optimization, and a failure only costs the next reply a longer
        prefill.
        """
        info = dict(tokens=0, reused=0, prefilled=0, chunks=0, yielded=False, ms=0.0)
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
            first = True
            while True:
                if self._waiting:
                    info["yielded"] = True
                    break
                with self._lock:
                    if self._waiting:
                        info["yielded"] = True
                        break
                    entry, common = cache.lookup(ids_t)
                    if first:
                        info["reused"] = min(common, L)
                        first = False
                    if common >= L:
                        break
                    reused = common if entry is not None else 0
                    end = min(L, reused + self.PREWARM_CHUNK)
                    kv_in, stored = (entry.k[0], int(entry.ids.numel())) if reused > 0 else (None, 0)
                    kv_out = torch.empty(self.engine.kv_floats(end))
                    self.engine.generate(
                        ids[:end], n=1, max_new=1, sampling=self._sampling(), eos_id=self.eos_id,
                        seed=0, greedy=True, start=reused, kv_in=kv_in, kv_in_len=stored, kv_out=kv_out,
                    )
                    cache.store(ids_t[:end], [kv_out], [])
                    info["prefilled"] += end - reused
                    info["chunks"] += 1
        except Exception as exc:  # pragma: no cover - never let warming hurt serving
            self.log.event("bot.error", where="prewarm", error=f"{type(exc).__name__}: {exc}")
        info["ms"] = round((time.perf_counter() - started) * 1000, 2)
        p = self._prewarm
        p["tokens"] += info["prefilled"]
        p["chunks"] += info["chunks"]
        p["yielded"] += int(info["yielded"])
        p["seconds"] += info["ms"] / 1000
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
                    kv_out = torch.empty(self.engine.kv_floats(P))
            seed = int(torch.randint(0, 2**62, (1,)).item())
            entered = time.perf_counter()
            out = self.engine.generate(
                ids,
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

    def benchmark_metadata(self) -> dict[str, object]:
        opts = [
            "native C++ engine (AVX2/FMA, no transformers)",
            "int8 weights dequantized in-register x fp32 activations",
            f"{self.engine.threads} engine threads",
            "shared prompt prefill",
            "grouped-expert batched decode",
            "row compaction",
        ]
        if self.prefix_cache.enabled:
            opts.append(f"prefix KV cache ({self.prefix_cache.max_bytes // (1024 * 1024)} MB)")
        opts.append("frequency/presence penalties on" if self._extra_penalties else "frequency/presence penalties off")
        return {
            "model": self.model_id,
            "params": self.param_count,
            "dtype": "int8/fp32",
            "backend": "hf-native",
            "runtime": "native",
            "optimizations": tuple(opts),
        }


__all__ = ["NativeEngine", "NativeGenerator", "NativeOutput", "NativeUnavailable"]
