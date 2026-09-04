"""Small, in-process inference benchmark for the Discord ``/bench`` command.

``run_benchmark`` is synchronous by design. The Discord adapter runs it with
``asyncio.to_thread`` while holding the same lock as ordinary generation, so it
uses the loaded model without blocking the event loop or racing a reply. The
workload is fixed, synthetic, and bounded; it never reads messages or trains.
"""

from __future__ import annotations

import platform
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

_BENCHMARK_PROMPT = "Write two sentences about a robot learning why people laugh."
_BENCHMARK_TOKENS = 64
_BENCHMARK_CANDIDATES = 4
_BENCHMARK_SEED = 0xBABB1E


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    prompt_tokens: int | None
    candidates: int
    candidate_token_counts: tuple[int, ...]
    aggregate_tokens: int
    selected_tokens: int
    selected_content_tokens: int
    selected_tps: float
    ttft_ms: float | None
    steady_tps: float | None
    e2e_tps: float
    elapsed_ms: float
    cpu_cores: float
    rss_mb: float | None
    peak_rss_mb: float | None
    torch_version: str
    transformers_version: str | None
    threads: int
    interop_threads: int
    machine: str
    model: str
    params: int
    dtype: str
    optimizations: tuple[str, ...]


class BenchmarkUnavailable(RuntimeError):
    """The loaded generator cannot provide honest step-level measurements."""


def _rss_mb() -> float | None:
    """Current resident memory on Linux, where the production bot runs."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def _peak_rss_mb() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _transformers_version() -> str | None:
    try:
        import transformers
    except ImportError:
        return None
    return str(transformers.__version__)


def run_benchmark(generator: Callable[[str], Any]) -> BenchmarkResult:
    """Measure one bounded inference run on an already-loaded generator.

    HF generators expose step-level timing and candidate counts without holding
    score history. A generator without that instrumentation is refused rather
    than presenting its whole-call time as TTFT or guessing candidate counts.
    Torch's global RNG is restored even if generation fails, so running
    ``/bench`` does not alter the next sampled reply.
    """
    settings = getattr(generator, "settings", None)
    configured_n = int(getattr(settings, "best_of", _BENCHMARK_CANDIDATES))
    candidates = min(_BENCHMARK_CANDIDATES, max(1, configured_n))
    prompt = _BENCHMARK_PROMPT
    conversation_formatter = getattr(generator, "conversation_prompt", None)
    if getattr(settings, "conversation_context", False) and callable(conversation_formatter):
        prompt = conversation_formatter(
            [],
            prompt,
            max_turns=int(getattr(settings, "conversation_max_turns", 6)),
            max_tokens=int(getattr(settings, "conversation_max_tokens", 512)),
            max_chars=int(getattr(settings, "conversation_max_chars", 6_000)),
        )
    prompt_tokens: int | None = None
    encode_prompt = getattr(generator, "_encode_prompt", None)
    if callable(encode_prompt):
        prompt_tokens = int(encode_prompt(prompt).shape[-1])

    rng_state = torch.random.get_rng_state()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    try:
        # HF serving is CPU-only. Do not also reseed an unrelated MPS/CUDA
        # generator when this helper is used from a development process.
        torch.random.default_generator.manual_seed(_BENCHMARK_SEED)
        benchmark_sample = getattr(generator, "benchmark_sample", None)
        if callable(benchmark_sample):
            stats = benchmark_sample(
                prompt,
                max_new_tokens=_BENCHMARK_TOKENS,
                best_of=candidates,
            )
            counts = tuple(int(value) for value in stats.candidate_token_counts)
            aggregate = sum(counts)
            selected = counts[int(stats.selected_index)]
            steady_tokens = max(0, aggregate - int(stats.first_step_tokens))
            steady_tps = steady_tokens / stats.steady_s if stats.steady_s > 0 else None
            ttft_ms = stats.ttft_s * 1000
            selected_content = int(stats.selected_content_tokens)
            backend = "hf"
        else:
            raise BenchmarkUnavailable(
                f"{type(generator).__name__} does not expose step-level benchmark instrumentation"
            )
    finally:
        torch.random.set_rng_state(rng_state)
    elapsed_s = time.perf_counter() - wall_started
    cpu_s = time.process_time() - cpu_started
    e2e_tps = aggregate / elapsed_s if elapsed_s > 0 else 0.0
    metadata_fn = getattr(generator, "benchmark_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    return BenchmarkResult(
        backend=backend,
        prompt_tokens=prompt_tokens,
        candidates=candidates,
        candidate_token_counts=counts,
        aggregate_tokens=aggregate,
        selected_tokens=selected,
        selected_content_tokens=selected_content,
        selected_tps=selected / elapsed_s if elapsed_s > 0 else 0.0,
        ttft_ms=ttft_ms,
        steady_tps=steady_tps,
        e2e_tps=e2e_tps,
        elapsed_ms=elapsed_s * 1000,
        cpu_cores=cpu_s / elapsed_s if elapsed_s > 0 else 0.0,
        rss_mb=_rss_mb(),
        peak_rss_mb=_peak_rss_mb(),
        torch_version=str(torch.__version__),
        transformers_version=_transformers_version(),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        machine=platform.machine() or "unknown",
        model=str(metadata.get("model", type(generator).__name__)),
        params=int(metadata.get("params", 0)),
        dtype=str(metadata.get("dtype", "unknown")),
        optimizations=tuple(str(value) for value in metadata.get("optimizations", ())),
    )


def format_benchmark(result: BenchmarkResult) -> str:
    """Render a compact Discord-safe benchmark report (well below 2,000 chars)."""
    runtime = f"torch {result.torch_version}"
    if result.transformers_version:
        runtime += f" · transformers {result.transformers_version}"
    prompt = f" · {result.prompt_tokens} prompt tokens" if result.prompt_tokens is not None else ""
    ttft = f"{result.ttft_ms:.0f} ms" if result.ttft_ms is not None else "n/a"
    steady = f"{result.steady_tps:.1f} tok/s" if result.steady_tps is not None else "n/a"
    rss = f"{result.rss_mb:.0f} MB" if result.rss_mb is not None else "n/a"
    peak = f"{result.peak_rss_mb:.0f} MB" if result.peak_rss_mb is not None else "n/a"
    counts = "/".join(str(value) for value in result.candidate_token_counts)
    return (
        "**Booper inference benchmark**\n"
        f"Fixed synthetic workload · {result.candidates} candidate(s) · up to "
        f"{_BENCHMARK_TOKENS} tokens each{prompt}\n"
        f"TTFT {ttft} · aggregate steady {steady}\n"
        f"End-to-end {result.e2e_tps:.1f} aggregate tok/s · "
        f"{result.selected_tps:.1f} selected-candidate tok/s "
        f"({result.elapsed_ms:.0f} ms)\n"
        f"Generated {result.aggregate_tokens} tokens aggregate ({counts}); selected "
        f"candidate {result.selected_tokens} tokens ({result.selected_content_tokens} content)\n"
        f"CPU {result.cpu_cores:.1f} cores average · threads {result.threads}+"
        f"{result.interop_threads} interop · RSS {rss} (peak {peak})\n"
        f"{result.model} · {result.params / 1e6:.1f}M params · {result.dtype} · "
        f"{result.backend}/{result.machine}\n"
        f"{runtime} · {', '.join(result.optimizations)}"
    )


__all__ = ["BenchmarkResult", "BenchmarkUnavailable", "format_benchmark", "run_benchmark"]
