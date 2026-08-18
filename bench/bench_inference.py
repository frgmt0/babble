"""Measure how fast babble actually is on this CPU box.

Everything ro asked for, measured rather than estimated:

* **TTFT** -- time to first token (prefill + first sample), warm process.
* **TPS** -- steady-state decode tokens/sec, and end-to-end tokens/sec over a
  whole `max_new_tokens` generation (which includes TTFT).
* **Cached vs uncached** -- the KV cache is PR #7's headline; both paths decode
  the *same* greedy sequence from the *same* prompt so the ratio is honest.
* **Cold vs warm** -- first generation after process start (includes any lazy
  init) versus steady state.
* **Thread sweep** -- torch intra-op threads 1/2/4/8, each in its own process so
  the count is set before any tensor work.
* **RAM** -- torch import baseline, model idle RSS, peak RSS during decode, and
  the model's own parameter footprint (params x 4 bytes) so the gap between "the
  model" and "the python process" is visible.
* **CPU** -- cores busy during decode, at each thread count.
* **Training** -- one pretrain run (`babble train --force`) on the live corpus:
  steps/sec, wall time, peak RSS, CPU.

Timing harness choices, stated so the numbers are reproducible:

* Decode is **greedy** (temperature 0, argmax). Two reasons: it makes the cached
  and uncached paths emit a byte-for-byte identical sequence, so their speed
  ratio is not muddied by different sampling luck; and it removes RNG variance
  from the timing. Sampling (multinomial over a 260-way vocab) adds a fixed,
  negligible per-step cost -- the realistic `best_of=4` bot reply path is timed
  separately as an end-to-end latency so that cost is not hidden.
* The harness always runs the **full** `max_new_tokens` -- it never stops on
  <eos>. A randomly initialised model almost never emits <eos>, and a trained
  one would stop at a data-dependent point; forcing the full length makes TPS a
  property of the decoder, not of what the weights happen to say today.

Run it:

    python bench/bench_inference.py all --runs 40 --json-out bench/results.json

Sub-commands (`all` orchestrates the rest, spawning a fresh process per thread
count so torch's thread pool is sized before it touches a tensor):

    infer    one thread count, prints a JSON blob (also the thread-sweep worker)
    train    one pretrain run, measured
    all      the whole report + BENCHMARKS-shaped tables
"""

from __future__ import annotations

import time

# Captured before torch is imported, so a cold-start figure taken against this
# includes torch's ~1s import -- the dominant cost of a bot restart, and the one
# an in-process timer taken after `import torch` silently drops.
_SCRIPT_START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

# How long torch itself took to import, and the RSS right after -- the "torch
# baseline" the model's own footprint is added on top of. Captured here so
# nothing this module does can inflate them first.
_TORCH_IMPORT_S = time.perf_counter() - _SCRIPT_START
_TORCH_BASELINE_RSS_KB = None


def _rss_kb(field_name: str = "VmRSS", pid: int | None = None) -> int | None:
    """A single field from /proc/<pid>/status, in kB. `VmHWM` is peak RSS."""
    path = f"/proc/{pid or 'self'}/status"
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(field_name + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


_TORCH_BASELINE_RSS_KB = _rss_kb()

from babble.config import Settings  # noqa: E402
from babble.cpu_runtime import configure_cpu, force_cpu_device  # noqa: E402
from babble.corpus import CorpusStore  # noqa: E402
from babble.generate import best_continuation  # noqa: E402
from babble.model import Babbler, ModelConfig, config_from_settings  # noqa: E402
from babble.tokenizer import text_context  # noqa: E402
from babble.trainer import corpus_rows  # noqa: E402

CLK_TCK = os.sysconf("SC_CLK_TCK")
NPROC = os.cpu_count() or 1


# --- small stats helpers --------------------------------------------------


def _summary(values: list[float]) -> dict:
    """Median plus the spread, so a single noisy sample never stands alone."""
    if not values:
        return {"median": None, "min": None, "max": None, "p25": None, "p75": None, "n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "p25": s[max(0, int(0.25 * (n - 1)))],
        "p75": s[min(n - 1, int(0.75 * (n - 1)))],
        "n": n,
    }


def _cpu_ticks(pid: int | None = None) -> int | None:
    """utime+stime for a pid, in clock ticks (fields 14,15 of /proc/<pid>/stat)."""
    path = f"/proc/{pid or 'self'}/stat"
    try:
        with open(path) as fh:
            data = fh.read()
    except OSError:
        return None
    # The comm field can contain spaces/parens; split after the trailing ')'.
    rparen = data.rfind(")")
    rest = data[rparen + 2 :].split()
    # rest[0] is state (field 3); utime=field 14 -> rest[11], stime=field 15 -> rest[12]
    return int(rest[11]) + int(rest[12])


# --- loading the model under test -----------------------------------------


def _resolve_checkpoint(settings: Settings, explicit: Path | None) -> tuple[Path | None, str]:
    """Which weights to benchmark, and a label for the report.

    Preference: an explicit `--checkpoint`, then the served `latest.pt`, then
    random init. Timing is essentially weight-independent -- what matters is
    the geometry -- but the label keeps the report honest about which bytes
    were loaded.
    """
    if explicit is not None:
        return explicit, f"explicit:{explicit.name}"
    if settings.latest_checkpoint.exists():
        return settings.latest_checkpoint, "latest.pt"
    return None, "random-init"


def load_model(settings: Settings, checkpoint: Path | None) -> tuple[Babbler, str, int]:
    """Load `checkpoint` (or a random model) onto CPU, eval mode. Returns the
    model, a source label, and the training step it stopped at."""
    path, label = _resolve_checkpoint(settings, checkpoint)
    device = force_cpu_device()
    if path is None:
        model = Babbler(config_from_settings(settings)).to(device)
        model.eval()
        return model, label, 0
    payload = torch.load(path, map_location=device, weights_only=True)
    model = Babbler(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, label, int(payload.get("step", 0))


def pick_prompts(settings: Settings) -> tuple[str, list[str]]:
    """A canonical median-length prompt plus a short/median/long spread, all real
    corpus rows so prefill sees realistic prompt lengths -- never a synthetic
    4-token toy prompt."""
    rows = corpus_rows(settings)
    texts = sorted({r.text for r in rows if r.text.strip()}, key=len)
    if not texts:
        texts = ["what's the weather"]
    canonical = texts[len(texts) // 2]
    spread = [texts[0], texts[len(texts) // 4], texts[len(texts) // 2],
              texts[(3 * len(texts)) // 4], texts[-1]]
    # de-dup while preserving order
    seen: set[str] = set()
    spread = [t for t in spread if not (t in seen or seen.add(t))]
    return canonical, spread


# --- the decode timing harness --------------------------------------------


@dataclass
class DecodeTiming:
    ttft_s: float                 # prefill + first token
    step_times_s: list[float]     # per-token time for tokens 2..N
    total_s: float                # whole generation, TTFT included

    @property
    def n_tokens(self) -> int:
        return len(self.step_times_s) + 1

    @property
    def steady_tps(self) -> float:
        body = sum(self.step_times_s)
        return (len(self.step_times_s) / body) if body > 0 else float("nan")

    @property
    def e2e_tps(self) -> float:
        return self.n_tokens / self.total_s if self.total_s > 0 else float("nan")


@torch.inference_mode()
def decode_cached(model: Babbler, context: list[int], max_new_tokens: int) -> DecodeTiming:
    """Prefill the prompt into a KV cache, then one single-token forward per step
    -- the path PR #7 added and the bot uses. Greedy, full length."""
    cache = model.new_cache(1, max_len=len(context) + max_new_tokens)
    t0 = time.perf_counter()
    logits = model(torch.tensor([context], dtype=torch.long), cache=cache)[:, -1]
    tok = int(logits.argmax(dim=-1)[0])
    ttft = time.perf_counter() - t0
    step_times: list[float] = []
    for _ in range(max_new_tokens - 1):
        s = time.perf_counter()
        logits = model(torch.tensor([[tok]], dtype=torch.long), cache=cache)[:, -1]
        tok = int(logits.argmax(dim=-1)[0])
        step_times.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return DecodeTiming(ttft, step_times, total)


@torch.inference_mode()
def decode_uncached(model: Babbler, context: list[int], max_new_tokens: int) -> DecodeTiming:
    """The pre-cache fallback: a full forward over the whole (growing) window on
    every step. Same greedy sequence as `decode_cached`, so the only difference
    is the redone attention over the prefix."""
    block = model.config.block_size
    seq = list(context)
    t0 = time.perf_counter()
    logits = model(torch.tensor([seq[-block:]], dtype=torch.long))[:, -1]
    tok = int(logits.argmax(dim=-1)[0])
    seq.append(tok)
    ttft = time.perf_counter() - t0
    step_times: list[float] = []
    for _ in range(max_new_tokens - 1):
        s = time.perf_counter()
        window = seq[-block:]
        logits = model(torch.tensor([window], dtype=torch.long))[:, -1]
        tok = int(logits.argmax(dim=-1)[0])
        seq.append(tok)
        step_times.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return DecodeTiming(ttft, step_times, total)


def _aggregate(timings: list[DecodeTiming]) -> dict:
    return {
        "ttft_ms": _summary([t.ttft_s * 1000 for t in timings]),
        "steady_tps": _summary([t.steady_tps for t in timings]),
        "e2e_tps": _summary([t.e2e_tps for t in timings]),
        "total_ms": _summary([t.total_s * 1000 for t in timings]),
        "tokens": timings[0].n_tokens if timings else 0,
    }


# --- one thread-count's worth of inference numbers ------------------------


def run_infer(
    settings: Settings,
    checkpoint: Path | None,
    *,
    threads: int,
    runs: int,
    warmup: int,
    max_new_tokens: int,
) -> dict:
    """All the inference numbers for a single torch thread count. This is both a
    stand-alone command and the worker `all` spawns once per thread count."""
    configure_cpu(threads)
    model, source, step = load_model(settings, checkpoint)
    rss_model_idle = _rss_kb()
    canonical, spread = pick_prompts(settings)
    ctx = text_context(canonical, model.config.block_size)

    # Warm up: first touch of each code path pays lazy-init and allocator costs
    # we do not want folded into the steady-state medians.
    for _ in range(max(1, warmup)):
        decode_cached(model, ctx, max_new_tokens)
        decode_uncached(model, ctx, max_new_tokens)

    cached = [decode_cached(model, ctx, max_new_tokens) for _ in range(runs)]
    uncached = [decode_uncached(model, ctx, max_new_tokens) for _ in range(runs)]

    # CPU: cores busy over a sustained cached-decode burst (~1.5s of work).
    t_wall0 = time.perf_counter()
    c0 = _cpu_ticks()
    burst = 0
    while time.perf_counter() - t_wall0 < 1.5:
        decode_cached(model, ctx, max_new_tokens)
        burst += 1
    wall = time.perf_counter() - t_wall0
    cpu_s = (_cpu_ticks() - c0) / CLK_TCK
    cores_busy = cpu_s / wall if wall > 0 else float("nan")

    # Peak RSS after everything above -- VmHWM is the process high-water mark.
    rss_peak = _rss_kb("VmHWM")

    # Prompt-length sensitivity of TTFT (prefill is O(prompt length)).
    ttft_by_len = []
    for p in spread:
        pctx = text_context(p, model.config.block_size)
        samples = [decode_cached(model, pctx, max_new_tokens).ttft_s * 1000
                   for _ in range(max(5, runs // 3))]
        ttft_by_len.append({"prompt_chars": len(p), "ctx_tokens": len(pctx),
                            "ttft_ms": _summary(samples)})

    cached_agg = _aggregate(cached)
    uncached_agg = _aggregate(uncached)
    speedup = None
    if uncached_agg["steady_tps"]["median"] and cached_agg["steady_tps"]["median"]:
        speedup = cached_agg["steady_tps"]["median"] / uncached_agg["steady_tps"]["median"]

    return {
        "threads": threads,
        "source": source,
        "step": step,
        "params": model.num_params(),
        "max_new_tokens": max_new_tokens,
        "runs": runs,
        "canonical_prompt": canonical,
        "canonical_prompt_chars": len(canonical),
        "canonical_ctx_tokens": len(ctx),
        "cached": cached_agg,
        "uncached": uncached_agg,
        "cache_speedup_steady": speedup,
        "cpu": {"cores_busy": cores_busy, "pct_one_core": cores_busy * 100,
                "cpu_s": cpu_s, "wall_s": wall, "burst_generations": burst},
        "ram_kb": {  # every field in kB, so they compare directly
            "torch_baseline_rss": _TORCH_BASELINE_RSS_KB,
            "model_idle_rss": rss_model_idle,
            "peak_rss": rss_peak,
            "param_footprint": model.num_params() * 4 // 1024,  # float32 weights
        },
        "ttft_by_prompt_len": ttft_by_len,
    }


# --- cold vs warm ---------------------------------------------------------


def run_cold_warm(checkpoint_arg: str, settings_env: dict, max_new_tokens: int, threads: int) -> dict:
    """Cold = a fresh process: interpreter + torch import + model load + first
    generation. Warm = steady state inside that same process. We spawn a child so
    "cold" genuinely includes process start, then read its self-reported split."""
    child = subprocess.run(
        [sys.executable, __file__, "_coldwarm",
         "--checkpoint", checkpoint_arg, "--tokens", str(max_new_tokens),
         "--threads", str(threads)],
        capture_output=True, text=True, env={**os.environ, **settings_env},
    )
    if child.returncode != 0:
        return {"error": child.stderr.strip()[-500:]}
    return json.loads(child.stdout.strip().splitlines()[-1])


def _coldwarm_main(settings: Settings, checkpoint: Path | None, max_new_tokens: int, threads: int) -> None:
    """Runs in the freshly-spawned child. `_SCRIPT_START` was taken before torch
    was imported, so the cold number includes torch's ~1s import."""
    proc_start = _SCRIPT_START
    configure_cpu(threads)
    t_load0 = time.perf_counter()
    model, source, _ = load_model(settings, checkpoint)
    load_s = time.perf_counter() - t_load0
    canonical, _ = pick_prompts(settings)
    ctx = text_context(canonical, model.config.block_size)

    t_first0 = time.perf_counter()
    first = decode_cached(model, ctx, max_new_tokens)
    first_gen_s = time.perf_counter() - t_first0
    cold_total_s = time.perf_counter() - proc_start

    # Warm: a handful after warmup, for contrast in the same process.
    for _ in range(3):
        decode_cached(model, ctx, max_new_tokens)
    warm = [decode_cached(model, ctx, max_new_tokens) for _ in range(20)]

    print(json.dumps({
        "source": source,
        "process_start_to_ready_s": cold_total_s,
        "torch_import_s": _TORCH_IMPORT_S,
        "model_load_s": load_s,
        "first_gen_s": first_gen_s,
        "first_gen_e2e_tps": first.e2e_tps,
        "warm_e2e_tps": _summary([w.e2e_tps for w in warm]),
        "warm_total_ms": _summary([w.total_s * 1000 for w in warm]),
    }))


# --- realistic bot reply latency ------------------------------------------


def run_bot_latency(settings: Settings, checkpoint: Path | None, threads: int, runs: int) -> dict:
    """The actual reply path: best_continuation(n=best_of, temperature, top_k).
    This is what a Discord user waits for, sampling and candidate-scoring
    included."""
    configure_cpu(threads)
    model, _, _ = load_model(settings, checkpoint)
    canonical, _ = pick_prompts(settings)
    gen = torch.Generator()
    latencies = []
    # warmup
    for _ in range(3):
        best_continuation(model, canonical, n=max(1, settings.best_of),
                          max_new_tokens=settings.max_new_tokens,
                          temperature=settings.temperature, top_k=settings.top_k, generator=gen)
    for i in range(runs):
        gen.manual_seed(1234 + i)
        t0 = time.perf_counter()
        best_continuation(model, canonical, n=max(1, settings.best_of),
                          max_new_tokens=settings.max_new_tokens,
                          temperature=settings.temperature, top_k=settings.top_k, generator=gen)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "best_of": settings.best_of,
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "max_new_tokens": settings.max_new_tokens,
        "reply_latency_ms": _summary(latencies),
    }


# --- training: a measured pretrain run -------------------------------------


def run_train(settings: Settings, threads: int) -> dict:
    """Run `babble train --force` as a low-priority child (exactly how the bot
    fires it) and watch its RSS and CPU from /proc while it runs."""
    env = {
        **os.environ,
        "BABBLE_DATA_DIR": str(settings.data_dir),
        "BABBLE_CHECKPOINT_DIR": str(settings.checkpoint_dir),
        "BABBLE_TRAIN_THREADS": str(threads),
    }
    peak_rss = {"kb": 0}
    stop = threading.Event()

    proc = subprocess.Popen(
        [sys.executable, "-m", "babble", "train", "--force", "--seed", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )

    def monitor() -> None:
        last_ticks = 0
        while not stop.is_set():
            rss = _rss_kb("VmRSS", pid=proc.pid)
            if rss:
                peak_rss["kb"] = max(peak_rss["kb"], rss)
            t = _cpu_ticks(pid=proc.pid)
            if t is not None:
                last_ticks = t
            monitor.last_ticks = last_ticks  # type: ignore[attr-defined]
            time.sleep(0.02)

    monitor.last_ticks = 0  # type: ignore[attr-defined]
    mon = threading.Thread(target=monitor, daemon=True)
    t0 = time.perf_counter()
    mon.start()
    out, _ = proc.communicate()
    wall = time.perf_counter() - t0
    stop.set()
    mon.join(timeout=1)

    cpu_s = monitor.last_ticks / CLK_TCK  # type: ignore[attr-defined]
    steps = settings.train_steps
    # Parse the reported final step out of the CLI line if present.
    reported = out.strip().splitlines()[-1] if out.strip() else ""
    return {
        "threads": threads,
        "train_steps": steps,
        "wall_s": wall,
        "steps_per_s": steps / wall if wall > 0 else None,
        "peak_rss_kb": peak_rss["kb"],
        "cpu_s": cpu_s,
        "cores_busy": cpu_s / wall if wall > 0 else None,
        "cli_line": reported,
    }


# --- orchestration --------------------------------------------------------


def _spawn_infer(checkpoint_arg: str, settings_env: dict, threads: int, runs: int,
                 warmup: int, max_new_tokens: int) -> dict:
    child = subprocess.run(
        [sys.executable, __file__, "infer", "--threads", str(threads),
         "--runs", str(runs), "--warmup", str(warmup), "--tokens", str(max_new_tokens),
         "--checkpoint", checkpoint_arg, "--json"],
        capture_output=True, text=True, env={**os.environ, **settings_env},
    )
    if child.returncode != 0:
        return {"threads": threads, "error": child.stderr.strip()[-800:]}
    return json.loads(child.stdout.strip().splitlines()[-1])


def _settings_env(settings: Settings) -> dict:
    return {
        "BABBLE_DATA_DIR": str(settings.data_dir),
        "BABBLE_CHECKPOINT_DIR": str(settings.checkpoint_dir),
    }


def live_bot_rss() -> dict | None:
    """Cross-reference: the live bot's RSS from /proc, read-only. Never touches
    the process."""
    try:
        pid = subprocess.run(
            ["systemctl", "--user", "show", "babble-bot", "-p", "MainPID", "--value"],
            capture_output=True, text=True,
        ).stdout.strip()
        if not pid or pid == "0":
            return None
        return {"pid": int(pid), "vmrss_kb": _rss_kb("VmRSS", pid=int(pid)),
                "vmhwm_kb": _rss_kb("VmHWM", pid=int(pid))}
    except Exception:
        return None


def run_all(settings: Settings, checkpoint_arg: str, threads_sweep: list[int],
            runs: int, warmup: int, max_new_tokens: int, default_threads: int) -> dict:
    env = _settings_env(settings)
    print(f"# babble inference benchmark — {len(threads_sweep)} thread counts, "
          f"{runs} runs each, {max_new_tokens} tokens/gen", file=sys.stderr, flush=True)

    sweep = {}
    for t in threads_sweep:
        print(f"  thread sweep: {t} thread(s)…", file=sys.stderr, flush=True)
        sweep[t] = _spawn_infer(checkpoint_arg, env, t, runs, warmup, max_new_tokens)

    print(f"  cold vs warm ({default_threads} threads)…", file=sys.stderr, flush=True)
    cold_warm = run_cold_warm(checkpoint_arg, env, max_new_tokens, default_threads)

    print(f"  realistic bot reply latency ({default_threads} threads)…", file=sys.stderr, flush=True)
    bot = run_bot_latency(settings, _ckpt_path(checkpoint_arg), default_threads, max(10, runs // 2))

    print("  training…", file=sys.stderr, flush=True)
    train = run_train(settings, settings.train_threads)

    return {
        "machine": _machine_info(),
        "commit": _git_commit(settings),
        "config": {
            "max_new_tokens": max_new_tokens,
            "runs_per_config": runs,
            "warmup": warmup,
            "default_threads": default_threads,
            "threads_sweep": threads_sweep,
        },
        "thread_sweep": sweep,
        "cold_warm": cold_warm,
        "bot_reply": bot,
        "training": train,
        "live_bot_rss": live_bot_rss(),
    }


# --- environment / provenance ---------------------------------------------


def _machine_info() -> dict:
    model_name = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_total_kb = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    return {
        "cpu": model_name,
        "logical_cpus": NPROC,
        "mem_total_kb": mem_total_kb,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "torch_default_threads": torch.get_num_threads(),
    }


def _git_commit(settings: Settings) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Settings.from_env().checkpoint_dir.parent,
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --- CLI ------------------------------------------------------------------


def _ckpt_path(arg: str) -> Path | None:
    return None if arg in ("", "auto") else Path(arg)


def _base_settings() -> Settings:
    return Settings.from_env()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench_inference", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default="auto",
                        help="checkpoint to load (default: auto -> latest.pt, or random)")
    common.add_argument("--threads", type=int, default=torch.get_num_threads())
    common.add_argument("--runs", type=int, default=40)
    common.add_argument("--warmup", type=int, default=3)
    common.add_argument("--tokens", type=int, default=96, help="max_new_tokens per generation")

    p_infer = sub.add_parser("infer", parents=[common], help="inference numbers at one thread count")
    p_infer.add_argument("--json", action="store_true", help="emit a JSON blob (thread-sweep worker mode)")

    sub.add_parser("_coldwarm", parents=[common], help="internal: cold-vs-warm child")

    p_train = sub.add_parser("train", help="measure one pretrain run")
    p_train.add_argument("--threads", type=int, default=None)

    p_all = sub.add_parser("all", parents=[common], help="the whole report")
    p_all.add_argument("--sweep", default="1,2,4,8", help="comma-separated thread counts")
    p_all.add_argument("--json-out", type=Path, default=None, help="also write raw JSON here")

    args = parser.parse_args(argv)
    settings = _base_settings()

    if args.command == "infer":
        result = run_infer(settings, _ckpt_path(args.checkpoint), threads=args.threads,
                           runs=args.runs, warmup=args.warmup, max_new_tokens=args.tokens)
        if args.json:
            print(json.dumps(result))
        else:
            _print_infer(result)
        return 0

    if args.command == "_coldwarm":
        _coldwarm_main(settings, _ckpt_path(args.checkpoint), args.tokens, args.threads)
        return 0

    if args.command == "train":
        threads = args.threads if args.threads is not None else settings.train_threads
        print(json.dumps(run_train(settings, threads), indent=2))
        return 0

    if args.command == "all":
        sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
        result = run_all(settings, args.checkpoint, sweep, args.runs, args.warmup,
                         args.tokens, args.threads)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2))
            print(f"wrote {args.json_out}", file=sys.stderr)
        _print_all(result)
        return 0

    parser.print_help()
    return 1


def _fmt(summ: dict, key: str = "median", nd: int = 1) -> str:
    v = summ.get(key)
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "n/a"


def _print_infer(r: dict) -> None:
    print(f"threads={r['threads']} source={r['source']} params={r['params']:,} "
          f"tokens={r['max_new_tokens']} runs={r['runs']}")
    print(f"  cached   TTFT {_fmt(r['cached']['ttft_ms'])}ms  "
          f"steady {_fmt(r['cached']['steady_tps'])} tps  e2e {_fmt(r['cached']['e2e_tps'])} tps")
    print(f"  uncached TTFT {_fmt(r['uncached']['ttft_ms'])}ms  "
          f"steady {_fmt(r['uncached']['steady_tps'])} tps  e2e {_fmt(r['uncached']['e2e_tps'])} tps")
    sp = r.get("cache_speedup_steady")
    print(f"  cache speedup (steady): {sp:.2f}x" if sp else "  cache speedup: n/a")
    print(f"  cpu: {r['cpu']['cores_busy']:.2f} cores busy  "
          f"ram: model-idle {r['ram_kb']['model_idle_rss']/1024:.0f}MB "
          f"peak {r['ram_kb']['peak_rss']/1024:.0f}MB "
          f"params {r['ram_kb']['param_footprint']/1024:.1f}MB")


def _print_all(r: dict) -> None:
    m = r["machine"]
    print(f"\n=== babble benchmark @ {r['commit'][:12]} ===")
    print(f"{m['cpu']} · {m['logical_cpus']} logical CPUs · "
          f"{(m['mem_total_kb'] or 0)/1024/1024:.0f}GB · torch {m['torch']} · py {m['python']}")
    print(f"\nthread sweep (cached decode, {r['config']['max_new_tokens']} tokens, "
          f"{r['config']['runs_per_config']} runs):")
    print(f"  {'thr':>3} {'TTFT ms':>9} {'steady tps':>11} {'e2e tps':>9} "
          f"{'cores':>6} {'peakRAM MB':>11}")
    for t, d in r["thread_sweep"].items():
        if "error" in d:
            print(f"  {t:>3}  ERROR: {d['error'][:60]}")
            continue
        print(f"  {int(t):>3} {_fmt(d['cached']['ttft_ms']):>9} "
              f"{_fmt(d['cached']['steady_tps']):>11} {_fmt(d['cached']['e2e_tps']):>9} "
              f"{d['cpu']['cores_busy']:>6.2f} {d['ram_kb']['peak_rss']/1024:>11.0f}")
    cw = r["cold_warm"]
    if "error" not in cw:
        print(f"\ncold vs warm: restart→ready {cw['process_start_to_ready_s']*1000:.0f}ms "
              f"(torch import {cw.get('torch_import_s', 0)*1000:.0f}ms, load {cw['model_load_s']*1000:.0f}ms, "
              f"first gen {cw['first_gen_s']*1000:.0f}ms) | warm e2e {_fmt(cw['warm_e2e_tps'])} tps")
    bot = r["bot_reply"]
    print(f"\nrealistic reply (best_of={bot['best_of']}, temp {bot['temperature']}, "
          f"top_k {bot['top_k']}, {bot['max_new_tokens']} tok): "
          f"median {_fmt(bot['reply_latency_ms'], nd=0)}ms "
          f"(p25 {_fmt(bot['reply_latency_ms'],'p25',0)}–p75 {_fmt(bot['reply_latency_ms'],'p75',0)})")
    tr = r["training"]
    print(f"\ntraining ({tr['threads']} threads, {tr['train_steps']} steps): "
          f"{tr['wall_s']:.1f}s wall, {tr['steps_per_s']:.1f} steps/s, "
          f"peak {tr['peak_rss_kb']/1024:.0f}MB, {tr['cores_busy']:.2f} cores")
    lb = r.get("live_bot_rss")
    if lb:
        print(f"\nlive bot (pid {lb['pid']}, untouched): VmRSS {lb['vmrss_kb']/1024:.0f}MB "
              f"VmHWM {lb['vmhwm_kb']/1024:.0f}MB")


if __name__ == "__main__":
    raise SystemExit(main())
