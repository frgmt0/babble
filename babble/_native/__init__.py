"""Build-on-first-use loader for the native inference engine (``engine.cpp``).

The engine is plain C++ with a C ABI, compiled with the system ``g++`` into a
shared library and loaded with ctypes -- no setuptools extension, so a live
install updated by ``git pull --ff-only`` (``deploy/update-live.sh``) picks up
a new engine with no manual step: the first process to need it compiles it
(~5-10 s) and every later start loads the cached build.

Cache: ``$BABBLE_NATIVE_CACHE`` or ``$XDG_CACHE_HOME/babble/native`` (default
``~/.cache/babble/native``) -- deliberately outside the repo tree and outside
/tmp. A build is keyed by the source hash, the compiler's identity, the flags
and the CPU features it relies on, so an edited engine, a compiler upgrade or
a different CPU each get their own library and a stale one is never loaded.
Builds hold an exclusive file lock and publish the library with an atomic
rename, so concurrent first starts neither race nor load a half-written file.

Anything that stops the engine from being usable -- not x86-64 Linux, no
AVX2/FMA/F16C, no compiler, a failed build, a library that will not load --
raises `NativeUnavailable`; `hfserve.make_generator` turns that into a logged
fallback to the lean runtime.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SOURCE = Path(__file__).resolve().with_name("engine.cpp")
ABI_VERSION = 4
# -march=haswell (AVX2 + FMA + F16C) rather than -march=native: the same build
# key then means the same instructions and the same float results on any CPU
# that passes the feature check.
CXXFLAGS = (
    "-O3",
    "-march=haswell",
    "-mtune=haswell",
    "-std=c++17",
    "-fPIC",
    "-shared",
    "-pthread",
    "-Wall",
    "-Wno-unused-function",
)
# Everything -march=haswell may emit, so a CPU that passes cannot SIGILL.
REQUIRED_CPU_FLAGS = ("avx2", "fma", "f16c", "bmi1", "bmi2", "abm", "movbe")
BUILD_TIMEOUT_S = 600


class NativeUnavailable(RuntimeError):
    """The native engine cannot run here (CPU, compiler, build or model shape)."""


@dataclass(frozen=True)
class BuildInfo:
    path: Path
    key: str
    built: bool  # compiled by this call (False: loaded from the cache)
    build_s: float


def cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def check_cpu() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise NativeUnavailable(f"native engine needs x86-64 Linux, this is {platform.system()}/{platform.machine()}")
    missing = [f for f in REQUIRED_CPU_FLAGS if f not in cpu_flags()]
    if missing:
        raise NativeUnavailable(f"CPU lacks {', '.join(missing)} (native engine needs AVX2+FMA+F16C)")


def cache_dir() -> Path:
    env = os.environ.get("BABBLE_NATIVE_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
    return Path(base) / "babble" / "native"


def compiler() -> str:
    cxx = os.environ.get("CXX", "").strip() or "g++"
    path = shutil.which(cxx)
    if not path:
        raise NativeUnavailable(f"no C++ compiler ({cxx!r} not on PATH) to build the native engine")
    return path


def _compiler_id(cxx: str) -> str:
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeUnavailable(f"cannot run {cxx}: {exc}") from exc
    return f"{cxx}\n{out.stdout.strip()}"


def build_key(cxx: str) -> str:
    h = hashlib.sha256()
    h.update(SOURCE.read_bytes())
    h.update(_compiler_id(cxx).encode())
    h.update(" ".join(CXXFLAGS).encode())
    h.update(" ".join(sorted(f for f in REQUIRED_CPU_FLAGS if f in cpu_flags())).encode())
    h.update(f"abi{ABI_VERSION}".encode())
    return h.hexdigest()[:20]


def build(*, force: bool = False) -> BuildInfo:
    """Compile (or find) the engine library for this source/compiler/CPU."""
    check_cpu()
    cxx = compiler()
    key = build_key(cxx)
    root = cache_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NativeUnavailable(f"cannot create native cache dir {root}: {exc}") from exc
    lib = root / f"libbabble_native-{key}.so"
    if lib.exists() and not force:
        return BuildInfo(lib, key, False, 0.0)
    with open(root / f".build-{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # another process may be building this very key
        if lib.exists() and not force:
            return BuildInfo(lib, key, False, 0.0)
        tmp = root / f".libbabble_native-{key}.{os.getpid()}.tmp"
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [cxx, *CXXFLAGS, "-o", str(tmp), str(SOURCE)],
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            tmp.unlink(missing_ok=True)
            raise NativeUnavailable(f"native engine build failed: {exc}") from exc
        if proc.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
            raise NativeUnavailable("native engine build failed: " + " | ".join(tail))
        os.replace(tmp, lib)
        return BuildInfo(lib, key, True, time.perf_counter() - started)


_LIBS: dict[Path, ctypes.CDLL] = {}


class SampleParams(ctypes.Structure):
    _fields_ = [
        ("greedy", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("no_repeat_ngram", ctypes.c_int),
        ("eos_id", ctypes.c_int),
        ("stop_at_eos", ctypes.c_int),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
    ]


def _declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    vp, i32, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.eng_abi_version.restype = i32
    lib.eng_abi_version.argtypes = []
    lib.eng_create.restype = vp
    lib.eng_create.argtypes = [i32, i32, i32, i32, i32, i32, i32, i32, f32]
    lib.eng_destroy.restype = None
    lib.eng_destroy.argtypes = [vp]
    lib.eng_set_matrix.restype = i32
    lib.eng_set_matrix.argtypes = [vp, i32, i32, i32, vp, vp, i32, i32]
    lib.eng_set_matrix_q4.restype = i32
    lib.eng_set_matrix_q4.argtypes = [vp, i32, i32, i32, vp, vp, i32, i32, i32]
    lib.eng_set_head2.restype = i32
    lib.eng_set_head2.argtypes = [vp, i32, f32]
    lib.eng_head2_stats.restype = None
    lib.eng_head2_stats.argtypes = [vp, vp]
    lib.eng_set_vector.restype = i32
    lib.eng_set_vector.argtypes = [vp, i32, i32, vp]
    lib.eng_set_rope.restype = None
    lib.eng_set_rope.argtypes = [vp, vp, vp]
    lib.eng_kv_bytes.restype = ctypes.c_longlong
    lib.eng_kv_bytes.argtypes = [vp, i32]
    lib.eng_set_kv_type.restype = i32
    lib.eng_set_kv_type.argtypes = [vp, i32]
    lib.eng_kv_type.restype = i32
    lib.eng_kv_type.argtypes = [vp]
    lib.eng_forward.restype = i32
    lib.eng_forward.argtypes = [vp, vp, i32, i32, vp, i32, vp, vp]
    lib.eng_forward_incremental.restype = i32
    lib.eng_forward_incremental.argtypes = [vp, vp, i32, i32, vp]
    lib.eng_generate.restype = i32
    lib.eng_generate.argtypes = [
        vp, vp, i32, i32, vp, i32, vp,  # engine, prompt, T, start, kv_in, kv_in_len, kv_out
        i32, i32, ctypes.POINTER(SampleParams), ctypes.c_uint64,  # ns, max_new, params, seed
        vp, vp, vp, vp,  # out_tokens, counts, logprob, timing
    ]
    lib.eng_spec_begin.restype = vp
    lib.eng_spec_begin.argtypes = [
        vp, vp, i32, i32, vp, i32, vp,  # engine, prompt, T, start, kv_in, kv_in_len, kv_out
        i32, i32, i32, ctypes.POINTER(SampleParams), ctypes.c_uint64,  # ns, max_new, kmax, params, seed
        vp, vp,  # first_tokens, timing
    ]
    lib.eng_spec_step.restype = i32
    lib.eng_spec_step.argtypes = [vp, vp, vp, vp, vp]  # session, drafts, nd, out, nout
    lib.eng_spec_end.restype = None
    lib.eng_spec_end.argtypes = [vp, vp, vp]  # session, counts, logprob
    lib.eng_forward_verify.restype = i32
    lib.eng_forward_verify.argtypes = [vp, vp, i32, i32, i32, i32, vp]
    lib.eng_warp_probs.restype = i32
    lib.eng_warp_probs.argtypes = [vp, i32, vp, i32, ctypes.POINTER(SampleParams), vp]
    return lib


def load(*, force_build: bool = False) -> tuple[ctypes.CDLL, BuildInfo]:
    """Build if needed and load the engine library; `NativeUnavailable` on any failure."""
    info = build(force=force_build)
    lib = _LIBS.get(info.path)
    if lib is None:
        try:
            lib = _declare(ctypes.CDLL(str(info.path)))
            abi = lib.eng_abi_version()
        except (OSError, AttributeError) as exc:
            if info.built:
                raise NativeUnavailable(f"native engine library does not load: {exc}") from exc
            # A cached file that does not load (truncated by a crash, say): rebuild once.
            info.path.unlink(missing_ok=True)
            return load(force_build=True)
        if abi != ABI_VERSION:
            raise NativeUnavailable(f"native engine ABI {abi} != expected {ABI_VERSION}")
        _LIBS[info.path] = lib
    return lib, info


__all__ = [
    "ABI_VERSION",
    "BuildInfo",
    "NativeUnavailable",
    "SampleParams",
    "build",
    "cache_dir",
    "check_cpu",
    "load",
]
