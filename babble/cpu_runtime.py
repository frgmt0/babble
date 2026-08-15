"""CPU-first runtime knobs for babble's small transformer.

The shipped model is ~3.3M parameters and is meant to train and reply on a
couple of CPU threads (see `trainer.be_polite`). These helpers keep torch on
the CPU path, turn on oneDNN/MKL-friendly settings, and avoid paying for CUDA
scaffolding that this project never uses.
"""

from __future__ import annotations

import os
from typing import Any

import torch


_CONFIGURED = False


def force_cpu_device() -> torch.device:
    """Always CPU. Babble is built against the CPU torch wheel on purpose."""
    return torch.device("cpu")


def configure_cpu(threads: int | None = None) -> dict[str, Any]:
    """Make torch prefer the fast CPU kernels for this process.

    Safe to call more than once; the heavyweight switches run only on the first
    call so a bot and a trainer in the same interpreter do not fight. Returns
    the settings that actually stuck, for logs.
    """
    global _CONFIGURED

    report: dict[str, Any] = {"device": "cpu"}

    # Never let an accidental CUDA build steal work — this project's uv source
    # pins the CPU wheel, but a system torch can still report cuda.is_available.
    if hasattr(torch, "set_default_device"):
        try:
            torch.set_default_device("cpu")
            report["default_device"] = "cpu"
        except Exception:
            report["default_device"] = "unchanged"

    if threads is not None:
        threads = max(1, int(threads))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        report["threads"] = threads
        report["interop_threads"] = 1

    if _CONFIGURED:
        return report

    # oneDNN / MKL-DNN: the win for our Linear + LayerNorm + SDPA stack on CPU.
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
        report["mkldnn"] = bool(torch.backends.mkldnn.enabled)

    # Flushing denormals avoids the classic CPU slow-path on near-zero grads.
    try:
        torch.set_flush_denormal(True)
        report["flush_denormal"] = True
    except Exception:
        report["flush_denormal"] = False

    # Keep matmul in float32; half/bfloat on CPU is usually slower for a model
    # this small, and we want stable loss curves more than a risky cast.
    if "OMP_NUM_THREADS" not in os.environ and threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    if "MKL_NUM_THREADS" not in os.environ and threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(threads)

    _CONFIGURED = True
    report["configured"] = True
    return report


def maybe_compile(model: torch.nn.Module, *, enabled: bool | None = None) -> torch.nn.Module:
    """Optionally `torch.compile` the module for CPU inductor speedups.

    Off by default: compile time dominates short voice-pass runs, and the
    Discord bot wants the first reply fast. Set `BABBLE_TORCH_COMPILE=1` (or
    pass `enabled=True`) when doing a long base pretrain.
    """
    if enabled is None:
        enabled = os.environ.get("BABBLE_TORCH_COMPILE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not enabled:
        return model
    try:
        return torch.compile(model, backend="inductor", mode="reduce-overhead")  # type: ignore[return-value]
    except Exception:
        return model
