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
import torch.nn as nn


_CONFIGURED = False
# Last oneDNN / denormal choices, so later configure_cpu calls can still report
# them in logs even though the heavyweight switches only run once.
_CPU_FLAGS: dict[str, Any] = {}


def force_cpu_device() -> torch.device:
    """Always CPU. Babble is built against the CPU torch wheel on purpose."""
    return torch.device("cpu")


def configure_cpu(threads: int | None = None) -> dict[str, Any]:
    """Make torch prefer the fast CPU kernels for this process.

    Safe to call more than once; the heavyweight switches run only on the first
    call so a bot and a trainer in the same interpreter do not fight. Returns
    the settings that actually stuck, for logs.

    Thread count is controlled only via `torch.set_num_threads` — OpenMP/MKL
    env vars are read at runtime init (already past by the time torch is
    imported), so setting them here would be a no-op.
    """
    global _CONFIGURED

    report: dict[str, Any] = {"device": "cpu", **_CPU_FLAGS}

    # Torch defaults to CPU already, and every model/tensor boundary in babble
    # uses an explicit CPU device. Installing the equivalent global
    # ``set_default_device("cpu")`` mode adds a Python dispatch hook to every
    # torch API call; remove a hook left by an earlier call instead.
    if hasattr(torch, "set_default_device"):
        try:
            torch.set_default_device(None)
            report["default_device"] = "cpu_native"
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
        _CPU_FLAGS["mkldnn"] = bool(torch.backends.mkldnn.enabled)

    # Flushing denormals avoids the classic CPU slow-path on near-zero grads.
    try:
        torch.set_flush_denormal(True)
        _CPU_FLAGS["flush_denormal"] = True
    except Exception:
        _CPU_FLAGS["flush_denormal"] = False

    report.update(_CPU_FLAGS)
    _CONFIGURED = True
    report["configured"] = True
    return report


def uncompiled(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying `nn.Module` behind a `torch.compile` wrapper, if any.

    `OptimizedModule.state_dict()` prefixes every key with `_orig_mod.`, which
    would make every checkpoint unloadable by a plain `Babbler`. Always save
    (and load into) the unwrapped module.
    """
    return getattr(model, "_orig_mod", model)


def model_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """Checkpoint-safe weights: never the compiled wrapper's prefixed keys."""
    return uncompiled(model).state_dict()


def maybe_compile(model: torch.nn.Module, *, enabled: bool | None = None) -> torch.nn.Module:
    """Optionally `torch.compile` the module for CPU inductor speedups.

    Off by default: compile time dominates a short training run, and the
    Discord bot wants the first reply fast. Set `BABBLE_TORCH_COMPILE=1` (or
    pass `enabled=True`) when doing a long one.

    Compile is lazy: this returns an `OptimizedModule` immediately and any
    inductor failure surfaces on the first forward, not here. Pair every save
    with `model_state_dict` / `uncompiled` so checkpoints stay plain-Babbler.
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
    return torch.compile(model, backend="inductor", mode="reduce-overhead")  # type: ignore[return-value]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def quantize_dynamic_linears(model: nn.Module) -> nn.Module:
    """Dynamic int8 on every `nn.Linear`. Embeddings stay fp32.

    On a bandwidth-bound CPU decode, this cuts Linear weight traffic ~4x.
    The output head is a Linear too (even when its weight was tied to the
    token embedding at construction): `quantize_dynamic` needs a real
    `nn.Linear` with its own Parameter, so a tied `lm_head.weight` is cloned
    first. That clone is decode-only and is never written back to a
    checkpoint.

    Requires a CPU torch build with a quantized engine (fbgemm / qnnpack).
    """
    prior = None
    # Some macOS CPU wheels expose qnnpack but leave the selected engine at
    # ``none``. Choose that supported CPU engine only in the unusable state;
    # Linux x86/fbgemm selections are left untouched.
    if (
        getattr(torch.backends.quantized, "engine", "none") == "none"
        and "qnnpack" in getattr(torch.backends.quantized, "supported_engines", ())
    ):
        torch.backends.quantized.engine = "qnnpack"
    if hasattr(model, "num_params"):
        try:
            prior = int(model.num_params())
        except Exception:
            prior = None
    lm_head = getattr(model, "lm_head", None)
    tok_emb = getattr(model, "tok_emb", None)
    if (
        isinstance(lm_head, nn.Linear)
        and tok_emb is not None
        and lm_head.weight.data_ptr() == tok_emb.weight.data_ptr()
    ):
        lm_head.weight = nn.Parameter(tok_emb.weight.detach().clone())
    model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    if prior is not None:
        model._params_before_quant = prior
    return model


def prepare_for_cpu_infer(
    model: nn.Module,
    *,
    quantize: bool | None = None,
    compile_model: bool | None = None,
) -> nn.Module:
    """Serving-time transforms. Never used on the training path.

    Dynamic int8 is off by default (`BABBLE_QUANTIZE=1` turns it on): it is a
    real ~1.4-2x decode speedup, but it is not free — measured bits/char on
    the Discord val split moves from the fp32 number by a few thousandths,
    and this project's bar (set by the #26 repetition-quality work landing
    alongside this) is that serving quality must not regress by default.
    Opt in with `BABBLE_QUANTIZE=1` if the speed is worth that tradeoff for
    your deployment. `torch.compile` stays off: the inductor tax on first
    forward is tens of seconds for a chatbot that must answer immediately
    after load.
    """
    if quantize is None:
        quantize = _env_flag("BABBLE_QUANTIZE", False)
    if compile_model is None:
        compile_model = _env_flag("BABBLE_TORCH_COMPILE", False)
    model.eval()
    if quantize:
        model = quantize_dynamic_linears(model)
    if compile_model:
        model = maybe_compile(model, enabled=True)
    return model
