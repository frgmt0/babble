"""`reference.py check` entry points for the production lean runtime.

    python bench/extreme/reference.py check bench.extreme.leanserve_check:int8_full
    python bench/extreme/reference.py check bench.extreme.leanserve_check:int8_decode
    python bench/extreme/reference.py check bench.extreme.leanserve_check:fp32_full
    python bench/extreme/reference.py check bench.extreme.leanserve_check:fp32_decode

``*_full`` runs one prefill over the whole sequence (the path prompts take);
``*_decode`` feeds one token at a time through the KV cache (the path every
sampled token takes). In int8 mode the top-64 logits of every position are
exact fp32 (serving rescores every sampling candidate exactly).
"""

from __future__ import annotations

import torch

from babble.leanserve import LeanModel
from bench.extreme.reference import MODEL_DIR

_MODELS: dict[str, LeanModel] = {}


def _model(precision: str) -> LeanModel:
    if precision not in _MODELS:
        torch.set_num_threads(4)
        _MODELS[precision] = LeanModel(MODEL_DIR, precision=precision)
    return _MODELS[precision]


def int8_full(ids):
    return _model("int8").full_logits(ids)


def int8_decode(ids):
    return _model("int8").decode_logits(ids)


def fp32_full(ids):
    return _model("fp32").full_logits(ids)


def fp32_decode(ids):
    return _model("fp32").decode_logits(ids)
