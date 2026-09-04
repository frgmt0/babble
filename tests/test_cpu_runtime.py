from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from babble.cpu_runtime import quantize_dynamic_linears


def test_dynamic_quantization_selects_qnnpack_when_engine_is_none() -> None:
    if torch.backends.quantized.engine != "none":
        pytest.skip("runtime already selected a working quantized engine")
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        pytest.skip("this CPU build does not provide qnnpack")

    quantize_dynamic_linears(nn.Sequential(nn.Linear(4, 4)).eval())

    assert torch.backends.quantized.engine == "qnnpack"
