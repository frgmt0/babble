"""Gate entry points for track w4 against the *live* model (longctx-v1).

The repo fixture (`artifacts/perf-ref/ref.pt`) was built from multiturn-v1;
`w4_quant_study.py ref` rebuilt the same 5 fixture cases from longctx-v1 into
$W4_DIR/ref-longctx.pt. These functions run the native engine on longctx-v1
with whatever `BABBLE_NATIVE_W4` / `BABBLE_NATIVE_HEAD2` is set in the env.

    python bench/extreme/w4_check.py decode|full|prefix
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bench.extreme.reference as ref  # noqa: E402

MODEL_DIR = Path(os.environ.get("W4_MODEL_DIR", "/home/jason/projects/babble/artifacts/hf-booper-longctx-v1"))
ref.REF_PATH = Path(os.environ.get("W4_DIR", "/tmp/maxperf/w4")) / "ref-longctx.pt"
_ENG = None


def engine():
    global _ENG
    if _ENG is None:
        from babble.nativeserve import NativeEngine

        _ENG = NativeEngine(MODEL_DIR, threads=int(os.environ.get("NATIVE_THREADS", "4")))
    return _ENG


def full(ids):
    return engine().full_logits(ids)


def decode(ids):
    return engine().decode_logits(ids, prefill=1)


if __name__ == "__main__":
    print(os.environ.get("BABBLE_NATIVE_W4", ""), sys.argv[1], ref.compare(globals()[sys.argv[1]]))
