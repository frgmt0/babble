"""`reference.py check` entry points for the production native runtime.

    python bench/extreme/reference.py check bench.extreme.nativeserve_check:full
    python bench/extreme/reference.py check bench.extreme.nativeserve_check:decode
    python bench/extreme/reference.py check bench.extreme.nativeserve_check:prefix
    python bench/extreme/reference.py check bench.extreme.nativeserve_check:verify

All four go through a `NativeGenerator` built by `hfserve.make_generator`
(BABBLE_HF_RUNTIME=native), i.e. the engine the bot would serve with:

* ``full``   -- one batched prefill over the whole sequence (the prompt path);
* ``decode`` -- a 1-token prefill, then token-by-token decode (every sampled token);
* ``prefix`` -- the first half is prefilled and exported as a prefix snapshot,
  then the rest is prefilled on top of the restored snapshot (the multi-turn
  prefix-KV-reuse path).
"""

from __future__ import annotations

import os

import torch

from bench.extreme.reference import MODEL_DIR

_GEN = None


def generator():
    global _GEN
    if _GEN is None:
        from babble.config import Settings
        from babble.hfserve import make_generator
        from babble.nativeserve import NativeGenerator

        s = Settings.for_root(MODEL_DIR.parent / ".native-check-unused")
        s.serve_backend = "hf"
        s.hf_runtime = "native"
        s.hf_model_dir = MODEL_DIR
        s.infer_threads = int(os.environ.get("NATIVE_THREADS", "4"))
        _GEN = make_generator(s)
        if not isinstance(_GEN, NativeGenerator):
            raise SystemExit("native runtime fell back; see the model.native_fallback reason on stderr")
    return _GEN


def full(ids):
    return generator().engine.full_logits(ids)


def decode(ids):
    return generator().engine.decode_logits(ids, prefill=1)


def prefix(ids):
    eng = generator().engine
    cut = max(1, len(ids) // 2)
    head, kv = eng.forward(ids[:cut], export=True)
    tail, _ = eng.forward(ids, start=cut, kv_in=kv, kv_in_len=cut)
    return torch.cat([head, tail])


def verify(ids):
    """Speculative-decoding verify path: 4-row chunks of one stream through the
    decode forward with per-row cache positions, each chunk first fed as junk
    and rolled back (a rejected draft), then fed for real."""
    return generator().engine.verify_logits(ids, prefill=1, chunk=4, junk=True)
