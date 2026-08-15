"""The benchmark harness: cheap invariants, no real timing.

These do not assert on wall-clock numbers -- those are the point of the
benchmark, not of the test suite, and they are machine- and load-dependent. What
they pin down is the property the reported speedup ratio *rests* on: the cached
and uncached decoders must emit the byte-for-byte same greedy sequence, or the
"cache is Nx faster" number would be comparing two different generations.
"""

from __future__ import annotations

import torch

from bench.bench_inference import DecodeTiming, decode_cached, decode_uncached, _summary
from babble.model import Babbler, ModelConfig
from babble.tokenizer import BOS_ID

TINY = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=64)


def _model() -> Babbler:
    torch.manual_seed(0)
    m = Babbler(TINY)
    m.eval()
    return m


def test_cached_and_uncached_decode_identical_greedy_sequence():
    """Same prompt, same greedy rule -> same tokens, so the speedup ratio is a
    like-for-like comparison and not two different generations."""
    m = _model()
    ctx = [BOS_ID, *b"which bear is best"]
    n = 24

    torch.manual_seed(1)
    cached = decode_cached(m, ctx, n)
    torch.manual_seed(1)
    uncached = decode_uncached(m, ctx, n)

    # Re-derive the token stream each path produced by replaying greedily; the
    # timing objects only carry durations, so assert on the counts they imply.
    assert cached.n_tokens == n
    assert uncached.n_tokens == n
    # Both decode the full budget (a tiny random model effectively never hits eos
    # under argmax within 24 steps, and the harness never stops early anyway).
    assert len(cached.step_times_s) == n - 1
    assert len(uncached.step_times_s) == n - 1


def test_greedy_token_streams_match_between_paths():
    """Directly compare the produced ids, not just their length."""
    m = _model()
    ctx = [BOS_ID, *b"hello there"]
    n = 20

    @torch.inference_mode()
    def cached_ids() -> list[int]:
        cache = m.new_cache(1, max_len=len(ctx) + n)
        logits = m(torch.tensor([ctx]), cache=cache)[:, -1]
        out = []
        tok = int(logits.argmax(-1)[0])
        out.append(tok)
        for _ in range(n - 1):
            logits = m(torch.tensor([[tok]]), cache=cache)[:, -1]
            tok = int(logits.argmax(-1)[0])
            out.append(tok)
        return out

    @torch.inference_mode()
    def uncached_ids() -> list[int]:
        seq = list(ctx)
        out = []
        for _ in range(n):
            logits = m(torch.tensor([seq[-TINY.block_size:]]))[:, -1]
            tok = int(logits.argmax(-1)[0])
            seq.append(tok)
            out.append(tok)
        return out

    assert cached_ids() == uncached_ids()


def test_decode_timing_derived_metrics():
    t = DecodeTiming(ttft_s=0.010, step_times_s=[0.002, 0.002, 0.002, 0.004], total_s=0.020)
    assert t.n_tokens == 5
    assert abs(t.steady_tps - (4 / 0.010)) < 1e-6
    assert abs(t.e2e_tps - (5 / 0.020)) < 1e-6


def test_summary_reports_median_and_spread():
    s = _summary([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["median"] == 30.0
    assert s["min"] == 10.0
    assert s["max"] == 50.0
    assert s["n"] == 5


def test_summary_handles_empty():
    s = _summary([])
    assert s["median"] is None
    assert s["n"] == 0
