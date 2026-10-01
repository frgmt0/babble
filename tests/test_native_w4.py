"""Opt-in native engine paths from track w4: int4 decode weights and the two-stage head.

* `BABBLE_NATIVE_W4` (int4 group-wise decode copies) is a quality trade on the
  real model; here it is checked for *kernel* exactness on a tiny snapshot
  whose int8 weights are exactly representable in int4 (so int4 decode must
  reproduce the int8 decode to float rounding).
* `BABBLE_NATIVE_HEAD2` (int4 screen + exact rescoring) must leave sampling
  untouched: same seeds give the same tokens and best-of scores.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from test_nativeserve import BOS, EOS, NATIVE_OK, VOCAB, WHY, _write_snapshot  # noqa: E402

from babble.leanserve import SamplingConfig  # noqa: E402
from babble.nativeserve import NativeEngine, NativeUnavailable, parse_head2, parse_w4, quantize_q4  # noqa: E402

needs_native = pytest.mark.skipif(not NATIVE_OK, reason=f"native engine unavailable: {WHY}")
G = 16


def _int4_exact(src: Path, out: Path) -> Path:
    """Rewrite every int8 matrix so that each 16-group holds q in [-7, 7] with a 7 in it."""
    import shutil

    from safetensors.torch import load_file, save_file

    shutil.copytree(src, out)
    packed = load_file(str(out / "model-int8.safetensors"))
    for name in list(packed):
        q = packed[name]
        if q.dtype != torch.int8:
            continue
        q4 = torch.clamp(torch.round(q.float() / 18), -7, 7)
        q4 = q4.reshape(q.shape[0], -1, G)
        q4[:, :, 0] = 7
        packed[name] = q4.reshape(q.shape).to(torch.int8).contiguous()
        s = packed[name + ".scale"].float() * 18
        packed[name + ".scale"] = torch.exp2(torch.round(torch.log2(s))).to(torch.bfloat16)
    save_file(packed, str(out / "model-int8.safetensors"))
    return out


@pytest.fixture(scope="module")
def snap4(tmp_path_factory) -> Path:
    base = _write_snapshot(tmp_path_factory.mktemp("tiny-w4") / "base")
    return _int4_exact(base, base.parent / "int4exact")


@pytest.fixture(scope="module")
def engines(snap4):
    if not NATIVE_OK:
        pytest.skip(WHY)
    plain = NativeEngine(snap4, threads=3, w4="", head2="")
    w4 = NativeEngine(snap4, threads=3, w4=f"all:{G}", head2="")
    yield plain, w4
    plain.close()
    w4.close()


def _ids(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return [BOS, *torch.randint(4, VOCAB, (n - 1,), generator=g).tolist()]


def test_quantize_q4_is_exact_on_int4_representable_rows() -> None:
    q = torch.randint(-7, 8, (32, 64), dtype=torch.int8)
    q.view(32, -1, G)[:, :, 0] = -7
    scale = torch.full((32,), 0.03125)
    q4, s4 = quantize_q4(q, scale, G)
    assert torch.equal(q4, q)
    assert torch.allclose(s4, torch.full_like(s4, 0.03125))


def test_parse_specs() -> None:
    assert parse_w4("") is None and parse_w4("off") is None
    kinds, g = parse_w4("noh:32")
    assert g == 32 and 7 not in kinds and 4 in kinds
    assert parse_w4("all")[1] == 64
    assert parse_head2("0") is None
    assert parse_head2("256") == (256, 0.5, 64)
    assert parse_head2("128:1.0:32") == (128, 1.0, 32)
    for bad in ("everything", "all:6", "all:x"):
        with pytest.raises(NativeUnavailable):
            parse_w4(bad)
    for bad in ("x", "0:1", "8:nan"):
        with pytest.raises(NativeUnavailable):
            parse_head2(bad)


@needs_native
@pytest.mark.parametrize("prefill", [1, 5])
def test_int4_decode_equals_int8_decode_when_representable(engines, prefill) -> None:
    plain, w4 = engines
    ids = _ids(40, 3)
    a = plain.decode_logits(ids, prefill=prefill)
    b = w4.decode_logits(ids, prefill=prefill)
    assert (a - b).abs().max() < 2e-4


@needs_native
def test_int4_prefill_stays_int8(engines) -> None:
    plain, w4 = engines
    ids = _ids(30, 4)
    assert torch.equal(plain.full_logits(ids), w4.full_logits(ids))


@needs_native
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_int4_best_of_n_matches_int8(engines, n) -> None:
    plain, w4 = engines
    cfg = SamplingConfig(temperature=0.7, top_k=10, top_p=0.95, repetition_penalty=1.15, no_repeat_ngram_size=3)
    ids = _ids(12, 5)
    a = plain.generate(ids, n=n, max_new=24, sampling=cfg, eos_id=EOS, seed=11, stop_at_eos=False)
    b = w4.generate(ids, n=n, max_new=24, sampling=cfg, eos_id=EOS, seed=11, stop_at_eos=False)
    assert [list(t) for t in a.tokens] == [list(t) for t in b.tokens]
    assert max(abs(x - y) for x, y in zip(a.mean_logprob, b.mean_logprob)) < 1e-4


@pytest.fixture(scope="module")
def head2_engines(tmp_path_factory):
    if not NATIVE_OK:
        pytest.skip(WHY)
    snap = _write_snapshot(tmp_path_factory.mktemp("tiny-head2"))
    plain = NativeEngine(snap, threads=3, w4="", head2="")
    two = NativeEngine(snap, threads=3, w4="", head2="8:0.5:16")
    yield plain, two
    plain.close()
    two.close()


@needs_native
@pytest.mark.parametrize("freq", [0.0, 0.12])
def test_head2_sampling_is_unchanged(head2_engines, freq) -> None:
    plain, two = head2_engines
    cfg = SamplingConfig(temperature=0.5, top_k=5, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4,
                         frequency_penalty=freq)
    for seed in range(6):
        ids = _ids(10 + seed, 20 + seed)
        a = plain.generate(ids, n=4, max_new=20, sampling=cfg, eos_id=EOS, seed=seed)
        b = two.generate(ids, n=4, max_new=20, sampling=cfg, eos_id=EOS, seed=seed)
        assert [list(t) for t in a.tokens] == [list(t) for t in b.tokens]
        assert a.best == b.best
        assert max(abs(x - y) for x, y in zip(a.mean_logprob, b.mean_logprob)) < 1e-5
    s = two.head2_stats()
    assert s["rows"] > 0 and s["candidates"] >= 8 * (s["rows"] - s["fallbacks"])


@needs_native
def test_head2_greedy_and_top_k_above_n_fall_back_exactly(head2_engines) -> None:
    plain, two = head2_engines
    ids = _ids(9, 31)
    greedy = SamplingConfig()
    a = plain.generate(ids, n=1, max_new=16, sampling=greedy, eos_id=EOS, seed=0, greedy=True, stop_at_eos=False)
    b = two.generate(ids, n=1, max_new=16, sampling=greedy, eos_id=EOS, seed=0, greedy=True, stop_at_eos=False)
    assert a.tokens == b.tokens
    before = two.head2_stats()["fallbacks"]
    wide = SamplingConfig(temperature=0.7, top_k=40, top_p=1.0)  # k > N: every row takes the exact head
    a = plain.generate(ids, n=2, max_new=8, sampling=wide, eos_id=EOS, seed=3, stop_at_eos=False)
    b = two.generate(ids, n=2, max_new=8, sampling=wide, eos_id=EOS, seed=3, stop_at_eos=False)
    assert [list(t) for t in a.tokens] == [list(t) for t in b.tokens]
    assert two.head2_stats()["fallbacks"] == before + 1 + 7 * 2  # shared first row, then 2 rows x 7 steps
