"""CPU-first model path: KV cache, oneDNN knobs, and decode parity."""

from __future__ import annotations

import torch

import babble.generate as generate
from babble.cpu_runtime import (
    configure_cpu,
    force_cpu_device,
    maybe_compile,
    model_state_dict,
    uncompiled,
)
from babble.generate import continue_many, continue_text, sample
from babble.model import Babbler, ModelConfig


TINY = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=64)


def test_configure_cpu_forces_cpu_and_is_idempotent():
    first = configure_cpu(threads=2)
    second = configure_cpu(threads=2)

    assert force_cpu_device().type == "cpu"
    assert first["device"] == "cpu"
    assert second["device"] == "cpu"
    assert torch.get_num_threads() == 2
    # Second call must still report the oneDNN / denormal flags, not None.
    assert "mkldnn" in second
    assert second.get("flush_denormal") is not None


def test_kv_cache_prefill_plus_decode_matches_full_forward():
    """Cached single-token steps must agree with a full no-cache forward."""
    torch.manual_seed(0)
    model = Babbler(TINY)
    model.eval()

    tokens = torch.randint(0, TINY.vocab_size, (2, 8), dtype=torch.long)
    with torch.inference_mode():
        full = model(tokens)

        cache = model.new_cache(batch_size=2, max_len=8)
        pref = model(tokens[:, :5], cache=cache)
        assert cache.length == 5
        assert cache.max_len == 8
        rest = model(tokens[:, 5:], cache=cache)
        cached = torch.cat([pref, rest], dim=1)

    assert cached.shape == full.shape
    assert torch.allclose(cached, full, atol=1e-5, rtol=1e-5)


def test_cached_matches_uncached_continue_text(monkeypatch):
    """Same seed + same prompt must yield identical text with or without KV cache."""
    model = Babbler(TINY)
    model.eval()
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        cached = continue_text(
            model, "hi", max_new_tokens=24, temperature=0.9, top_k=40, generator=g
        )
        monkeypatch.setattr(generate, "_can_cache", lambda *a, **k: False)
        g = torch.Generator().manual_seed(seed)
        uncached = continue_text(
            model, "hi", max_new_tokens=24, temperature=0.9, top_k=40, generator=g
        )
        monkeypatch.undo()
        assert cached == uncached


def test_cached_matches_uncached_continue_many(monkeypatch):
    """Batched decode has its own finished-mask bookkeeping — pin parity too."""
    model = Babbler(TINY)
    model.eval()
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        cached = continue_many(
            model, "ab", 3, max_new_tokens=16, temperature=0.9, top_k=40, generator=g
        )
        monkeypatch.setattr(generate, "_can_cache", lambda *a, **k: False)
        g = torch.Generator().manual_seed(seed)
        uncached = continue_many(
            model, "ab", 3, max_new_tokens=16, temperature=0.9, top_k=40, generator=g
        )
        monkeypatch.undo()
        assert cached == uncached


def test_overflow_path_is_deterministic():
    """When context + max_new exceeds block_size, the uncached fallback stays greedy-stable."""
    torch.manual_seed(3)
    model = Babbler(TINY)
    model.eval()
    long_prefix = "x" * 50
    overflow = continue_text(
        model, long_prefix, max_new_tokens=32, temperature=0.0, top_k=40
    )
    overflow2 = continue_text(
        model, long_prefix, max_new_tokens=32, temperature=0.0, top_k=40
    )
    assert overflow == overflow2


def test_batched_overflow_matches_uncached_and_is_not_truncated(monkeypatch):
    """best_of's batched path must keep generating after the KV window fills.

    The live bot uses continue_many (best_of=4). Truncating when
    context+max_new > block_size was a silent quality/length regression.
    """
    torch.manual_seed(3)
    model = Babbler(TINY)
    model.eval()
    long_prefix = "x" * 50
    kwargs = dict(max_new_tokens=32, temperature=0.0, top_k=40)
    single = continue_text(model, long_prefix, **kwargs)
    batched = continue_many(model, long_prefix, 2, **kwargs)
    monkeypatch.setattr(generate, "_can_cache", lambda *a, **k: False)
    uncached = continue_many(model, long_prefix, 2, **kwargs)
    monkeypatch.undo()
    assert batched[0] == batched[1] == single == uncached[0]


def test_batched_cache_decode_runs_and_returns_n():
    torch.manual_seed(1)
    model = Babbler(TINY)
    outs = continue_many(model, "ab", 3, max_new_tokens=6, temperature=0.0, top_k=8)
    assert len(outs) == 3
    assert all(isinstance(o, str) for o in outs)
    assert len(set(outs)) == 1  # greedy + shared prompt => identical


def test_dropout_zero_uses_identity():
    model = Babbler(TINY)
    assert type(model.drop).__name__ == "Identity"
    assert type(model.blocks[0].mlp[3]).__name__ == "Identity"


def test_gelu_is_tanh_approximate_for_cpu():
    gelu = Babbler(TINY).blocks[0].mlp[1]
    assert gelu.approximate == "tanh"


def test_maybe_compile_off_by_default_returns_same_module():
    model = Babbler(TINY)
    assert maybe_compile(model, enabled=False) is model


def test_model_state_dict_unwraps_compiled_prefix():
    """Compiled checkpoints must save plain Babbler keys, never `_orig_mod.*`."""
    plain = Babbler(TINY)
    plain_keys = set(plain.state_dict())
    compiled = maybe_compile(plain, enabled=True)
    if compiled is plain:
        # Inductor unavailable in this environment — still assert the helper.
        assert set(model_state_dict(plain)) == plain_keys
        return
    wrapped_keys = set(compiled.state_dict())
    assert all(k.startswith("_orig_mod.") for k in wrapped_keys)
    assert set(model_state_dict(compiled)) == plain_keys
    # Round-trip into a fresh Babbler the way resume does.
    fresh = Babbler(TINY)
    fresh.load_state_dict(model_state_dict(compiled))
    assert uncompiled(compiled) is not compiled


def test_pair_sample_still_works_on_cpu_path():
    text = sample(Babbler(TINY), "hi", max_new_tokens=4, temperature=0.0)
    assert isinstance(text, str)


def test_dynamic_int8_still_decodes():
    from babble.cpu_runtime import quantize_dynamic_linears

    torch.manual_seed(0)
    model = Babbler(TINY)
    model.eval()
    q = quantize_dynamic_linears(model)
    text = continue_text(q, "hi", max_new_tokens=8, temperature=0.0)
    assert isinstance(text, str)
    assert len(text) > 0


def test_can_cache_when_new_tokens_would_overflow_block():
    """A long max_new_tokens must not disable the cache for a short prompt."""
    model = Babbler(TINY)
    assert generate._can_cache(model, context_len=8, max_new_tokens=10_000)
    assert not generate._can_cache(model, context_len=TINY.block_size, max_new_tokens=1)

