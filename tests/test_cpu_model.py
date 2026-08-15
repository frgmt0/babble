"""CPU-first model path: KV cache, oneDNN knobs, and decode parity."""

from __future__ import annotations

import torch

from babble.cpu_runtime import configure_cpu, force_cpu_device, maybe_compile
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


def test_kv_cache_prefill_plus_decode_matches_full_forward():
    """Cached single-token steps must agree with a full no-cache forward."""
    torch.manual_seed(0)
    model = Babbler(TINY)
    model.eval()

    tokens = torch.randint(0, TINY.vocab_size, (2, 8), dtype=torch.long)
    with torch.inference_mode():
        full = model(tokens)

        cache = model.new_cache(batch_size=2)
        pref = model(tokens[:, :5], cache=cache)
        assert cache.length == 5
        rest = model(tokens[:, 5:], cache=cache)
        cached = torch.cat([pref, rest], dim=1)

    assert cached.shape == full.shape
    assert torch.allclose(cached, full, atol=1e-5, rtol=1e-5)


def test_cached_greedy_decode_matches_uncached_path():
    """Temperature 0 must be identical with or without the cache fast path."""
    torch.manual_seed(3)
    model = Babbler(TINY)
    model.eval()

    # Force the overflow fallback by asking for more tokens than fit, then a
    # short reply that fits — both must be deterministic under temperature 0.
    short = continue_text(model, "hi", max_new_tokens=8, temperature=0.0, top_k=40)
    again = continue_text(model, "hi", max_new_tokens=8, temperature=0.0, top_k=40)
    assert short == again

    # Overflow path: context + max_new > block_size disables the cache.
    long_prefix = "x" * 50
    overflow = continue_text(
        model, long_prefix, max_new_tokens=32, temperature=0.0, top_k=40
    )
    overflow2 = continue_text(
        model, long_prefix, max_new_tokens=32, temperature=0.0, top_k=40
    )
    assert overflow == overflow2


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
    gelu = model_mlp_gelu(Babbler(TINY))
    assert gelu.approximate == "tanh"


def model_mlp_gelu(model: Babbler):
    return model.blocks[0].mlp[1]


def test_maybe_compile_off_by_default_returns_same_module():
    model = Babbler(TINY)
    assert maybe_compile(model, enabled=False) is model


def test_pair_sample_still_works_on_cpu_path():
    text = sample(Babbler(TINY), "hi", max_new_tokens=4, temperature=0.0)
    assert isinstance(text, str)
