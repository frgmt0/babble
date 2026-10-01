"""The native C++ hf runtime (`BABBLE_HF_RUNTIME=native`).

No network. Most tests use a tiny randomly-initialised Mixtral written in the
same int8 safetensors layout as the live snapshot (the engine takes its
geometry at runtime, so the same build serves both). The engine is compiled on
first use into the normal build cache (~/.cache/babble/native, or
$BABBLE_NATIVE_CACHE); tests that exercise the build machinery itself use a
throwaway cache and a trivial source file. Real-artifact checks skip cleanly
when artifacts/hf-booper-multiturn-v1 is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from babble import _native  # noqa: E402
from babble.hfserve import _CandidateTracker, _load_int8, make_generator  # noqa: E402
from babble.leanserve import LeanGenerator, SamplingConfig  # noqa: E402

try:
    _native.check_cpu()
    _native.compiler()
    NATIVE_OK, WHY = True, ""
except _native.NativeUnavailable as exc:  # pragma: no cover - depends on the box
    NATIVE_OK, WHY = False, str(exc)

needs_native = pytest.mark.skipif(not NATIVE_OK, reason=f"native engine unavailable: {WHY}")

from babble.nativeserve import NativeEngine, NativeGenerator, NativeUnavailable  # noqa: E402

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"]
WORDS = [f"w{i}" for i in range(60)]
VOCAB = len(SPECIALS) + len(WORDS)  # 64
PAD, BOS, SEP, EOS = range(4)

ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL = ROOT / "artifacts" / "hf-booper-multiturn-v1"
has_real = pytest.mark.skipif(
    not (REAL_MODEL / "model-int8.safetensors").exists(), reason="real snapshot not present under artifacts/"
)


# --------------------------------------------------------------------------
# tiny snapshot
# --------------------------------------------------------------------------


def _quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (w.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8).to(torch.bfloat16)
    q = torch.round(w / scale.float()).clamp(-127, 127).to(torch.int8)
    return q, scale


def _write_snapshot(out: Path, *, hidden: int = 64, heads: int = 4, inter: int = 96, seed: int = 1234) -> Path:
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(seed)
    config = MixtralConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        num_local_experts=3,
        num_experts_per_tok=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        initializer_range=0.15,
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )
    model = MixtralForCausalLM(config).eval()
    state = {k: v.detach().float() for k, v in model.state_dict().items()}
    packed: dict[str, torch.Tensor] = {}

    def put(name: str, w: torch.Tensor) -> None:
        if w.ndim == 2:
            packed[name], packed[name + ".scale"] = _quantize(w)
        else:
            packed[name] = w.to(torch.bfloat16)

    for name, w in state.items():
        if name == "lm_head.weight" or ".mlp." in name or ".block_sparse_moe." in name:
            continue
        put(name, w)
    put("lm_head.weight", state["model.embed_tokens.weight"])
    for layer in range(config.num_hidden_layers):
        new = f"model.layers.{layer}.block_sparse_moe"
        fused = f"model.layers.{layer}.mlp"
        if fused + ".experts.gate_up_proj" in state:
            put(new + ".gate.weight", state[fused + ".gate.weight"])
            gate_up, down = state[fused + ".experts.gate_up_proj"], state[fused + ".experts.down_proj"]
            for e in range(config.num_local_experts):
                put(f"{new}.experts.{e}.w1.weight", gate_up[e, :inter])
                put(f"{new}.experts.{e}.w3.weight", gate_up[e, inter:])
                put(f"{new}.experts.{e}.w2.weight", down[e])
        else:
            put(new + ".gate.weight", state[new + ".gate.weight"])
            for e in range(config.num_local_experts):
                for w in ("w1", "w2", "w3"):
                    put(f"{new}.experts.{e}.{w}.weight", state[f"{new}.experts.{e}.{w}.weight"])
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in packed.items()}, str(out / "model-int8.safetensors"))
    config.save_pretrained(out)
    tok = Tokenizer(WordLevel({t: i for i, t in enumerate([*SPECIALS, *WORDS])}, unk_token="<pad>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(SPECIALS)
    tok.save(str(out / "tokenizer.json"))
    return out


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory) -> Path:
    return _write_snapshot(tmp_path_factory.mktemp("tiny-mixtral-native"))


@pytest.fixture(scope="module")
def hf_model(snapshot):
    return _load_int8(snapshot)[0]


@pytest.fixture(scope="module")
def engine(snapshot):
    if not NATIVE_OK:
        pytest.skip(WHY)
    eng = NativeEngine(snapshot, threads=3)
    yield eng
    eng.close()


def _ids(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return [BOS, *torch.randint(4, VOCAB, (n,), generator=g).tolist(), SEP]


def _hf_logits(model, ids: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        return model(torch.tensor([ids])).logits[0].float()


GREEDY = SamplingConfig()


def _hf_greedy(model, ids: list[int], n: int) -> list[int]:
    """transformers greedy for exactly `n` tokens with <eos> an ordinary token
    (min_new_tokens would instead mask it, which the engine's probe does not)."""
    model.generation_config.eos_token_id = None
    with torch.inference_mode():
        out = model.generate(
            torch.tensor([ids]), do_sample=False, max_new_tokens=n, eos_token_id=None,
            pad_token_id=model.config.pad_token_id,
        )
    return out[0, len(ids):].tolist()


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, name: str, **fields) -> None:
        self.events.append((name, fields))


# --------------------------------------------------------------------------
# forward parity against transformers (tiny model)
# --------------------------------------------------------------------------


@needs_native
def test_full_prefill_matches_transformers(engine, hf_model) -> None:
    for seed in range(3):
        ids = _ids(40 + 7 * seed, seed)
        ref = _hf_logits(hf_model, ids)
        got = engine.full_logits(ids)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), float((got - ref).abs().max())


@needs_native
def test_decode_path_matches_transformers(engine, hf_model) -> None:
    ids = _ids(33, 5)
    ref = _hf_logits(hf_model, ids)
    for prefill in (1, 9, len(ids)):
        got = engine.decode_logits(ids, prefill=prefill)
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (prefill, float((got - ref).abs().max()))


@needs_native
@pytest.mark.parametrize("cut", [1, 7, 16, 23, 40])
def test_prefix_restore_equals_full_prefill(engine, cut) -> None:
    ids = _ids(60, 11)
    full, _ = engine.forward(ids)
    head, kv = engine.forward(ids[:cut], export=True)
    assert kv is not None and kv.numel() == engine.kv_floats(cut)
    tail, _ = engine.forward(ids, start=cut, kv_in=kv, kv_in_len=cut)
    got = torch.cat([head, tail])
    # Same arithmetic, different row tiling: agreement to float rounding.
    assert float((got - full).abs().max()) <= 1e-5


@needs_native
def test_prefix_restore_uses_only_the_matching_part_of_a_longer_snapshot(engine) -> None:
    a = _ids(50, 3)
    b = a[:30] + _ids(25, 4)[1:]  # shares 30 tokens with a, then diverges
    _, kv = engine.forward(a, export=True)
    full, _ = engine.forward(b)
    tail, _ = engine.forward(b, start=30, kv_in=kv, kv_in_len=len(a))
    assert float((tail - full[30:]).abs().max()) <= 1e-5
    # and a snapshot of *different* tokens really is different (the test has teeth)
    wrong, _ = engine.forward(b, start=40, kv_in=kv, kv_in_len=len(a))
    assert float((wrong - full[40:]).abs().max()) > 1e-3


@needs_native
def test_forward_rejects_bad_arguments(engine) -> None:
    with pytest.raises(ValueError):
        engine.forward(_ids(10), start=5)  # prefix positions without a snapshot
    with pytest.raises(ValueError):
        engine.forward(list(range(300)))  # beyond max_position_embeddings=256


# --------------------------------------------------------------------------
# decode consistency (ported from bench/extreme/native/consistency.py)
# --------------------------------------------------------------------------


@needs_native
def test_greedy_batch4_equals_batch1(engine) -> None:
    for seed in range(3):
        ids = _ids(20, seed)
        kw = dict(max_new=40, sampling=GREEDY, eos_id=EOS, seed=0, greedy=True, stop_at_eos=False)
        one = engine.generate(ids, n=1, **kw).tokens[0]
        four = engine.generate(ids, n=4, **kw).tokens
        assert len(one) == 40
        assert all(s == one for s in four)


@needs_native
def test_greedy_matches_transformers_greedy(engine, hf_model) -> None:
    for seed in range(3):
        ids = _ids(20, seed)
        hf = _hf_greedy(hf_model, ids, 48)
        nat = engine.generate(ids, n=1, max_new=48, sampling=GREEDY, eos_id=EOS, seed=0, greedy=True, stop_at_eos=False)
        assert nat.tokens[0] == hf


@needs_native
def test_sampled_best_of_invariants(engine) -> None:
    ids = _ids(30, 2)
    cfg = SamplingConfig(0.9, 20, 0.95, 1.15, 3)
    stopped = False
    for seed in range(20):
        out = engine.generate(ids, n=4, max_new=24, sampling=cfg, eos_id=EOS, seed=seed)
        for row, count, mean in zip(out.tokens, out.counts, out.mean_logprob):
            assert len(row) == count >= 1
            assert all(0 <= t < VOCAB for t in row)
            assert EOS not in row[:-1]  # a finished candidate left the batch
            stopped |= row[-1] == EOS and len(row) < 24
            seq = ids + row
            grams = [tuple(seq[i:i + 3]) for i in range(len(seq) - 2)]
            assert len(grams) == len(set(grams))  # no-repeat-3gram over prompt+reply
            assert mean <= 0.0 and mean > -50
        assert out.best == max(range(4), key=lambda i: (out.mean_logprob[i], -i))
        assert out.ttft_s <= out.last_s <= out.total_s
    assert stopped


@needs_native
def test_same_seed_same_output_and_seeds_differ(engine) -> None:
    ids = _ids(12, 8)
    cfg = SamplingConfig(1.0, 0, 1.0, 1.0, 0)
    a = engine.generate(ids, n=3, max_new=16, sampling=cfg, eos_id=EOS, seed=7)
    b = engine.generate(ids, n=3, max_new=16, sampling=cfg, eos_id=EOS, seed=7)
    c = engine.generate(ids, n=3, max_new=16, sampling=cfg, eos_id=EOS, seed=8)
    assert a.tokens == b.tokens and a.mean_logprob == b.mean_logprob
    assert a.tokens != c.tokens


# --------------------------------------------------------------------------
# sampler equivalence with hfserve's processor chain
# --------------------------------------------------------------------------


def _hf_probs(hist: list[int], scores: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    from transformers.generation.logits_process import (
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
    )

    h = torch.tensor([hist])
    s = scores.clone()[None]
    if cfg.repetition_penalty != 1.0:
        s = RepetitionPenaltyLogitsProcessor(cfg.repetition_penalty)(h, s)
    if cfg.no_repeat_ngram_size:
        s = NoRepeatNGramLogitsProcessor(cfg.no_repeat_ngram_size)(h, s)
    tracker = _CandidateTracker(
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        pad_id=PAD,
        eos_id=EOS,
    )
    return tracker(h, s)[0].softmax(-1)


SAMPLER_CONFIGS = [
    SamplingConfig(0.5, 40, 0.9, 1.15, 4),  # the live shape
    SamplingConfig(0.7, 0, 0.95, 1.3, 3, 0.12, 0.1),  # frequency/presence on
    SamplingConfig(1.0, 5, 1.0, 1.0, 2),
    SamplingConfig(1.3, 0, 1.0, 0.9, 0, 0.5, 0.0),
    SamplingConfig(0.5, 1, 0.9, 1.15, 1),
    SamplingConfig(0.8, 300, 0.8, 1.1, 4),  # top-k beyond the streaming buffer
]


@needs_native
@pytest.mark.parametrize("cfg", SAMPLER_CONFIGS)
@pytest.mark.parametrize("V", [200, 16384])
def test_sampler_distribution_matches_transformers(engine, cfg, V) -> None:
    g = torch.Generator().manual_seed(hash((cfg, V)) % 2**31)
    for trial in range(8):
        # A small alphabet makes repeated n-grams (and so bans) common.
        hist = torch.randint(0, 12 if trial % 2 else V, (40,), generator=g).tolist()
        scores = torch.randn(V, generator=g) * 3
        ref = _hf_probs(hist, scores, cfg)
        got = engine.warp_probs(scores, hist, cfg)
        assert torch.equal(ref > 0, got > 0), trial
        assert torch.allclose(ref, got, atol=2e-6), float((ref - got).abs().max())


@needs_native
def test_draws_follow_the_warped_distribution(engine) -> None:
    ids = _ids(10, 21)
    cfg = SamplingConfig(1.0, 12, 0.97, 1.2, 0)
    logits = engine.full_logits(ids)[-1]
    want = engine.warp_probs(logits, ids, cfg)
    hits = torch.zeros(VOCAB)
    draws = 0
    for seed in range(40):
        out = engine.generate(ids, n=64, max_new=1, sampling=cfg, eos_id=EOS, seed=seed)
        for row in out.tokens:
            hits[row[0]] += 1
            draws += 1
    emp = hits / draws
    assert float((emp - want).abs().sum()) / 2 < 0.04  # total variation, 2560 draws
    assert bool(((hits > 0) <= (want > 0)).all())  # never draws a masked token


# --------------------------------------------------------------------------
# the generator seam
# --------------------------------------------------------------------------


@pytest.fixture
def native_settings(settings, snapshot):
    settings.serve_backend = "hf"
    settings.hf_model_dir = snapshot
    settings.hf_runtime = "native"
    settings.infer_threads = 2
    settings.max_new_tokens = 12
    settings.best_of = 4
    settings.temperature, settings.top_k, settings.top_p = 0.9, 20, 0.95
    settings.repetition_penalty, settings.no_repeat_ngram_size = 1.15, 3
    return settings


@needs_native
def test_make_generator_selects_native(native_settings) -> None:
    log = _Log()
    gen = make_generator(native_settings, log)
    assert isinstance(gen, NativeGenerator) and gen.runtime == "native"
    load = dict(log.events)["model.load"]
    assert load["runtime"] == "native" and load["native_build"] in {"compiled", "cached"}
    assert gen.param_count > 0
    lean = LeanGenerator(native_settings)
    assert gen._encode_prompt("w1 w2").tolist() == lean._encode_prompt("w1 w2").tolist()
    kw = dict(max_turns=6, max_tokens=512, max_chars=6000)
    assert gen.conversation_prompt([], "w1 w2", **kw) == lean.conversation_prompt([], "w1 w2", **kw)


@needs_native
def test_generator_contract_and_benchmark(native_settings) -> None:
    from babble.benchmark import run_benchmark

    gen = make_generator(native_settings)
    out = gen("w1 w2")
    assert all(tok in WORDS for tok in out.text.split())
    assert out.max_new_tokens == 12 and out.top_k == 20 and out.no_repeat_ngram_size == 3
    assert out.frequency_penalty == 0.0 and out.presence_penalty == 0.0  # gated off like hf
    result = run_benchmark(gen)
    assert result.backend == "hf-native"
    assert result.transformers_version is None
    assert len(result.candidate_token_counts) == 4
    assert result.selected_tokens in result.candidate_token_counts
    assert result.ttft_ms is not None and result.ttft_ms > 0


@needs_native
def test_generator_is_seed_deterministic(native_settings) -> None:
    native_settings.lean_prefix_cache_mb = 0
    gen = make_generator(native_settings)
    outs = []
    for _ in range(2):
        torch.manual_seed(99)
        outs.append(gen("w1 w2 w3").text)
    assert outs[0] == outs[1]
    texts = set()
    for seed in range(5):
        torch.manual_seed(seed)
        texts.add(gen("w1 w2 w3").text)
    assert len(texts) > 1


@needs_native
def test_generator_prefix_cache_hits_across_turns(native_settings) -> None:
    from babble.conversation import ConversationTurn

    gen = make_generator(native_settings)
    kw = dict(max_turns=6, max_tokens=512, max_chars=6_000)
    first = gen.conversation_prompt([], "w1 w2 w3 w4", **kw)
    reply = gen(first).text
    second = gen.conversation_prompt([ConversationTurn("w1 w2 w3 w4", reply)], "w5 w6", **kw)
    torch.manual_seed(5)
    _text, hit = gen._generate(second, max_new_tokens=8, best_of=2)
    stats = gen.prefix_cache.stats()
    assert stats["misses"] == 1 and stats["hits"] == 1 and stats["entries"] == 1
    assert hit.prefix_reused >= len(gen._encode_prompt(first)[0]) - 2
    # Same seed, cache bypassed: same reply (restored KV == recomputed KV).
    torch.manual_seed(5)
    _text2, cold = gen._generate(second, max_new_tokens=8, best_of=2, use_prefix_cache=False)
    assert cold.prefix_reused == 0
    assert cold.tokens == hit.tokens

    before = gen.prefix_cache.stats()
    gen.benchmark_sample(second, max_new_tokens=4, best_of=2)
    assert gen.prefix_cache.stats() == before


@needs_native
def test_frequency_penalties_follow_the_hf_gate(native_settings, monkeypatch) -> None:
    native_settings.frequency_penalty, native_settings.presence_penalty = 0.3, 0.2
    gen = make_generator(native_settings)
    assert gen._sampling().frequency_penalty == 0.0
    monkeypatch.setenv("BABBLE_HF_FREQUENCY_PENALTIES", "1")
    gen = make_generator(native_settings)
    cfg = gen._sampling()
    assert (cfg.frequency_penalty, cfg.presence_penalty) == (0.3, 0.2)
    assert gen("w1 w2").frequency_penalty == 0.3


# --------------------------------------------------------------------------
# fallback to lean
# --------------------------------------------------------------------------


def _assert_fell_back(settings, log, needle: str) -> None:
    gen = make_generator(settings, log)
    assert isinstance(gen, LeanGenerator) and not isinstance(gen, NativeGenerator)
    reasons = [f["reason"] for name, f in log.events if name == "model.native_fallback"]
    assert reasons and needle in reasons[0], reasons
    assert gen("w1 w2").text is not None


def test_fallback_when_build_fails(native_settings, monkeypatch, tmp_path, capsys) -> None:
    bad = tmp_path / "engine.cpp"
    bad.write_text("this is not C++\n")
    monkeypatch.setattr(_native, "SOURCE", bad)
    monkeypatch.setenv("BABBLE_NATIVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(_native, "check_cpu", lambda: None)
    if not NATIVE_OK:
        monkeypatch.setattr(_native, "compiler", lambda: "/bin/false")
    _assert_fell_back(native_settings, _Log(), "build failed")
    assert "falling back to lean" in capsys.readouterr().err
    assert not list((tmp_path / "cache").glob("*.so"))  # nothing half-built published


def test_fallback_when_cpu_lacks_avx2(native_settings, monkeypatch) -> None:
    monkeypatch.setattr(_native, "cpu_flags", lambda: {"sse2", "fma"})
    _assert_fell_back(native_settings, _Log(), "AVX2")


def test_fallback_without_compiler(native_settings, monkeypatch) -> None:
    monkeypatch.setattr(_native, "check_cpu", lambda: None)
    monkeypatch.setenv("CXX", "definitely-not-a-compiler-xyz")
    _assert_fell_back(native_settings, _Log(), "no C++ compiler")


@needs_native
def test_fallback_on_unsupported_shape(native_settings, tmp_path) -> None:
    # hidden 72 / 4 heads = head_dim 18: not a multiple of 16
    native_settings.hf_model_dir = _write_snapshot(tmp_path / "odd", hidden=72, heads=4, inter=96)
    _assert_fell_back(native_settings, _Log(), "multiples of 16")


def test_unusable_snapshot_errors_are_not_swallowed(native_settings, tmp_path) -> None:
    native_settings.hf_model_dir = tmp_path / "missing"
    with pytest.raises(Exception, match="not a directory"):
        make_generator(native_settings)


def test_config_refusals_become_native_unavailable(snapshot, tmp_path) -> None:
    if not NATIVE_OK:
        pytest.skip(WHY)
    odd = tmp_path / "gqa"
    odd.mkdir()
    for f in snapshot.iterdir():
        (odd / f.name).symlink_to(f)
    (odd / "config.json").unlink()
    cfg = json.loads((snapshot / "config.json").read_text())
    cfg["num_key_value_heads"] = 2
    (odd / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(NativeUnavailable, match="GQA"):
        NativeEngine(odd)
    cfg["num_key_value_heads"] = 4
    cfg["num_experts_per_tok"] = 2
    (odd / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(NativeUnavailable, match="num_experts_per_tok"):
        NativeEngine(odd)


# --------------------------------------------------------------------------
# build cache
# --------------------------------------------------------------------------

_TRIVIAL = 'extern "C" int eng_abi_version() { return %d; }\n'


@needs_native
def test_build_is_cached_keyed_and_atomic(monkeypatch, tmp_path) -> None:
    src = tmp_path / "engine.cpp"
    src.write_text(_TRIVIAL % _native.ABI_VERSION)
    monkeypatch.setattr(_native, "SOURCE", src)
    monkeypatch.setenv("BABBLE_NATIVE_CACHE", str(tmp_path / "cache"))
    first = _native.build()
    assert first.built and first.path.exists() and first.path.parent == tmp_path / "cache"
    again = _native.build()
    assert not again.built and again.path == first.path
    # an edited source is a different build, never the stale library
    src.write_text(_TRIVIAL % _native.ABI_VERSION + "// edited\n")
    edited = _native.build()
    assert edited.built and edited.key != first.key
    assert not list((tmp_path / "cache").glob("*.tmp"))


@needs_native
def test_truncated_cached_library_is_rebuilt(monkeypatch, tmp_path) -> None:
    src = tmp_path / "engine.cpp"
    src.write_text(_TRIVIAL % _native.ABI_VERSION)
    monkeypatch.setattr(_native, "SOURCE", src)
    monkeypatch.setattr(_native, "_declare", lambda lib: lib)
    monkeypatch.setenv("BABBLE_NATIVE_CACHE", str(tmp_path / "cache"))
    info = _native.build()
    info.path.write_bytes(b"\x7fELF garbage")
    lib, reloaded = _native.load()
    assert reloaded.built and lib.eng_abi_version() == _native.ABI_VERSION


def test_cache_dir_defaults_outside_repo_and_tmp(monkeypatch) -> None:
    monkeypatch.delenv("BABBLE_NATIVE_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    d = _native.cache_dir()
    assert d == Path.home() / ".cache" / "babble" / "native"
    monkeypatch.setenv("BABBLE_NATIVE_CACHE", "/srv/x")
    assert _native.cache_dir() == Path("/srv/x")


# --------------------------------------------------------------------------
# real snapshot (skipped without artifacts/)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_engine():
    if not NATIVE_OK:
        pytest.skip(WHY)
    eng = NativeEngine(REAL_MODEL, threads=4)
    yield eng
    eng.close()


def _real_prompt() -> list[int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(REAL_MODEL / "tokenizer.json"))
    text = "user: hey booper whats up\nassistant: not much hbu\nuser: did you see the game last night"
    return [tok.token_to_id("<bos>"), *tok.encode(text, add_special_tokens=False).ids, tok.token_to_id("<sep>")]


@has_real
@needs_native
def test_real_greedy_matches_transformers_for_128_tokens(real_engine) -> None:
    ids = _real_prompt()
    model, config = _load_int8(REAL_MODEL)
    hf = _hf_greedy(model, ids, 128)
    nat = real_engine.generate(
        ids, n=1, max_new=128, sampling=GREEDY, eos_id=config.eos_token_id, seed=0, greedy=True, stop_at_eos=False
    )
    assert len(hf) == 128 and nat.tokens[0] == hf


@has_real
@needs_native
def test_real_greedy_batch4_equals_batch1_and_prefix_restore(real_engine) -> None:
    ids = _real_prompt()
    kw = dict(max_new=64, sampling=GREEDY, eos_id=real_engine.cfg.vocab - 1, seed=0, greedy=True, stop_at_eos=False)
    one = real_engine.generate(ids, n=1, **kw).tokens[0]
    assert all(s == one for s in real_engine.generate(ids, n=4, **kw).tokens)
    cut = len(ids) - 5
    _, kv = real_engine.forward(ids[:cut], export=True)
    restored = real_engine.generate(ids, n=1, start=cut, kv_in=kv, kv_in_len=cut, **kw).tokens[0]
    assert restored == one


# --------------------------------------------------------------------------
# speculative decoding (BABBLE_NATIVE_SPEC)
# --------------------------------------------------------------------------


class _ScriptDrafter:
    """Test drafter: proposes ``f(history)`` (same begin/push/draft protocol as
    `NgramDrafter`)."""

    def __init__(self, f):
        self.f = f

    def begin(self, prompt, n):
        return [list(prompt) for _ in range(n)]

    def push(self, state, s, toks):
        state[s].extend(toks)

    def draft(self, state, s, k):
        return self.f(state[s])[:k]


def _greedy_oracle(engine, ids, n_tokens, corrupt_every=0):
    """Drafter function proposing the true greedy continuation (optionally with
    every `corrupt_every`-th position replaced, to exercise rejection)."""
    full = engine.generate(ids, n=1, max_new=n_tokens + 8, sampling=GREEDY, eos_id=EOS, seed=0, greedy=True,
                           stop_at_eos=False).tokens[0]
    seq = ids + full

    def f(hist):
        if hist != seq[: len(hist)]:
            return []
        out = list(seq[len(hist):len(hist) + 8])
        if corrupt_every:
            out = [(t + 1) % VOCAB if (len(hist) + i) % corrupt_every == 0 else t for i, t in enumerate(out)]
        return out

    return f


@needs_native
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_spec_greedy_equals_generate(engine, k) -> None:
    """The verify path (per-row stream-cache positions, KV rollback on
    rejection) reproduces plain greedy decoding token for token."""
    from babble.nativeserve import NgramDrafter, build_ngram_tables

    g = torch.Generator().manual_seed(3)
    corpus = [torch.randint(4, VOCAB, (40,), generator=g).tolist() for _ in range(30)]
    ngram = NgramDrafter(build_ngram_tables(corpus, order=3, min_conf=0.0, vocab=VOCAB), vocab=VOCAB, no_repeat_ngram=3)
    for seed in range(4):
        ids = _ids(18, 40 + seed)
        drafters = [ngram, _ScriptDrafter(_greedy_oracle(engine, ids, 40)),
                    _ScriptDrafter(_greedy_oracle(engine, ids, 40, corrupt_every=3)),
                    _ScriptDrafter(lambda h: [int(h[-1]) % VOCAB] * 8)]
        for n in (1, 3):
            for stop in (False, True):
                kw = dict(n=n, max_new=40, sampling=SamplingConfig(1.0, 0, 1.0, 1.15, 3), eos_id=EOS, seed=seed,
                          greedy=True, stop_at_eos=stop)
                want = engine.generate(ids, **kw)
                for d in drafters:
                    got = engine.spec_generate(ids, drafter=d, k=k, **kw)
                    assert got.tokens == want.tokens and got.counts == want.counts, (seed, n, stop, d)


@needs_native
def test_spec_accepts_a_perfect_draft_in_few_steps(engine) -> None:
    ids = _ids(18, 50)
    calls = []
    oracle = _greedy_oracle(engine, ids, 32)

    def f(h):
        calls.append(len(h))
        return oracle(h)

    kw = dict(n=1, max_new=32, sampling=GREEDY, eos_id=EOS, seed=0, greedy=True, stop_at_eos=False)
    got = engine.spec_generate(ids, drafter=_ScriptDrafter(f), k=7, **kw)
    assert got.tokens == engine.generate(ids, **kw).tokens
    assert len(calls) == 4  # 1 prefill token + 4 steps x (7 drafts + 1 bonus) >= 32


def _exact_marginals(engine, ids, cfg, depth):
    """Exact distributions of reply tokens 1..depth under the warp (enumerated)."""
    margs = [torch.zeros(VOCAB) for _ in range(depth)]
    frontier = [(list(ids), 1.0)]
    for d in range(depth):
        nxt = []
        for hist, pr in frontier:
            p = engine.warp_probs(engine.full_logits(hist)[-1], hist, cfg).double()
            margs[d] += pr * p
            if d + 1 < depth:
                nxt += [(hist + [t], pr * float(p[t])) for t in torch.nonzero(p).flatten().tolist()]
        frontier = nxt
    return [m.float() for m in margs]


@needs_native
def test_spec_sampling_matches_the_target_distribution(engine) -> None:
    """Exact speculative sampling leaves the output distribution unchanged:
    reply tokens 2 and 3 (made by the verify path, with history-dependent
    repetition / no-repeat-ngram warps) follow the enumerated distribution,
    for a good drafter (mostly accepted) and a bad one (mostly rejected, so
    residual draws), as closely as plain sampling does."""
    ids = _ids(6, 77)
    cfg = SamplingConfig(1.0, 6, 0.95, 1.3, 2)
    exact = _exact_marginals(engine, ids, cfg, 3)

    def top_draft(h):  # the target's own modes, history-aware: high acceptance
        p = engine.warp_probs(engine.full_logits(h)[-1], h, cfg)
        a = int(p.argmax())
        p2 = engine.warp_probs(engine.full_logits(h + [a])[-1], h + [a], cfg)
        return [a, int(p2.argmax())]

    drafters = {
        "good": _ScriptDrafter(top_draft),
        "bad": _ScriptDrafter(lambda h: [int(h[-1]), (int(h[-1]) + 1) % VOCAB]),
    }
    calls = 250
    draws = 64 * calls
    for name, d in [("plain", None), *drafters.items()]:
        hits = [torch.zeros(VOCAB) for _ in range(3)]
        for seed in range(calls):
            kw = dict(n=64, max_new=3, sampling=cfg, eos_id=EOS, seed=1000 + seed, stop_at_eos=False)
            out = engine.generate(ids, **kw) if d is None else engine.spec_generate(ids, drafter=d, k=2, **kw)
            for row in out.tokens:
                assert len(row) == 3
                for j in range(3):
                    hits[j][row[j]] += 1
        for j in range(3):
            emp = hits[j] / draws
            tv = float((emp - exact[j]).abs().sum()) / 2
            assert tv < 0.03, (name, j, tv)
            assert bool(((hits[j] > 0) <= (exact[j] > 0)).all()), (name, j)  # never a masked token


@needs_native
def test_spec_sampled_invariants_and_logprob_scale(engine) -> None:
    from babble.nativeserve import NgramDrafter, build_ngram_tables

    ids = _ids(30, 2)
    cfg = SamplingConfig(0.9, 20, 0.95, 1.15, 3)
    g = torch.Generator().manual_seed(9)
    corpus = [torch.randint(4, VOCAB, (60,), generator=g).tolist() for _ in range(50)]
    d = NgramDrafter(build_ngram_tables(corpus, order=2, min_conf=0.0, vocab=VOCAB), vocab=VOCAB, no_repeat_ngram=3)
    stopped = False
    plain, spec = [], []
    for seed in range(20):
        a = engine.generate(ids, n=4, max_new=24, sampling=cfg, eos_id=EOS, seed=seed)
        out = engine.spec_generate(ids, n=4, max_new=24, sampling=cfg, eos_id=EOS, seed=seed, drafter=d, k=3)
        assert [r[0] for r in out.tokens] == [r[0] for r in a.tokens]  # same prefill draw
        for row, count in zip(out.tokens, out.counts):
            assert len(row) == count >= 1
            assert EOS not in row[:-1]
            stopped |= row[-1] == EOS and len(row) < 24
            seq = ids + row
            grams = [tuple(seq[i:i + 3]) for i in range(len(seq) - 2)]
            assert len(grams) == len(set(grams))
        assert out.best == max(range(4), key=lambda i: (out.mean_logprob[i], -i))
        assert out.ttft_s <= out.last_s <= out.total_s
        plain += a.mean_logprob
        spec += out.mean_logprob
    assert stopped
    # same quantity (mean post-warp log-prob of the chosen tokens), same scale
    assert abs(sum(plain) / len(plain) - sum(spec) / len(spec)) < 0.25


def test_ngram_drafter_respects_no_repeat_ngram_and_roundtrips(tmp_path) -> None:
    from babble.nativeserve import NgramDrafter, build_ngram_tables, load_ngram_tables, save_ngram_tables

    seqs = [[5, 6, 7, 8, 5, 6, 7, 9]] * 3 + [[10, 11, 12]]
    tables = build_ngram_tables(seqs, order=2, min_conf=0.0, vocab=VOCAB)
    save_ngram_tables(tables, tmp_path / "t.pt", vocab=VOCAB)
    loaded, v = load_ngram_tables(tmp_path / "t.pt")
    assert v == VOCAB and loaded == tables
    d = NgramDrafter(loaded, vocab=VOCAB, no_repeat_ngram=0)
    st = d.begin([1, 10], 1)
    assert d.draft(st, 0, 2) == [11, 12]
    # no-repeat-3gram: (10, 11) was already followed by 12 -> 12 is never proposed after it
    d3 = NgramDrafter(loaded, vocab=VOCAB, no_repeat_ngram=3)
    st = d3.begin([10, 11, 12, 1, 10], 1)
    assert d3.draft(st, 0, 3) == [11]
    d3.push(st, 0, [11, 13])
    assert d3.draft(st, 0, 3) == []  # 13 has no continuation
    # a context whose top continuation is not confident enough drafts nothing
    weak = NgramDrafter(build_ngram_tables([[1, 2], [1, 3], [1, 4]], order=1, min_conf=0.5, vocab=VOCAB), vocab=VOCAB)
    assert weak.draft(weak.begin([1], 1), 0, 2) == []


@needs_native
def test_generator_spec_flag(native_settings, monkeypatch, tmp_path) -> None:
    from babble.nativeserve import build_ngram_tables, save_ngram_tables

    log = _Log()
    monkeypatch.setenv("BABBLE_NATIVE_SPEC", "1")
    monkeypatch.setenv("BABBLE_NATIVE_SPEC_TABLE", str(tmp_path / "missing.pt"))
    gen = make_generator(native_settings, log)
    assert gen.drafter is None and "no n-gram table" in dict(log.events)["model.load"]["native_spec"]
    g = torch.Generator().manual_seed(1)
    corpus = [torch.randint(4, VOCAB, (40,), generator=g).tolist() for _ in range(20)]
    save_ngram_tables(build_ngram_tables(corpus, order=3, min_conf=0.0, vocab=VOCAB), tmp_path / "t.pt", vocab=VOCAB)
    monkeypatch.setenv("BABBLE_NATIVE_SPEC_TABLE", str(tmp_path / "t.pt"))
    monkeypatch.setenv("BABBLE_NATIVE_SPEC_K", "2")
    native_settings.lean_prefix_cache_mb = 0
    gen = make_generator(native_settings)
    assert gen.drafter is not None and gen.spec_k == 2
    assert any("speculative" in o for o in gen.benchmark_metadata()["optimizations"])
    out = gen("w1 w2")
    assert all(tok in WORDS for tok in out.text.split())
    monkeypatch.setenv("BABBLE_NATIVE_SPEC", "0")
    assert make_generator(native_settings).drafter is None


@needs_native
@pytest.mark.parametrize("chunk", [1, 3, 4, 7])
def test_verify_forward_matches_full_prefill(engine, chunk) -> None:
    """Multi-row verify steps (per-row cache index), with every chunk first fed
    as rejected junk and rolled back, give the full-prefill logits."""
    ids = _ids(45, 13)
    full = engine.full_logits(ids)
    for prefill in (1, 9):
        for junk in (False, True):
            got = engine.verify_logits(ids, prefill=prefill, chunk=chunk, junk=junk)
            assert torch.allclose(got, full, atol=2e-4), (prefill, junk, float((got - full).abs().max()))
