"""The lean hf runtime against the transformers path it replaces.

No network, no real model: a tiny randomly-initialised Mixtral is written in
the same int8 safetensors layout the live snapshot uses, and both runtimes
load it from disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from babble.hfserve import HFGenerator, _CandidateTracker, _load_int8, make_generator  # noqa: E402
from babble.leanserve import (  # noqa: E402
    KVCache,
    LeanConfig,
    LeanGenerator,
    LeanModel,
    PrefixKVCache,
    SamplingConfig,
    generate,
    int8_kernel_available,
    ngram_bans,
    warp,
)

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"]
WORDS = [f"w{i}" for i in range(60)]
VOCAB = len(SPECIALS) + len(WORDS)  # 64
PAD, BOS, SEP, EOS = range(4)


# --------------------------------------------------------------------------
# tiny snapshot
# --------------------------------------------------------------------------


def _quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (w.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8).to(torch.bfloat16)
    q = torch.round(w / scale.float()).clamp(-127, 127).to(torch.int8)
    return q, scale


def _write_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {tok: i for i, tok in enumerate([*SPECIALS, *WORDS])}
    tok = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(SPECIALS)
    tok.save(str(path))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory) -> Path:
    from safetensors.torch import save_file
    from transformers import MixtralConfig, MixtralForCausalLM

    out = tmp_path_factory.mktemp("tiny-mixtral")
    torch.manual_seed(1234)
    config = MixtralConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
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
            gate_up = state[fused + ".experts.gate_up_proj"]
            down = state[fused + ".experts.down_proj"]
            inter = config.intermediate_size
            for e in range(config.num_local_experts):
                put(f"{new}.experts.{e}.w1.weight", gate_up[e, :inter])
                put(f"{new}.experts.{e}.w3.weight", gate_up[e, inter:])
                put(f"{new}.experts.{e}.w2.weight", down[e])
        else:
            put(new + ".gate.weight", state[new + ".gate.weight"])
            for e in range(config.num_local_experts):
                for w in ("w1", "w2", "w3"):
                    put(f"{new}.experts.{e}.{w}.weight", state[f"{new}.experts.{e}.{w}.weight"])
    save_file({k: v.contiguous() for k, v in packed.items()}, str(out / "model-int8.safetensors"))
    config.save_pretrained(out)
    _write_tokenizer(out / "tokenizer.json")
    return out


@pytest.fixture(scope="module")
def hf_model(snapshot):
    model, _config = _load_int8(snapshot)
    return model


def _ids(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return [BOS, *torch.randint(4, VOCAB, (n,), generator=g).tolist(), SEP]


def _hf_logits(model, ids: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        return model(torch.tensor([ids])).logits[0].float()


# --------------------------------------------------------------------------
# forward parity
# --------------------------------------------------------------------------


def test_fp32_full_sequence_matches_transformers(snapshot, hf_model) -> None:
    lean = LeanModel(snapshot, precision="fp32")
    for seed in range(3):
        ids = _ids(40, seed)
        ref = _hf_logits(hf_model, ids)
        got = lean.full_logits(ids)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_fp32_decode_and_prefix_resume_match_transformers(snapshot, hf_model) -> None:
    lean = LeanModel(snapshot, precision="fp32")
    ids = _ids(30, 7)
    ref = _hf_logits(hf_model, ids)
    assert torch.allclose(lean.decode_logits(ids), ref, atol=1e-4, rtol=1e-4)

    # Resuming from a cached prefix (masked chunk prefill) is the same model.
    with torch.inference_mode():
        cache = lean.new_cache(1, len(ids))
        a = lean.forward(torch.tensor([ids[:20]]), cache, 0, all_positions=True)[0]
        b = lean.forward(torch.tensor([ids[20:]]), cache, 20, all_positions=True)[0]
        got = lean.exact_logits(torch.cat([a, b]))
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not int8_kernel_available(), reason="no int8 CPU kernel in this torch build")
def test_int8_full_and_decode_track_transformers(snapshot, hf_model) -> None:
    lean = LeanModel(snapshot, precision="int8")
    agree = total = 0
    for seed in range(3):
        ids = _ids(40, seed)
        ref = _hf_logits(hf_model, ids)
        full = lean.full_logits(ids)
        # Prefill-sized matmuls use the fp32 copy: essentially exact.
        assert torch.allclose(full, ref, atol=1e-3, rtol=1e-3)
        decode = lean.decode_logits(ids)
        # Decode rounds activations to bf16; an occasional router flip makes
        # the max error spiky, so bound the typical error instead.
        err = (decode - ref).abs()
        assert float(err.median()) < 0.02 * float(ref.std())
        assert float(err.mean()) < 0.1 * float(ref.std())
        agree += int((decode.argmax(-1) == ref.argmax(-1)).sum())
        total += ref.shape[0]
    # A random tiny model has near-flat logits (many argmax near-ties), so the
    # real-model gate (top-1 >= 0.995, bench/extreme/reference.py) is the
    # quality bar; this only catches a broken decode path.
    assert agree / total >= 0.9


def test_int8_candidate_rescoring_is_exact(snapshot) -> None:
    if not int8_kernel_available():
        pytest.skip("no int8 CPU kernel")
    lean = LeanModel(snapshot, precision="int8")
    h = torch.randn(3, 64)
    idx = torch.randint(0, VOCAB, (3, 5))
    exact = torch.nn.functional.linear(h, lean.embed).gather(1, idx)
    assert torch.allclose(lean.exact_logits_at(h, idx), exact, atol=1e-5)


def test_kv_cache_compaction_keeps_each_rows_history(snapshot) -> None:
    lean = LeanModel(snapshot, precision="fp32")
    rows = torch.tensor([_ids(10, s) for s in range(3)])
    nxt = torch.tensor([[5], [6], [7]])
    with torch.inference_mode():
        ref_cache = lean.new_cache(3, 16)
        lean.forward(rows, ref_cache, 0)
        ref = lean.forward(nxt, ref_cache, rows.shape[1])

        cache = lean.new_cache(3, 16)
        lean.forward(rows, cache, 0)
        cache.compact(torch.tensor([0, 2]), rows.shape[1])
        got = lean.forward(nxt[[0, 2]], cache, rows.shape[1])
    assert torch.allclose(got, ref[[0, 2]], atol=1e-5)


def test_config_refuses_what_it_does_not_implement() -> None:
    base = {
        "model_type": "mixtral",
        "num_experts_per_tok": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_local_experts": 2,
        "intermediate_size": 8,
        "vocab_size": 8,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 32,
    }
    assert LeanConfig.from_dict(base).head_dim == 16
    for bad in ({"num_experts_per_tok": 2}, {"sliding_window": 8}, {"model_type": "llama"}):
        with pytest.raises(Exception, match="BABBLE_HF_RUNTIME=transformers"):
            LeanConfig.from_dict({**base, **bad})


# --------------------------------------------------------------------------
# sampler vs transformers' processors
# --------------------------------------------------------------------------


def _hf_pipeline(hist: torch.Tensor, scores: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    from transformers.generation.logits_process import (
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
    )

    s = scores.clone()
    if cfg.repetition_penalty != 1.0:
        s = RepetitionPenaltyLogitsProcessor(cfg.repetition_penalty)(hist, s)
    if cfg.no_repeat_ngram_size:
        s = NoRepeatNGramLogitsProcessor(cfg.no_repeat_ngram_size)(hist, s)
    tracker = _CandidateTracker(
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        pad_id=PAD,
        eos_id=EOS,
    )
    return tracker(hist, s)


def _ours(hist: torch.Tensor, scores: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    B, V = scores.shape
    counts = torch.zeros(B, V).scatter_add_(1, hist, torch.ones(hist.shape, dtype=torch.float32))
    vals, idx = warp(scores, counts, ngram_bans(hist, cfg.no_repeat_ngram_size, V), cfg)
    return torch.full((B, V), -math.inf).scatter(1, idx, vals)


CONFIGS = [
    SamplingConfig(0.5, 40, 0.9, 1.15, 4),  # the live shape
    SamplingConfig(0.7, 0, 0.95, 1.3, 3, 0.12, 0.1),
    SamplingConfig(1.0, 5, 1.0, 1.0, 2),
    SamplingConfig(1.3, 0, 1.0, 0.9, 0, 0.5, 0.0),
    SamplingConfig(0.5, 1, 0.9, 1.15, 1),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_sampler_masks_and_probs_match_transformers(cfg: SamplingConfig) -> None:
    g = torch.Generator().manual_seed(hash(cfg) % 2**31)
    V = 200
    for trial in range(20):
        B = 4
        # A small alphabet makes repeated n-grams (and so bans) common.
        hist = torch.randint(0, 12 if trial % 2 else V, (B, 40), generator=g)
        scores = torch.randn(B, V, generator=g) * 3
        ref = _hf_pipeline(hist, scores, cfg)
        got = _ours(hist, scores, cfg)
        assert torch.equal(torch.isfinite(ref), torch.isfinite(got)), trial
        assert torch.allclose(ref.softmax(-1), got.softmax(-1), atol=1e-6)
        # Best-of scores the post-warp log-probability of the chosen token.
        finite = torch.isfinite(ref)
        assert torch.allclose(ref.log_softmax(-1)[finite], got.log_softmax(-1)[finite], atol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_ngram_bans_match_transformers(n: int) -> None:
    from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

    g = torch.Generator().manual_seed(n)
    for length in (1, n - 1, n, 30):
        if length < 1:
            continue
        hist = torch.randint(0, 6, (3, length), generator=g)
        ref = torch.isinf(NoRepeatNGramLogitsProcessor(n)(hist, torch.zeros(3, 10)))
        ban = ngram_bans(hist, n, 10)
        got = torch.zeros(3, 10, dtype=torch.bool) if ban is None else ban
        assert torch.equal(ref, got), (n, length)


# --------------------------------------------------------------------------
# generation loop bookkeeping (compaction, stats, best-of)
# --------------------------------------------------------------------------


class _ToyModel:
    """Next-token candidates are a function of the row's *cached* history.

    Each row may continue with f(history), g(history) or <eos>, uniformly. A
    compaction bug that hands a row another row's cache shows up as a token
    outside its own candidate set.
    """

    V = 32

    def __init__(self) -> None:
        self.cfg = LeanConfig(
            hidden=1, n_heads=1, n_kv=1, head_dim=1, n_layers=1, n_experts=1, inter=1,
            vocab=self.V, eps=1e-5, max_pos=512, rope_theta=1e4, tied=True,
        )
        self.approx_head = False

    def new_cache(self, rows: int, tmax: int) -> KVCache:
        return KVCache(self.cfg, rows, tmax)

    @staticmethod
    def state(tokens) -> int:
        return sum((i + 1) * int(t) for i, t in enumerate(tokens)) % 997

    def forward(self, ids, cache, start, *, all_positions=False):
        B, T = ids.shape
        cache.k[0][:B, 0, start : start + T, 0] = ids.float()
        hist = cache.k[0][:B, 0, : start + T, 0].long()
        return torch.tensor([[float(self.state(row.tolist()))] for row in hist])

    @classmethod
    def candidates(cls, s: int) -> set[int]:
        return {4 + s % 10, 14 + s % 7, EOS}

    def exact_logits(self, h):
        out = torch.full((h.shape[0], self.V), -1e4)
        for r, s in enumerate(h[:, 0].long().tolist()):
            for tok in self.candidates(s):
                out[r, tok] = 0.0
        return out

    head_logits = exact_logits


def test_generate_compacts_rows_without_mixing_histories() -> None:
    model = _ToyModel()
    prompt = [BOS, 5, 6, 7, SEP]
    cfg = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0)
    saw_early_finish = False
    for seed in range(20):
        torch.manual_seed(seed)
        result = generate(model, prompt, n=4, max_new=12, cfg=cfg, pad_id=PAD, eos_id=EOS)
        lengths = set()
        for row in result.tokens:
            hist = list(prompt)
            for tok in row:
                assert tok in model.candidates(model.state(hist))
                hist.append(tok)
            assert EOS not in row[:-1]
            lengths.add(len(row))
        assert result.stats.candidate_token_counts == tuple(len(r) for r in result.tokens)
        assert result.stats.selected_index == result.best
        finished = [r for r in result.tokens if r and r[-1] == EOS]
        saw_early_finish |= len(lengths) > 1 and bool(finished)
        # Every token here has post-warp probability 1/3 (or 1/2 when f == g).
        assert all(m <= math.log(0.5) + 1e-6 for m in result.mean_logprob)
    assert saw_early_finish


# --------------------------------------------------------------------------
# prefix cache
# --------------------------------------------------------------------------


def _kv(n: int) -> tuple[list, list]:
    return [torch.zeros(1, 1, n, 4)], [torch.zeros(1, 1, n, 4)]


def test_prefix_cache_hit_miss_dominance_and_eviction() -> None:
    per_tok = 2 * 4 * 4  # k+v, 4 floats each
    cache = PrefixKVCache(max_entries=3, max_bytes=per_tok * 25)
    a = torch.tensor([1, 2, 3, 4, 9])
    assert cache.lookup(a) == (None, 0)
    cache.store(a, *_kv(5))

    # Next turn of the same conversation: the old prompt minus <sep> matches.
    a2 = torch.tensor([1, 2, 3, 4, 7, 8, 9])
    entry, common = cache.lookup(a2)
    assert entry is not None and common == 4
    cache.store(a2, *_kv(7))
    assert len(cache) == 1  # the turn-1 entry was superseded, not duplicated

    b = torch.tensor([1, 5, 6, 9])
    entry, common = cache.lookup(b)
    assert common == 1
    cache.store(b, *_kv(4))
    c = torch.tensor([2, 2, 2, 9])
    assert cache.lookup(c) == (None, 0)
    cache.store(c, *_kv(4))
    assert len(cache) == 3

    # Touch `a2`, then add a fourth: least recently used (`b`) goes.
    cache.lookup(torch.tensor([1, 2, 3]))
    cache.store(torch.tensor([3, 3, 3, 9]), *_kv(4))
    assert len(cache) == 3 and cache.evictions == 1
    assert cache.lookup(torch.tensor([1, 5, 6]))[1] == 1  # only a2's leading 1

    # Byte bound: a big entry evicts until it fits; oversized entries are skipped.
    cache.store(torch.tensor([8] * 20), *_kv(20))
    assert cache.bytes <= cache.max_bytes
    cache.store(torch.tensor([7] * 30), *_kv(30))
    assert cache.lookup(torch.tensor([7] * 30))[1] == 0


def test_prefix_cache_disabled_by_zero_budget() -> None:
    cache = PrefixKVCache(max_entries=0, max_bytes=1 << 20)
    cache.store(torch.tensor([1, 2]), *_kv(2))
    assert not cache.enabled and len(cache) == 0


# --------------------------------------------------------------------------
# the generator seam
# --------------------------------------------------------------------------


@pytest.fixture
def lean_settings(settings, snapshot):
    settings.serve_backend = "hf"
    settings.hf_model_dir = snapshot
    settings.hf_runtime = "lean"
    settings.lean_precision = "fp32"
    settings.max_new_tokens = 12
    settings.best_of = 4
    settings.temperature, settings.top_k, settings.top_p = 0.9, 20, 0.95
    settings.repetition_penalty, settings.no_repeat_ngram_size = 1.15, 3
    return settings


def test_make_generator_selects_runtime(lean_settings) -> None:
    gen = make_generator(lean_settings)
    assert isinstance(gen, LeanGenerator)
    assert gen.param_count > 0
    lean_settings.hf_runtime = "transformers"
    assert isinstance(make_generator(lean_settings), HFGenerator)
    lean_settings.hf_runtime = "onnx"
    with pytest.raises(ValueError, match="hf_runtime"):
        make_generator(lean_settings)


def test_settings_read_runtime_flags(monkeypatch, tmp_path) -> None:
    from babble.config import Settings

    assert Settings.from_env(tmp_path).hf_runtime == "transformers"
    monkeypatch.setenv("BABBLE_HF_RUNTIME", "lean")
    monkeypatch.setenv("BABBLE_LEAN_PRECISION", "fp32")
    monkeypatch.setenv("BABBLE_LEAN_PREFIX_CACHE_MB", "0")
    s = Settings.from_env(tmp_path)
    assert (s.hf_runtime, s.lean_precision, s.lean_prefix_cache_mb) == ("lean", "fp32", 0)


def test_lean_and_transformers_param_counts_agree(lean_settings) -> None:
    lean = LeanGenerator(lean_settings)
    lean_settings.hf_runtime = "transformers"
    hf = HFGenerator(lean_settings)
    assert lean.param_count == hf.param_count
    assert lean._encode_prompt("w1 w2").tolist() == hf._encode_prompt("w1 w2").tolist()


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_generator_is_seed_deterministic(lean_settings, precision) -> None:
    if precision == "int8" and not int8_kernel_available():
        pytest.skip("no int8 CPU kernel")
    lean_settings.lean_precision = precision
    lean_settings.lean_prefix_cache_mb = 0
    gen = LeanGenerator(lean_settings)
    outs = []
    for _ in range(2):
        torch.manual_seed(99)
        outs.append(gen("w1 w2 w3").text)
    assert outs[0] == outs[1]
    assert all(tok in WORDS for tok in outs[0].split())
    # A different seed explores (with overwhelming probability).
    texts = set()
    for seed in range(5):
        torch.manual_seed(seed)
        texts.add(gen("w1 w2 w3").text)
    assert len(texts) > 1


def test_generator_prefix_cache_hits_across_turns(lean_settings) -> None:
    from babble.conversation import ConversationTurn

    gen = LeanGenerator(lean_settings)
    kwargs = dict(max_turns=6, max_tokens=512, max_chars=6_000)
    first = gen.conversation_prompt([], "w1 w2 w3 w4", **kwargs)
    reply = gen(first).text
    second = gen.conversation_prompt([ConversationTurn("w1 w2 w3 w4", reply)], "w5 w6", **kwargs)
    gen(second)
    stats = gen.prefix_cache.stats()
    assert stats["misses"] == 1 and stats["hits"] == 1
    assert stats["reused_tokens"] >= len(gen._encode_prompt(first)[0]) - 2
    assert stats["entries"] == 1

    # The benchmark never reads or fills the cache.
    before = gen.prefix_cache.stats()
    gen.benchmark_sample(second, max_new_tokens=4, best_of=2)
    assert gen.prefix_cache.stats() == before


def test_generator_output_matches_generation_contract(lean_settings) -> None:
    from babble.benchmark import run_benchmark

    gen = LeanGenerator(lean_settings)
    out = gen("w1 w2")
    assert out.max_new_tokens == 12 and out.top_k == 20 and out.no_repeat_ngram_size == 3
    assert out.frequency_penalty == 0.0 and out.presence_penalty == 0.0  # gated off like hf
    result = run_benchmark(gen)
    assert result.backend == "hf-lean"
    assert result.transformers_version is None
    assert len(result.candidate_token_counts) == 4
    assert result.selected_tokens in result.candidate_token_counts
