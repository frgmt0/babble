from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from babble.benchmark import BenchmarkUnavailable, format_benchmark, run_benchmark


class InstrumentedGenerator:
    settings = SimpleNamespace(best_of=4)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        return torch.zeros((1, 7), dtype=torch.long)

    def benchmark_sample(self, prompt: str, *, max_new_tokens: int, best_of: int):
        assert "robot" in prompt
        assert max_new_tokens == 64
        assert best_of == 4
        # Deliberately consume RNG; run_benchmark must restore it.
        torch.rand(3)
        return SimpleNamespace(
            ttft_s=0.01,
            total_s=0.1,
            steady_s=0.08,
            candidate_token_counts=(64, 32, 48, 16),
            first_step_tokens=4,
            selected_index=2,
            selected_content_tokens=47,
        )

    def benchmark_metadata(self):
        return {
            "model": "fixture-model",
            "params": 149_000_000,
            "dtype": "float32",
            "optimizations": ("KV cache", "streamed best-of scores"),
        }


class ConversationInstrumentedGenerator(InstrumentedGenerator):
    settings = SimpleNamespace(
        best_of=4,
        conversation_context=True,
        conversation_max_turns=3,
        conversation_max_tokens=99,
        conversation_max_chars=200,
    )

    def conversation_prompt(self, history, current_user, **bounds):
        assert history == []
        assert bounds == {"max_turns": 3, "max_tokens": 99, "max_chars": 200}
        return f"user: {current_user}"

    def benchmark_sample(self, prompt: str, *, max_new_tokens: int, best_of: int):
        assert prompt.startswith("user: ")
        return super().benchmark_sample(prompt, max_new_tokens=max_new_tokens, best_of=best_of)


def test_benchmark_reports_actual_aggregate_and_selected_counts_and_restores_rng() -> None:
    generator = InstrumentedGenerator()
    before = torch.random.get_rng_state().clone()
    result = run_benchmark(generator)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert result.aggregate_tokens == 160
    assert result.selected_tokens == 48
    assert result.selected_content_tokens == 47
    assert result.prompt_tokens == 7
    assert result.model == "fixture-model"
    assert result.selected_tps < result.e2e_tps
    rendered = format_benchmark(result)
    assert "aggregate tok/s" in rendered
    assert "selected-candidate tok/s" in rendered
    assert "149.0M params" in rendered
    assert len(rendered) < 2_000


def test_benchmark_refuses_generator_without_step_instrumentation() -> None:
    with pytest.raises(BenchmarkUnavailable, match="step-level"):
        run_benchmark(lambda prompt: None)


def test_benchmark_uses_backend_conversation_prompt_when_enabled() -> None:
    result = run_benchmark(ConversationInstrumentedGenerator())
    assert result.prompt_tokens == 7
