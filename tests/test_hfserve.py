from __future__ import annotations

from types import SimpleNamespace

import torch

from babble.conversation import ConversationTurn
from babble.hfserve import _CandidateTracker, HFGenerator


class _IdentityWarper:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, input_ids, scores):
        return scores


def test_hf_frequency_and_presence_penalties_count_prompt_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        "babble.hfserve._require_hf_sampling",
        lambda: (_IdentityWarper, _IdentityWarper, _IdentityWarper),
    )
    tracker = _CandidateTracker(
        frequency_penalty=0.5,
        presence_penalty=0.25,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        pad_id=0,
        eos_id=3,
    )
    scores = tracker(torch.tensor([[1, 1, 2]]), torch.zeros((1, 4)))

    assert scores.tolist() == [[0.0, -1.25, -0.75, 0.0]]


class _Tokenized:
    def __init__(self, text: str) -> None:
        self.ids = list(text.encode("utf-8"))


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> _Tokenized:
        return _Tokenized(text)


def test_hf_conversation_prompt_keeps_complete_turns_inside_model_budget() -> None:
    generator = object.__new__(HFGenerator)
    generator.config = SimpleNamespace(max_position_embeddings=80)
    generator.settings = SimpleNamespace(max_new_tokens=16)
    generator.tokenizer = _Tokenizer()
    history = [
        ConversationTurn("old question", "old answer"),
        ConversationTurn("new question", "new answer"),
    ]

    prompt = generator.conversation_prompt(
        history,
        "current",
        max_turns=2,
        max_tokens=512,
        max_chars=1_000,
    )

    assert prompt == "user: new question\nassistant: new answer\nuser: current"
    assert len(prompt.encode()) <= 62  # 80 context - 16 reply - two structural tokens
