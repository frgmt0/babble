"""The model itself: random at birth, byte-clean, and small enough to be polite."""

from __future__ import annotations

import torch

from babble.generate import continue_text, sample
from babble.model import Babbler, ModelConfig, sequence_loss
from babble.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    VOCAB_SIZE,
    build_example,
    decode,
    encode,
    prompt_budget,
    prompt_context,
)

TINY = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=64)


# --- tokenizer ----------------------------------------------------------


def test_every_kind_of_text_round_trips():
    for text in [
        "hello",
        "https://tenor.com/view/cat-typing-98765",  # gif corrections are urls
        "👍🫠 ❤️",
        "こんにちは",
        "a\"b'c\\d\n\ttabs",
        "!@#$%^&*()_+-=[]{};:,.<>/?`~",
        "1234567890",
    ]:
        assert decode(encode(text)) == text


def test_random_bytes_decode_instead_of_raising():
    # An untrained model emits mostly invalid UTF-8. That must not be an error.
    assert isinstance(decode(list(range(200, 256))), str)


def test_an_example_masks_the_prompt_and_trains_on_the_response():
    example = build_example("hi", "hey", block_size=64)

    assert example.tokens[0] == BOS_ID
    assert example.tokens[-1] == EOS_ID
    assert SEP_ID in example.tokens
    # 1 exactly on the response bytes plus the closing <eos>.
    assert sum(example.mask) == len("hey") + 1
    separator = example.tokens.index(SEP_ID)
    assert example.mask[:separator + 1] == [0] * (separator + 1)


def test_long_input_is_truncated_to_fit_the_block():
    example = build_example("p" * 500, "r" * 500, block_size=64)

    assert len(example.tokens) <= 64


def test_the_prompt_is_trimmed_from_the_left_and_the_response_takes_what_is_left():
    # The prompt is trimmed by a rule that depends only on the block size --
    # `prompt_budget(13)` is 7 -- so the trainer and `prompt_context` agree on
    # where <sep> lands. Its head goes first, because its tail is the relevant
    # part. The response then gets whatever the 10-byte budget has left.
    example = build_example("a long prompt", "abcdefgh", block_size=13)

    assert prompt_budget(13) == 7
    assert len(example.tokens) == 13
    assert decode(example.tokens) == " promptabc"


def test_prompt_context_ends_ready_to_generate():
    assert prompt_context("hi", 64)[-1] == SEP_ID


# --- model --------------------------------------------------------------


def test_weights_start_random_and_unseeded():
    torch.manual_seed(1)
    a = Babbler(TINY)
    torch.manual_seed(2)
    b = Babbler(TINY)

    assert not torch.allclose(a.tok_emb.weight, b.tok_emb.weight)
    # Not constant, not zeros: actual noise.
    assert a.tok_emb.weight.std().item() > 0.001


def test_the_default_model_is_a_few_million_parameters():
    count = Babbler().num_params()

    assert 1_000_000 < count < 8_000_000
    # float32 weights plus two AdamW moments must stay well inside a couple of GB.
    assert count * 4 * 3 < 200 * 1024 * 1024


def test_forward_produces_a_distribution_over_every_byte():
    logits = Babbler(TINY)(torch.zeros((2, 5), dtype=torch.long))

    assert logits.shape == (2, 5, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_a_training_step_produces_gradients():
    model = Babbler(TINY)
    tokens = torch.tensor([[BOS_ID, 104, SEP_ID, 105, EOS_ID]])
    mask = torch.tensor([[0, 0, 0, 1, 1]])

    loss = sequence_loss(model, tokens, mask, torch.tensor([1.0]))
    loss.backward()

    assert loss.item() > 0
    assert model.tok_emb.weight.grad is not None
    assert model.tok_emb.weight.grad.abs().sum().item() > 0


def test_a_zero_weight_row_cannot_move_the_model():
    model = Babbler(TINY)
    tokens = torch.tensor([[BOS_ID, 104, SEP_ID, 105, EOS_ID]])
    mask = torch.tensor([[0, 0, 0, 1, 1]])

    sequence_loss(model, tokens, mask, torch.tensor([0.0])).backward()

    assert model.tok_emb.weight.grad.abs().sum().item() == 0


def test_padding_is_excluded_from_the_loss():
    model = Babbler(TINY)
    torch.manual_seed(0)
    real = torch.tensor([[BOS_ID, 104, SEP_ID, 105, EOS_ID]])
    mask = torch.tensor([[0, 0, 0, 1, 1]])
    padded = torch.tensor([[BOS_ID, 104, SEP_ID, 105, EOS_ID, PAD_ID, PAD_ID]])
    padded_mask = torch.tensor([[0, 0, 0, 1, 1, 0, 0]])

    a = sequence_loss(model, real, mask, torch.tensor([1.0]))
    b = sequence_loss(model, padded, padded_mask, torch.tensor([1.0]))

    assert torch.allclose(a, b, atol=1e-5)


# --- sampling -----------------------------------------------------------


class _Rigged(torch.nn.Module):
    """A model that always wants one particular token."""

    def __init__(self, favourite: int) -> None:
        super().__init__()
        self.config = TINY
        self.favourite = favourite

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(idx.shape[0], idx.shape[1], VOCAB_SIZE)
        logits[..., self.favourite] = 100.0
        return logits


def test_generation_stops_at_end_of_sequence():
    assert sample(_Rigged(EOS_ID), "hi", max_new_tokens=32, temperature=0) == ""


def test_structural_tokens_can_never_be_emitted():
    for banned in (SEP_ID, BOS_ID, PAD_ID):
        text = sample(_Rigged(banned), "hi", max_new_tokens=8, temperature=0)
        assert "<sep>" not in text and "<bos>" not in text and "<pad>" not in text
        assert len(text) > 0  # it kept generating rather than emitting the token


def test_sampling_is_reproducible_with_a_seeded_generator():
    model = Babbler(TINY)
    args = dict(max_new_tokens=16, temperature=1.0, top_k=8)

    first = sample(model, "hi", generator=torch.Generator().manual_seed(5), **args)
    second = sample(model, "hi", generator=torch.Generator().manual_seed(5), **args)

    assert first == second


def test_a_fresh_model_babbles_rather_than_crashing():
    text = sample(Babbler(TINY), "hello", max_new_tokens=24)

    assert isinstance(text, str)


def test_sampling_leaves_the_model_in_training_mode():
    model = Babbler(TINY)
    model.train()

    sample(model, "hi", max_new_tokens=2)

    assert model.training, "the trainer would silently lose dropout otherwise"


def _repeated_ngrams(text: str, n: int = 3) -> int:
    grams = [text[i : i + n] for i in range(max(0, len(text) - n + 1))]
    return 0 if not grams else len(grams) - len(set(grams))


class _LoopBias(torch.nn.Module):
    """Always prefers one byte, with a weaker runner-up — greedy without a
    penalty emits a solid run of the favourite, which is the loop we want
    the penalty to break.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = TINY

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(idx.shape[0], idx.shape[1], VOCAB_SIZE)
        logits[..., 97] = 10.0  # 'a'
        logits[..., 98] = 9.0  # 'b' — above 10/1.15 so a penalty of 1.15 switches
        return logits


def test_repetition_penalty_reduces_repeated_ngrams():
    model = _LoopBias()
    args = dict(max_new_tokens=32, temperature=0.0, top_k=0, top_p=1.0)
    off = continue_text(model, "x", repetition_penalty=1.0, **args)
    on = continue_text(model, "x", repetition_penalty=1.15, **args)

    def overlapping(text: str, n: int = 3) -> int:
        return sum(1 for i in range(len(text) - n + 1) if text[i : i + n] == "a" * n)

    assert _repeated_ngrams(on) < _repeated_ngrams(off)
    assert overlapping(on) < overlapping(off)
    assert "b" in on and "b" not in off


class _StubbornLoopBias(torch.nn.Module):
    """Gap wide enough that a *flat* repetition penalty can never close it --
    10/1.15 is still far above -5, so `repetition_penalty` alone loops
    forever no matter how many times 'a' repeats. Mirrors the live incident:
    "boop boop boop" in, ~85 consecutive "boop" out, with
    `BABBLE_REPETITION_PENALTY=1.15` nominally active the whole time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = TINY

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.full((idx.shape[0], idx.shape[1], VOCAB_SIZE), -100.0)
        logits[..., 97] = 10.0  # 'a'
        logits[..., 98] = -5.0  # 'b' -- /1.15 never brings 'a' this low, but still the
        # clear second choice once something finally does (everything else is -100)
        return logits


def test_flat_repetition_penalty_alone_cannot_break_a_wide_enough_loop():
    """Proves the failure mode is real, not hypothetical, before proving the fix."""
    model = _StubbornLoopBias()
    text = continue_text(
        model,
        "x",
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.15,
    )
    assert text == "a" * 32


def test_frequency_penalty_breaks_a_loop_flat_repetition_penalty_cannot():
    model = _StubbornLoopBias()
    text = continue_text(
        model,
        "x",
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.15,
        frequency_penalty=2.0,
    )
    assert "b" in text, "a penalty that grows with count must eventually outweigh a flat one"


class _TwoTokenPreference(torch.nn.Module):
    """Strong preference for 'a', clear second-place 'b', everything else far
    behind -- so once no-repeat-ngram bans a token, which one wins next is
    unambiguous."""

    def __init__(self) -> None:
        super().__init__()
        self.config = TINY

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.full((idx.shape[0], idx.shape[1], VOCAB_SIZE), -100.0)
        logits[..., 97] = 10.0  # 'a'
        logits[..., 98] = 5.0  # 'b'
        return logits


def test_no_repeat_ngram_blocks_a_repeated_bigram():
    model = _TwoTokenPreference()
    args = dict(max_new_tokens=8, temperature=0.0, top_k=0, top_p=1.0)
    off = continue_text(model, "x", no_repeat_ngram_size=0, **args)
    on = continue_text(model, "x", no_repeat_ngram_size=2, **args)

    assert off == "a" * 8
    assert on != "a" * 8
    assert "b" in on
