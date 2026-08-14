"""The training signal: what is optimised, what is reported, and what comes out.

The regression gate for the "loss 0.02 but it still babbles" bug lives here.
That bug was never one mistake -- it was a reported number that averaged the
problem away, and a sampler hot enough to act on whatever uncertainty was left.
So these tests check both halves, plus the thing built on top of the fix: the
bot picking the best of several draws instead of the first one.

The trainer's actual objective is plain next-token prediction over an
unlabelled text corpus -- no prompt, no chosen answer, nothing paired with
anything -- so most of these tests build `CorpusRow`s and train through
`to_examples`/`text_examples`. The old `<bos> prompt <sep> response` pair
layout is still real, still used to score correction pairs, and still tested
here, but it is no longer what the loss is computed over.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from babble.config import Settings
from babble.corpus import SOURCE_MENTION, CorpusRow, make_corpus_id
from babble.generate import (
    best_continuation,
    best_of,
    continue_many,
    continue_text,
    sample,
    sample_many,
    score,
    score_continuations,
    score_many,
)
from babble.model import Babbler, ModelConfig, config_from_settings, sequence_loss
from babble.tokenizer import (
    BOS_ID,
    EOS_ID,
    SEP_ID,
    VOCAB_SIZE,
    build_continuation_example,
    build_example,
    prompt_budget,
    prompt_context,
    text_context,
    text_examples,
)
from babble.trainer import _stack_examples, leading_words, make_batch, measure, to_examples

# Three rows, deliberately short and deliberately distinct. Small enough that
# the test model can memorise them in a few seconds, which is what makes a
# memorisation assertion affordable in a unit test at all.
FIXTURE = [
    ("hi", "hey there"),
    ("bye", "see ya"),
    ("thanks", "np"),
]

# The corpus equivalent of FIXTURE: flat rows of writing, not prompt/response
# pairs. Each one is still short and distinct enough to memorise fast.
CORPUS_FIXTURE = [
    "hi hey there",
    "bye see ya",
    "thanks np",
]


def shipped(tmp_path) -> Settings:
    """Settings with every knob at its shipped default.

    `Settings.for_root` only redirects the paths, so temperature, top_k and
    best_of are exactly what a fresh install runs with -- which is the whole
    point of a gate: it must fail if someone turns the sampler back up.
    """
    return Settings.for_root(tmp_path)


def tiny_model(block_size: int = 64) -> Babbler:
    return Babbler(ModelConfig(block_size=block_size, n_layer=2, n_head=2, n_embd=64))


def fit(model: Babbler, rows, *, block_size: int = 64, steps: int = 900, lr: float = 3e-3):
    """Train `model` to convergence on prompt/response `rows`, pair layout.

    Still used by the tests below the "pair layout" section, which exercise
    `sample`/`score`/`best_of` -- functions the corpus change left unchanged.
    Returns the final examples.
    """
    examples = [build_example(p, r, block_size) for p, r in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    model.train()
    for _ in range(steps):
        tokens, mask, weights = make_batch(examples, 4, rng)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return examples


def fit_corpus(model: Babbler, texts, *, block_size: int = 64, steps: int = 900, lr: float = 3e-3):
    """Train `model` to convergence on plain corpus `texts`, the real objective.

    Same shape as `fit`, but through `text_examples` and with no mask held
    back: every token after `<bos>` is a target. Returns the final examples.
    """
    examples = [example for text in texts for example in text_examples(text, block_size)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    model.train()
    for _ in range(steps):
        tokens, mask, weights = make_batch(examples, 4, rng)
        loss = sequence_loss(model, tokens, mask, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return examples


class _AlwaysWants(torch.nn.Module):
    """A model that only ever wants one token. Lets the decoder be tested
    without a trained model in the way."""

    def __init__(self, favourite: int, block_size: int = 64) -> None:
        super().__init__()
        self.config = ModelConfig(block_size=block_size, n_layer=2, n_head=2, n_embd=64)
        self.favourite = favourite

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(idx.shape[0], idx.shape[1], VOCAB_SIZE)
        logits[..., self.favourite] = 100.0
        return logits


def make_row(text: str, *, author: str = "u_1", source: str = SOURCE_MENTION) -> CorpusRow:
    return CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=source)


# --- the gate ------------------------------------------------------------


@pytest.mark.slow
def test_a_memorised_row_comes_back_memorised_at_the_shipped_defaults(tmp_path):
    """The end-to-end gate: trained to convergence, it must continue itself.

    A model that has driven its loss to the floor on a three-row corpus must
    continue a real prefix from that corpus with exactly the rest of that
    row -- using the temperature, top_k and best-of-n the bot actually ships
    with, not a hand-picked greedy decode.

    Measured honestly, this particular assertion also passes at the *old*
    temperature of 1.0: three short rows converge to a loss around 1e-5, and at
    that point there is no residual uncertainty left for a hot sampler to cash
    in. So this gates the training path -- the corpus objective and layout --
    while `test_the_shipped_defaults_reproduce_the_corpus_where_the_old_ones_did_not`
    gates the sampling path, at the partially-converged loss where the sampler
    actually decides the outcome.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit_corpus(model, CORPUS_FIXTURE)
    settings = shipped(tmp_path)

    for text in CORPUS_FIXTURE:
        prefix = leading_words(text)
        want = text[len(prefix) :]
        got = best_continuation(
            model,
            prefix,
            n=settings.best_of,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
        )
        assert got == want, f"{prefix!r} -> {got!r}, wanted {want!r}"


@pytest.mark.slow
def test_an_unseen_prefix_lands_on_the_corpus_rather_than_on_noise(tmp_path):
    """The other half of "it should at least continue with something real".

    A three-row model will not produce a whole memorised continuation for a
    prefix it has never seen -- it produces pieces of what it knows ("hey",
    "np") -- and claiming otherwise would be a test that only passes by luck.
    What it must not do is what ro reported: byte soup. So the assertion is
    that whatever comes out is something this model finds far more probable
    than the noise it used to emit.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit_corpus(model, CORPUS_FIXTURE)
    settings = shipped(tmp_path)

    got = best_continuation(
        model,
        "hi there",
        n=settings.best_of,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
    )

    assert got, "an empty reply is not an answer"
    assert (
        score_continuations(model, "hi there", [got])[0]
        < score_continuations(model, "hi there", ["smeu.nnuccrl,"])[0]
    )


# --- what the loss is actually measured over ------------------------------


def test_the_loss_converges_over_the_whole_sequence_with_nothing_held_back():
    """The new invariant, replacing the old prompt-masking hypothesis.

    A corpus row has no prompt to hold back -- every token after `<bos>` is a
    target -- so training to convergence must drive the loss over the *whole*
    sequence down, not just some trained half of it. If any part of the
    sequence were still going unmasked-but-untrained, the mean would stay
    stuck near its random-init value (around 5.5 nats, `ln(256)`) no matter how
    long training ran; 0.15 is nowhere near that floor and well past the point
    a held-back sequence could reach by memorising only half of itself. It
    does not reach zero: three rows with three different opening bytes leave a
    genuine ambiguity right after `<bos>`, with nothing yet said to resolve it.
    """
    torch.manual_seed(0)
    model = tiny_model()
    rows = [make_row(t) for t in CORPUS_FIXTURE]
    examples = fit_corpus(model, CORPUS_FIXTURE)

    report = measure(model, examples, rows)

    assert report.mean < 0.15, "nothing in the sequence should be left untrained"


def test_the_worst_row_is_reported_next_to_the_mean():
    """The measurement that makes "loss 0.02 but it babbles" visible.

    This is the shape of the real bug: a corpus that is already memorised, plus
    one fresh row the model has not learned yet. The reported loss is a mean
    over *tokens*, so the short new row barely moves it -- it can sit at a loss
    of 8 while the headline number still reads like everything is fine.
    Reporting the worst row alongside the mean is what stops it hiding.
    """
    torch.manual_seed(0)
    memorised = CORPUS_FIXTURE + [
        "not much yet, honestly, but i am learning",
        "ro did, mostly, and it shows",
        "inside a checkpoint file on someone's laptop",
    ]
    model = tiny_model()
    known = fit_corpus(model, memorised)
    fresh_text = "zzzz"
    rows = [make_row(t) for t in memorised]
    rows.append(make_row(fresh_text))

    report = measure(model, known + text_examples(fresh_text, 64), rows)

    assert report.mean < 0.5, "the mean still looks respectable..."
    assert report.worst_row > 4.0, "...while one row is complete noise"
    assert report.worst_row > report.mean * 8
    assert report.worst_text == fresh_text


def test_the_worst_row_is_named_correctly_when_an_earlier_row_expands_into_several_examples():
    """`_example_owner` walks `to_examples`' row order to name the worst
    example, so a row long enough to become several examples must not throw
    off the index of every row that comes after it -- the real off-by-one
    risk in the new code, now that one row no longer means one example.
    """
    torch.manual_seed(0)
    block_size = 16
    long_row = "abcdefghijklmnopqrstuvwxyz0123456789"
    assert len(text_examples(long_row, block_size)) > 1, "fixture must actually chunk"
    memorised = [long_row, "hi", "bye"]
    model = tiny_model(block_size)
    known = fit_corpus(model, memorised, block_size=block_size)
    fresh_text = "never seen zzqx"
    rows = [make_row(t) for t in memorised]
    rows.append(make_row(fresh_text))

    report = measure(model, known + text_examples(fresh_text, block_size), rows)

    assert report.worst_text == fresh_text


def test_measuring_an_empty_corpus_says_nothing_rather_than_dividing_by_zero():
    assert measure(tiny_model(), []).mean is None


# --- train / inference prompt layout --------------------------------------


def test_the_inference_prefix_is_byte_for_byte_the_trained_prefix():
    """Hypothesis three. Any drift between these two puts <sep> -- and with
    learned positions, everything after it -- somewhere the model was never
    trained to see it.
    """
    for prompt, response in FIXTURE + [("x" * 400, "y" * 400), ("", "hello")]:
        example = build_example(prompt, response, 64)
        prefix = prompt_context(prompt, 64)
        separator = example.tokens.index(SEP_ID)

        assert example.tokens[: separator + 1] == prefix, f"drift on {prompt[:20]!r}"


def test_the_corpus_inference_prefix_is_byte_for_byte_the_trained_prefix():
    """The corpus-layout counterpart of the test above. `text_context` is what
    the bot and the checkpoint probe generate from; `build_continuation_example`
    is how a candidate continuation gets scored. Any drift between the two
    means a continuation is judged in a context it was never generated from.
    """
    for prefix, continuation in FIXTURE + [("x" * 400, "y" * 400), ("", "hello")]:
        example = build_continuation_example(prefix, continuation, 64)
        head = text_context(prefix, 64)

        assert example.tokens[: len(head)] == head, f"drift on {prefix[:20]!r}"


def test_a_prompt_longer_than_the_block_is_trimmed_the_same_way_on_both_paths():
    long_prompt = "".join(str(i % 10) for i in range(500))

    example = build_example(long_prompt, "short", 64)
    prefix = prompt_context(long_prompt, 64)

    assert prefix[0] == BOS_ID and prefix[-1] == SEP_ID
    assert len(prefix) == prompt_budget(64) + 2
    assert example.tokens[: example.tokens.index(SEP_ID) + 1] == prefix


def test_a_long_prompt_never_squeezes_the_response_out_of_the_example():
    example = build_example("p" * 500, "the response", 64)

    assert sum(example.mask) > 1, "the response must still have room to be trained on"
    assert len(example.tokens) <= 64


# --- corpus example weight --------------------------------------------------


def test_every_corpus_example_carries_weight_one_so_no_row_can_dominate_the_batch():
    """Per-row weighting is gone from the training path: a corpus row is one
    piece of writing, not a correction to be trusted more or less than
    another, so every example it produces must carry the same weight -- and
    `to_examples` must not even accept a settings argument to apply one with.
    """
    rows = [make_row("hi there"), make_row("a somewhat longer row of plain corpus text, more of it")]

    examples = to_examples(rows, 64)

    assert examples, "fixture must actually produce examples"
    assert all(example.weight == pytest.approx(1.0) for example in examples)
    _, _, weights = _stack_examples(examples)
    assert torch.allclose(weights, torch.ones_like(weights))
    with pytest.raises(TypeError):
        to_examples(rows, 64, Settings.for_root(Path("/nonexistent")))


# --- best-of-n: the pair layout ---------------------------------------------


def test_scoring_prefers_the_response_the_model_was_trained_on():
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, FIXTURE)

    assert score(model, "hi", "hey there") < score(model, "hi", "qX7 zzz mmm")


def test_batched_scoring_matches_scoring_one_at_a_time():
    """Padding must not leak into the score. A short candidate batched next to a
    long one has to come out with exactly the number it would have alone --
    otherwise best-of-n is picking by candidate length.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, FIXTURE, steps=300)
    candidates = ["np", "hey there", "a much longer candidate answer than the others"]

    batched = score_many(model, "hi", candidates)
    alone = [score(model, "hi", c) for c in candidates]

    assert batched == pytest.approx(alone, abs=1e-5)


def test_scoring_nothing_is_an_empty_list_not_a_crash():
    assert score_many(tiny_model(), "hi", []) == []


def test_best_of_one_is_a_plain_sample():
    torch.manual_seed(0)
    model = tiny_model()

    text = best_of(model, "hi", n=1, max_new_tokens=8, temperature=0.0)

    assert text == sample(model, "hi", max_new_tokens=8, temperature=0.0)


# Six rows rather than three, and stopped short of the floor: a 3-row model
# converges so hard that even temperature 1.0 reproduces it perfectly, which
# would make the comparison below meaningless. This sits where the live bot
# sits -- learned, but with real residual uncertainty on a few bytes.
PARTIAL_CORPUS = CORPUS_FIXTURE + [
    "what can you do not much yet honestly but i am learning",
    "who made you ro did mostly and it shows",
    "where do you live inside a checkpoint file on someones laptop",
]


@pytest.mark.slow
def test_the_shipped_defaults_reproduce_the_corpus_where_the_old_ones_did_not(tmp_path):
    """The regression gate for the *sampling* half of the bug.

    Best-of-n at a cooler temperature exists because one unlucky byte ruins a
    whole continuation -- a byte-level model has no way back once it leaves
    the memorised path -- and because the model can tell afterwards that it
    did.

    This is a distributional claim, so it is measured over many draws rather
    than one: at this loss the old defaults miss a handful out of sixty, and the
    shipped defaults must miss none.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit_corpus(model, PARTIAL_CORPUS, steps=150)
    settings = shipped(tmp_path)
    draws = 10
    probes = [(leading_words(text), text[len(leading_words(text)) :]) for text in PARTIAL_CORPUS]

    old_defaults = sum(
        continue_text(model, prefix, max_new_tokens=96, temperature=1.0, top_k=40) == want
        for prefix, want in probes
        for _ in range(draws)
    )
    shipped_defaults = sum(
        best_continuation(
            model, prefix, n=settings.best_of, max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature, top_k=settings.top_k,
        )
        == want
        for prefix, want in probes
        for _ in range(draws)
    )

    total = len(probes) * draws
    assert shipped_defaults >= old_defaults
    assert shipped_defaults >= total - 1, (
        f"the shipped defaults reproduced {shipped_defaults}/{total}; they are what the "
        f"bot answers with, so they have to be effectively perfect on a memorised corpus"
    )


def test_sampling_many_returns_one_candidate_per_draw_and_they_differ():
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, FIXTURE, steps=200)  # part-trained, so there is real uncertainty

    candidates = sample_many(model, "hi", 6, max_new_tokens=32, temperature=1.0, top_k=40)

    assert len(candidates) == 6
    assert all(isinstance(c, str) for c in candidates)
    assert len(set(candidates)) > 1, "batched draws must be independent, not copies"


def test_batched_sampling_is_the_same_decoder_as_the_single_one():
    """Greedy is deterministic, so the batched path must reproduce the single
    path exactly. If these ever diverge, best-of-n is choosing between
    candidates the bot would never actually have said.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, FIXTURE, steps=300)

    one = sample(model, "hi", max_new_tokens=32, temperature=0.0)
    many = sample_many(model, "hi", 3, max_new_tokens=32, temperature=0.0)

    assert many == [one, one, one]


def test_batched_sampling_never_emits_a_structural_token():
    for candidate in sample_many(
        _AlwaysWants(SEP_ID), "hi", 3, max_new_tokens=8, temperature=0.0
    ):
        assert candidate and "<sep>" not in candidate


def test_batched_sampling_stops_each_row_at_its_own_eos():
    assert sample_many(_AlwaysWants(EOS_ID), "hi", 3, max_new_tokens=8, temperature=0.0) == [""] * 3


def test_best_of_n_never_returns_an_empty_reply_when_it_has_a_real_one():
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, [("hi", "hey there")], steps=400)

    assert best_of(model, "hi", n=6, max_new_tokens=64, temperature=0.5, top_k=40)


# --- best-of-n: the continuation layout -------------------------------------


def test_continuation_scoring_prefers_the_text_the_model_was_trained_on():
    torch.manual_seed(0)
    model = tiny_model()
    fit_corpus(model, CORPUS_FIXTURE)

    assert (
        score_continuations(model, "hi", [" hey there"])[0]
        < score_continuations(model, "hi", ["qX7 zzz mmm"])[0]
    )


def test_batched_continuation_scoring_matches_scoring_one_at_a_time():
    """The continuation-layout counterpart of the padding test above."""
    torch.manual_seed(0)
    model = tiny_model()
    fit_corpus(model, CORPUS_FIXTURE, steps=300)
    candidates = [" np", " hey there", " a much longer continuation than the others"]

    batched = score_continuations(model, "hi", candidates)
    alone = [score_continuations(model, "hi", [c])[0] for c in candidates]

    assert batched == pytest.approx(alone, abs=1e-5)


def test_scoring_no_continuations_is_an_empty_list_not_a_crash():
    assert score_continuations(tiny_model(), "hi", []) == []


def test_best_continuation_with_n_equal_to_one_is_a_plain_continue_text():
    torch.manual_seed(0)
    model = tiny_model()

    text = best_continuation(model, "hi", n=1, max_new_tokens=8, temperature=0.0)

    assert text == continue_text(model, "hi", max_new_tokens=8, temperature=0.0)


def test_continuing_many_returns_one_candidate_per_draw_and_they_differ():
    torch.manual_seed(0)
    model = tiny_model()
    fit_corpus(model, CORPUS_FIXTURE, steps=35)  # part-trained, so there is real uncertainty

    candidates = continue_many(model, "hi", 6, max_new_tokens=32, temperature=1.0, top_k=40)

    assert len(candidates) == 6
    assert all(isinstance(c, str) for c in candidates)
    assert len(set(candidates)) > 1, "batched draws must be independent, not copies"


# --- the model behind the checkpoint ---------------------------------------


def test_config_from_settings_still_round_trips_the_model_shape():
    """Hypothesis four: a config mismatch would silently build a differently
    shaped model behind an unchanged checkpoint filename.
    """
    settings = Settings.for_root(Path("/nonexistent"))
    config = config_from_settings(settings)

    assert ModelConfig.from_dict(config.to_dict()) == config
    assert (config.n_layer, config.n_embd, config.block_size) == (
        settings.n_layer,
        settings.n_embd,
        settings.block_size,
    )
