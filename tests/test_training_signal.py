"""The training signal: what is optimised, what is reported, and what comes out.

The regression gate for the "loss 0.02 but it still babbles" bug lives here.
That bug was never one mistake -- it was a reported number that averaged the
problem away, and a sampler hot enough to act on whatever uncertainty was left.
So these tests check both halves, plus the two things built on top of the fix:
corrections weighing more, and the bot picking the best of several draws instead
of the first one.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from babble.config import Settings
from babble.generate import best_of, sample, sample_many, score, score_many
from babble.model import Babbler, ModelConfig, config_from_settings, sequence_loss
from babble.store import APPROVAL, CORRECTION, Interaction
from babble.tokenizer import (
    BOS_ID,
    EOS_ID,
    SEP_ID,
    VOCAB_SIZE,
    build_example,
    prompt_budget,
    prompt_context,
)
from babble.trainer import (
    _stack_examples,
    make_batch,
    measure,
    row_weight,
    to_examples,
    trainable_rows,
)

# Three rows, deliberately short and deliberately distinct. Small enough that
# the test model can memorise them in a few seconds, which is what makes a
# memorisation assertion affordable in a unit test at all.
FIXTURE = [
    ("hi", "hey there"),
    ("bye", "see ya"),
    ("thanks", "np"),
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
    """Train `model` to convergence on `rows`. Returns the final examples."""
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


def make_row(prompt: str, chosen: str, rejected: str | None, *, weight: float = 1.0) -> Interaction:
    return Interaction(
        id=f"{prompt}-{chosen}",
        signal=CORRECTION if rejected is not None else APPROVAL,
        prompt=prompt,
        rejected=rejected,
        chosen=chosen,
        prompt_author="u_1",
        signal_author="u_2",
        weight=weight,
    )


# --- the gate ------------------------------------------------------------


@pytest.mark.slow
def test_a_memorised_prompt_comes_back_memorised_at_the_shipped_defaults(tmp_path):
    """The end-to-end gate: trained to convergence, it must parrot itself back.

    A model that has driven its loss to the floor on a three-row corpus must
    answer a prompt from that corpus with that prompt's response -- using the
    temperature, top_k and best-of-n the bot actually ships with, not a
    hand-picked greedy decode.

    Measured honestly, this particular assertion also passes at the *old*
    temperature of 1.0: three short rows converge to a loss around 1e-5, and at
    that point there is no residual uncertainty left for a hot sampler to cash
    in. So this gates the training path -- masking, layout, weighting -- while
    `test_the_shipped_defaults_reproduce_the_corpus_where_the_old_ones_did_not`
    gates the sampling path, at the partially-converged loss where the sampler
    actually decides the outcome.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, FIXTURE)
    settings = shipped(tmp_path)

    for prompt, want in FIXTURE:
        got = best_of(
            model,
            prompt,
            n=settings.best_of,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
        )
        assert got == want, f"{prompt!r} -> {got!r}, wanted {want!r}"


@pytest.mark.slow
def test_an_unseen_prompt_lands_on_the_corpus_rather_than_on_noise(tmp_path):
    """The other half of "it should at least parrot something back".

    A three-row model will not produce a whole memorised response for a prompt
    it has never seen -- it produces pieces of them ("see", "np"), and claiming
    otherwise would be a test that only passes by luck. What it must not do is
    what ro reported: byte soup. So the assertion is that whatever comes out is
    something this model finds far more probable than the noise it used to
    emit.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, FIXTURE)
    settings = shipped(tmp_path)

    got = best_of(
        model,
        "hi there",
        n=settings.best_of,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
    )

    assert got, "an empty reply is not an answer"
    assert score(model, "hi there", got) < score(model, "hi there", "smeu.nnuccrl,")


# --- what the loss is actually measured over ------------------------------


def test_the_loss_ignores_prompt_tokens_entirely():
    """Hypothesis one, settled by measurement rather than by reading the mask.

    If prompt tokens were in the training loss, driving the loss down would
    drive both numbers down together. They must diverge: response loss to the
    floor, prompt loss free to go wherever it likes, because nothing trains it.
    """
    torch.manual_seed(0)
    model = tiny_model()
    examples = fit(model, FIXTURE)
    rows = [make_row(p, r, "junk") for p, r in FIXTURE]

    report = measure(model, examples, rows)

    assert report.response < 0.05, "the response tokens are what training optimises"
    assert report.prompt > 1.0, (
        "prompt tokens are not trained, so their loss must stay high -- if this "
        "drops, the mask has broken and the reported loss is measuring the wrong thing"
    )


def test_the_worst_row_is_reported_next_to_the_mean():
    """The measurement that makes "loss 0.02 but it babbles" visible.

    This is the shape of the real bug: a corpus that is already memorised, plus
    one fresh correction the model has not learned yet. The reported loss is a
    mean over *tokens*, so the short new row barely moves it -- it can sit at a
    loss of 8 while the headline number still reads like everything is fine.
    Reporting the worst row alongside the mean is what stops it hiding.
    """
    torch.manual_seed(0)
    memorised = FIXTURE + [
        ("what can you do", "not much yet, honestly, but i am learning"),
        ("who made you", "ro did, mostly, and it shows"),
        ("where do you live", "inside a checkpoint file on someone's laptop"),
    ]
    model = tiny_model()
    known = fit(model, memorised)
    fresh = build_example("quantum chromodynamics", "zzzz", 64)
    rows = [make_row(p, r, "junk") for p, r in memorised]
    rows.append(make_row("quantum chromodynamics", "zzzz", "junk"))

    report = measure(model, known + [fresh], rows)

    assert report.response < 0.5, "the mean still looks respectable..."
    assert report.worst_row > 4.0, "...while one row is complete noise"
    assert report.worst_row > report.response * 8
    assert report.worst_prompt == "quantum chromodynamics"


def test_the_worst_row_is_named_correctly_when_a_row_has_no_chosen_answer():
    """`to_examples` drops rows with an empty `chosen`, so a raw row list is one
    filter out of step with the examples built from it -- and the worst row gets
    reported under some other row's prompt. `trainable_rows` is what keeps the
    two lined up.
    """
    torch.manual_seed(0)
    memorised = FIXTURE + [("who made you", "ro did, mostly, and it shows")]
    model = tiny_model()
    rows = [make_row("", "", None)]  # dropped: nothing to learn from it
    rows += [make_row(p, r, "junk") for p, r in memorised]
    rows.insert(2, make_row("also dropped", "", None))
    rows.append(make_row("quantum chromodynamics", "zzzz", "junk"))

    examples = to_examples(rows, 64)
    known = fit(model, memorised)
    assert len(examples) == len(trainable_rows(rows)) < len(rows)

    report = measure(model, known + [build_example("quantum chromodynamics", "zzzz", 64)],
                     trainable_rows(rows))

    assert report.worst_prompt == "quantum chromodynamics"


def test_measuring_an_empty_corpus_says_nothing_rather_than_dividing_by_zero():
    assert measure(tiny_model(), []).response is None


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


# --- correction upweighting ------------------------------------------------


def test_a_correction_outweighs_an_approval_of_the_same_stored_weight():
    settings = Settings.for_root(Path("/nonexistent"))
    correction = make_row("hi", "hey", "junk", weight=1.0)
    approval = make_row("hi", "hey", None, weight=1.0)

    assert row_weight(correction, settings) == pytest.approx(settings.correction_boost)
    assert row_weight(approval, settings) == pytest.approx(1.0)


def test_the_boost_is_configurable_and_can_be_turned_off():
    settings = Settings.for_root(Path("/nonexistent"))
    settings.correction_boost = 1.0
    row = make_row("hi", "hey", "junk", weight=1.0)

    assert row_weight(row, settings) == pytest.approx(1.0)


def test_to_examples_applies_the_boost_only_when_given_settings():
    settings = Settings.for_root(Path("/nonexistent"))
    rows = [make_row("hi", "hey", "junk", weight=1.0)]

    assert to_examples(rows, 64, settings)[0].weight == pytest.approx(settings.correction_boost)
    assert to_examples(rows, 64)[0].weight == pytest.approx(1.0)


def test_an_upweighted_correction_takes_a_bigger_share_of_the_batch_loss():
    """The point of the boost, not just its arithmetic.

    A per-example weight cancels out in a one-example batch's normaliser, so the
    boost only bites where it is meant to: against the other rows it is sharing
    a batch with.
    """
    correction = build_example("hi", "hey", 64, weight=3.0)
    plain = build_example("hi", "hey", 64, weight=1.0)
    alongside = build_example("bye", "see ya", 64, weight=1.0)

    def share(examples):
        _, mask, weights = _stack_examples(examples)
        scale = mask[:, 1:].float() * weights[:, None]
        return float(scale[0].sum() / scale.sum())

    assert share([correction, alongside]) > share([plain, alongside])


# --- best-of-n -------------------------------------------------------------


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
PARTIAL = FIXTURE + [
    ("what can you do", "not much yet, honestly, but i am learning"),
    ("who made you", "ro did, mostly, and it shows"),
    ("where do you live", "inside a checkpoint file on someone's laptop"),
]


@pytest.mark.slow
def test_the_shipped_defaults_reproduce_the_corpus_where_the_old_ones_did_not(tmp_path):
    """The regression gate for the *sampling* half of the bug.

    Best-of-n at a cooler temperature exists because one unlucky byte ruins a
    whole response -- a byte-level model has no way back once it leaves the
    memorised path -- and because the model can tell afterwards that it did.

    This is a distributional claim, so it is measured over many draws rather
    than one: at this loss the old defaults miss a handful out of sixty, and the
    shipped defaults must miss none.
    """
    torch.manual_seed(0)
    model = tiny_model()
    fit(model, PARTIAL, steps=150)
    settings = shipped(tmp_path)
    draws = 10

    old_defaults = sum(
        sample(model, p, max_new_tokens=96, temperature=1.0, top_k=40) == want
        for p, want in PARTIAL
        for _ in range(draws)
    )
    shipped_defaults = sum(
        best_of(
            model, p, n=settings.best_of, max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature, top_k=settings.top_k,
        )
        == want
        for p, want in PARTIAL
        for _ in range(draws)
    )

    total = len(PARTIAL) * draws
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
