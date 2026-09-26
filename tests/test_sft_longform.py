from babble.conversation import ConversationTurn, conversation_prompt
import json
from types import SimpleNamespace

import pytest

from sft.sft_longform import (
    GIF_TAG_RE,
    GifStats,
    SFTRecord,
    _batch_groups,
    batches,
    _chatml_turns,
    _dedupe_groups,
    _discord_group,
    _messages_group,
    _restore_prompt_metadata,
    _split_grouped,
    _source_gate,
    _tokenize_records,
    duty_sleep_seconds,
    gif_url_to_tag,
    on_battery,
    reaction_gif_words,
    rewrite_gifs,
)


RAW = """<|im_start|>user
one<|im_end|>
<|im_start|>assistant
two<|im_end|>
<|im_start|>user
three<|im_end|>
<|im_start|>assistant
four<|im_end|><|end_of_text|>"""


def test_chatml_conversation_yields_every_assistant_turn_with_runtime_format():
    assert _chatml_turns(RAW) == [
        ("user", "one"),
        ("assistant", "two"),
        ("user", "three"),
        ("assistant", "four"),
    ]
    records = _discord_group(RAW, history_turns=3)
    assert [
        (
            conversation_prompt(record.history, record.current_user, max_turns=3, max_chars=0),
            record.response,
        )
        for record in records
    ] == [
        ("user: one", "two"),
        ("user: one\nassistant: two\nuser: three", "four"),
    ]
    assert len({record.group_id for record in records}) == 1


def test_history_is_bounded_by_completed_exchange_count():
    raw = RAW.replace("<|end_of_text|>", "") + """
<|im_start|>user
five<|im_end|>
<|im_start|>assistant
six<|im_end|>"""
    records = _discord_group(raw, history_turns=1)
    assert conversation_prompt(
        records[-1].history, records[-1].current_user, max_turns=1, max_chars=0
    ) == "user: three\nassistant: four\nuser: five"


def test_group_split_never_leaks_sibling_targets():
    records = [
        SFTRecord("discord", "conversation-a", "p1", "a1"),
        SFTRecord("discord", "conversation-a", "p2", "a2"),
        SFTRecord("discord", "conversation-b", "p3", "a3"),
        SFTRecord("discord", "conversation-c", "p4", "a4"),
    ]
    train, val = _split_grouped(records, 1, seed=7)
    assert {r.group_id for r in train}.isdisjoint({r.group_id for r in val})
    assert len(train) + len(val) == len(records)


def test_dedup_is_cross_source_and_drops_the_whole_conversation_group():
    seen: set[str] = set()
    first, dropped = _dedupe_groups(
        [SFTRecord("one", "g1", "same prompt", "same answer")], seen
    )
    second, dropped_second = _dedupe_groups(
        [
            SFTRecord("two", "g2", "same prompt", "same answer"),
            SFTRecord("two", "g2", "follow up", "response"),
        ],
        seen,
    )
    assert first and dropped == 0
    assert second == [] and dropped_second == 1


def test_token_budget_keeps_role_boundary_instead_of_slicing_mid_history():
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    class CharTokenizer:
        specials = {"<bos>": 1000, "<sep>": 1001, "<eos>": 1002}

        def token_to_id(self, token):
            return self.specials[token]

        def encode(self, text, add_special_tokens=False):
            return Encoded([ord(c) for c in text])

    record = SFTRecord(
        "discord",
        "conversation",
        "the newest question is also much too long",
        "answer",
        history=(
            ConversationTurn(user="old question", assistant="old answer"),
        ),
    )
    args = SimpleNamespace(prompt_budget=24, history_turns=3, min_response=1, seq_len=128)
    ((ids, n_prompt),) = _tokenize_records(CharTokenizer(), [record], args)
    prompt = "".join(chr(i) for i in ids[1 : n_prompt - 1])
    assert prompt.startswith("user: ")
    assert "assistant:" not in prompt


def test_source_gate_fails_closed_on_missing_or_nonfinite_metrics():
    baseline = {
        "discord": 1.0,
        "discord_single": 1.1,
        "discord_legacy": 0.9,
        "discord_multiturn": 1.2,
    }
    candidate = {
        "discord": 1.01,
        "discord_single": 0.94,
        "discord_legacy": 0.91,
        "discord_multiturn": 1.19,
    }
    assert _source_gate({"discord": 0.9}, baseline, 0.05)[0] is False
    assert _source_gate({**candidate, "discord_legacy": float("nan")}, baseline, 0.05)[0] is False
    passed, regressions = _source_gate(candidate, baseline, 0.05)
    assert passed is True
    assert regressions["discord_migration"] == candidate["discord_single"] - baseline["discord_legacy"]
    assert _source_gate({**candidate, "discord_single": 0.96}, baseline, 0.05)[0] is False
    assert _source_gate({**candidate, "discord_multiturn": 1.2}, baseline, 0.05)[0] is False


def test_reexport_uses_saved_prompt_metadata_without_inventing_it(tmp_path):
    config = SimpleNamespace(
        babble_prompt_format="role_transcript_v1",
        babble_history_turns=3,
        babble_prompt_budget=512,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mixtral"}))
    _restore_prompt_metadata(config, tmp_path)
    assert not hasattr(config, "babble_prompt_format")
    assert not hasattr(config, "babble_history_turns")
    assert not hasattr(config, "babble_prompt_budget")

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "babble_prompt_format": "role_transcript_v1",
                "babble_history_turns": 2,
                "babble_prompt_budget": 384,
            }
        )
    )
    _restore_prompt_metadata(config, tmp_path)
    assert config.babble_prompt_format == "role_transcript_v1"
    assert config.babble_history_turns == 2
    assert config.babble_prompt_budget == 384


# ------------------------------------------------------------ gif tags ---


@pytest.mark.parametrize(
    "url, tag",
    [
        ("https://tenor.com/view/cat-laughing-funny-gif-12345", "[gif: cat laughing funny]"),
        ("https://tenor.com/en-GB/view/spongebob-dance-gif-26211870", "[gif: spongebob dance]"),
        ("<https://tenor.com/view/the-office-michael-scott-no-god-please-no-gif-1234>", None),  # 7 words
        ("https://giphy.com/gifs/reaction-shocked-pikachu-l0HlBO7eyXzSZkJri", "[gif: reaction shocked pikachu]"),
        ("https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif", None),
        ("https://tenor.com/bQ3xY.gif", None),
        ("https://cdn.discordapp.com/attachments/1/2/dancing_fungus.gif", "[gif: dancing fungus]"),
        ("https://tenor.com/view/wow-gif-999", None),  # one word is not a search
    ],
)
def test_gif_url_to_tag_uses_slug_and_drops_ids(url, tag):
    assert gif_url_to_tag(url) == tag
    if tag:
        assert GIF_TAG_RE.fullmatch(tag)


def test_rewrite_gifs_moves_tag_to_end_and_leaves_no_gif_url():
    text, made, dropped = rewrite_gifs("look https://tenor.com/view/cat-laughing-funny-gif-12345 lol")
    assert (text, made, dropped) == ("look lol [gif: cat laughing funny]", 1, 0)

    text, made, dropped = rewrite_gifs("https://media.giphy.com/media/abcDEF123xyz/giphy.gif")
    assert (text, made, dropped) == ("", 0, 1)

    # Non-gif URLs are left for the caller to judge.
    assert rewrite_gifs("see https://example.com/page")[0] == "see https://example.com/page"


def test_rewrite_gifs_handles_discord_attachment_filenames():
    assert rewrite_gifs("SipsBubble.gif")[0] == "[gif: sips bubble]"
    assert rewrite_gifs("Danny devito clapping and crying.gif")[0] == "[gif: danny devito clapping and crying]"
    assert rewrite_gifs("name it jerma_cute.gif")[0] == "name it [gif: jerma cute]"
    # single words and prose that merely mentions .gif are untouched
    assert rewrite_gifs("gasp.gif") == ("gasp.gif", 0, 0)
    prose = "Try downloading it and changing it into a.gif"
    assert rewrite_gifs(prose) == (prose, 0, 0)


def test_discord_targets_with_raw_urls_are_skipped_but_kept_as_history():
    raw = """<|im_start|>user
send a gif<|im_end|>
<|im_start|>assistant
https://tenor.com/view/dog-dancing-gif-555<|im_end|>
<|im_start|>user
where is that from<|im_end|>
<|im_start|>assistant
https://example.com/source<|im_end|>
<|im_start|>user
ok<|im_end|>
<|im_start|>assistant
cool<|im_end|>"""
    stats = GifStats()
    records = _discord_group(raw, history_turns=8, gif=stats)
    assert [r.response for r in records] == ["[gif: dog dancing]", "cool"]
    assert records[-1].history[-1].assistant == "https://example.com/source"
    assert stats.real == 1 and stats.url_targets_skipped == 1 and stats.targets == 2


def test_reaction_synthesis_is_deterministic_and_capped():
    assert reaction_gif_words("LMAOOOO") is not None
    assert reaction_gif_words("bruh.") is not None
    assert reaction_gif_words("no way") is not None
    assert reaction_gif_words("what are you doing tonight") is None

    def run(rate, frac):
        stats = GifStats(synth_rate=rate, synth_frac=frac, seed=3)
        out = []
        for i in range(1000):
            text = ["lmao", "tell me more about it", "bruh", "no way", "ok sure"][i % 5]
            made = stats.maybe_synthesize(f"g{i}", text)
            reply = made or text
            stats.count_target(reply, synthetic=bool(made))
            out.append(reply)
        return stats, out

    stats, out = run(1.0, 0.03)
    assert 0 < stats.synthetic <= 0.03 * stats.targets
    assert all(GIF_TAG_RE.search(o) and o.endswith("]") for o in out if "[gif:" in o)
    assert run(1.0, 0.03)[1] == out  # deterministic by content
    assert run(0.0, 0.03)[0].synthetic == 0


# ------------------------------------------------------- new sources ---


def test_messages_group_targets_every_assistant_turn_with_history():
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "what is rust?"},
        {"role": "assistant", "content": "a systems language"},
        {"role": "user", "content": "is it fast?"},
        {"role": "assistant", "content": "yes, like C++"},
        {"role": "user", "content": "thanks"},
    ]
    records = _messages_group("ultrachat", messages, GifStats())
    assert [(r.current_user, r.response, len(r.history)) for r in records] == [
        ("what is rust?", "a systems language", 0),
        ("is it fast?", "yes, like C++", 1),
    ]
    assert {r.source for r in records} == {"ultrachat"}
    assert len({r.group_id for r in records}) == 1
    # a conversation that breaks alternation is dropped entirely
    bad = messages[:3] + [
        {"role": "assistant", "content": "again"},
        {"role": "user", "content": "hm"},
        {"role": "assistant", "content": "yes"},
    ]
    assert _messages_group("ultrachat", bad) == []


def test_long_response_trims_history_to_fit_sequence_instead_of_skipping():
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    class CharTokenizer:
        specials = {"<bos>": 1000, "<sep>": 1001, "<eos>": 1002}

        def token_to_id(self, token):
            return self.specials[token]

        def encode(self, text, add_special_tokens=False):
            return Encoded([ord(c) for c in text])

    record = SFTRecord(
        "ultrachat",
        "g",
        "and then?",
        "x" * 80,
        history=(ConversationTurn(user="first question here", assistant="y" * 40),),
    )
    args = SimpleNamespace(prompt_budget=100, history_turns=8, min_response=1, seq_len=128)
    ((ids, n_prompt),) = _tokenize_records(CharTokenizer(), [record], args)
    assert len(ids) <= 128
    assert "".join(chr(i) for i in ids[1 : n_prompt - 1]) == "user: and then?"
    # batching accepts the compact uint16 arrays, padded to a shape multiple
    assert _batch_groups([(ids, n_prompt)], 4096) == [[0]]
    ((batch_ids, labels),) = list(batches([(ids, n_prompt)], 4096, 0, pad_multiple=64))
    assert batch_ids.shape == (1, 128) and int((labels != -100).sum()) == 81
    # fixed rows: filler rows carry only a BOS and no loss
    ((batch_ids, labels),) = list(batches([(ids, n_prompt)], 1024, 0, pad_multiple=256, fixed_rows=True))
    assert batch_ids.shape == (4, 256)
    assert batch_ids[1:, 0].tolist() == [1000] * 3 and int((batch_ids[1:, 1:] != 0).sum()) == 0
    assert int((labels != -100).sum()) == 81


def test_batch_shape_set_is_small_with_fixed_rows():
    import random

    from sft.sft_longform import batch_shape

    rng = random.Random(0)
    examples = [([1] * rng.randint(4, 2048), 1) for _ in range(3000)]
    groups = _batch_groups(examples, 2048, shuffle_seed=3, pad_multiple=256)
    shapes = {
        batch_shape(len(g), max(len(examples[i][0]) for i in g), 2048, pad_multiple=256, fixed_rows=True)
        for g in groups
    }
    assert len(shapes) <= 8
    assert all(rows * width <= 2048 for rows, width in shapes)
    assert sorted(i for g in groups for i in g) == list(range(3000))


def test_bucketed_experts_match_the_eager_mixtral_loop():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from sft.sft_longform import use_bucketed_experts

    config = transformers.MixtralConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, num_local_experts=5, num_experts_per_tok=2,
    )
    torch.manual_seed(0)
    model = transformers.MixtralForCausalLM(config).eval()
    ids = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        eager = model(input_ids=ids).logits
    assert use_bucketed_experts(model, bucket=4) == 1
    with torch.no_grad():
        grouped = model(input_ids=ids).logits
    torch.testing.assert_close(grouped, eager, rtol=1e-4, atol=1e-4)
    # gradients reach the fp32 expert weights
    model.train()
    model(input_ids=ids).logits.sum().backward()
    experts = next(m for m in model.modules() if type(m).__name__ == "MixtralExperts")
    assert experts.gate_up_proj.grad is not None and experts.gate_up_proj.grad.abs().sum() > 0


# -------------------------------------------------------------- duty ---


def test_duty_cycle_sleep_math():
    assert duty_sleep_seconds(2.0, 1.0) == 0.0
    assert duty_sleep_seconds(2.0, 0.5) == pytest.approx(2.0)
    assert duty_sleep_seconds(1.0, 0.25) == pytest.approx(3.0)
    compute = 1.7
    nap = duty_sleep_seconds(compute, 0.6)
    assert compute / (compute + nap) == pytest.approx(0.6)
    assert duty_sleep_seconds(-1.0, 0.5) == 0.0
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            duty_sleep_seconds(1.0, bad)


def test_battery_parse():
    assert on_battery("Now drawing from 'Battery Power'\n -InternalBattery-0 80%; discharging;")
    assert not on_battery("Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged;")
    assert not on_battery("")
