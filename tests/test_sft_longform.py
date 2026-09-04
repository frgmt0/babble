from babble.conversation import ConversationTurn, conversation_prompt
from types import SimpleNamespace

from sft.sft_longform import (
    SFTRecord,
    _chatml_turns,
    _dedupe_groups,
    _discord_group,
    _split_grouped,
    _source_gate,
    _tokenize_records,
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
    baseline = {"discord": 1.0, "discord_legacy": 1.2}
    assert _source_gate({"discord": 0.9}, baseline, 0.05)[0] is False
    assert _source_gate({"discord": 0.9, "discord_legacy": float("nan")}, baseline, 0.05)[0] is False
    assert _source_gate({"discord": 0.9, "discord_legacy": 1.24}, baseline, 0.05)[0] is True
