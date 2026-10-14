"""Unhinged-run data helpers (sft/vibe.py + the new sft_longform sources). No network."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

import sft.sft_longform as sl
from babble.blocklist import Blocklist
from sft.sft_longform import SFTRecord, _data_extras, _discord_vibe_group, _skip_seen, build_examples
from sft.vibe import (
    LocalStats,
    coherence_checks,
    conversation_verdict,
    corpus_prompts,
    engagement_reason,
    is_meta_instruction,
    local_snapshot_groups,
    local_snapshot_records,
    snapshot_sha256,
    trim_to_user_start,
    vibe_features,
    vibe_score,
)

BLOCK = Blocklist(frozenset({"badword"}))


# ------------------------------------------------------------- scoring ---


def test_vibe_rewards_discord_energy_over_dry_text():
    assert vibe_score("LMAOOO no way 💀") > 2.5
    assert vibe_score("bro really said that 😭😭") > 2.5
    assert vibe_score("hiiii bestie!!") > 1.5
    assert vibe_score("I think the answer depends on the context of the question.") == 0
    feats = vibe_features("LMAOOO no way 💀")
    assert {"laughter", "elongated", "emoji", "caps_burst"} <= feats.keys()


def test_vibe_penalizes_assistant_speak_fillers_urls_spam_and_walls():
    assert vibe_score("Certainly! I would be happy to help you with that.") <= -2.5
    assert "assistant_speak" in vibe_features("As an AI, I cannot have opinions.")
    assert vibe_features("ok")["filler"] < 0 and vibe_score("ok") < 0
    assert vibe_features("okkk").get("filler", 0) < 0
    assert vibe_features("ok lmao that is wild").get("filler") is None  # filler only when it's the whole reply
    assert vibe_features("look https://example.com/x")["url"] < 0
    assert vibe_features("12345678901234")["spam"] < 0
    assert vibe_features("aaaaaaaaaaaaaaaaaaaa")["spam"] < 0
    assert vibe_features(" ".join(["copy pasta"] * 10))["copypasta"] < 0
    assert vibe_features("word " * 300)["wall_of_text"] <= -1.0


def test_vibe_features_saturate_and_sum_to_score():
    text = "lol lol lol lol lol lol"
    assert vibe_features(text)["laughter"] == pytest.approx(1.6)  # capped at 2 hits
    for t in ("LMAOOO no way 💀", "ok", "*bonks you* go to sleep", "[gif: cat laughing]"):
        assert vibe_score(t) == pytest.approx(sum(vibe_features(t).values()))
    assert vibe_features("[gif: cat laughing]")["gif_tag"] > 0


# ---------------------------------------------------------- conversation ---


def _conv(*pairs):
    turns = []
    for u, a in pairs:
        turns += [("user", u), ("assistant", a)]
    return turns


VIBEY = _conv(
    ("did you see the cat video", "LMAOOO the cat video 💀"),
    ("the cat fell off the table", "nooo the poor cat 😭 lmao"),
)


def test_engagement_rules():
    assert engagement_reason("do you like pizza?", "yeah its fire") == "answer"
    assert engagement_reason("my cat is sick", "oh no is your cat ok") == "overlap"
    assert engagement_reason("what are you studying?", "everything") == "answer"
    assert engagement_reason("he tripped on stage", "LMAOOO") == "reaction"
    assert engagement_reason("good luck bro", "thanks man") == "ack"
    assert engagement_reason("hello", "the moon is cheese") is None


def test_conversation_verdict_keeps_vibey_engaged_multi_turn():
    v = conversation_verdict(VIBEY, min_vibe=0.75, min_targets=2, blocklist=BLOCK)
    assert v.ok and v.reason == "ok" and v.targets == 2 and v.engaged_frac == 1.0


def test_conversation_verdict_rejections():
    assert conversation_verdict(VIBEY[:2], min_targets=2).reason == "too_few_targets"
    one_word = _conv(("hi", "yo"), ("sup", "nm"), ("cool", "lmao"))
    assert conversation_verdict(one_word, min_vibe=0, min_targets=2).reason == "one_word"
    non_seq = _conv(("my exam is tomorrow", "LMAOOO the moon 💀"), ("i studied all week", "bananas are great lol 😭"))
    assert conversation_verdict(non_seq, min_vibe=0, min_targets=2).reason == "low_engagement"
    dry = _conv(("did you see the game", "Yes, the game was good."), ("the ending was close", "The ending was very close."))
    assert conversation_verdict(dry, min_vibe=0.75, min_targets=2).reason == "low_vibe"
    robot = _conv(("can you help with my essay", "Certainly! I would be happy to help with your essay."), ("thanks the essay", "the essay lol 💀"))
    assert conversation_verdict(robot, min_vibe=0, min_targets=2).reason == "bad_target"
    blocked = _conv(("the cat badword video", "LMAOOO the cat 💀"), ("the cat fell", "nooo the cat 😭"))
    assert conversation_verdict(blocked, min_vibe=0, min_targets=2, blocklist=BLOCK).reason == "blocklist"
    assert conversation_verdict([("assistant", "x"), ("user", "y")]).reason == "malformed"


def test_trim_to_user_start_drops_leading_assistant_turns():
    assert trim_to_user_start([("assistant", "a"), ("user", "b"), ("assistant", "c")]) == [("user", "b"), ("assistant", "c")]


def _chatml(turns):
    return "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in turns) + "<|end_of_text|>"


def test_discord_vibe_group_uses_discord_group_id_and_respects_blocklist():
    raw = _chatml([("assistant", "opening line"), *VIBEY])
    records = _discord_vibe_group(raw, min_vibe=0.75, min_targets=2, blocklist=BLOCK)
    assert [r.response for r in records] == ["LMAOOO the cat video 💀", "nooo the poor cat 😭 lmao"]
    assert {r.source for r in records} == {"discord_vibe"}
    assert {r.group_id for r in records} == {sl._group_id(raw)}  # same id `discord` would use
    stats = sl.VibeFilterStats()
    bad = _chatml(_conv(("the cat badword video", "LMAOOO the cat 💀"), ("the cat fell", "nooo the cat 😭")))
    assert _discord_vibe_group(bad, min_vibe=0, min_targets=2, blocklist=BLOCK, stats=stats) == []
    assert stats.rejected["blocklist"] == 1


def test_discord_vibe_gif_handling_matches_discord():
    turns = _conv(
        ("did you see the cat", "LMAOOO the cat https://tenor.com/view/cat-laughing-funny-gif-123"),
        ("the cat fell off", "nooo the poor cat 😭 lmao"),
    )
    gif = sl.GifStats(0.0, 0.0, 0)
    records = _discord_vibe_group(_chatml(turns), min_vibe=0.75, min_targets=2, blocklist=BLOCK, gif=gif)
    assert records[0].response == "LMAOOO the cat [gif: cat laughing funny]"
    assert gif.real == 1 and gif.targets == 2


# -------------------------------------------------------- local snapshot ---


def _write_snapshot(root: Path, rows: list[dict], corpus: list[dict] | None = None) -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if corpus is not None:
        (root / "data" / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in corpus), encoding="utf-8")
    return root


def _row(prompt, chosen, signal="correction", rejected="meh"):
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected if signal == "correction" else None, "signal": signal, "weight": 1.0}


@pytest.mark.parametrize(
    "chosen",
    [
        'say something like "nice try buddy" whenever someone says the toaster is broken',
        'A comet is ice and dust. when someone is asking you "what is a comet", you\'d say a dirty snowball.',
        "reply with something funnier next time",
        "you should say hi back",
        "booper should be nicer here",
        "instead of saying that, ask how they are",
        "if someone says hello, wave",
    ],
)
def test_meta_instruction_corrections_are_detected(chosen):
    assert is_meta_instruction(chosen)


@pytest.mark.parametrize("chosen", ["hey bestie", "TACO TACO TACO", "only on tuesdays", "huh", "i am a small potato"])
def test_direct_reply_corrections_are_not_meta(chosen):
    assert not is_meta_instruction(chosen)


def test_local_snapshot_keeps_approvals_and_direct_corrections_only(tmp_path):
    snap = _write_snapshot(tmp_path / "snap", [
        _row("hi", "hiii :3", signal="approval"),
        _row("are you a robot", "beep boop im a toaster"),
        _row("my bot is bad", 'say something like "lol" whenever someone says bot'),
        _row("question", "you'd say whatever when someone is asking you that"),
        _row("", "empty prompt"),
        _row("blocked", "this has badword in it"),
        _row("link", "https://example.com"),
        _row("hi", "hiii :3", signal="approval"),  # duplicate
        {"prompt": "x", "chosen": "y", "signal": "other"},
    ])
    stats = LocalStats()
    records = local_snapshot_records(snap, blocklist=BLOCK, stats=stats)
    assert [(r.current_user, r.response) for r in records] == [("hi", "hiii :3"), ("are you a robot", "beep boop im a toaster")]
    assert {r.source for r in records} == {"booper_local"} and all(not r.history for r in records)
    assert stats.rows == 9 and stats.kept == 2
    assert stats.rejected == {"meta_instruction": 2, "empty": 1, "blocklist": 1, "url": 1, "duplicate": 1, "unknown_signal": 1}
    assert stats.by_signal == {"approval": 1, "correction": 1}


def test_local_snapshot_groups_by_normalized_prompt(tmp_path):
    snap = _write_snapshot(tmp_path / "s", [_row("Hi!", "yo"), _row("hi", "hey there"), _row("bye", "cya")])
    groups = list(local_snapshot_groups(snap, blocklist=BLOCK))
    assert [len(g) for g in groups] == [2, 1]
    assert len({r.group_id for r in groups[0]}) == 1


def test_local_snapshot_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        local_snapshot_records(tmp_path / "missing", blocklist=BLOCK)
    empty = _write_snapshot(tmp_path / "empty", [])
    with pytest.raises(ValueError, match="no usable rows"):
        local_snapshot_records(empty, blocklist=BLOCK)
    meta_only = _write_snapshot(tmp_path / "meta", [_row("x", "you should say hi whenever someone says x")])
    with pytest.raises(ValueError, match="no usable rows"):
        local_snapshot_records(meta_only, blocklist=BLOCK)
    # one usable dir alongside an empty one is fine
    good = _write_snapshot(tmp_path / "good", [_row("hi", "hiii")])
    assert len(local_snapshot_records([good, empty], blocklist=BLOCK)) == 1


def test_snapshot_sha256_tracks_file_bytes_not_paths(tmp_path):
    a = _write_snapshot(tmp_path / "a", [_row("hi", "hiii")])
    b = _write_snapshot(tmp_path / "b", [_row("hi", "hiii")])
    assert snapshot_sha256([a]) == snapshot_sha256([b])
    before = snapshot_sha256([a])
    _write_snapshot(a, [_row("hi", "hiii"), _row("yo", "yooo")])
    assert snapshot_sha256([a]) != before


def test_corpus_prompts_use_direct_sources_and_blocklist(tmp_path):
    snap = _write_snapshot(tmp_path / "s", [], corpus=[
        {"text": "hey booper", "source": "mention"},
        {"text": "dm text", "source": "dm"},
        {"text": "ambient chatter", "source": "ambient"},
        {"text": "badword here", "source": "mention"},
        {"text": "hey booper", "source": "reply"},
    ])
    assert corpus_prompts(snap, blocklist=BLOCK) == ["hey booper", "dm text"]


# ------------------------------------------------- build integration ---


class _FakeDS:
    def __init__(self, rows):
        self.rows = rows

    def skip(self, n):
        return _FakeDS(self.rows[n:])

    def __iter__(self):
        return iter(self.rows)


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _CharTok:
    specials = {"<bos>": 1, "<sep>": 2, "<eos>": 3, "<pad>": 0}

    def token_to_id(self, token):
        return self.specials[token]

    def encode(self, text, add_special_tokens=False):
        return _Enc([4 + ord(c) % 1000 for c in text])


def _build_args(**over):
    base = dict(
        mix_story=0.0, mix_wp=0.0, mix_norobots=0.0, repeat_norobots=1, mix_smoltalk=0.0, smoltalk_multiturn=False,
        mix_discord=0.5, mix_ultrachat=0.0, mix_discord_vibe=0.5, discord_vibe_min=0.75, discord_vibe_min_targets=2,
        discord_vibe_skip_rows=0, mix_local=0.0, repeat_local=1, local_snapshot=None, examples=24, val_examples=8,
        seed=3, history_turns=8, discord_revision="r", gif_tags=False, gif_synth_rate=0.0, gif_synth_max_frac=0.0,
        prompt_budget=512, seq_len=1024, min_response=1, log_examples=0,
    )
    base.update(over)
    return Namespace(**base)


def _vibey_rows(n):
    return [{"text": _chatml(_conv(
        (f"did you see the cat{i} video", f"LMAOOO the cat{i} video 💀"),
        (f"the cat{i} fell off the table", f"nooo the poor cat{i} 😭 lmao"),
    ))} for i in range(n)]


def test_discord_and_discord_vibe_never_share_a_conversation(monkeypatch):
    """Same dataset, same head of the stream: discord_vibe must skip whatever discord took (train or val)."""
    import datasets

    rows = _vibey_rows(40)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeDS(rows))
    captured: dict[str, list[SFTRecord]] = {}
    real_split = sl._split_grouped

    def spy(records, val_examples, *, seed):
        if records:
            captured[records[0].source] = list(records)
        return real_split(records, val_examples, seed=seed)

    monkeypatch.setattr(sl, "_split_grouped", spy)
    _, val, counts = build_examples(_CharTok(), _build_args(), lambda m: None)
    discord, vibe = captured["discord"], captured["discord_vibe"]
    assert discord and vibe
    assert {r.group_id for r in discord}.isdisjoint({r.group_id for r in vibe})
    assert {(r.current_user, r.response) for r in discord}.isdisjoint({(r.current_user, r.response) for r in vibe})
    assert counts["discord_vibe"]["vibe_skipped_seen"] == counts["discord"]["groups"]
    assert val["discord_vibe"] and val["discord_vibe_multiturn"]


def test_skip_seen_matches_group_id_even_when_targets_differ():
    seen_groups = {"g1"}
    groups = [[SFTRecord("discord_vibe", "g1", "p", "different synthetic gif")], [SFTRecord("discord_vibe", "g2", "p2", "r2")]]
    stats = sl.VibeFilterStats()
    kept = list(_skip_seen(groups, set(), seen_groups, stats))
    assert [g[0].group_id for g in kept] == ["g2"] and stats.skipped_seen == 1


def test_booper_local_is_train_only_and_repeated(monkeypatch, tmp_path):
    import datasets

    snap = _write_snapshot(tmp_path / "s", [_row(f"hello there {i}", f"hiii friend {i} :3") for i in range(5)])
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeDS(_vibey_rows(40)))
    args = _build_args(mix_discord=0.6, mix_discord_vibe=0.0, mix_local=0.4, repeat_local=3, local_snapshot=[str(snap)])
    train, val, counts = build_examples(_CharTok(), args, lambda m: None)
    assert "booper_local" not in val
    assert counts["booper_local"]["train"] == 9  # int(24 * 0.4) = 9 <= 5 unique x 3
    assert counts["booper_local"]["val"] == 0 and counts["booper_local"]["local_kept"] == 5


# -------------------------------------------------- metadata / cache key ---


def test_new_knobs_enter_data_extras_and_cache_key(tmp_path, monkeypatch):
    old = Namespace(mix_ultrachat=0.28, ultrachat_revision="r", smoltalk_multiturn=True, gif_tags=True,
                    gif_synth_rate=0.5, gif_synth_max_frac=0.025)
    assert not any(k.startswith(("mix_discord_vibe", "mix_local", "vibe_", "local_", "blocklist")) for k in _data_extras(old))
    snap = _write_snapshot(tmp_path / "s", [_row("hi", "hiii")])
    new = Namespace(**vars(old), mix_discord_vibe=0.45, discord_vibe_min=0.75, discord_vibe_min_targets=1,
                    discord_vibe_skip_rows=233989, mix_local=0.01, repeat_local=3, local_snapshot=[str(snap)])
    extras = _data_extras(new)
    assert extras["mix_discord_vibe"] == 0.45 and extras["discord_vibe_min"] == 0.75
    assert extras["discord_vibe_min_targets"] == 1 and extras["discord_vibe_skip_rows"] == 233989
    assert extras["mix_local"] == 0.01 and extras["repeat_local"] == 3
    assert extras["local_snapshot_sha256"] == snapshot_sha256([snap])
    assert "hiii" not in json.dumps(extras) and str(snap) not in json.dumps(extras)
    assert extras["vibe_data_version"] == sl.VIBE_DATA_VERSION and len(extras["blocklist_sha256"]) == 64

    # _cached_build keys on _data_extras: a changed snapshot or threshold rebuilds.
    keys = []

    def fake_build(tok, args, log):
        keys.append(args)
        return [], {}, {}

    monkeypatch.setattr(sl, "build_examples", fake_build)
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text("{}", encoding="utf-8")
    full = Namespace(**vars(new), examples=10, val_examples=2, seed=0, seq_len=64, prompt_budget=32, history_turns=2,
                     min_response=1, mix_story=0, mix_wp=0, mix_norobots=0, repeat_norobots=1, mix_smoltalk=0,
                     mix_discord=0.5, tinystories_revision=None, writingprompts_revision=None, no_robots_revision=None,
                     smoltalk_revision=None, discord_revision="r")
    run = tmp_path / "run"
    run.mkdir()
    sl._cached_build(None, tok_path, full, run, lambda m: None)
    sl._cached_build(None, tok_path, full, run, lambda m: None)
    assert len(keys) == 1  # second call reused the cache
    _write_snapshot(snap, [_row("hi", "hiii"), _row("yo", "yooo")])
    sl._cached_build(None, tok_path, full, run, lambda m: None)
    assert len(keys) == 2
    full.discord_vibe_min = 1.0
    sl._cached_build(None, tok_path, full, run, lambda m: None)
    assert len(keys) == 3


def test_mix_local_without_snapshot_is_an_error():
    with pytest.raises(ValueError, match="--local-snapshot"):
        _data_extras(Namespace(mix_local=0.1, repeat_local=1, local_snapshot=None))


def test_unhinged_preset_keeps_longctx_serving_contract():
    root = Path(__file__).resolve().parents[1] / "configs" / "sft"
    longctx = json.loads((root / "longctx-mac.json").read_text())
    preset = json.loads((root / "unhinged-mac.json").read_text())
    for key in ("seq_len", "prompt_budget", "history_turns", "gif_tags", "discord_revision"):
        assert preset[key] == longctx[key]
    assert preset["repeat_local"] <= 3 and preset["repeat_persona"] <= 3
    assert {"discord", "persona"} <= set(preset["guard_sources"])
    assert preset["multiturn_must_improve"]
    assert "local_snapshot" not in preset  # machine-local paths go on the command line


# ------------------------------------------------------------- vibe_eval ---


def test_coherence_checks():
    assert coherence_checks("do you like pizza?", "yeah pizza is fire lol")["coherent"]
    assert coherence_checks("what's 2+2", "what's 2+2")["echo"]
    assert coherence_checks("hi", "the the the the the the the the")["degenerate"]
    assert coherence_checks("hi", "")["empty"]
    assert not coherence_checks("my exam is tomorrow", "bananas are yellow")["coherent"]


def test_vibe_eval_summary_and_prompt_pick():
    from sft.vibe_eval import compare, pick_prompts, score_replies, summarize

    prompts = [f"p{i}" for i in range(50)]
    assert pick_prompts(prompts, 10, 0) == pick_prompts(list(reversed(prompts)), 10, 0)
    assert pick_prompts(prompts, 10, 0) != pick_prompts(prompts, 10, 1)
    rows = score_replies([
        {"prompt": "do you like pizza?", "reply": "yeah pizza is fire lmaooo", "tokens": 6},
        {"prompt": "hello", "reply": "hello", "tokens": 1},
    ])
    s = summarize(rows)
    assert s["replies"] == 2 and s["echo"] == 0.5 and s["coherent"] == 0.5
    assert s["vibe_mean"] == pytest.approx((rows[0]["vibe"] + rows[1]["vibe"]) / 2, abs=1e-4)
    assert compare({"summary": s}, {"summary": {**s, "coherent": 1.0}})["coherent"] == 0.5


def test_vibe_eval_runs_a_tiny_int8_export_through_sample(tmp_path):
    """End-to-end model path on a tiny random Mixtral in the live INT8 layout."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from sft.vibe_eval import run_eval, summarize

    specials = ["<pad>", "<bos>", "<sep>", "<eos>"]
    words = ["user", ":", "hi", "booper", "lol", "yo", "cat", "what"]
    vocab = {t: i for i, t in enumerate(specials + words)}
    tok = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(specials)
    src, out = tmp_path / "src", tmp_path / "tiny"
    src.mkdir()
    tok.save(str(src / "tokenizer.json"))
    torch.manual_seed(0)
    config = transformers.MixtralConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, num_local_experts=2, num_experts_per_tok=1, max_position_embeddings=128,
        tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=3,
    )
    config.babble_prompt_budget = 64
    model = transformers.MixtralForCausalLM(config).eval()
    sl.export_int8(model, config, src / "tokenizer.json", src, out, lambda m: None)
    rows = run_eval(out, ["hi booper", "what cat lol"], max_new=6, seed=0, log=lambda m: None)
    assert len(rows) == 2 and {r["prompt"] for r in rows} == {"hi booper", "what cat lol"}
    assert all(isinstance(r["vibe"], float | int) and "coherent" in r["checks"] for r in rows)
    assert summarize(rows)["replies"] == 2
