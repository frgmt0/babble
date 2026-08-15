"""The two-stage pipeline: a frozen base, a voice pass that always restarts from
it, the +N-row trigger, and the 512 geometry. The consent gate is untouched --
the external corpus never enters it, and the human pass reads through it exactly
as the old trainer did."""

from __future__ import annotations

import json

import pytest
import torch

from babble.cli import build_parser
from babble.consent import SCOPE_CORPUS, ConsentStore
from babble.corpus import SOURCE_MENTION, CorpusRow, CorpusStore, make_corpus_id
from babble.external import prepare_base_corpus
from babble.pretrain import (
    VoiceAutoTrigger,
    archive_existing_checkpoints,
    pretrain_base,
    voice_pass,
    voice_trigger,
)


# --- helpers -------------------------------------------------------------


def _prepare_base_corpus(settings, tmp_path):
    words = tmp_path / "w.txt"
    words.write_text("apple\nbanana\ncherry\ndog\ncat\nhouse\n", encoding="utf-8")
    stories = tmp_path / "s.txt"
    stories.write_text("A cat sat on a mat.<|endoftext|>A dog ran home fast.", encoding="utf-8")
    return prepare_base_corpus(settings, wordlist_path=words, stories_path=stories)


def _seed_human(settings, ids, texts, *, author_raw="alice-raw"):
    ConsentStore(settings.consent_path).grant(author_raw, SCOPE_CORPUS)
    store = CorpusStore(settings.corpus_path)
    author = ids.user(author_raw)
    for text in texts:
        store.append(
            CorpusRow(id=make_corpus_id(text, author), text=text, author=author, source=SOURCE_MENTION)
        )


def _model_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def _states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- stage 1: base -------------------------------------------------------


def test_base_pretrain_writes_a_frozen_base_and_reports_loss(settings, tmp_path):
    _prepare_base_corpus(settings, tmp_path)

    result = pretrain_base(settings, steps=6, seed=1, echo=False)

    assert result.stage == "base"
    assert result.final_step == 6
    assert result.checkpoints_written >= 1
    assert settings.base_checkpoint.exists()
    # It does NOT write latest.pt -- that is the voice pass's job.
    assert not settings.latest_checkpoint.exists()
    # Loss is reported the way the trainer reports it: appended to loss.jsonl.
    curve = settings.loss_curve_path.read_text(encoding="utf-8").splitlines()
    assert curve and "loss" in json.loads(curve[-1])


def test_base_pretrain_archives_existing_checkpoints_never_deletes(settings, tmp_path):
    # A stale checkpoint from before the geometry change is sitting in the dir.
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (settings.latest_checkpoint).write_bytes(b"stale-latest")
    (settings.checkpoint_dir / "ckpt-0000042.pt").write_bytes(b"stale-archive")

    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=2, seed=1, echo=False)

    # The stale files are gone from the live dir but preserved under archive/.
    assert not (settings.checkpoint_dir / "ckpt-0000042.pt").exists()
    archived = list(settings.checkpoint_archive_dir.rglob("*"))
    names = {p.name for p in archived if p.is_file()}
    assert "ckpt-0000042.pt" in names
    assert "latest.pt" in names


def test_base_pretrain_without_prepared_corpus_fails_loud(settings):
    from babble.external import EmptyCorpusError

    with pytest.raises(EmptyCorpusError):
        pretrain_base(settings, steps=2, echo=False)


# --- stage 2: voice ------------------------------------------------------


def test_voice_pass_starts_from_base_and_writes_latest(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=4, seed=1, echo=False)
    _seed_human(settings, ids, ["hey there friend", "wacky sentence", "babble babble"])

    result = voice_pass(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert result.ran and result.reason == "trained"
    assert result.rows_trained == 3
    assert settings.latest_checkpoint.exists()


def test_voice_pass_never_overwrites_the_base(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=4, seed=1, echo=False)
    base_bytes = settings.base_checkpoint.read_bytes()
    _seed_human(settings, ids, ["hey there", "wacky sentence"])

    voice_pass(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    voice_pass(settings, force=True, steps=4, seed=1, echo=False, ids=ids)

    assert settings.base_checkpoint.read_bytes() == base_bytes


def test_voice_pass_always_restarts_from_base_not_the_previous_voice(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=4, seed=1, echo=False)
    _seed_human(settings, ids, ["hey there", "wacky sentence"])

    voice_pass(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    first = _model_state(settings.latest_checkpoint)

    # Poison latest.pt: if the voice pass ever resumed from it instead of the
    # frozen base, the second run would diverge from the first.
    settings.latest_checkpoint.write_bytes(b"not a checkpoint at all")

    voice_pass(settings, force=True, steps=4, seed=1, echo=False, ids=ids)
    second = _model_state(settings.latest_checkpoint)

    assert _states_equal(first, second)


def test_voice_pass_with_no_base_reports_no_base(settings, tmp_path, ids):
    _seed_human(settings, ids, ["hey there"])
    result = voice_pass(settings, force=True, steps=2, echo=False, ids=ids)
    assert not result.ran and result.reason == "no_base"


def test_voice_pass_with_no_consented_rows_reports_no_data(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=2, seed=1, echo=False)
    result = voice_pass(settings, force=True, steps=2, echo=False, ids=ids)
    assert not result.ran and result.reason == "no_data"


# --- the +N-row trigger --------------------------------------------------


def test_trigger_fires_only_after_the_threshold_of_new_rows(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=2, seed=1, echo=False)
    settings.voice_trigger_rows = 5

    _seed_human(settings, ids, [f"row number {i}" for i in range(4)])  # 4 < 5
    assert voice_trigger(settings).due is False

    _seed_human(settings, ids, ["row number 4"])  # now 5 >= 5
    assert voice_trigger(settings).due is True


def test_voice_pass_respects_the_trigger_and_persists_last_count(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=2, seed=1, echo=False)
    settings.voice_trigger_rows = 3
    _seed_human(settings, ids, ["a a a", "b b b"])  # 2 < 3

    # Not forced and not due -> a no-op that says why, and writes no latest.pt.
    skipped = voice_pass(settings, steps=2, echo=False, ids=ids)
    assert not skipped.ran and skipped.reason == "not_due"
    assert not settings.latest_checkpoint.exists()

    _seed_human(settings, ids, ["c c c"])  # 3 >= 3 -> due
    assert voice_trigger(settings).due is True

    ran = voice_pass(settings, steps=2, echo=False, ids=ids)  # fires without --force
    assert ran.ran

    # The last-trained count is persisted, so it does not re-fire on the next call.
    assert voice_trigger(settings).due is False
    again = voice_pass(settings, steps=2, echo=False, ids=ids)
    assert not again.ran and again.reason == "not_due"


def test_trigger_off_when_threshold_is_zero(settings, tmp_path, ids):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=2, seed=1, echo=False)
    settings.voice_trigger_rows = 0
    _seed_human(settings, ids, [f"row {i}" for i in range(50)])
    assert voice_trigger(settings).due is False


def test_auto_trigger_launches_a_subprocess_only_when_due(settings, tmp_path, ids, monkeypatch):
    _prepare_base_corpus(settings, tmp_path)
    pretrain_base(settings, steps=2, seed=1, echo=False)
    settings.voice_trigger_rows = 2

    launches = []

    class FakePopen:
        def __init__(self, argv, **kw):
            launches.append(argv)
            self.pid = 4321

        def poll(self):
            return None  # still running

    monkeypatch.setattr("babble.pretrain.subprocess.Popen", FakePopen)
    trigger = VoiceAutoTrigger(settings)

    trigger.maybe_run()  # 0 rows -> not due -> no launch
    assert launches == []

    _seed_human(settings, ids, ["one one", "two two"])  # 2 >= 2 -> due
    trigger.maybe_run()
    assert len(launches) == 1
    assert launches[0][1:4] == ["-m", "babble", "voice-pass"]

    trigger.maybe_run()  # a pass is still "running" -> do not stack a second
    assert len(launches) == 1


# --- 512 geometry --------------------------------------------------------


def test_base_pretrain_runs_at_the_512_geometry(settings, tmp_path):
    # The real default geometry, just a 1-layer model so the wide forward is fast.
    settings.block_size = 512
    settings.n_layer, settings.n_head, settings.n_embd = 1, 1, 8
    settings.batch_size, settings.checkpoint_every = 2, 2
    _prepare_base_corpus(settings, tmp_path)

    result = pretrain_base(settings, steps=2, seed=1, echo=False)

    assert settings.base_checkpoint.exists()
    payload = torch.load(settings.base_checkpoint, map_location="cpu", weights_only=True)
    assert payload["config"]["block_size"] == 512
    assert result.final_step == 2


# --- CLI wiring ----------------------------------------------------------


@pytest.mark.parametrize("command", ["prepare-base", "base-pretrain", "voice-pass", "voice-status"])
def test_new_subcommands_are_registered(command):
    args = build_parser().parse_args([command])
    assert args.command == command
