"""The Claude-CLI LLM call path: never shells out to the real binary in
tests -- every case here stubs `subprocess.run` so the suite stays fast,
free, and network-independent."""

from __future__ import annotations

import json
import subprocess

import pytest

from babble.llm import ClaudeCLIClient, LLMError, client_from_settings


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_complete_returns_the_result_field(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert cmd[-1] == "hello"
        return _completed(json.dumps({"is_error": False, "result": "hi there"}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCLIClient()
    assert client.complete("hello") == "hi there"


def test_nonzero_exit_raises_llm_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="boom"))
    client = ClaudeCLIClient()
    with pytest.raises(LLMError, match="boom"):
        client.complete("hello")


def test_is_error_envelope_raises_llm_error(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(json.dumps({"is_error": True, "result": "rate limited"})),
    )
    client = ClaudeCLIClient()
    with pytest.raises(LLMError, match="rate limited"):
        client.complete("hello")


def test_non_json_stdout_raises_llm_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("not json at all"))
    client = ClaudeCLIClient()
    with pytest.raises(LLMError):
        client.complete("hello")


def test_empty_result_raises_llm_error(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _completed(json.dumps({"is_error": False, "result": "   "}))
    )
    client = ClaudeCLIClient()
    with pytest.raises(LLMError):
        client.complete("hello")


def test_timeout_raises_llm_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCLIClient(timeout_seconds=1)
    with pytest.raises(LLMError, match="timed out"):
        client.complete("hello")


def test_missing_binary_raises_llm_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCLIClient(binary="not-a-real-binary")
    with pytest.raises(LLMError, match="not-a-real-binary"):
        client.complete("hello")


def test_client_from_settings_uses_configured_knobs(settings):
    settings.paraphrase_model = "opus"
    settings.paraphrase_timeout_seconds = 5.0
    settings.paraphrase_bin = "claude"
    client = client_from_settings(settings)
    assert isinstance(client, ClaudeCLIClient)
    assert client.model == "opus"
    assert client.timeout_seconds == 5.0
