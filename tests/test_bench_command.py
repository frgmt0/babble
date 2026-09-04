"""Slash benchmark access, serialization and failure recovery, without Discord."""
import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from babble.bot import BabbleClient


def interaction(*, admin=True, guild=True):
    return SimpleNamespace(
        guild=object() if guild else None,
        permissions=discord.Permissions(manage_guild=admin),
        response=SimpleNamespace(
            send_message=AsyncMock(), defer=AsyncMock(), is_done=lambda: True,
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def test_bench_is_registered_with_server_permission(settings, log, brain):
    client = BabbleClient(settings, log, brain)
    command = client.tree.get_command("bench")
    assert command is not None and command.guild_only
    assert command.default_permissions.manage_guild


def test_bench_rejects_unprivileged_users_and_dms(settings, log, brain):
    client = BabbleClient(settings, log, brain)
    for request in (interaction(admin=False), interaction(guild=False)):
        asyncio.run(client._bench(request))
        request.response.send_message.assert_awaited_once()
        request.response.defer.assert_not_awaited()
    assert client._bench_last_started is None


def test_bench_uses_generator_without_capture_and_enforces_cooldown(settings, log, brain, monkeypatch):
    client = BabbleClient(settings, log, brain)
    calls = []

    def run(generator):
        assert client._lock.locked()
        calls.append(generator)
        return {"tps": 123}

    monkeypatch.setitem(sys.modules, "babble.benchmark", SimpleNamespace(
        run_benchmark=run, format_benchmark=lambda r: "123 TPS",
    ))
    request, repeated = interaction(), interaction()

    async def scenario():
        await client._bench(request)
        await client._bench(repeated)

    asyncio.run(scenario())
    assert calls == [brain.generator]
    request.followup.send.assert_awaited_once_with("123 TPS", ephemeral=True)
    repeated.response.defer.assert_not_awaited()
    assert "wait" in repeated.response.send_message.call_args.args[0]
    assert not client._lock.locked() and not client._bench_running
    assert brain.generator.prompts == []


def test_bench_declines_while_chat_is_busy(settings, log, brain):
    client = BabbleClient(settings, log, brain)
    request = interaction()

    async def scenario():
        async with client._lock:
            await client._bench(request)

    asyncio.run(scenario())
    request.response.defer.assert_not_awaited()
    assert "busy" in request.response.send_message.call_args.args[0]


def test_bench_failure_releases_chat_lock(settings, log, brain, monkeypatch, read_log):
    client = BabbleClient(settings, log, brain)

    def fail(generator):
        raise RuntimeError("test failure")

    monkeypatch.setitem(sys.modules, "babble.benchmark", SimpleNamespace(
        run_benchmark=fail, format_benchmark=str,
    ))
    request = interaction()
    asyncio.run(client._bench(request))
    assert not client._lock.locked() and not client._bench_running
    assert "failed" in request.followup.send.call_args.args[0]
    assert read_log("bot.error")[-1]["where"] == "bench"


def test_bench_cli_serializes_dataclass_without_discord(monkeypatch, capsys):
    from dataclasses import dataclass
    from babble import hfserve
    from babble.cli import main

    @dataclass
    class Measurement:
        e2e_tps: float = 123.0

    sentinel = object()
    monkeypatch.setattr(hfserve, "make_generator", lambda settings: sentinel)
    monkeypatch.setitem(sys.modules, "babble.benchmark", SimpleNamespace(
        run_benchmark=lambda generator: Measurement(), format_benchmark=str,
    ))
    assert main(["bench", "--json"]) == 0
    assert '"e2e_tps": 123.0' in capsys.readouterr().out


def test_bench_cli_reports_unsupported_backend(monkeypatch, capsys):
    from babble import hfserve
    from babble.cli import main

    def unavailable(generator):
        raise RuntimeError("backend has no token instrumentation")

    monkeypatch.setattr(hfserve, "make_generator", lambda settings: object())
    monkeypatch.setitem(sys.modules, "babble.benchmark", SimpleNamespace(
        run_benchmark=unavailable, format_benchmark=str,
    ))
    assert main(["bench"]) == 1
    assert "no token instrumentation" in capsys.readouterr().err


def test_cancelled_bench_keeps_model_locked_until_worker_finishes(settings, log, brain, monkeypatch):
    client = BabbleClient(settings, log, brain)
    started, release = threading.Event(), threading.Event()

    def run(generator):
        started.set()
        assert release.wait(3)
        return {}

    monkeypatch.setitem(sys.modules, "babble.benchmark", SimpleNamespace(
        run_benchmark=run, format_benchmark=str,
    ))

    async def scenario():
        task = asyncio.create_task(client._bench(interaction()))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            assert client._lock.locked() and client._bench_running
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client._lock.locked() and not client._bench_running

    asyncio.run(scenario())
