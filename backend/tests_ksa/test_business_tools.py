"""MCP discovery is insurance-only, concurrent-safe and recoverable; no real MCP."""

import asyncio

import pytest

from app import mcp_tools
from voice_agent import business_tools


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.setattr(business_tools, "_catalog", None)
    monkeypatch.setattr(business_tools, "_lock", None)
    monkeypatch.setattr(business_tools, "_loop", None)


async def test_concurrent_discovery_is_single_flight_and_never_loads_feishu(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    insurance = {"transport": "http", "url": "http://127.0.0.1:1/isolated-mcp"}
    monkeypatch.setattr(mcp_tools, "_mcp_servers", lambda: {
        "insurance": insurance, "feishu": {"transport": "http", "url": "https://unavailable.invalid"},
    })
    first, second = object(), object()

    async def load_one(name, config):
        calls.append((name, config))
        entered.set()
        await release.wait()
        return [first, second]

    monkeypatch.setattr(mcp_tools, "_load_one", load_one)
    pending = [asyncio.create_task(business_tools.load_business_tools()) for _ in range(6)]
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert calls == [("insurance", insurance)]
        assert all(not task.done() for task in pending)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert calls == [("insurance", insurance)]
        assert all(result == [first, second] for result in results)
        assert len({id(result) for result in results}) == len(results)
        results[0].clear()
        assert await business_tools.load_business_tools() == [first, second]
        assert len(calls) == 1
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def test_unavailable_insurance_source_is_not_cached_and_recovers_on_next_task(monkeypatch):
    calls = []
    tool = object()
    monkeypatch.setattr(mcp_tools, "_mcp_servers", lambda: {"insurance": {"isolated": True}})

    async def load_one(name, config):
        calls.append(name)
        return [] if len(calls) == 1 else [tool]

    monkeypatch.setattr(mcp_tools, "_load_one", load_one)
    assert await business_tools.load_business_tools() == []
    assert business_tools._catalog is None
    assert await business_tools.load_business_tools() == [tool]
    assert await business_tools.load_business_tools() == [tool]
    assert calls == ["insurance", "insurance"]


async def test_feishu_only_configuration_does_not_attempt_any_tool_connection(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_mcp_servers", lambda: {"feishu": {"isolated": True}})

    async def never_load(*args, **kwargs):
        raise AssertionError("No insurance configuration may load no external source")

    monkeypatch.setattr(mcp_tools, "_load_one", never_load)
    assert await business_tools.load_business_tools() == []
    assert business_tools._catalog is None
