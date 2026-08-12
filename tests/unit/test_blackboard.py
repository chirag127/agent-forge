"""Unit tests — blackboard."""

from __future__ import annotations

import asyncio

import pytest

from agent_forge.blackboard import Blackboard, BlackboardEntry


@pytest.mark.asyncio
async def test_write_and_read() -> None:
    bb = Blackboard()
    await bb.write("agent1", "result", 42)
    val = await bb.read("result")
    assert val == 42


@pytest.mark.asyncio
async def test_read_default() -> None:
    bb = Blackboard()
    val = await bb.read("missing_key", default="fallback")
    assert val == "fallback"


@pytest.mark.asyncio
async def test_history_records_writes() -> None:
    bb = Blackboard()
    await bb.write("agentA", "k1", "v1")
    await bb.write("agentB", "k2", "v2")
    history = bb.history()
    assert len(history) == 2
    assert history[0].agent == "agentA"
    assert history[1].key == "k2"


@pytest.mark.asyncio
async def test_snapshot() -> None:
    bb = Blackboard()
    await bb.write("a", "x", 1)
    await bb.write("a", "y", 2)
    snap = bb.snapshot()
    assert snap == {"x": 1, "y": 2}


@pytest.mark.asyncio
async def test_subscribe_receives_notification() -> None:
    bb = Blackboard()
    q = await bb.subscribe("status")
    await bb.write("planner", "status", "done")
    entry = q.get_nowait()
    assert isinstance(entry, BlackboardEntry)
    assert entry.value == "done"
    assert entry.agent == "planner"


@pytest.mark.asyncio
async def test_overwrite_key() -> None:
    bb = Blackboard()
    await bb.write("a", "key", "first")
    await bb.write("a", "key", "second")
    val = await bb.read("key")
    assert val == "second"
    assert len(bb.history()) == 2


@pytest.mark.asyncio
async def test_concurrent_writes() -> None:
    bb = Blackboard()

    async def write_many(agent: str, n: int) -> None:
        for i in range(n):
            await bb.write(agent, f"{agent}_{i}", i)

    await asyncio.gather(write_many("a", 10), write_many("b", 10))
    snap = bb.snapshot()
    assert len(snap) == 20
