"""Shared blackboard for multi-agent message passing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class BlackboardEntry:
    agent: str
    key: str
    value: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Blackboard:
    """Thread-safe shared key-value store + message bus."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._history: list[BlackboardEntry] = []
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = {}

    async def write(self, agent: str, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = value
            entry = BlackboardEntry(agent=agent, key=key, value=value)
            self._history.append(entry)
            # notify subscribers
            for q in self._subscribers.get(key, []):
                await q.put(entry)

    async def read(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._store.get(key, default)

    async def subscribe(self, key: str) -> asyncio.Queue[Any]:
        async with self._lock:
            q: asyncio.Queue[Any] = asyncio.Queue()
            self._subscribers.setdefault(key, []).append(q)
            return q

    def snapshot(self) -> dict[str, Any]:
        return dict(self._store)

    def history(self) -> list[BlackboardEntry]:
        return list(self._history)
