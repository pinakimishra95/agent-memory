"""AsyncMemoryStore — async wrapper around MemoryStore.

Wraps the synchronous MemoryStore in a single-threaded ThreadPoolExecutor
so all methods are awaitable. The single-threaded executor serialises SQLite
access and avoids thread-safety issues.

Usage:
    import asyncio
    from agentmemory import AsyncMemoryStore

    async def main():
        memory = AsyncMemoryStore(agent_id="my-agent")

        await memory.remember("User is Alice, prefers concise answers")
        results = await memory.recall("user preferences")
        context = await memory.get_context("answer a question")

        print(context)
        memory.close()

    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentmemory.store import MemoryStore


class AsyncMemoryStore:
    """
    Async-compatible wrapper around MemoryStore.

    All MemoryStore methods are exposed as coroutines. The underlying sync store
    runs in a dedicated single-threaded executor, ensuring safe SQLite access
    from async contexts.

    Constructor arguments are identical to MemoryStore — just add ``await``
    to every method call.

    Example:
        memory = AsyncMemoryStore(agent_id="my-agent")
        await memory.remember("User's name is Alice")
        ctx = await memory.get_context("Who is the user?")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        All arguments are forwarded to MemoryStore.__init__().
        See MemoryStore for full parameter documentation.
        """
        self._store = MemoryStore(*args, **kwargs)
        # max_workers=1 serialises DB access — prevents SQLite threading issues
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentmemory")

    # ── Internal helper ────────────────────────────────────────────────────────

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a sync callable in the executor and return its result."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: fn(*args, **kwargs),
        )

    # ── Core API ───────────────────────────────────────────────────────────────

    async def remember(
        self,
        content: str,
        importance: int = 5,
        store_semantic: bool = True,
    ) -> None:
        """Async version of MemoryStore.remember()."""
        await self._run(self._store.remember, content,
                        importance=importance, store_semantic=store_semantic)

    async def recall(
        self,
        query: str,
        n: int = 5,
        include_episodic: bool = True,
    ) -> list[dict]:
        """Async version of MemoryStore.recall()."""
        return await self._run(self._store.recall, query,
                               n=n, include_episodic=include_episodic)

    async def get_context(
        self,
        query: str | None = None,
        max_tokens: int = 500,
        n: int = 5,
    ) -> str:
        """Async version of MemoryStore.get_context()."""
        return await self._run(self._store.get_context,
                               query=query, max_tokens=max_tokens, n=n)

    async def add_message(self, role: str, content: str) -> None:
        """Async version of MemoryStore.add_message()."""
        await self._run(self._store.add_message, role, content)

    async def get_messages(self) -> list[dict]:
        """Async version of MemoryStore.get_messages()."""
        return await self._run(self._store.get_messages)

    async def compress(self) -> None:
        """Async version of MemoryStore.compress()."""
        await self._run(self._store.compress)

    async def clear(self, tiers: list[str] | None = None) -> None:
        """Async version of MemoryStore.clear()."""
        await self._run(self._store.clear, tiers=tiers)

    async def stats(self) -> dict:
        """Async version of MemoryStore.stats()."""
        return await self._run(self._store.stats)

    async def export_json(
        self,
        path: str | None = None,
        tiers: list[str] | None = None,
    ) -> dict:
        """Async version of MemoryStore.export_json()."""
        return await self._run(self._store.export_json, path=path, tiers=tiers)

    async def import_json(
        self,
        source: str | dict,
        merge: bool = False,
    ) -> int:
        """Async version of MemoryStore.import_json()."""
        return await self._run(self._store.import_json, source, merge=merge)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the executor. Call when done to release threads."""
        self._executor.shutdown(wait=False)

    async def __aenter__(self) -> AsyncMemoryStore:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.close()

    # ── Pass-through properties ────────────────────────────────────────────────

    @property
    def working(self):  # type: ignore[override]
        """Direct access to the underlying WorkingMemory tier (sync)."""
        return self._store.working

    @property
    def episodic(self):  # type: ignore[override]
        """Direct access to the underlying EpisodicMemory tier (sync)."""
        return self._store.episodic

    @property
    def semantic(self):  # type: ignore[override]
        """Direct access to the underlying SemanticMemory tier (sync)."""
        return self._store.semantic

    @property
    def agent_id(self) -> str:
        return self._store.agent_id
