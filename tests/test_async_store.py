"""Tests for AsyncMemoryStore."""
from __future__ import annotations

import asyncio

import pytest

from agentmemory import AsyncMemoryStore


def run(coro):
    """Run a coroutine synchronously (no pytest-asyncio dependency)."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def memory(tmp_path):
    store = AsyncMemoryStore(
        agent_id="async-test",
        persist_dir=str(tmp_path),
        enable_dedup=False,
        auto_compress=False,
        auto_extract_facts=False,
    )
    yield store
    run(store.clear())
    store.close()


class TestAsyncMemoryStore:
    def test_remember_and_recall(self, memory):
        async def _test():
            await memory.remember("User is Alice", importance=7)
            results = await memory.recall("user", n=5)
            assert any("Alice" in r["content"] for r in results)

        run(_test())

    def test_get_context_empty_returns_empty_string(self, memory):
        async def _test():
            ctx = await memory.get_context()
            assert ctx == ""

        run(_test())

    def test_get_context_with_memories(self, memory):
        async def _test():
            await memory.remember("System uses microservices", importance=6)
            ctx = await memory.get_context("architecture")
            assert isinstance(ctx, str)

        run(_test())

    def test_add_and_get_messages(self, memory):
        async def _test():
            await memory.add_message("user", "Hello agent")
            await memory.add_message("assistant", "Hello user")
            msgs = await memory.get_messages()
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"

        run(_test())

    def test_stats_returns_nested_dict(self, memory):
        async def _test():
            s = await memory.stats()
            assert "working" in s
            assert "episodic" in s
            assert "semantic" in s
            assert "message_count" in s["working"]
            assert "count" in s["episodic"]

        run(_test())

    def test_clear_episodic_tier(self, memory):
        async def _test():
            await memory.remember("To be cleared", importance=5)
            s_before = await memory.stats()
            assert s_before["episodic"]["count"] == 1

            await memory.clear(["episodic"])
            s_after = await memory.stats()
            assert s_after["episodic"]["count"] == 0

        run(_test())

    def test_clear_all_tiers(self, memory):
        async def _test():
            await memory.remember("To be wiped", importance=5)
            await memory.add_message("user", "test message")
            await memory.clear()
            s = await memory.stats()
            assert s["episodic"]["count"] == 0
            assert s["working"]["message_count"] == 0

        run(_test())

    def test_export_and_import_roundtrip(self, memory):
        async def _test():
            await memory.remember("Architecture: event-driven", importance=8)
            await memory.remember("Stack: Python + FastAPI", importance=7)

            data = await memory.export_json()
            assert len(data["episodic"]) == 2
            assert data["agent_id"] == "async-test"

            await memory.clear(["episodic"])
            count = await memory.import_json(data)
            assert count == 2

            rows = memory.episodic.recall_recent(n=10)
            assert len(rows) == 2

        run(_test())

    def test_async_context_manager(self, tmp_path):
        async def _test():
            async with AsyncMemoryStore(
                agent_id="ctx-mgr",
                persist_dir=str(tmp_path),
                enable_dedup=False,
            ) as mem:
                await mem.remember("Context manager test", importance=5)
                s = await mem.stats()
                assert s["episodic"]["count"] == 1

        run(_test())

    def test_working_property_accessible(self, memory):
        assert hasattr(memory.working, "messages")

    def test_episodic_property_accessible(self, memory):
        assert hasattr(memory.episodic, "count")
        assert callable(memory.episodic.count)

    def test_semantic_property_accessible(self, memory):
        assert hasattr(memory.semantic, "count")

    def test_compress_does_not_raise_on_empty(self, memory):
        async def _test():
            await memory.compress()  # nothing to compress

        run(_test())

    def test_agent_id_property(self, memory):
        assert memory.agent_id == "async-test"

    def test_multiple_remember_recall(self, memory):
        async def _test():
            items = [
                ("FastAPI handles HTTP requests", 6),
                ("PostgreSQL stores relational data", 7),
                ("Redis is used for caching", 5),
            ]
            for content, importance in items:
                await memory.remember(content, importance=importance)

            results = await memory.recall("database", n=5)
            assert isinstance(results, list)
            assert len(results) >= 1

        run(_test())

    def test_close_is_idempotent(self, tmp_path):
        store = AsyncMemoryStore(
            agent_id="close-test",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        store.close()
        store.close()  # should not raise
