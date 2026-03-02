"""Tests for the AutoGen memory adapter."""
from __future__ import annotations

import pytest

from agentmemory import MemoryStore
from agentmemory.adapters.autogen import AutoGenMemoryHook, get_autogen_memory_context


@pytest.fixture()
def memory(tmp_path):
    store = MemoryStore(
        agent_id="autogen-test",
        persist_dir=str(tmp_path),
        enable_dedup=False,
        auto_compress=False,
        auto_extract_facts=False,
    )
    yield store
    store.clear()


@pytest.fixture()
def hook(memory):
    return AutoGenMemoryHook(memory_store=memory, importance=6)


def _msgs(*contents):
    return [{"role": "assistant", "content": c} for c in contents]


class TestAutoGenMemoryHook:
    def test_hook_stores_message_on_reply(self, hook, memory):
        hook.on_agent_reply(None, _msgs("The codebase uses FastAPI for the web layer."), None, None)
        recalled = memory.recall("FastAPI web layer", n=5, include_episodic=True)
        assert any("FastAPI" in r["content"] for r in recalled)

    def test_hook_returns_false_none(self, hook):
        result = hook.on_agent_reply(None, _msgs("hello"), None, None)
        assert result == (False, None)

    def test_hook_skips_empty_content(self, hook, memory):
        hook.on_agent_reply(None, [{"role": "assistant", "content": ""}], None, None)
        assert memory.episodic.count() == 0

    def test_hook_skips_whitespace_content(self, hook, memory):
        hook.on_agent_reply(None, [{"role": "assistant", "content": "   "}], None, None)
        assert memory.episodic.count() == 0

    def test_hook_skips_none_messages(self, hook, memory):
        result = hook.on_agent_reply(None, None, None, None)
        assert result == (False, None)
        assert memory.episodic.count() == 0

    def test_hook_truncates_long_content(self, hook, memory):
        long_text = "x" * 2000
        hook.on_agent_reply(None, [{"role": "assistant", "content": long_text}], None, None)
        rows = memory.episodic.recall_recent(n=1)
        assert len(rows[0]["content"]) <= 1000

    def test_hook_uses_only_last_message(self, hook, memory):
        msgs = _msgs("First message", "Second message", "Third message")
        hook.on_agent_reply(None, msgs, None, None)
        rows = memory.episodic.recall_recent(n=5)
        assert len(rows) == 1
        assert "Third message" in rows[0]["content"]

    def test_hook_importance_stored(self, memory):
        hook_high = AutoGenMemoryHook(memory, importance=9)
        hook_high.on_agent_reply(None, _msgs("Critical info"), None, None)
        rows = memory.episodic.recall_recent(n=1)
        assert rows[0]["importance"] == 9

    def test_on_message_received_str(self, hook, memory):
        hook.on_message_received("User said hello")
        rows = memory.episodic.recall_recent(n=1)
        assert rows[0]["content"] == "User said hello"

    def test_on_message_received_dict(self, hook, memory):
        hook.on_message_received({"content": "Dict message content"})
        rows = memory.episodic.recall_recent(n=1)
        assert rows[0]["content"] == "Dict message content"

    def test_on_message_received_skips_empty(self, hook, memory):
        hook.on_message_received("")
        assert memory.episodic.count() == 0

    def test_on_message_received_truncates_long(self, hook, memory):
        hook.on_message_received("y" * 2000)
        rows = memory.episodic.recall_recent(n=1)
        assert len(rows[0]["content"]) <= 1000


class TestGetAutogenMemoryContext:
    def test_returns_empty_when_no_memories(self, memory):
        result = get_autogen_memory_context(memory, role="Research Assistant")
        assert result == ""

    def test_returns_context_string_after_remember(self, memory):
        memory.remember("Python is preferred over Java", importance=7)
        result = get_autogen_memory_context(memory, role="Code Reviewer")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_respects_max_tokens(self, memory):
        for i in range(20):
            memory.remember(f"Memory item {i}: " + "detail " * 10, importance=5)
        result = get_autogen_memory_context(memory, role="Agent", max_tokens=100)
        assert len(result) < 800  # rough bound

    def test_uses_role_and_goal_for_query(self, memory):
        memory.remember("LLM benchmarks were reviewed", importance=6)
        result = get_autogen_memory_context(
            memory, role="Research Assistant", goal="literature review on LLMs"
        )
        assert isinstance(result, str)

    def test_returns_context_with_header_when_memories_exist(self, memory):
        memory.remember("Test memory for header check", importance=5)
        result = get_autogen_memory_context(memory, role="Assistant")
        if result:
            assert "[Memory Context]" in result
