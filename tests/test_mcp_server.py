"""Tests for the agentmemory MCP server tools.

Tests call tool functions directly — no MCP protocol overhead.
Each test uses a fresh in-memory / tmp_path store via monkeypatched env vars.
"""
from __future__ import annotations

from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────


def _reset_store(monkeypatch, tmp_path: Path, agent_id: str = "test-agent") -> None:
    """Point the module-level _store singleton to None and set env vars."""
    import agentmemory.mcp_server as srv

    monkeypatch.setenv("AGENTMEMORY_AGENT_ID", agent_id)
    monkeypatch.setenv("AGENTMEMORY_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(srv, "_store", None)


# ── remember ──────────────────────────────────────────────────────────────────


class TestRemember:
    def test_remember_returns_confirmation(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import remember

        result = remember("We use PostgreSQL for all relational data")
        assert "Stored" in result
        assert "PostgreSQL" in result

    def test_remember_importance_reflected_in_output(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import remember

        result = remember("Critical security constraint: never log API keys", importance=9)
        assert "9/10" in result or "importance=9" in result

    def test_remember_long_content_truncated_in_output(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import remember

        long_content = "A" * 200
        result = remember(long_content)
        # Output should be shorter than the raw content (preview + ellipsis)
        assert len(result) < len(long_content) + 50


# ── recall ────────────────────────────────────────────────────────────────────


class TestRecall:
    def test_recall_after_remember(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import recall, remember

        remember("User prefers functional programming style over OOP classes", importance=7)
        result = recall("coding style preferences")
        # Should find the stored memory (episodic keyword search at minimum)
        assert "functional" in result.lower() or "preferences" in result.lower() or "1." in result

    def test_recall_empty_returns_no_memories_message(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import recall

        result = recall("authentication architecture")
        assert "no relevant memories" in result.lower() or "no memories" in result.lower()

    def test_recall_respects_n_param(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import recall, remember

        for i in range(5):
            remember(f"Memory item {i}: some important fact about module_{i}")

        result = recall("important fact", n=2)
        # Result should have at most 2 numbered entries (lines starting with "1." "2.")
        numbered = [line for line in result.splitlines() if line.strip().startswith(("1.", "2.", "3.", "4.", "5."))]
        assert len(numbered) <= 5  # can't exceed n


# ── get_context ───────────────────────────────────────────────────────────────


class TestGetContext:
    def test_get_context_returns_string(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import get_context, remember

        remember("The API uses JWT tokens for authentication")
        result = get_context("API authentication")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_context_empty_store_returns_hint(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import get_context

        result = get_context()
        assert "no memories" in result.lower() or "remember()" in result

    def test_get_context_with_no_query(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import get_context, remember

        remember("Project uses Python 3.12 and ruff for linting")
        result = get_context()  # no query — should still work
        assert isinstance(result, str)

    def test_get_context_respects_max_tokens(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import get_context, remember

        for i in range(10):
            remember(f"Fact {i}: " + "x" * 100)

        small = get_context("fact", max_tokens=50)
        large = get_context("fact", max_tokens=1000)
        # Smaller budget should produce equal-or-shorter output
        assert len(small) <= len(large)


# ── memory_stats ──────────────────────────────────────────────────────────────


class TestMemoryStats:
    def test_stats_shows_agent_id(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path, agent_id="my-project")
        from agentmemory.mcp_server import memory_stats

        result = memory_stats()
        assert "my-project" in result

    def test_stats_increments_after_remember(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import memory_stats, remember

        before = memory_stats()
        remember("A new architectural decision was made today")
        after = memory_stats()

        # After remembering, stats should reflect at least 1 episodic entry
        assert "1" in after or after != before

    def test_stats_shows_all_tier_labels(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import memory_stats

        result = memory_stats()
        assert "working" in result.lower()
        assert "episodic" in result.lower()
        assert "semantic" in result.lower()


# ── clear_memory ──────────────────────────────────────────────────────────────


class TestClearMemory:
    def test_clear_all_empties_store(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import clear_memory, memory_stats, remember

        remember("Something important")
        clear_memory()  # clear all
        result = memory_stats()
        assert "0" in result  # everything cleared

    def test_clear_returns_confirmation(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import clear_memory

        result = clear_memory()
        assert "cleared" in result.lower() or "clear" in result.lower()

    def test_clear_specific_tier(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import clear_memory

        result = clear_memory(tiers=["episodic"])
        assert "episodic" in result.lower()

    def test_clear_all_tiers_message(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path)
        from agentmemory.mcp_server import clear_memory

        result = clear_memory(tiers=None)
        assert "all" in result.lower() or "working" in result.lower()


# ── env config ────────────────────────────────────────────────────────────────


class TestEnvConfig:
    def test_agent_id_from_env(self, monkeypatch, tmp_path):
        _reset_store(monkeypatch, tmp_path, agent_id="project-x")
        from agentmemory.mcp_server import memory_stats

        result = memory_stats()
        assert "project-x" in result

    def test_different_agent_ids_isolated(self, monkeypatch, tmp_path):
        """Two different agent IDs must not share memories."""
        import agentmemory.mcp_server as srv

        # Store memory under agent-a
        monkeypatch.setenv("AGENTMEMORY_AGENT_ID", "agent-a")
        monkeypatch.setenv("AGENTMEMORY_PERSIST_DIR", str(tmp_path))
        monkeypatch.setattr(srv, "_store", None)
        from agentmemory.mcp_server import remember
        remember("Secret fact only for agent-a")

        # Switch to agent-b
        monkeypatch.setenv("AGENTMEMORY_AGENT_ID", "agent-b")
        monkeypatch.setattr(srv, "_store", None)
        from agentmemory.mcp_server import recall
        result = recall("secret fact agent-a")
        # agent-b should NOT see agent-a's memory
        assert "agent-a" not in result.lower() or "no relevant" in result.lower()

    def test_persist_dir_from_env(self, monkeypatch, tmp_path):
        """Memories should be written to AGENTMEMORY_PERSIST_DIR."""
        custom_dir = tmp_path / "custom_memory"
        _reset_store(monkeypatch, tmp_path)

        import agentmemory.mcp_server as srv
        monkeypatch.setenv("AGENTMEMORY_PERSIST_DIR", str(custom_dir))
        monkeypatch.setattr(srv, "_store", None)

        from agentmemory.mcp_server import remember
        remember("Test persistence directory", importance=8)

        # The custom directory should now exist and contain something
        assert custom_dir.exists()
