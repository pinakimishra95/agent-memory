"""
Tests for the MemoryStore class and memory tiers.
These tests run without any external API keys or vector DB dependencies.
"""

import os
import tempfile

from agentmemory.dedup import MemoryDeduplicator
from agentmemory.tiers.episodic import EpisodicMemory
from agentmemory.tiers.working import WorkingMemory

# ---------------------------------------------------------------------------
# EpisodicMemory tests (no external dependencies)
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_episodic.db")
        self.mem = EpisodicMemory(agent_id="test-agent", db_path=self.db_path)

    def test_store_and_recall(self):
        self.mem.store("User's name is Alice")
        self.mem.store("User prefers Python")
        results = self.mem.recall_recent(n=10)
        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "User's name is Alice" in contents
        assert "User prefers Python" in contents

    def test_recall_is_ordered_by_recency(self):
        self.mem.store("First memory")
        self.mem.store("Second memory")
        self.mem.store("Third memory")
        results = self.mem.recall_recent(n=3)
        assert results[0]["content"] == "Third memory"
        assert results[1]["content"] == "Second memory"
        assert results[2]["content"] == "First memory"

    def test_search_by_keyword(self):
        self.mem.store("User likes coffee")
        self.mem.store("User dislikes tea")
        self.mem.store("User is a developer")
        results = self.mem.search("coffee")
        assert len(results) == 1
        assert results[0]["content"] == "User likes coffee"

    def test_importance_affects_eviction_order(self):
        mem = EpisodicMemory(agent_id="test-evict", db_path=self.db_path + "_evict", max_entries=3)
        mem.store("Low importance", importance=1)
        mem.store("High importance", importance=9)
        mem.store("Medium importance", importance=5)
        # Add one more to trigger eviction
        mem.store("New memory", importance=5)
        results = mem.recall_recent(n=10)
        contents = [r["content"] for r in results]
        # Low importance should be evicted
        assert "Low importance" not in contents
        assert "High importance" in contents

    def test_count(self):
        assert self.mem.count() == 0
        self.mem.store("Memory 1")
        self.mem.store("Memory 2")
        assert self.mem.count() == 2

    def test_clear(self):
        self.mem.store("Memory 1")
        self.mem.clear()
        assert self.mem.count() == 0

    def test_agent_namespacing(self):
        mem_a = EpisodicMemory(agent_id="agent-a", db_path=self.db_path)
        mem_b = EpisodicMemory(agent_id="agent-b", db_path=self.db_path)
        mem_a.store("Memory for A")
        mem_b.store("Memory for B")
        assert mem_a.count() == 1
        assert mem_b.count() == 1
        a_results = mem_a.recall_recent(n=10)
        assert all(r["content"] == "Memory for A" for r in a_results)

    def test_metadata_roundtrip(self):
        self.mem.store("Memory with metadata", metadata={"source": "test", "version": 2})
        results = self.mem.recall_recent(n=1)
        assert results[0]["metadata"]["source"] == "test"
        assert results[0]["metadata"]["version"] == 2


# ---------------------------------------------------------------------------
# WorkingMemory tests
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    def setup_method(self):
        self.wm = WorkingMemory(max_tokens=100, compression_threshold=0.8)

    def test_add_and_retrieve_messages(self):
        self.wm.add_message("user", "Hello")
        self.wm.add_message("assistant", "Hi there!")
        messages = self.wm.messages
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_token_counting(self):
        self.wm.add_message("user", "A" * 40)  # ~10 tokens
        assert self.wm.token_count > 0

    def test_needs_compression_trigger(self):
        wm = WorkingMemory(max_tokens=20, compression_threshold=0.8)
        # Add enough content to exceed 80% of 20 tokens = 16 tokens
        wm.add_message("user", "A" * 80)  # ~20 tokens
        assert wm.needs_compression

    def test_pop_oldest_messages(self):
        self.wm.add_message("user", "First")
        self.wm.add_message("assistant", "Second")
        self.wm.add_message("user", "Third")
        popped = self.wm.pop_oldest_messages(2)
        assert len(popped) == 2
        assert popped[0].content == "First"
        assert len(self.wm.messages) == 1

    def test_system_messages_excluded_from_compression_targets(self):
        self.wm.add_message("system", "You are a bot")
        self.wm.add_message("user", "Hello")
        targets = self.wm.get_messages_for_compression()
        # System message should not be in compression targets
        roles = [m.role for m in targets]
        assert "system" not in roles

    def test_inject_context(self):
        self.wm.inject_context("[Memory Context]\n- User is Alice")
        assert "Alice" in self.wm.get_system_context()

    def test_clear(self):
        self.wm.add_message("user", "Hello")
        self.wm.clear()
        assert len(self.wm.messages) == 0

    def test_stats(self):
        self.wm.add_message("user", "Hello world")
        stats = self.wm.get_stats()
        assert "token_count" in stats
        assert "utilization" in stats
        assert stats["token_limit"] == 100


# ---------------------------------------------------------------------------
# MemoryDeduplicator tests (no external dependencies — uses exact match path)
# ---------------------------------------------------------------------------

class TestMemoryDeduplicator:
    def setup_method(self):
        # Use a threshold that falls back to exact matching in tests
        # (sentence-transformers not guaranteed in test env)
        self.dedup = MemoryDeduplicator(threshold=0.99)

    def test_exact_duplicate_detected(self):
        existing = ["User's name is Alice"]
        assert self.dedup.is_duplicate("User's name is Alice", existing)

    def test_non_duplicate_passes(self):
        existing = ["User's name is Alice"]
        assert not self.dedup.is_duplicate("User prefers Python", existing)

    def test_empty_existing_always_passes(self):
        assert not self.dedup.is_duplicate("Any memory", [])

    def test_deduplicate_list(self):
        candidates = ["Fact A", "Fact B", "Fact A", "Fact C"]
        unique = self.dedup.deduplicate(candidates)
        assert len(unique) == 3
        assert "Fact A" in unique
        assert "Fact B" in unique
        assert "Fact C" in unique

    def test_deduplicate_against_existing(self):
        existing = ["Fact A"]
        candidates = ["Fact A", "Fact B"]
        unique = self.dedup.deduplicate(candidates, existing=existing)
        assert "Fact A" not in unique
        assert "Fact B" in unique


# ---------------------------------------------------------------------------
# Integration: MemoryStore without external dependencies
# ---------------------------------------------------------------------------

class TestMemoryStoreIntegration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_store_instantiates_without_vector_db(self):
        """MemoryStore should instantiate without importing chromadb."""
        from agentmemory import MemoryStore
        # Instantiation should succeed — vector DB is lazily initialized
        store = MemoryStore(
            agent_id="test",
            persist_dir=self.tmpdir,
            enable_dedup=False,
        )
        assert store is not None

    def test_working_memory_integration(self):
        from agentmemory import MemoryStore
        store = MemoryStore(
            agent_id="test-wm",
            persist_dir=self.tmpdir,
            auto_compress=False,
            enable_dedup=False,
        )
        store.add_message("user", "Hello, I'm Bob")
        store.add_message("assistant", "Nice to meet you, Bob!")
        messages = store.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_episodic_remember_and_recall_recent(self):
        from agentmemory import MemoryStore
        store = MemoryStore(
            agent_id="test-episodic",
            persist_dir=self.tmpdir,
            auto_compress=False,
            enable_dedup=False,
        )
        store.episodic.store("User's name is Charlie")
        store.episodic.store("User likes jazz music")
        recent = store.episodic.recall_recent(n=5)
        contents = [r["content"] for r in recent]
        assert "User's name is Charlie" in contents

    def test_stats(self):
        from agentmemory import MemoryStore
        store = MemoryStore(
            agent_id="test-stats",
            persist_dir=self.tmpdir,
            auto_compress=False,
            enable_dedup=False,
        )
        store.add_message("user", "Hello")
        stats = store.stats()
        assert stats["agent_id"] == "test-stats"
        assert "working" in stats
        assert "episodic" in stats
        assert "semantic" in stats

    def test_clear_working_only(self):
        from agentmemory import MemoryStore
        store = MemoryStore(
            agent_id="test-clear",
            persist_dir=self.tmpdir,
            auto_compress=False,
            enable_dedup=False,
        )
        store.add_message("user", "Hello")
        store.episodic.store("Persistent fact")
        store.clear(tiers=["working"])
        assert len(store.get_messages()) == 0
        assert store.episodic.count() == 1
