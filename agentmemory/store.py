"""
MemoryStore — the main entry point for agentmemory.

Orchestrates all three memory tiers:
  - WorkingMemory  (in-context, current session)
  - EpisodicMemory (recent history, SQLite)
  - SemanticMemory (long-term knowledge, vector DB)

Handles automatic compression, deduplication, and context injection.
"""

from __future__ import annotations

from pathlib import Path

from .compression import ContextCompressor
from .dedup import MemoryDeduplicator
from .tiers.episodic import EpisodicMemory
from .tiers.semantic import SemanticMemory
from .tiers.working import WorkingMemory


class MemoryStore:
    """
    Production-ready persistent memory for any AI agent.

    Quick start:
        from agentmemory import MemoryStore

        memory = MemoryStore(agent_id="my-agent")
        memory.remember("User's name is Alice, she prefers concise answers")
        context = memory.get_context("What do we know about the user?")

    Works with any LLM framework — see adapters for LangChain, CrewAI, AutoGen.
    """

    def __init__(
        self,
        agent_id: str,
        # Storage config
        persist_dir: str | None = None,
        # Working memory config
        max_working_tokens: int = 4096,
        compression_threshold: float = 0.8,
        # Semantic memory config
        semantic_backend: str = "chromadb",
        embedding_provider: str = "sentence-transformers",
        qdrant_url: str | None = None,
        # Compression config
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        api_key: str | None = None,
        # Dedup config
        dedup_threshold: float = 0.92,
        enable_dedup: bool = True,
        # Feature flags
        auto_compress: bool = True,
        auto_extract_facts: bool = True,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent. Memories are namespaced by this.
            persist_dir: Directory for storing memories. Defaults to ~/.agentmemory/
            max_working_tokens: Token budget for working memory before compression triggers.
            compression_threshold: Fraction of max_working_tokens that triggers compression.
            semantic_backend: Vector DB backend ("chromadb" or "qdrant").
            embedding_provider: Embeddings for semantic search ("sentence-transformers" or "openai").
            qdrant_url: Qdrant server URL when semantic_backend="qdrant"
                        (e.g. "http://localhost:6333"). Ignored for chromadb.
            llm_provider: LLM used for compression ("anthropic" or "openai").
            llm_model: Specific model for compression. Defaults to fast/cheap models.
            api_key: API key (falls back to env vars ANTHROPIC_API_KEY / OPENAI_API_KEY).
            dedup_threshold: Cosine similarity threshold for duplicate detection.
            enable_dedup: Whether to deduplicate before storing semantic memories.
            auto_compress: Automatically compress working memory when token limit is near.
            auto_extract_facts: Extract and store long-term facts during compression.
        """
        self._agent_id = agent_id
        self.agent_id = agent_id
        self.auto_compress = auto_compress
        self.auto_extract_facts = auto_extract_facts

        if persist_dir is None:
            persist_dir = str(Path.home() / ".agentmemory")

        # Ensure persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Build backend-specific kwargs (e.g. Qdrant URL)
        backend_kwargs: dict = {}
        if qdrant_url:
            backend_kwargs["url"] = qdrant_url

        # Initialize tiers
        self.working = WorkingMemory(
            max_tokens=max_working_tokens,
            compression_threshold=compression_threshold,
        )
        self.episodic = EpisodicMemory(
            agent_id=agent_id,
            db_path=str(Path(persist_dir) / f"{agent_id}_episodic.db"),
        )
        self.semantic = SemanticMemory(
            agent_id=agent_id,
            backend=semantic_backend,
            embedding_provider=embedding_provider,
            persist_dir=str(Path(persist_dir) / "semantic"),
            **backend_kwargs,
        )

        self._compressor = ContextCompressor(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
        )
        self._deduplicator = MemoryDeduplicator(
            threshold=dedup_threshold,
            embedding_provider=embedding_provider,
        ) if enable_dedup else None

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    def remember(self, content: str, importance: int = 5, store_semantic: bool = True):
        """
        Store a memory across the appropriate tiers.

        Args:
            content: The fact or observation to remember.
            importance: 1 (low) to 10 (critical). High importance memories are
                        evicted last from episodic storage.
            store_semantic: Also embed and store in the vector DB for semantic search.
        """
        # Dedup check before storing — skip gracefully if semantic backend not installed
        if self._deduplicator and store_semantic:
            try:
                existing_results = self.semantic.search(content, n=5, min_similarity=0.5)
                existing_texts = [r["content"] for r in existing_results]
                if self._deduplicator.is_duplicate(content, existing_texts):
                    return  # Skip duplicate
            except (ImportError, Exception):
                pass  # Semantic backend unavailable — store without dedup check

        self.episodic.store(content, importance=importance)
        if store_semantic:
            try:
                self.semantic.store(content)
            except (ImportError, Exception):
                pass  # Semantic backend unavailable — episodic store succeeded

    def recall(self, query: str, n: int = 5, include_episodic: bool = True) -> list[dict]:
        """
        Retrieve memories most relevant to the query.

        Combines semantic search (meaning) and recent episodic memories.
        Returns a ranked list of memory dicts with 'content' and 'source' keys.
        """
        results = []

        # Semantic search (meaning-based) — skipped gracefully if backend not installed
        try:
            semantic_results = self.semantic.search(query, n=n)
        except (ImportError, Exception):
            semantic_results = []
        for r in semantic_results:
            results.append({
                "content": r["content"],
                "source": "semantic",
                "score": r.get("similarity", 0.0),
            })

        # Recent episodic memories
        if include_episodic:
            recent = self.episodic.recall_recent(n=min(n, 10))
            for r in recent:
                results.append({
                    "content": r["content"],
                    "source": "episodic",
                    "score": r.get("importance", 5) / 10.0,
                })

        # Deduplicate results
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["content"] not in seen:
                seen.add(r["content"])
                unique.append(r)

        return unique[:n]

    def get_context(
        self,
        query: str | None = None,
        max_tokens: int = 500,
        n: int = 5,
    ) -> str:
        """
        Get a formatted memory context string ready to inject into a system prompt.

        Args:
            query: Optional query to focus retrieved memories on.
                   If None, returns the most recent/important memories.
            max_tokens: Approximate token budget for the context string.
            n: Max number of memories to include.

        Returns:
            A formatted string like:
            "[Memory Context]\\n- User's name is Alice\\n- Prefers Python"
        """
        if query:
            memories = self.recall(query, n=n)
        else:
            recent = self.episodic.recall_recent(n=n)
            memories = [{"content": r["content"], "source": "episodic"} for r in recent]

        if not memories:
            return ""

        lines = ["[Memory Context]"]
        token_budget = max_tokens
        for mem in memories:
            line = f"- {mem['content']}"
            estimated_tokens = len(line) // 4
            if token_budget - estimated_tokens < 0:
                break
            lines.append(line)
            token_budget -= estimated_tokens

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def add_message(self, role: str, content: str):
        """
        Add a conversation message to working memory.
        Automatically triggers compression if the context window is near full.

        Args:
            role: "user", "assistant", or "system"
            content: Message content
        """
        self.working.add_message(role, content)

        if self.auto_compress and self.working.needs_compression:
            self._compress_working_memory()

    def get_messages(self) -> list[dict]:
        """Return the current working memory messages in {role, content} format."""
        return self.working.messages

    # -------------------------------------------------------------------------
    # Compression
    # -------------------------------------------------------------------------

    def _compress_working_memory(self):
        """
        Summarize and evict the oldest working memory messages.
        Stores the summary in episodic memory and extracted facts in semantic memory.
        """
        messages_to_compress = self.working.get_messages_for_compression()
        if not messages_to_compress:
            return

        # Summarize the old context
        summary = self._compressor.summarize(messages_to_compress)
        if summary:
            self.episodic.store(summary, importance=7, metadata={"type": "compression_summary"})

        # Extract and store long-term facts
        if self.auto_extract_facts:
            facts = self._compressor.extract_facts(messages_to_compress)
            for fact in facts:
                self.remember(fact, importance=8)

        # Evict compressed messages from working memory
        n_to_evict = len(messages_to_compress)
        self.working.pop_oldest_messages(n_to_evict)

    def compress(self):
        """Manually trigger working memory compression."""
        self._compress_working_memory()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def clear(self, tiers: list[str] | None = None):
        """
        Clear memory. Pass tiers=["working", "episodic", "semantic"] to clear specific tiers,
        or call with no args to clear all.
        """
        if tiers is None:
            tiers = ["working", "episodic", "semantic"]
        if "working" in tiers:
            self.working.clear()
        if "episodic" in tiers:
            self.episodic.clear()
        if "semantic" in tiers:
            self.semantic.clear()

    def stats(self) -> dict:
        """Return memory usage statistics."""
        return {
            "agent_id": self.agent_id,
            "working": self.working.get_stats(),
            "episodic": {"count": self.episodic.count()},
            "semantic": {"count": self.semantic.count()},
        }

    # -------------------------------------------------------------------------
    # Export / Import
    # -------------------------------------------------------------------------

    def export_json(
        self,
        path: str | None = None,
        tiers: list[str] | None = None,
    ) -> dict:
        """
        Export memories to a dict and optionally write to a JSON file.

        Working memory is session-scoped and is not exported. Semantic memory
        is re-derived from episodic content on import, so only episodic is
        serialised.

        Args:
            path: Optional file path to write the JSON export to. If omitted,
                  the export dict is returned but not written to disk.
            tiers: Which tiers to export. Currently supports ["episodic"].
                   Defaults to ["episodic"].

        Returns:
            dict with keys: version, agent_id, exported_at, episodic.

        Example:
            data = memory.export_json("backup.json")
            # → writes backup.json, returns the dict
        """
        import json  # noqa: PLC0415
        import time  # noqa: PLC0415

        tiers = tiers or ["episodic"]
        data: dict = {
            "version": "1.0",
            "agent_id": self._agent_id,
            "exported_at": time.time(),
        }

        if "episodic" in tiers:
            rows = self.episodic.recall_recent(n=10_000)
            data["episodic"] = [
                {
                    "content": r["content"],
                    "metadata": r.get("metadata", {}),
                    "created_at": r["created_at"],
                    "importance": r.get("importance", 5),
                }
                for r in rows
            ]

        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        return data

    def import_json(
        self,
        source: str | dict,
        merge: bool = False,
    ) -> int:
        """
        Import episodic memories from a JSON file path or previously exported dict.

        Args:
            source: File path string or dict produced by export_json().
            merge:  If False (default), clears episodic memory before importing
                    so the result is an exact restoration of the export.
                    If True, new records are merged alongside existing ones.

        Returns:
            Number of records imported.

        Example:
            # Backup
            memory.export_json("backup.json")

            # Restore on another machine / agent
            new_memory = MemoryStore(agent_id="new-agent")
            new_memory.import_json("backup.json")
        """
        import json  # noqa: PLC0415

        if isinstance(source, str):
            with open(source) as f:
                data = json.load(f)
        else:
            data = dict(source)

        if not merge:
            self.episodic.clear()

        records = data.get("episodic", [])
        for r in records:
            self.episodic.store(
                content=r["content"],
                metadata=r.get("metadata", {}),
                importance=r.get("importance", 5),
            )
        return len(records)
