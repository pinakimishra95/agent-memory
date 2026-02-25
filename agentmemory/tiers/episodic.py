"""
Episodic memory tier — recent session history stored in SQLite.
Searchable by recency and keyword. No external dependencies.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional


class EpisodicMemory:
    """
    Stores recent interactions and facts in a local SQLite database.
    Automatically evicts oldest entries when a size cap is reached.
    """

    def __init__(self, agent_id: str, db_path: Optional[str] = None, max_entries: int = 1000):
        self.agent_id = agent_id
        self.max_entries = max_entries

        if db_path is None:
            data_dir = Path.home() / ".agentmemory"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / f"{agent_id}_episodic.db")

        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    importance INTEGER DEFAULT 5
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_time ON memories (agent_id, created_at DESC)"
            )

    def store(self, content: str, metadata: Optional[dict] = None, importance: int = 5):
        """Store a memory. importance: 1 (low) to 10 (critical)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (agent_id, content, metadata, created_at, importance) VALUES (?, ?, ?, ?, ?)",
                (self.agent_id, content, json.dumps(metadata or {}), time.time(), importance),
            )
            # Evict oldest low-importance entries if over limit
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ?", (self.agent_id,)
            ).fetchone()[0]
            if count > self.max_entries:
                conn.execute("""
                    DELETE FROM memories WHERE id IN (
                        SELECT id FROM memories
                        WHERE agent_id = ?
                        ORDER BY importance ASC, created_at ASC
                        LIMIT ?
                    )
                """, (self.agent_id, count - self.max_entries))

    def recall_recent(self, n: int = 20) -> list[dict]:
        """Return the n most recent memories."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content, metadata, created_at, importance FROM memories "
                "WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
                (self.agent_id, n),
            ).fetchall()
        return [
            {
                "content": r["content"],
                "metadata": json.loads(r["metadata"]),
                "created_at": r["created_at"],
                "importance": r["importance"],
            }
            for r in rows
        ]

    def search(self, query: str, n: int = 10) -> list[dict]:
        """
        Simple keyword search over episodic memories.
        For semantic search, use the SemanticMemory tier.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content, metadata, created_at, importance FROM memories "
                "WHERE agent_id = ? AND content LIKE ? "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (self.agent_id, f"%{query}%", n),
            ).fetchall()
        return [
            {
                "content": r["content"],
                "metadata": json.loads(r["metadata"]),
                "created_at": r["created_at"],
                "importance": r["importance"],
            }
            for r in rows
        ]

    def clear(self):
        """Remove all episodic memories for this agent."""
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE agent_id = ?", (self.agent_id,))

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ?", (self.agent_id,)
            ).fetchone()[0]
