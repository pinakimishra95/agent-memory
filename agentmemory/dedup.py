"""
Semantic deduplication — prevents storing near-duplicate memories
that would waste vector DB space and pollute retrieval results.
"""

from __future__ import annotations

from typing import Optional


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryDeduplicator:
    """
    Checks whether a new memory is semantically similar to existing ones.
    Uses embeddings to detect near-duplicates before storing.

    Typical usage:
        dedup = MemoryDeduplicator(threshold=0.92)
        if not dedup.is_duplicate(new_fact, existing_facts):
            memory.store(new_fact)
    """

    def __init__(
        self,
        threshold: float = 0.92,
        embedding_provider: str = "sentence-transformers",
    ):
        """
        threshold: cosine similarity above which two memories are considered duplicates.
                   0.92 works well for factual sentences; lower for broader deduplication.
        """
        self.threshold = threshold
        self.embedding_provider = embedding_provider
        self._embedder = None

    def _get_embedder(self):
        if self._embedder:
            return self._embedder

        if self.embedding_provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def embed(self, text: str) -> list[float]:
        embedder = self._get_embedder()
        vec = embedder.encode([text])[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)

    def is_duplicate(self, candidate: str, existing: list[str]) -> bool:
        """
        Returns True if candidate is semantically similar to any item in existing.
        Falls back to exact match check if no embedder is available.
        """
        if not existing:
            return False

        # Fast exact match first
        if candidate in existing:
            return True

        try:
            candidate_vec = self.embed(candidate)
            for mem in existing:
                mem_vec = self.embed(mem)
                sim = cosine_similarity(candidate_vec, mem_vec)
                if sim >= self.threshold:
                    return True
        except Exception:
            # If embedding fails, fall back to exact match only
            pass

        return False

    def deduplicate(self, candidates: list[str], existing: Optional[list[str]] = None) -> list[str]:
        """
        Filter a list of candidate memories, removing duplicates against each other
        and against the existing memories list.
        Returns the unique candidates.
        """
        existing = list(existing or [])
        unique = []
        seen = list(existing)

        for candidate in candidates:
            if not self.is_duplicate(candidate, seen):
                unique.append(candidate)
                seen.append(candidate)

        return unique
