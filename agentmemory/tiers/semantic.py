"""
Semantic memory tier — long-term knowledge stored in a vector database.
Searchable by meaning using embeddings. Supports ChromaDB (default) and Qdrant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


class SemanticMemory:
    """
    Stores long-term facts and knowledge as vector embeddings for semantic search.

    Backends:
    - chromadb (default, local, zero-config)
    - qdrant (production scale, requires qdrant-client)

    Embedding providers:
    - sentence-transformers (default, fully local, no API key needed)
    - openai (requires OPENAI_API_KEY)
    - anthropic (uses voyage-3-lite via Voyage API)
    """

    def __init__(
        self,
        agent_id: str,
        backend: str = "chromadb",
        embedding_provider: str = "sentence-transformers",
        persist_dir: Optional[str] = None,
        **backend_kwargs,
    ):
        self.agent_id = agent_id
        self.backend = backend
        self.embedding_provider = embedding_provider

        if persist_dir is None:
            persist_dir = str(Path.home() / ".agentmemory" / "semantic")

        self._persist_dir = persist_dir
        self._collection = None
        self._embedder = None
        self._backend_kwargs = backend_kwargs
        self._client = None

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder

        if self.embedding_provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings.\n"
                    "Install with: pip install sentence-transformers"
                )
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        elif self.embedding_provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required: pip install openai")

            import os
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            class _OpenAIEmbedder:
                def encode(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    resp = client.embeddings.create(
                        model="text-embedding-3-small", input=texts
                    )
                    return [e.embedding for e in resp.data]

            self._embedder = _OpenAIEmbedder()

        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")

        return self._embedder

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        if self.backend == "chromadb":
            try:
                import chromadb
            except ImportError:
                raise ImportError(
                    "chromadb is required for semantic memory.\n"
                    "Install with: pip install chromadb"
                )
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=f"agentmemory_{self.agent_id}",
                metadata={"hnsw:space": "cosine"},
            )

        elif self.backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
            except ImportError:
                raise ImportError("qdrant-client required: pip install qdrant-client")

            url = self._backend_kwargs.get("url", ":memory:")
            self._client = QdrantClient(url=url)
            collection_name = f"agentmemory_{self.agent_id}"
            try:
                self._client.get_collection(collection_name)
            except Exception:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            self._collection = collection_name

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._collection

    def store(self, content: str, metadata: Optional[dict] = None, doc_id: Optional[str] = None):
        """Store a piece of knowledge with its embedding."""
        if doc_id is None:
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        embedder = self._get_embedder()
        collection = self._get_collection()

        if self.backend == "chromadb":
            embedding = embedder.encode([content])
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            elif isinstance(embedding, list) and hasattr(embedding[0], "tolist"):
                embedding = [e.tolist() for e in embedding]
            collection.upsert(
                ids=[doc_id],
                documents=[content],
                embeddings=embedding,
                metadatas=[metadata or {}],
            )

        elif self.backend == "qdrant":
            from qdrant_client.models import PointStruct
            embedding = embedder.encode([content])
            if hasattr(embedding[0], "tolist"):
                vec = embedding[0].tolist()
            else:
                vec = list(embedding[0])
            point_id = abs(hash(doc_id)) % (2**31)
            self._client.upsert(
                collection_name=collection,
                points=[PointStruct(id=point_id, vector=vec, payload={"content": content, **(metadata or {})})],
            )

    def search(self, query: str, n: int = 5, min_similarity: float = 0.3) -> list[dict]:
        """Return the n most semantically similar memories to the query."""
        embedder = self._get_embedder()
        collection = self._get_collection()

        if self.backend == "chromadb":
            embedding = embedder.encode([query])
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            elif isinstance(embedding, list) and hasattr(embedding[0], "tolist"):
                embedding = [e.tolist() for e in embedding]
            results = collection.query(
                query_embeddings=embedding,
                n_results=min(n, max(1, collection.count())),
                include=["documents", "metadatas", "distances"],
            )
            memories = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1.0 - dist  # cosine distance → similarity
                if similarity >= min_similarity:
                    memories.append({"content": doc, "metadata": meta, "similarity": round(similarity, 4)})
            return memories

        elif self.backend == "qdrant":
            embedding = embedder.encode([query])
            if hasattr(embedding[0], "tolist"):
                vec = embedding[0].tolist()
            else:
                vec = list(embedding[0])
            hits = self._client.search(
                collection_name=collection,
                query_vector=vec,
                limit=n,
                score_threshold=min_similarity,
            )
            return [
                {"content": h.payload.get("content", ""), "metadata": h.payload, "similarity": round(h.score, 4)}
                for h in hits
            ]

        return []

    def delete(self, doc_id: str):
        collection = self._get_collection()
        if self.backend == "chromadb":
            collection.delete(ids=[doc_id])

    def clear(self):
        if self.backend == "chromadb" and self._client:
            self._client.delete_collection(f"agentmemory_{self.agent_id}")
            self._collection = None

    def count(self) -> int:
        # Return 0 without triggering lazy initialization (avoids ImportError when backend not installed)
        if self._collection is None:
            return 0
        try:
            collection = self._get_collection()
            if self.backend == "chromadb":
                return collection.count()
        except (ImportError, Exception):
            pass
        return 0
