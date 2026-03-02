"""
Qdrant backend example for agentmemory.

Uses Qdrant as the vector store for semantic memory instead of the default ChromaDB.
Suitable for production deployments where you need a standalone vector DB.

Prerequisites:
    # Start a local Qdrant instance
    docker run -p 6333:6333 qdrant/qdrant

    # Install dependencies
    pip install agentcortex[qdrant]

Run:
    python examples/qdrant_agent.py
"""
from agentmemory import MemoryStore

# Connect to a local Qdrant server
# For Qdrant Cloud: qdrant_url="https://<cluster>.cloud.qdrant.io"
memory = MemoryStore(
    agent_id="my-agent",
    semantic_backend="qdrant",
    qdrant_url="http://localhost:6333",
    embedding_provider="sentence-transformers",  # local, no API key needed
)

# Store knowledge — same API regardless of backend
memory.remember("Production architecture uses microservices", importance=8)
memory.remember("API gateway handles auth and rate limiting", importance=7)
memory.remember("PostgreSQL for relational data, Redis for caching", importance=6)

print("Memories stored in Qdrant.")

# Semantic search works the same way
results = memory.recall("architecture and infrastructure", n=5)
print("\nRecalled memories:")
for r in results:
    print(f"  [{r['source']}] {r['content']}")

# Get context for a prompt
context = memory.get_context("current system design")
print(f"\nContext for system prompt:\n{context}")

# Stats
stats = memory.stats()
print(f"\nMemory stats: {stats['episodic']['count']} episodic, {stats['semantic']['count']} semantic")
