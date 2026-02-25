"""
agentmemory — Production-ready persistent memory for AI agents.

Add persistent memory to any AI agent in 3 lines:

    from agentmemory import MemoryStore

    memory = MemoryStore(agent_id="my-agent")
    memory.remember("User's name is Alice, she prefers concise answers")
    context = memory.get_context("What do we know about the user?")

Works with LangChain, CrewAI, AutoGen, and raw Anthropic/OpenAI SDKs.
"""

from .store import MemoryStore
from .tiers.episodic import EpisodicMemory
from .tiers.semantic import SemanticMemory
from .tiers.working import WorkingMemory
from .compression import ContextCompressor
from .dedup import MemoryDeduplicator

__version__ = "0.1.0"
__all__ = [
    "MemoryStore",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "ContextCompressor",
    "MemoryDeduplicator",
]
