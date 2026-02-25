"""
CrewAI adapter for agentmemory.

Provides a mixin and task callback that gives CrewAI agents persistent
long-term memory across crew runs.

Usage:
    from crewai import Agent, Task, Crew
    from agentmemory.adapters.crewai import MemoryMixin, memory_task_callback

    memory = MemoryStore(agent_id="research-crew")

    agent = Agent(
        role="Researcher",
        goal="Research topics thoroughly",
        backstory="Expert researcher",
        # Inject memory context into agent background
        backstory=memory.get_context("researcher background") + "\\nExpert researcher",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..store import MemoryStore


class CrewMemoryCallback:
    """
    A CrewAI task callback that stores task outputs in agentmemory.

    Pass this as the callback parameter on CrewAI Tasks to automatically
    capture outputs as persistent memories.

    Usage:
        from crewai import Task
        from agentmemory.adapters.crewai import CrewMemoryCallback

        memory = MemoryStore(agent_id="my-crew")
        callback = CrewMemoryCallback(memory_store=memory)

        task = Task(
            description="Research quantum computing",
            expected_output="Summary of key findings",
            callback=callback,
        )
    """

    def __init__(self, memory_store: MemoryStore, importance: int = 7):
        self.memory_store = memory_store
        self.importance = importance

    def __call__(self, output: Any):
        """Called by CrewAI when a task completes."""
        try:
            content = str(output.raw) if hasattr(output, "raw") else str(output)
            if content and len(content) > 10:
                # Truncate very long outputs to a reasonable memory size
                if len(content) > 1000:
                    content = content[:997] + "..."
                self.memory_store.remember(content, importance=self.importance)
        except Exception:
            pass


def get_memory_context_for_agent(
    memory_store: MemoryStore,
    role: str,
    goal: str | None = None,
    max_tokens: int = 400,
) -> str:
    """
    Get formatted memory context to inject into a CrewAI agent's backstory.

    Args:
        memory_store: MemoryStore instance.
        role: The agent's role (used as query for memory retrieval).
        goal: The agent's goal (also used as context for retrieval).
        max_tokens: Token budget for injected context.

    Returns:
        A string to prepend to the agent's backstory.
    """
    query = f"{role} {goal or ''}".strip()
    context = memory_store.get_context(query=query, max_tokens=max_tokens)
    return context
