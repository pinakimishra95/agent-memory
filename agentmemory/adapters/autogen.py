"""AutoGen adapter for agentmemory.

Provides two things:
  1. AutoGenMemoryHook  — a register_reply hook that captures agent messages
                          and stores them in persistent memory.
  2. get_autogen_memory_context — returns a memory context string ready to
                                   inject into an AutoGen agent's system_message.

No hard dependency on pyautogen — the module imports cleanly without it.
pyautogen is only required if you actually run AutoGen agents.

Usage:
    from agentmemory import MemoryStore
    from agentmemory.adapters.autogen import AutoGenMemoryHook, get_autogen_memory_context
    import autogen

    memory = MemoryStore(agent_id="my-agent")

    # Inject past context into agent system_message
    context = get_autogen_memory_context(memory, role="Research Assistant")

    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message=context + "\\nYou are a helpful research assistant.",
    )

    # Register the hook so replies are captured to memory
    hook = AutoGenMemoryHook(memory)
    assistant.register_reply(
        trigger=autogen.ConversableAgent,
        reply_func=hook.on_agent_reply,
        position=0,
    )
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentmemory.store import MemoryStore


class AutoGenMemoryHook:
    """
    AutoGen register_reply hook that captures agent replies and stores them
    in agentmemory for use across sessions.

    This hook observes replies without intercepting them — it always returns
    ``(False, None)`` so the normal reply chain continues unmodified.

    Usage:
        hook = AutoGenMemoryHook(memory_store, importance=6)
        agent.register_reply(
            trigger=autogen.ConversableAgent,
            reply_func=hook.on_agent_reply,
            position=0,
        )
    """

    def __init__(self, memory_store: MemoryStore, importance: int = 6) -> None:
        """
        Args:
            memory_store: The MemoryStore instance to write to.
            importance:   Importance level 1-10 for stored memories (default 6).
                          Use higher values for critical reasoning, lower for
                          routine conversational turns.
        """
        self._store = memory_store
        self._importance = importance

    def on_agent_reply(
        self,
        recipient: Any,
        messages: list[dict] | None,
        sender: Any,
        config: Any,
    ) -> tuple[bool, None]:
        """
        AutoGen register_reply callback signature.

        Reads the last message in the chain and stores its content to memory.
        Returns ``(False, None)`` — does NOT intercept or modify the reply.

        Args:
            recipient: The agent receiving the message (unused).
            messages:  Current message list.
            sender:    The agent that sent the message (unused).
            config:    AutoGen config dict (unused).
        """
        if messages:
            last = messages[-1]
            content = last.get("content", "")
            if content and isinstance(content, str) and content.strip():
                # Truncate very long messages to avoid storing noise
                self._store.remember(content[:1000], importance=self._importance)
        return False, None  # observe only — do not intercept

    def on_message_received(self, message: str | dict) -> None:
        """
        Convenience method for manually storing a received message.

        Args:
            message: Raw message string or dict with a "content" key.
        """
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = str(message)
        if content and content.strip():
            self._store.remember(content[:1000], importance=self._importance)


def get_autogen_memory_context(
    memory_store: MemoryStore,
    role: str,
    goal: str | None = None,
    max_tokens: int = 400,
) -> str:
    """
    Return a memory context string for injection into an AutoGen agent's
    ``system_message``.

    Retrieves relevant memories from agentmemory based on the agent's role and
    optional goal, and returns them as a formatted context block. Prepend this
    to the agent's system_message to give it codebase/task memory.

    Args:
        memory_store: The MemoryStore instance to query.
        role:         Agent role name, e.g. "Research Assistant", "Code Reviewer".
        goal:         Optional goal description to narrow memory retrieval.
        max_tokens:   Maximum token budget for the returned context (default 400).

    Returns:
        A formatted context string, or an empty string if no memories exist yet.

    Example:
        context = get_autogen_memory_context(memory, "Research Assistant",
                                             goal="literature review on LLMs")
        agent = autogen.AssistantAgent(
            name="researcher",
            system_message=context + "\\nYou are a research assistant.",
        )
    """
    query = f"{role} {goal or ''}".strip()
    return memory_store.get_context(query=query, max_tokens=max_tokens)
