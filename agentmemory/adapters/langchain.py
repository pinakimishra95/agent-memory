"""
LangChain / LangGraph adapter for agentmemory.

Provides a LangChain-compatible BaseChatMessageHistory and a RunnableWithMemory
wrapper that automatically injects memory context into chains.

Usage:
    from agentmemory.adapters.langchain import MemoryHistory, with_memory

    memory = MemoryStore(agent_id="my-agent")
    history = MemoryHistory(memory_store=memory)

    # Use with LangChain LCEL chains:
    chain = prompt | llm
    chain_with_memory = with_memory(chain, memory)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from ..store import MemoryStore


class MemoryHistory:
    """
    A LangChain-compatible chat message history backed by agentmemory.

    Implements the BaseChatMessageHistory interface so it can be used
    with RunnableWithMessageHistory and other LangChain components.
    """

    def __init__(self, memory_store: "MemoryStore"):
        self.memory_store = memory_store

    def add_user_message(self, message: str) -> None:
        self.memory_store.add_message("user", message)

    def add_ai_message(self, message: str) -> None:
        self.memory_store.add_message("assistant", message)

    def add_message(self, message: Any) -> None:
        try:
            role = message.type if hasattr(message, "type") else str(type(message).__name__)
            content = message.content if hasattr(message, "content") else str(message)
            # Normalize LangChain message types
            if role in ("human", "HumanMessage"):
                role = "user"
            elif role in ("ai", "AIMessage"):
                role = "assistant"
            self.memory_store.add_message(role, content)
        except Exception:
            pass

    def add_messages(self, messages: Sequence[Any]) -> None:
        for m in messages:
            self.add_message(m)

    @property
    def messages(self) -> list[dict]:
        return self.memory_store.get_messages()

    def clear(self) -> None:
        self.memory_store.clear(tiers=["working"])


def inject_memory_context(
    messages: list[dict],
    memory_store: "MemoryStore",
    query: Optional[str] = None,
    max_tokens: int = 500,
) -> list[dict]:
    """
    Inject memory context into a list of LangChain messages.

    Prepends the memory context to the first system message,
    or inserts a new system message at the front if none exists.

    Args:
        messages: List of {role, content} message dicts.
        memory_store: MemoryStore instance.
        query: Optional semantic query to focus memory retrieval.
        max_tokens: Token budget for injected context.

    Returns:
        Modified messages list with memory context injected.
    """
    context = memory_store.get_context(query=query, max_tokens=max_tokens)
    if not context:
        return messages

    messages = list(messages)  # Don't mutate the original

    # Find first system message and prepend context
    for i, msg in enumerate(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "system":
            if isinstance(msg, dict):
                messages[i] = {"role": "system", "content": f"{context}\n\n{msg['content']}"}
            return messages

    # No system message found — prepend one
    messages.insert(0, {"role": "system", "content": context})
    return messages
