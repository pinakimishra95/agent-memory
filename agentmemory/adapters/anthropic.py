"""
Anthropic SDK adapter for agentmemory.

Provides a thin wrapper around anthropic.Anthropic that automatically:
- Tracks conversation history in working memory
- Injects relevant memory context into each request
- Persists memories across sessions

Usage:
    from agentmemory.adapters.anthropic import MemoryAnthropic

    client = MemoryAnthropic(agent_id="my-agent")
    response = client.chat("Tell me about the weather")
    # Memory automatically persists across calls and Python sessions
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..store import MemoryStore


class MemoryAnthropic:
    """
    An anthropic.Anthropic wrapper that adds persistent memory to any chat interaction.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "claude-haiku-4-5",
        api_key: Optional[str] = None,
        memory_store: Optional["MemoryStore"] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        **memory_kwargs,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        from ..store import MemoryStore

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.memory = memory_store or MemoryStore(
            agent_id=agent_id,
            llm_provider="anthropic",
            **memory_kwargs,
        )
        self._client = __import__("anthropic").Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response, with automatic memory management.

        Args:
            user_message: The user's message.

        Returns:
            The assistant's response text.
        """
        # Add user message to working memory
        self.memory.add_message("user", user_message)

        # Build system prompt with injected memory context
        memory_context = self.memory.get_context(query=user_message, max_tokens=600)
        system_content = self.system_prompt
        if memory_context:
            system_content = f"{self.system_prompt}\n\n{memory_context}"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_content,
            messages=self.memory.get_messages(),
        )
        response_text = response.content[0].text

        # Track the response in working memory
        self.memory.add_message("assistant", response_text)

        return response_text

    def remember(self, fact: str, importance: int = 7):
        """Explicitly store a fact in memory."""
        self.memory.remember(fact, importance=importance)

    def recall(self, query: str, n: int = 5) -> list[dict]:
        """Retrieve memories relevant to a query."""
        return self.memory.recall(query, n=n)

    def stats(self) -> dict:
        return self.memory.stats()
