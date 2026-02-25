"""
Raw OpenAI SDK adapter for agentmemory.

Provides a thin wrapper around openai.OpenAI that automatically:
- Tracks conversation history in working memory
- Injects relevant memory context into each request
- Stores important facts from responses

Usage:
    from agentmemory.adapters.openai import MemoryOpenAI

    client = MemoryOpenAI(agent_id="my-agent")
    response = client.chat("Tell me about the weather")
    # Memory automatically persists across calls and Python sessions
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..store import MemoryStore


class MemoryOpenAI:
    """
    An openai.OpenAI wrapper that adds persistent memory to any chat interaction.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        memory_store: Optional["MemoryStore"] = None,
        system_prompt: str = "You are a helpful assistant.",
        **memory_kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        from ..store import MemoryStore

        self.model = model
        self.system_prompt = system_prompt
        self.memory = memory_store or MemoryStore(
            agent_id=agent_id,
            llm_provider="openai",
            **memory_kwargs,
        )
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def chat(self, user_message: str, remember_response: bool = True) -> str:
        """
        Send a message and get a response, with automatic memory management.

        Args:
            user_message: The user's message.
            remember_response: Store key facts from the response in memory.

        Returns:
            The assistant's response text.
        """
        # Add user message to working memory
        self.memory.add_message("user", user_message)

        # Build messages: system + memory context + conversation
        memory_context = self.memory.get_context(query=user_message, max_tokens=600)
        system_content = self.system_prompt
        if memory_context:
            system_content = f"{self.system_prompt}\n\n{memory_context}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.memory.get_messages())

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        response_text = response.choices[0].message.content

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
