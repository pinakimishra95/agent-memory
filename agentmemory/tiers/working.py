"""
Working memory tier — the active context window for the current session.
Tracks the running conversation and triggers compression when nearing token limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            # Rough estimate: ~4 chars per token
            self.token_estimate = max(1, len(self.content) // 4)


class WorkingMemory:
    """
    Manages the active in-context conversation window.

    Tracks token usage and signals when compression is needed
    (i.e., when messages should be summarized and moved to episodic/semantic tiers).
    """

    def __init__(self, max_tokens: int = 4096, compression_threshold: float = 0.8):
        """
        max_tokens: soft limit before compression is triggered.
        compression_threshold: compress when token usage exceeds this fraction of max_tokens.
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self._messages: list[Message] = []
        self._injected_context: str = ""

    @property
    def messages(self) -> list[dict]:
        """Return messages in the standard {role, content} format."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    @property
    def token_count(self) -> int:
        return sum(m.token_estimate for m in self._messages)

    @property
    def needs_compression(self) -> bool:
        return self.token_count >= int(self.max_tokens * self.compression_threshold)

    def add_message(self, role: str, content: str):
        """Add a message to working memory."""
        self._messages.append(Message(role=role, content=content))

    def inject_context(self, context: str):
        """
        Set the retrieved memory context to be prepended to system prompts.
        Called by MemoryStore after recall().
        """
        self._injected_context = context

    def get_system_context(self) -> str:
        """Returns the memory context string to inject into the system prompt."""
        return self._injected_context

    def pop_oldest_messages(self, n: int) -> list[Message]:
        """
        Remove and return the n oldest non-system messages.
        These will be summarized and stored in episodic/semantic memory.
        """
        non_system = [m for m in self._messages if m.role != "system"]
        to_evict = non_system[:n]
        ids_to_remove = set(id(m) for m in to_evict)
        self._messages = [m for m in self._messages if id(m) not in ids_to_remove]
        return to_evict

    def get_messages_for_compression(self) -> list[Message]:
        """
        Return the oldest half of messages (excluding system) for compression.
        Does not remove them — call pop_oldest_messages after compression.
        """
        non_system = [m for m in self._messages if m.role != "system"]
        half = max(1, len(non_system) // 2)
        return non_system[:half]

    def clear(self):
        self._messages = []
        self._injected_context = ""

    def get_stats(self) -> dict:
        return {
            "message_count": len(self._messages),
            "token_count": self.token_count,
            "token_limit": self.max_tokens,
            "utilization": round(self.token_count / self.max_tokens, 2),
            "needs_compression": self.needs_compression,
        }
