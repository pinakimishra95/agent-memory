"""
Context compression — summarizes old conversation turns so they
don't consume the entire context window, while preserving key facts.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tiers.working import Message


COMPRESSION_PROMPT = """You are a memory compression system for an AI agent.
Your job is to summarize the following conversation excerpt into a compact, factual summary that preserves:
- Key facts stated by the user (name, preferences, goals, constraints)
- Important decisions made
- Unresolved questions or tasks
- Context needed to continue the conversation coherently

Be concise. Output only the summary, no preamble.

Conversation to compress:
{conversation}"""

FACT_EXTRACTION_PROMPT = """Extract atomic facts from this conversation that are worth remembering long-term.
Format: one fact per line, starting with a dash.
Only include facts that would be useful in future conversations (not transient context).
Examples of good facts:
- User's name is Alice
- User prefers Python over JavaScript
- User is building a web scraper for real estate data
- User has a deadline of March 15

Conversation:
{conversation}"""


def _messages_to_text(messages: list["Message"]) -> str:
    lines = []
    for m in messages:
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n".join(lines)


class ContextCompressor:
    """
    Compresses old conversation turns into summaries and extracts long-term facts.
    Uses the configured LLM provider (Anthropic or OpenAI).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
            import os
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)
            if self.model is None:
                self.model = "claude-haiku-4-5"  # Fast and cheap for compression

        elif self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required: pip install openai")
            import os
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key)
            if self.model is None:
                self.model = "gpt-4o-mini"

        return self._client

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()

        if self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        elif self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()

        return ""

    def summarize(self, messages: list["Message"]) -> str:
        """
        Compress a list of messages into a concise summary.
        Returns the summary string.
        """
        conversation_text = _messages_to_text(messages)
        prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)
        return self._call_llm(prompt)

    def extract_facts(self, messages: list["Message"]) -> list[str]:
        """
        Extract discrete long-term facts from messages.
        Returns a list of fact strings.
        """
        conversation_text = _messages_to_text(messages)
        prompt = FACT_EXTRACTION_PROMPT.format(conversation=conversation_text)
        raw = self._call_llm(prompt)
        facts = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("-"):
                fact = line.lstrip("- ").strip()
                if fact:
                    facts.append(fact)
        return facts
