"""
Tests for the ContextCompressor and compression pipeline.
These tests mock the LLM calls so no API key is needed.
"""

import pytest
from unittest.mock import MagicMock

from agentmemory.tiers.working import Message
from agentmemory.compression import ContextCompressor


class TestContextCompressor:
    def _make_messages(self) -> list[Message]:
        return [
            Message(role="user", content="Hi, I'm Alice and I'm building a travel app"),
            Message(role="assistant", content="That's exciting! What kind of travel app?"),
            Message(role="user", content="It's for budget travelers, mainly backpackers"),
        ]

    def test_summarize_calls_llm(self):
        # Inject mock client directly — no need to import anthropic
        compressor = ContextCompressor(provider="anthropic")
        compressor.model = "claude-haiku-4-5"
        messages = self._make_messages()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary: Alice is building a travel app for backpackers.")]

        instance = MagicMock()
        instance.messages.create.return_value = mock_response
        compressor._client = instance

        summary = compressor.summarize(messages)

        assert len(summary) > 0
        instance.messages.create.assert_called_once()

    def test_extract_facts_returns_list(self):
        compressor = ContextCompressor(provider="anthropic")
        compressor.model = "claude-haiku-4-5"
        messages = self._make_messages()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="- Alice is building a travel app\n- Target users are backpackers")]

        instance = MagicMock()
        instance.messages.create.return_value = mock_response
        compressor._client = instance

        facts = compressor.extract_facts(messages)

        assert isinstance(facts, list)

    def test_extract_facts_parses_dashes(self):
        compressor = ContextCompressor(provider="anthropic")
        compressor.model = "claude-haiku-4-5"
        messages = self._make_messages()

        raw_output = "- User is Alice\n- User builds travel apps\n- Target: backpackers"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=raw_output)]

        instance = MagicMock()
        instance.messages.create.return_value = mock_response
        compressor._client = instance

        facts = compressor.extract_facts(messages)

        assert "User is Alice" in facts
        assert "User builds travel apps" in facts
        assert "Target: backpackers" in facts

    def test_openai_provider(self):
        compressor = ContextCompressor(provider="openai")
        compressor.model = "gpt-4o-mini"
        messages = self._make_messages()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI summary"))]

        instance = MagicMock()
        instance.chat.completions.create.return_value = mock_response
        compressor._client = instance

        summary = compressor.summarize(messages)

        assert summary == "OpenAI summary"
        instance.chat.completions.create.assert_called_once()
