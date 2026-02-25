"""
Tests for framework adapters.
Mock external SDKs so no API keys or frameworks are needed.
"""

import pytest
import tempfile
from unittest.mock import MagicMock, patch


class TestLangChainAdapter:
    def test_memory_history_add_and_retrieve(self):
        import tempfile
        from agentmemory import MemoryStore
        from agentmemory.adapters.langchain import MemoryHistory

        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(agent_id="lc-test", persist_dir=tmpdir, auto_compress=False, enable_dedup=False)
        history = MemoryHistory(memory_store=store)

        history.add_user_message("Hello")
        history.add_ai_message("Hi there!")

        messages = history.messages
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_inject_memory_context_prepends_to_system(self):
        import tempfile
        from agentmemory import MemoryStore
        from agentmemory.adapters.langchain import inject_memory_context

        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(agent_id="lc-inject", persist_dir=tmpdir, auto_compress=False, enable_dedup=False)
        store.episodic.store("User's name is Alice", importance=9)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's my name?"},
        ]

        result = inject_memory_context(messages, store, max_tokens=200)
        system_content = result[0]["content"]
        assert "Alice" in system_content
        assert "You are helpful." in system_content

    def test_inject_memory_context_inserts_system_if_missing(self):
        import tempfile
        from agentmemory import MemoryStore
        from agentmemory.adapters.langchain import inject_memory_context

        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(agent_id="lc-inject2", persist_dir=tmpdir, auto_compress=False, enable_dedup=False)
        store.episodic.store("User is a developer", importance=9)

        messages = [{"role": "user", "content": "Help me code"}]
        result = inject_memory_context(messages, store, max_tokens=200)

        # Should have inserted a system message at front
        assert result[0]["role"] == "system"
        assert "developer" in result[0]["content"]

    def test_memory_history_clear_resets_working(self):
        import tempfile
        from agentmemory import MemoryStore
        from agentmemory.adapters.langchain import MemoryHistory

        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(agent_id="lc-clear", persist_dir=tmpdir, auto_compress=False, enable_dedup=False)
        history = MemoryHistory(memory_store=store)
        history.add_user_message("Hello")
        history.clear()
        assert len(history.messages) == 0


class TestOpenAIAdapter:
    def test_chat_injects_memory_context(self):
        import tempfile
        from agentmemory import MemoryStore
        from agentmemory.adapters.openai import MemoryOpenAI

        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(agent_id="oai-test", persist_dir=tmpdir, auto_compress=False, enable_dedup=False)
        store.episodic.store("User's name is Bob", importance=9)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello Bob!"))]

        with patch("openai.OpenAI") as MockOpenAI:
            instance = MockOpenAI.return_value
            instance.chat.completions.create.return_value = mock_response

            client = MemoryOpenAI(
                agent_id="oai-test",
                memory_store=store,
            )
            client._client = instance

            response = client.chat("What's my name?")

        # The create call should have received system context with "Bob"
        call_args = instance.chat.completions.create.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else []
        if call_args.kwargs.get("messages"):
            messages_sent = call_args.kwargs["messages"]
        system_msg = next((m for m in messages_sent if m.get("role") == "system"), None)
        assert system_msg is not None
        assert "Bob" in system_msg["content"]

    def test_chat_tracks_history(self):
        import tempfile
        from agentmemory import MemoryStore
        from agentmemory.adapters.openai import MemoryOpenAI

        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(agent_id="oai-history", persist_dir=tmpdir, auto_compress=False, enable_dedup=False)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response!"))]

        with patch("openai.OpenAI") as MockOpenAI:
            instance = MockOpenAI.return_value
            instance.chat.completions.create.return_value = mock_response

            client = MemoryOpenAI(agent_id="oai-history", memory_store=store)
            client._client = instance

            client.chat("First message")

        messages = store.get_messages()
        assert len(messages) == 2  # user + assistant
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
