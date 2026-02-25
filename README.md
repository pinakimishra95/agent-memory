# agentmemory 🧠

**Your AI agent forgets everything. AgentMemory fixes that in 3 lines.**

[![PyPI version](https://badge.fury.io/py/agentmemory.svg)](https://badge.fury.io/py/agentmemory)
[![Tests](https://github.com/yourusername/agent-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/agent-memory/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Every time your agent starts a new session, it starts from zero.

```python
# What happens today — every single time
agent = MyAgent()
agent.chat("Hi, I'm Alice and I'm building a fraud detection system")
# → "Nice to meet you, Alice!"

# Next session...
agent = MyAgent()
agent.chat("What's my name?")
# → "I don't know your name — could you tell me?"  ❌
```

This isn't an AI limitation. It's a missing infrastructure layer.

---

## The Solution

```python
from agentmemory import MemoryStore

memory = MemoryStore(agent_id="my-agent")
memory.remember("User's name is Alice, building a fraud detection system in Python")

context = memory.get_context("What do we know about the user?")
# → "[Memory Context]\n- User's name is Alice, building a fraud detection system in Python"
```

**That's it.** Memory persists to disk. It's there next session, and the one after that.

---

## Install

```bash
# Minimal install (SQLite episodic memory only, no external dependencies)
pip install agentmemory

# With semantic search + local embeddings (recommended)
pip install "agentmemory[chromadb,local]"

# Batteries included
pip install "agentmemory[all]"
```

---

## Quick Start

### With Anthropic

```python
from agentmemory import MemoryStore
import anthropic

memory = MemoryStore(agent_id="my-agent")
client = anthropic.Anthropic()

def chat(user_input: str) -> str:
    memory.add_message("user", user_input)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=f"You are a helpful assistant.\n\n{memory.get_context(user_input)}",
        messages=memory.get_messages(),
    )
    reply = response.content[0].text
    memory.add_message("assistant", reply)
    return reply

chat("Hi, I'm Alice and I'm building a fraud detection system")
chat("I prefer concise code examples")
# ... restart Python ...
chat("What do you know about me?")
# → "You're Alice, and you're building a fraud detection system in Python.
#    You prefer concise code examples."  ✅
```

### With OpenAI

```python
from agentmemory.adapters.openai import MemoryOpenAI

client = MemoryOpenAI(agent_id="my-agent")
client.chat("Hi, I'm Alice")
client.chat("I'm building a fraud detection system")
# Next session...
client.chat("What's my name?")  # → "Your name is Alice." ✅
```

### With LangChain

```python
from agentmemory import MemoryStore
from agentmemory.adapters.langchain import MemoryHistory, inject_memory_context
from langchain_anthropic import ChatAnthropic

memory = MemoryStore(agent_id="my-agent")
history = MemoryHistory(memory_store=memory)
llm = ChatAnthropic(model="claude-opus-4-6")

history.add_user_message("Hello, I'm Alice")
messages = inject_memory_context(history.messages, memory, query="Alice")
response = llm.invoke(messages)
```

### With CrewAI

```python
from agentmemory import MemoryStore
from agentmemory.adapters.crewai import CrewMemoryCallback, get_memory_context_for_agent
from crewai import Agent, Task

memory = MemoryStore(agent_id="research-crew")

agent = Agent(
    role="Researcher",
    goal="Research AI topics",
    backstory=get_memory_context_for_agent(memory, "Researcher") + "\nExpert researcher.",
)

task = Task(
    description="Research memory systems for AI agents",
    expected_output="Structured research findings",
    agent=agent,
    callback=CrewMemoryCallback(memory),  # Auto-stores task output
)
```

---

## How It Works

AgentMemory uses a **three-tier architecture** that mirrors how human memory works:

```
┌─────────────────────────────────────────────────────────┐
│                    Your LLM / Agent                     │
└─────────────────────┬───────────────────────────────────┘
                      │  get_context() / add_message()
┌─────────────────────▼───────────────────────────────────┐
│                   MemoryStore                           │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Working   │  │   Episodic   │  │   Semantic    │  │
│  │   Memory    │  │   Memory     │  │   Memory      │  │
│  │             │  │              │  │               │  │
│  │ Current     │  │ Recent       │  │ Long-term     │  │
│  │ session     │  │ history      │  │ knowledge     │  │
│  │ (in-RAM)    │  │ (SQLite)     │  │ (ChromaDB)    │  │
│  │             │  │              │  │               │  │
│  │ Auto-       │  │ Persists     │  │ Semantic      │  │
│  │ compresses  │  │ forever      │  │ search        │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Working Memory** — the current conversation window. Automatically compresses old messages into summaries when it nears the token limit.

**Episodic Memory** — recent interactions stored in SQLite. No setup required. Evicts least-important entries when full.

**Semantic Memory** — long-term facts stored as vector embeddings (ChromaDB). Retrieved by meaning, not keyword.

---

## Features

- **Framework-agnostic** — works with LangChain, CrewAI, AutoGen, or any raw SDK
- **Local-first** — runs entirely on your machine, no cloud required
- **Auto-compression** — context window never overflows; old messages are summarized automatically
- **Semantic deduplication** — stops storing near-identical facts that pollute retrieval
- **Importance scoring** — critical memories survive longer; low-priority ones get evicted first
- **Pluggable backends** — ChromaDB (local) or Qdrant (production scale) for semantic memory
- **Zero-config defaults** — just `MemoryStore(agent_id="x")` and you're running

---

## API Reference

### `MemoryStore`

```python
MemoryStore(
    agent_id: str,                        # Unique ID — memories are namespaced by this
    persist_dir: str = "~/.agentmemory", # Where to store memories
    max_working_tokens: int = 4096,      # Token budget before compression triggers
    semantic_backend: str = "chromadb",  # "chromadb" | "qdrant"
    embedding_provider: str = "sentence-transformers",  # "sentence-transformers" | "openai"
    llm_provider: str = "anthropic",     # LLM for compression: "anthropic" | "openai"
    enable_dedup: bool = True,           # Deduplicate before storing
    auto_compress: bool = True,          # Auto-compress when window fills
)
```

| Method | Description |
|---|---|
| `memory.remember(content, importance=5)` | Store a fact in episodic + semantic memory |
| `memory.recall(query, n=5)` | Retrieve top-n relevant memories by meaning |
| `memory.get_context(query, max_tokens=500)` | Get formatted context string for system prompt |
| `memory.add_message(role, content)` | Track a conversation turn in working memory |
| `memory.get_messages()` | Get current working memory as `[{role, content}]` |
| `memory.compress()` | Manually trigger compression of working memory |
| `memory.stats()` | Get memory usage stats across all tiers |
| `memory.clear(tiers=None)` | Clear specific or all memory tiers |

---

## Comparison

| | MemGPT | LangChain Memory | **AgentMemory** |
|---|---|---|---|
| Framework | MemGPT only | LangChain only | Any framework |
| Composable library | No | Partial | **Yes** |
| Local-first | Partial | No | **Yes** |
| Auto-compression | Yes | No | **Yes** |
| Semantic search | Yes | Partial | **Yes** |
| Deduplication | No | No | **Yes** |
| PyPI installable | No | Yes | **Yes** |
| Zero config | No | Partial | **Yes** |

---

## Roadmap

- [ ] AutoGen adapter
- [ ] Qdrant production backend examples
- [ ] Memory export/import (JSON)
- [ ] Memory visualization CLI (`agentmemory inspect`)
- [ ] Async support (`AsyncMemoryStore`)
- [ ] MCP server integration

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/yourusername/agent-memory
cd agent-memory
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT. See [LICENSE](LICENSE).

---

**Star this repo** if you're tired of your agents forgetting everything. 🌟
