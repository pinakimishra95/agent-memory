# agentmemory 🧠

**Your AI agent forgets everything. AgentMemory fixes that in 3 lines.**

[![PyPI version](https://badge.fury.io/py/agentcortex.svg)](https://badge.fury.io/py/agentcortex)
[![Tests](https://github.com/pinakimishra95/agent-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/pinakimishra95/agent-memory/actions)
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
pip install agentcortex

# With semantic search + local embeddings (recommended)
pip install "agentcortex[chromadb,local]"

# Batteries included
pip install "agentcortex[all]"
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

## Claude Code Integration — Persistent Codebase Memory

Give Claude Code a permanent brain for your project. Every session, it remembers
architecture decisions, bug fixes, your coding preferences, and known gotchas — and
recalls the relevant ones automatically before touching any file.

```bash
pip install "agentcortex[mcp]"
```

Copy `example.mcp.json` to your project root and rename it `.mcp.json`, then set your
project name:

```json
{
  "mcpServers": {
    "agentmemory": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "agentmemory.mcp_server"],
      "env": {
        "AGENTMEMORY_AGENT_ID": "your-project-name"
      }
    }
  }
}
```

Open Claude Code and run `/mcp` — you'll see `agentmemory` connected with 5 tools.

Claude will now automatically:
- Call `get_context("current task")` at the start of each session
- Call `remember(...)` after fixing bugs, making architectural decisions, or learning about the codebase
- Call `recall("payment module")` before touching any file it's worked on before

### What it looks like in practice

```
Session 1 — You:  "Fix the race condition in payment/process_transaction.py"
Claude fixes it, then calls:
  remember("payment/process_transaction.py had a race condition — fixed with
   a DB-level lock in process_transaction(). Do NOT use in-memory locks here,
   they don't work across workers.", importance=9)

Session 2 (next week) — You: "Add retry logic to the payment module"
Claude calls:  get_context("payment module retry logic")
→ retrieves:   "process_transaction.py: use DB-level locks, not in-memory"
Claude:        "I remember this module had a concurrency issue before.
                I'll make sure the retry logic respects the DB-level lock..."
```

### Available MCP tools

| Tool | Description |
|---|---|
| `get_context(query, max_tokens)` | Call at session start — returns relevant memories for current task |
| `remember(content, importance)` | Store a fact, decision, or gotcha (importance 1-10) |
| `recall(query, n)` | Semantic search over all stored memories |
| `memory_stats()` | Show counts across working / episodic / semantic tiers |
| `clear_memory(tiers)` | Reset memories (irreversible) |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `AGENTMEMORY_AGENT_ID` | `"default"` | Memory namespace — use your project name |
| `AGENTMEMORY_PERSIST_DIR` | `~/.agentmemory` | Where memories are stored on disk |
| `AGENTMEMORY_LLM_PROVIDER` | `"anthropic"` | LLM for auto-compression: `"anthropic"` or `"openai"` |

Works with Claude Code, Cursor, and any MCP-compatible AI coding assistant.

---

## AutoGen Integration

Give AutoGen agents persistent memory that survives across sessions.

```python
from agentmemory import MemoryStore
from agentmemory.adapters.autogen import AutoGenMemoryHook, get_autogen_memory_context
import autogen

memory = MemoryStore(agent_id="my-autogen-agent")

# Inject past context into the agent's system_message
context = get_autogen_memory_context(memory, role="Research Assistant",
                                     goal="literature review on LLMs")

assistant = autogen.AssistantAgent(
    name="researcher",
    system_message=context + "\nYou are a helpful research assistant.",
    llm_config={"model": "gpt-4o-mini"},
)

# Hook captures every reply and stores it in memory
hook = AutoGenMemoryHook(memory, importance=6)
assistant.register_reply(
    trigger=autogen.ConversableAgent,
    reply_func=hook.on_agent_reply,
    position=0,
)
```

Install: `pip install "agentcortex[autogen]"`

---

## Qdrant Production Backend

Scale to millions of vectors with a dedicated vector database.

```python
from agentmemory import MemoryStore

# docker run -p 6333:6333 qdrant/qdrant
memory = MemoryStore(
    agent_id="my-agent",
    semantic_backend="qdrant",
    qdrant_url="http://localhost:6333",      # or Qdrant Cloud URL
    embedding_provider="sentence-transformers",
)

memory.remember("Production architecture uses microservices", importance=8)
results = memory.recall("architecture")
```

Install: `pip install "agentcortex[qdrant]"`

---

## Memory Export / Import (JSON)

Back up and restore episodic memories across machines or agent instances.

```python
from agentmemory import MemoryStore

memory = MemoryStore(agent_id="my-agent")
memory.remember("PostgreSQL is our main database", importance=8)

# Export to JSON file
memory.export_json("backup.json")

# Restore on another machine / new agent
new_memory = MemoryStore(agent_id="new-agent")
count = new_memory.import_json("backup.json")
print(f"Imported {count} memories")

# Merge instead of replacing
new_memory.import_json("backup.json", merge=True)

# Or work with the dict directly
data = memory.export_json()   # no path → returns dict only
new_memory.import_json(data)
```

---

## Memory CLI

Inspect and manage memories from the command line.

```bash
# Inspect stored memories
agentmemory inspect --agent-id my-project

# agentmemory — agent: my-project
# ════════════════════════════════════════
# EPISODIC MEMORY  (3 entries)
# ────────────────────────────────────────
#   #   IMP   Created              Content
#   1    9    2026-02-28 14:23:01  We use PostgreSQL for relational...
#   2    7    2026-02-27 09:14:55  payment/process_transaction.py h...
#   3    5    2026-02-26 18:30:12  User prefers functional style ove...

# Export memories to JSON
agentmemory export --agent-id my-project --output memories.json

# Import memories (restores; use --merge to add alongside existing)
agentmemory import memories.json --agent-id new-project --merge
```

Install: `pip install agentcortex`  (the CLI is always included)

---

## Async Support

Use agentmemory in FastAPI, aiohttp, or any async Python application.

```python
import asyncio
from agentmemory import AsyncMemoryStore

async def main():
    # Identical API to MemoryStore — just add await
    memory = AsyncMemoryStore(agent_id="my-async-agent")

    await memory.remember("User prefers Python over JavaScript", importance=7)
    results = await memory.recall("tech stack")
    context = await memory.get_context("What do we know?")

    # Export / import work the same way
    data = await memory.export_json()
    await memory.import_json(data)

    memory.close()

# Or use as an async context manager
async def with_context_manager():
    async with AsyncMemoryStore(agent_id="my-agent") as memory:
        await memory.remember("Context manager closes executor automatically")
        ctx = await memory.get_context()
        print(ctx)

asyncio.run(main())
```

Install: `pip install agentcortex`  (`AsyncMemoryStore` is always included)

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

- [x] AutoGen adapter (`pip install "agentcortex[autogen]"`)
- [x] Qdrant production backend (`pip install "agentcortex[qdrant]"`)
- [x] Memory export/import (JSON) — `memory.export_json()` / `memory.import_json()`
- [x] Memory visualization CLI — `agentmemory inspect / export / import`
- [x] Async support — `AsyncMemoryStore` with full `await` API
- [x] MCP server integration (`pip install "agentcortex[mcp]"`)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/pinakimishra95/agent-memory
cd agent-memory
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT. See [LICENSE](LICENSE).

---

**Star this repo** if you're tired of your agents forgetting everything. 🌟
