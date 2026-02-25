# Social Media Posts — AgentCortex / agent-memory

Copy-paste these to post yourself.

---

## Reddit — r/LocalLLaMA

**Title:** I built a framework-agnostic persistent memory library for AI agents — add memory to any agent in 3 lines (open source)

**Post body:**
```
Every agent I built had the same problem: it forgets everything the moment the session ends.

MemGPT solves this but requires rewriting your entire agent around their framework. I wanted something I could drop into an existing LangChain/CrewAI/AutoGen agent without changing my architecture.

So I built **agentcortex** — a composable Python library for persistent agent memory.

**GitHub:** https://github.com/pinakimishra95/agent-memory
**PyPI:** pip install agentcortex

---

**How it works — three-tier memory:**

```
Working Memory  ← current session (RAM, auto-compresses when window fills)
Episodic Memory ← recent history (SQLite, zero dependencies)
Semantic Memory ← long-term knowledge (ChromaDB/Qdrant, semantic search)
```

**3-line quickstart:**
```python
from agentmemory import MemoryStore

memory = MemoryStore(agent_id="my-agent")
memory.remember("User's name is Alice, building a fraud detection system")
context = memory.get_context("What do we know about the user?")
# → "[Memory Context]\n- User's name is Alice, building a fraud detection system"
```

Run the demo twice to see memories survive a Python restart:
```bash
git clone https://github.com/pinakimishra95/agent-memory
cd agent-memory && pip install -e .
python examples/demo.py   # stores memories
python examples/demo.py   # recalls them from disk
```

**Works with:**
- LangChain/LangGraph
- CrewAI
- AutoGen
- Raw Anthropic/OpenAI SDK

**Key features:**
- Auto-compresses old messages when context window fills (uses cheap model, e.g. claude-haiku)
- Semantic deduplication before storing (won't save near-identical facts twice)
- Importance scoring — critical memories survive longer under eviction pressure
- Local-first: runs entirely on-device, no cloud required

Happy to answer questions — still early (v0.1.0) but core functionality is solid with 26 passing tests.
```

---

## Reddit — r/MachineLearning or r/artificial

**Title:** AgentCortex: Open-source three-tier memory architecture for AI agents (working/episodic/semantic)

**Post body:**
```
Released an open-source Python library for persistent AI agent memory inspired by cognitive memory research.

Most agent frameworks treat memory as an afterthought. AgentCortex implements a proper three-tier architecture:

**Working memory** — the active context window. Tracked per-session, auto-compressed when nearing the token limit (old messages get summarized into episodic memory automatically).

**Episodic memory** — recent interaction history stored in SQLite. Searchable by recency and keyword. Survives Python restarts. No external dependencies.

**Semantic memory** — long-term factual knowledge stored as vector embeddings (ChromaDB local or Qdrant for production scale). Retrieved by meaning, not keyword.

The library is framework-agnostic — it's a composable library, not a framework. Drop it into any existing LangChain, CrewAI, AutoGen, or raw SDK agent.

GitHub: https://github.com/pinakimishra95/agent-memory
pip install agentcortex

Would appreciate feedback from anyone building production agent systems.
```

---

## Hacker News — "Show HN"

**Title:** Show HN: AgentCortex – Persistent memory for AI agents in 3 lines (open source)

**Post body:**
```
Your AI agent forgets everything when the session ends. AgentCortex fixes that.

It's a Python library — not a framework — so you can drop it into any existing agent without rewriting anything. Works with LangChain, CrewAI, AutoGen, raw Anthropic/OpenAI.

  from agentmemory import MemoryStore
  memory = MemoryStore(agent_id="my-agent")
  memory.remember("User's name is Alice, building a fraud detection system")
  context = memory.get_context("What do we know about the user?")

Three-tier architecture: working (in-context) → episodic (SQLite, no deps) → semantic (ChromaDB/Qdrant). Auto-compresses when the context window fills so your agent never runs out of context.

Demo: clone the repo and run `python examples/demo.py` twice — the second run recalls everything from the first, including across a full Python restart.

GitHub: https://github.com/pinakimishra95/agent-memory
PyPI: pip install agentcortex

Happy to answer questions on the architecture choices (why SQLite for episodic, why three tiers vs two, how the compression prompts work).
```

---

## X / Twitter (Thread)

**Tweet 1 (hook):**
```
Your AI agent forgets everything the moment the session ends.

I built agentcortex — add persistent memory to any agent in 3 lines.

Works with LangChain, CrewAI, AutoGen, raw Anthropic/OpenAI.

GitHub: https://github.com/pinakimishra95/agent-memory

🧵 How it works:
```

**Tweet 2 (architecture):**
```
Three-tier memory architecture (inspired by how human memory works):

🟡 Working Memory → current conversation (in-context, auto-compresses)
🟠 Episodic Memory → recent sessions (SQLite, survives restarts)
🔵 Semantic Memory → long-term facts (ChromaDB, semantic search)

Old messages auto-summarize into episodic. Important facts extract into semantic.
```

**Tweet 3 (code demo):**
```
The whole API in 4 lines:

from agentmemory import MemoryStore

memory = MemoryStore(agent_id="my-agent")
memory.remember("Alice is building a fraud detection system")
context = memory.get_context("What do we know about the user?")
# → "[Memory Context]\n- Alice is building a fraud detection system"

That context goes straight into your system prompt.
```

**Tweet 4 (differentiators):**
```
Why not MemGPT?

MemGPT: replace your entire agent with their framework (35K stars but not a library)
agentcortex: drop into your existing agent in 3 lines ✅

✅ LangChain adapter
✅ CrewAI adapter
✅ AutoGen adapter
✅ Raw Anthropic/OpenAI adapter
✅ Local-first (SQLite + local embeddings, no cloud)
✅ pip install agentcortex
```

**Tweet 5 (CTA):**
```
Open source, MIT license, 26 passing tests.

pip install agentcortex

⭐ Star it if you're building agents and tired of them forgetting everything:
https://github.com/pinakimishra95/agent-memory
```

---

## LinkedIn

```
I just open-sourced agentcortex — a Python library that gives any AI agent persistent memory in 3 lines of code.

The problem I kept hitting: every AI agent I built would forget everything the moment the session ended. Existing solutions (MemGPT) required rewriting my entire agent around their framework.

So I built something composable instead.

agentcortex uses a three-tier memory architecture:
• Working Memory — the active context window, auto-compresses when full
• Episodic Memory — recent history in SQLite (no external deps, survives restarts)
• Semantic Memory — long-term facts in ChromaDB with vector similarity search

It works with whatever framework you're already using: LangChain, CrewAI, AutoGen, or raw Anthropic/OpenAI.

Quick example:
from agentmemory import MemoryStore
memory = MemoryStore(agent_id="my-agent")
memory.remember("User's name is Alice, building fraud detection")
context = memory.get_context("What do we know about the user?")

GitHub: https://github.com/pinakimishra95/agent-memory
PyPI: pip install agentcortex

Would love feedback from anyone building production agents. What memory patterns have you found most useful?

#AI #MachineLearning #OpenSource #Python #LLM #AgentAI #LangChain
```

---

## GIF Recording Instructions

To create the terminal demo GIF for the README:

### Option A: vhs (recommended — clean output)
```bash
brew install vhs
```

Create `demo.tape`:
```
Output demo.gif
Set FontSize 14
Set Width 800
Set Height 500
Set Theme "Dracula"

Type "python examples/demo.py"
Enter
Sleep 3s

Type "python examples/demo.py"
Enter
Sleep 3s
```

Run: `vhs demo.tape`

### Option B: asciinema (browser-embeddable)
```bash
brew install asciinema
asciinema rec demo.cast
# run: python examples/demo.py && python examples/demo.py
# Ctrl+D to stop
asciinema upload demo.cast
```

Add the GIF/asciinema link to the README right below the "The Solution" section for maximum impact.
