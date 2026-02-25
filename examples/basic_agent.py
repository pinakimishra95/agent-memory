"""
basic_agent.py — The "3 lines" demo.

Shows that agentmemory persists across Python sessions.
Run this script twice to see memories survive a restart.

Requirements:
    pip install agentmemory anthropic
    export ANTHROPIC_API_KEY=...
"""

from agentmemory import MemoryStore

# 1. Create a memory store (persists to ~/.agentmemory/ by default)
memory = MemoryStore(agent_id="demo-agent")

# 2. Remember something
memory.remember("User's name is Alice and she is building a web scraper in Python")
memory.remember("Alice prefers concise answers with code examples")

# 3. Get context to inject into your system prompt
context = memory.get_context("What do we know about the user?")
print("Memory context:\n", context)

# --- Full multi-turn example with Anthropic ---
import anthropic
import os

client = anthropic.Anthropic()

print("\n--- Multi-turn conversation with persistent memory ---\n")

turns = [
    "Hi! I'm Alice.",
    "I'm building a Python web scraper. Any tips?",
    "What's my name again?",  # Should recall from memory after restart
]

for user_input in turns:
    # Add user message to working memory (auto-compresses when window fills)
    memory.add_message("user", user_input)

    # Get relevant context for this query
    ctx = memory.get_context(query=user_input, max_tokens=400)

    system = "You are a helpful coding assistant."
    if ctx:
        system += f"\n\n{ctx}"

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        system=system,
        messages=memory.get_messages(),
    )
    reply = response.content[0].text
    memory.add_message("assistant", reply)

    print(f"User: {user_input}")
    print(f"Agent: {reply}\n")

# Stats
print("Memory stats:", memory.stats())
