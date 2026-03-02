"""
AsyncMemoryStore example — persistent memory in async applications.

Demonstrates using agentmemory with async/await for FastAPI, aiohttp,
or any other async Python framework.

Install dependencies:
    pip install agentcortex

Run:
    python examples/async_agent.py
"""
import asyncio

from agentmemory import AsyncMemoryStore


async def main():
    # AsyncMemoryStore has the same API as MemoryStore — just add await
    memory = AsyncMemoryStore(agent_id="async-agent")

    # Store memories asynchronously
    await memory.remember("User prefers Python over JavaScript", importance=7)
    await memory.remember("Project uses FastAPI for the web layer", importance=8)
    await memory.remember("Tests must pass before any PR is merged", importance=6)

    # Recall memories relevant to a query
    results = await memory.recall("tech stack and preferences", n=5)
    print("Recalled memories:")
    for r in results:
        print(f"  [{r['source']}] {r['content']}")

    # Get formatted context for injection into a system prompt
    context = await memory.get_context("What do we know about the project?")
    print(f"\nMemory context:\n{context}")

    # Working memory (conversation history)
    await memory.add_message("user", "What's the tech stack?")
    await memory.add_message("assistant", "We use FastAPI + Python with pytest for testing.")
    msgs = await memory.get_messages()
    print(f"\nWorking memory: {len(msgs)} messages")

    # Export memories to JSON for backup
    data = await memory.export_json()
    print(f"\nExported {len(data['episodic'])} episodic memories")

    # Stats
    stats = await memory.stats()
    print(f"Stats: {stats}")

    # Use as an async context manager (auto-closes executor on exit)
    async with AsyncMemoryStore(agent_id="ctx-mgr-agent") as mem2:
        await mem2.remember("Context manager handles cleanup automatically", importance=5)
        ctx = await mem2.get_context()
        print(f"\nContext manager example context:\n{ctx}")

    memory.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
