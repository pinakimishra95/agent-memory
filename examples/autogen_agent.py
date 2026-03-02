"""
AutoGen + agentmemory example.

This example shows how to give an AutoGen AssistantAgent persistent memory
so it can recall facts from previous sessions.

Install dependencies:
    pip install agentcortex pyautogen

Run:
    python examples/autogen_agent.py
"""
from agentmemory import MemoryStore
from agentmemory.adapters.autogen import AutoGenMemoryHook, get_autogen_memory_context

# 1. Create a persistent memory store
memory = MemoryStore(agent_id="autogen-researcher")

# 2. Build a memory context string from past sessions
context = get_autogen_memory_context(
    memory,
    role="Research Assistant",
    goal="literature review on LLMs",
)

# 3. Wire up the hook and agent (requires: pip install pyautogen)
try:
    import autogen  # noqa: PLC0415

    assistant = autogen.AssistantAgent(
        name="researcher",
        system_message=(context + "\nYou are a helpful research assistant.").strip(),
        llm_config={"model": "gpt-4o-mini"},  # replace with your config
    )

    # Register the hook so every reply is saved to memory
    hook = AutoGenMemoryHook(memory, importance=6)
    assistant.register_reply(
        trigger=autogen.ConversableAgent,
        reply_func=hook.on_agent_reply,
        position=0,
    )

    print("AutoGen agent ready with persistent memory.")
    print(f"Current memory context:\n{context or '(empty — first run)'}")

except ImportError:
    # Demo without AutoGen installed
    memory.remember("Transformer models dominate NLP benchmarks", importance=7)
    memory.remember("GPT-4 outperforms previous SOTA on reasoning tasks", importance=8)

    hook = AutoGenMemoryHook(memory, importance=6)
    fake_messages = [{"role": "assistant", "content": "Reviewed 5 papers on attention mechanisms"}]
    hook.on_agent_reply(None, fake_messages, None, None)

    print("AutoGen not installed — demo mode.")
    print("\nMemories stored:")
    for r in memory.recall("LLM research", n=5):
        print(f"  [{r['source']}] {r['content']}")

    ctx = get_autogen_memory_context(memory, role="Research Assistant")
    print(f"\nMemory context that would be injected:\n{ctx}")
