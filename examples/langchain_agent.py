"""
langchain_agent.py — LangChain integration demo.

Shows how to use agentmemory as a drop-in memory backend for LangChain chains.
Memory persists across Python sessions automatically.

Requirements:
    pip install agentmemory langchain-core langchain-anthropic chromadb sentence-transformers
    export ANTHROPIC_API_KEY=...
"""

from agentmemory import MemoryStore
from agentmemory.adapters.langchain import MemoryHistory, inject_memory_context

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


def main():
    # Initialize persistent memory
    memory = MemoryStore(agent_id="langchain-demo")

    # Pre-load some known facts
    memory.remember("User is a senior Python developer at a fintech startup")
    memory.remember("User is building a real-time fraud detection system")
    memory.remember("User prefers production-ready code with error handling")

    # Create LangChain history backed by agentmemory
    history = MemoryHistory(memory_store=memory)

    # Standard LangChain setup
    llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)

    # Example: Manual integration with inject_memory_context
    def chat(user_input: str) -> str:
        history.add_user_message(user_input)

        # Inject memory context into the message list
        messages = inject_memory_context(
            messages=history.messages,
            memory_store=memory,
            query=user_input,
            max_tokens=400,
        )

        response = llm.invoke(messages)
        reply = response.content
        history.add_ai_message(reply)
        return reply

    print("--- LangChain + agentmemory demo ---\n")
    prompts = [
        "What tech stack should I use for my project?",
        "How do I handle false positives in fraud detection?",
        "What do you know about my background?",  # Tests memory recall
    ]

    for prompt in prompts:
        print(f"User: {prompt}")
        reply = chat(prompt)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
