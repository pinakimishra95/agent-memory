"""
crewai_agent.py — CrewAI integration demo.

Shows how to give a CrewAI crew persistent memory across runs.
Task outputs are automatically stored and recalled in future runs.

Requirements:
    pip install agentmemory crewai chromadb sentence-transformers
    export ANTHROPIC_API_KEY=...
"""

from agentmemory import MemoryStore
from agentmemory.adapters.crewai import CrewMemoryCallback, get_memory_context_for_agent

from crewai import Agent, Task, Crew


def main():
    # Shared memory store for the whole crew
    memory = MemoryStore(agent_id="research-crew")

    # Get memory context relevant to each agent's role
    researcher_context = get_memory_context_for_agent(
        memory_store=memory,
        role="Researcher",
        goal="Research AI agent architectures",
    )
    writer_context = get_memory_context_for_agent(
        memory_store=memory,
        role="Writer",
        goal="Write technical summaries",
    )

    # Inject memory into agent backstories
    researcher = Agent(
        role="Senior AI Researcher",
        goal="Research AI agent architectures and memory systems",
        backstory=(
            f"{researcher_context}\n\n" if researcher_context else ""
        ) + "Expert researcher specializing in LLM applications and agent systems.",
        verbose=True,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Create clear technical summaries of research findings",
        backstory=(
            f"{writer_context}\n\n" if writer_context else ""
        ) + "Experienced technical writer who specializes in AI and ML content.",
        verbose=True,
    )

    # Memory callback — stores task output as a memory for future runs
    memory_callback = CrewMemoryCallback(memory_store=memory, importance=8)

    # Tasks
    research_task = Task(
        description=(
            "Research the top 3 approaches to persistent memory in AI agents. "
            "Focus on practical production implementations."
        ),
        expected_output="A structured summary of 3 memory approaches with pros/cons.",
        agent=researcher,
        callback=memory_callback,  # Auto-store the output
    )

    write_task = Task(
        description="Write a concise technical blog post based on the research findings.",
        expected_output="A 300-word technical blog post ready for publication.",
        agent=writer,
        callback=memory_callback,
    )

    # Run the crew
    crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], verbose=True)
    result = crew.kickoff()

    print("\n--- Crew Result ---")
    print(result)
    print("\n--- Memory Stats ---")
    print(memory.stats())
    print("\nMemories stored this run:")
    for m in memory.episodic.recall_recent(n=5):
        print(f"  [{m['importance']}] {m['content'][:100]}...")


if __name__ == "__main__":
    main()
