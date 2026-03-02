"""MCP server for agentmemory — gives Claude Code persistent codebase memory.

Exposes agentmemory's MemoryStore as MCP tools so any MCP-compatible AI coding
assistant (Claude Code, Cursor, etc.) can remember and recall knowledge about
your codebase across sessions.

Usage:
    # Start the server
    python -m agentmemory.mcp_server

    # Or via the installed entry point
    agentmemory-mcp

Configure via environment variables:
    AGENTMEMORY_AGENT_ID     Memory namespace (default: "default" — use your project name)
    AGENTMEMORY_PERSIST_DIR  Storage directory (default: ~/.agentmemory)
    AGENTMEMORY_LLM_PROVIDER LLM for auto-compression: "anthropic" | "openai" (default: "anthropic")
    AGENTMEMORY_LLM_MODEL    Override default compression model
    AGENTMEMORY_API_KEY      API key (falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY)

Claude Code setup (.mcp.json in project root):
    {
      "mcpServers": {
        "agentmemory": {
          "type": "stdio",
          "command": "python",
          "args": ["-m", "agentmemory.mcp_server"],
          "env": { "AGENTMEMORY_AGENT_ID": "your-project-name" }
        }
      }
    }
"""
from __future__ import annotations

import os
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: mcp package not installed. Run: pip install 'agentcortex[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

mcp = FastMCP("agentmemory")

# Lazy singleton — MemoryStore (and heavy deps like sentence-transformers)
# are only imported and initialised on the first tool call, not at server start.
_store = None


def _get_store():
    """Return the shared MemoryStore, creating it on first call."""
    global _store
    if _store is None:
        from agentmemory import MemoryStore  # noqa: PLC0415

        agent_id = os.environ.get("AGENTMEMORY_AGENT_ID", "default")
        persist_dir = os.environ.get("AGENTMEMORY_PERSIST_DIR") or None
        llm_provider = os.environ.get("AGENTMEMORY_LLM_PROVIDER", "anthropic")
        llm_model = os.environ.get("AGENTMEMORY_LLM_MODEL") or None
        api_key = os.environ.get("AGENTMEMORY_API_KEY") or None

        print(
            f"[agentmemory] initialising store: agent_id={agent_id!r}",
            file=sys.stderr,
        )
        _store = MemoryStore(
            agent_id=agent_id,
            persist_dir=persist_dir,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key,
        )
    return _store


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def remember(content: str, importance: int = 5) -> str:
    """Store a fact, decision, preference, or insight in persistent memory.

    Call this proactively when you discover something important about the codebase
    that future sessions should know. Good candidates:

    - Architectural decisions and WHY they were made
      ("We use PostgreSQL instead of MongoDB because the reporting queries
       need complex joins — tested MongoDB and it was 3x slower")
    - User coding preferences and style rules
      ("User prefers functional style, no classes unless necessary")
    - Known bugs, gotchas, or fragile areas
      ("payment/process_transaction.py had a race condition — fixed Feb 2026
       by using a DB-level lock. Don't use in-memory locks here.")
    - How a tricky problem was solved
    - Key files and what they own

    Args:
        content: The fact or insight to remember (plain English, be specific)
        importance: How critical this memory is, 1-10 (10=critical, 1=trivial).
                    Default 5. Use 8-10 for security issues, core patterns,
                    breaking constraints. Use 1-3 for minor style preferences.
    """
    store = _get_store()
    store.remember(content, importance=importance)
    preview = content[:80] + ("..." if len(content) > 80 else "")
    return f"Stored (importance={importance}/10): {preview}"


@mcp.tool()
def recall(query: str, n: int = 5) -> str:
    """Retrieve memories relevant to a query using semantic search.

    Call this before starting work on any file or feature to surface relevant
    past context — known issues, prior decisions, user preferences for that
    area of the codebase.

    Examples:
      recall("payment module")           → prior bugs, patterns, constraints
      recall("database migrations")      → how we handle schema changes
      recall("user prefers")             → coding style + preferences
      recall("authentication security")  → known vulnerabilities, design choices

    Args:
        query: What to search for. Natural language — semantic match, not exact.
        n: Max number of memories to return (default 5).
    """
    store = _get_store()
    memories = store.recall(query, n=n)
    if not memories:
        return "No relevant memories found."
    lines = []
    for i, m in enumerate(memories, 1):
        source = m.get("source", "memory")
        content = m.get("content", "")
        lines.append(f"{i}. [{source}] {content}")
    return "\n".join(lines)


@mcp.tool()
def get_context(query: str | None = None, max_tokens: int = 500) -> str:
    """Get a formatted block of relevant memories to guide your current task.

    Call this at the START of every session or when switching to a new task.
    Provide a brief description of what you're about to work on — the server
    will return the most relevant memories as a pre-formatted context block
    ready to inject into your reasoning.

    Example:
      get_context("add rate limiting to the API")
      → returns memories about: existing middleware, Redis usage, past
        rate-limit attempts, user's preference for no new dependencies

    Args:
        query: What you're about to work on (optional but recommended).
        max_tokens: Max token budget for the context block (default 500).
    """
    store = _get_store()
    context = store.get_context(query=query, max_tokens=max_tokens)
    if not context or context.strip() == "[Memory Context]":
        return "No memories stored yet. Use remember() to start building codebase memory."
    return context


@mcp.tool()
def memory_stats() -> str:
    """Show how many memories are stored across each tier.

    Use this to get a quick overview of what agentmemory has stored for the
    current project, or to verify that remember() calls are persisting correctly.
    """
    store = _get_store()
    stats = store.stats()
    agent_id = os.environ.get("AGENTMEMORY_AGENT_ID", "default")
    working_msgs = stats.get("working", {}).get("message_count", 0)
    episodic_count = stats.get("episodic", {}).get("count", 0)
    semantic_count = stats.get("semantic", {}).get("count", 0)
    total = episodic_count + semantic_count
    lines = [
        f"Agent ID       : {agent_id}",
        f"Working memory : {working_msgs} messages",
        f"Episodic memory: {episodic_count} entries",
        f"Semantic memory: {semantic_count} entries",
        f"Total stored   : {total} memories",
    ]
    return "\n".join(lines)


@mcp.tool()
def clear_memory(tiers: list[str] | None = None) -> str:
    """Clear stored memories. WARNING: this is permanent and cannot be undone.

    Only call this if explicitly asked by the user, or to reset a corrupt state.

    Args:
        tiers: Which tiers to clear. Options: ["working"], ["episodic"],
               ["semantic"], or any combination. Pass None (default) to clear ALL.
    """
    store = _get_store()
    store.clear(tiers=tiers)
    if tiers:
        cleared = ", ".join(tiers)
        return f"Cleared memory tiers: {cleared}"
    return "Cleared all memory tiers (working, episodic, semantic)."


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Start the agentmemory MCP server (stdio transport)."""
    agent_id = os.environ.get("AGENTMEMORY_AGENT_ID", "default")
    print(
        f"[agentmemory] MCP server starting — agent_id={agent_id!r}",
        file=sys.stderr,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
