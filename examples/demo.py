"""
demo.py — Self-contained memory persistence demo.

Run this TWICE to see memories survive a Python session restart.
No API key needed — uses only the SQLite episodic tier.

Usage:
    python examples/demo.py
"""

import sys
import time
from pathlib import Path

# Add project root to path for running before pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentmemory import MemoryStore

DEMO_AGENT_ID = "demo-persistent"

memory = MemoryStore(
    agent_id=DEMO_AGENT_ID,
    auto_compress=False,   # no API key needed
    enable_dedup=False,    # no embeddings needed
)

run_count = memory.episodic.count()

print()
print("=" * 55)
print("  agentmemory — Persistence Demo")
print("=" * 55)

if run_count == 0:
    # ── FIRST RUN ──────────────────────────────────────────
    print("\n  [First run — storing memories]\n")

    facts = [
        "User's name is Alice",
        "Alice is building a real-time fraud detection system",
        "Alice prefers Python and concise code examples",
        "Alice works at a fintech startup in San Francisco",
    ]

    for fact in facts:
        memory.remember(fact, store_semantic=False)
        print(f"  + Stored: {fact}")
        time.sleep(0.2)

    print(f"\n  Memories saved to: ~/.agentmemory/{DEMO_AGENT_ID}_episodic.db")
    print("\n  Now run this script AGAIN to see them recalled.")

else:
    # ── SECOND RUN (or later) ──────────────────────────────
    print(f"\n  [Subsequent run — recalling {run_count} memories from disk]\n")

    recent = memory.episodic.recall_recent(n=10)
    for m in recent:
        print(f"  ✓ Recalled: {m['content']}")
        time.sleep(0.15)

    print()
    ctx = memory.get_context(max_tokens=300)
    print("  Context injected into system prompt:")
    print("  " + "\n  ".join(ctx.splitlines()))
    print()
    print("  The agent remembered everything — across a full restart.")

print()
print("  Stats:", memory.stats()["episodic"])
print("=" * 55)
print()
