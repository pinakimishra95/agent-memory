"""agentmemory CLI — inspect, export, and import agent memories.

Entry point: agentmemory

Subcommands:
    inspect   Display stored memories in a formatted table
    export    Export memories to a JSON file
    import    Import memories from a JSON file

Configuration is read from CLI args or environment variables:
    AGENTMEMORY_AGENT_ID    — agent namespace (default: "default")
    AGENTMEMORY_PERSIST_DIR — storage directory (default: ~/.agentmemory)

Usage:
    agentmemory inspect --agent-id my-project
    agentmemory export  --agent-id my-project --output memories.json
    agentmemory import  memories.json --agent-id new-project
    agentmemory import  memories.json --agent-id new-project --merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# ── helpers ───────────────────────────────────────────────────────────────────


def _get_store(agent_id: str, persist_dir: str | None):
    """Lazily import and construct a MemoryStore."""
    from agentmemory import MemoryStore  # noqa: PLC0415

    return MemoryStore(
        agent_id=agent_id,
        persist_dir=persist_dir,
        auto_compress=False,
        auto_extract_facts=False,
        enable_dedup=False,
    )


def _fmt_ts(ts: float) -> str:
    """Format a Unix timestamp as a human-readable string."""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _truncate(text: str, width: int = 55) -> str:
    """Truncate text to fit in a fixed-width column."""
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def _resolve(args: argparse.Namespace, attr: str, env_var: str, default: str | None) -> str | None:
    """Resolve a value from CLI arg → env var → default."""
    val = getattr(args, attr, None)
    if val:
        return val
    return os.environ.get(env_var) or default


# ── subcommands ───────────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    """Print a formatted table of stored memories."""
    agent_id = _resolve(args, "agent_id", "AGENTMEMORY_AGENT_ID", "default")
    persist_dir = _resolve(args, "persist_dir", "AGENTMEMORY_PERSIST_DIR", None)
    limit = getattr(args, "limit", 50) or 50

    store = _get_store(agent_id, persist_dir)  # type: ignore[arg-type]
    stats = store.stats()

    # Header
    width = 72
    print()
    print(f"  agentmemory — agent: {agent_id}")
    print("  " + "═" * width)

    # ── Episodic memories ─────────────────────────────────────────────────────
    ep_count = stats.get("episodic", {}).get("count", 0)
    print(f"\n  EPISODIC MEMORY  ({ep_count} entries)")
    print("  " + "─" * width)

    rows = store.episodic.recall_recent(n=limit)
    if not rows:
        print("  (empty)")
    else:
        print(f"  {'#':<4} {'IMP':<5} {'Created':<22} Content")
        print("  " + "─" * width)
        for i, r in enumerate(rows, 1):
            imp = r.get("importance", 5)
            ts = _fmt_ts(r.get("created_at", 0))
            content = _truncate(r.get("content", ""), 45)
            print(f"  {i:<4} {imp:<5} {ts:<22} {content}")

    # ── Working memory ────────────────────────────────────────────────────────
    wm = stats.get("working", {})
    wm_count = wm.get("message_count", 0)
    print(f"\n  WORKING MEMORY  ({wm_count} messages — session-scoped)")
    print("  " + "─" * width)
    msgs = store.working.get_stats()
    if wm_count == 0:
        print("  (empty — working memory is cleared between sessions)")
    else:
        print(f"  {wm_count} messages  ·  ~{msgs.get('token_count', 0)} tokens")

    # ── Semantic memory ───────────────────────────────────────────────────────
    sem_count = stats.get("semantic", {}).get("count", 0)
    print(f"\n  SEMANTIC MEMORY  ({sem_count} vector embeddings)")
    print("  " + "─" * width)
    if sem_count == 0:
        print("  (empty or vector backend not installed)")
    else:
        print(f"  {sem_count} entries indexed for semantic search")

    print()


def cmd_export(args: argparse.Namespace) -> None:
    """Export episodic memories to a JSON file."""
    agent_id = _resolve(args, "agent_id", "AGENTMEMORY_AGENT_ID", "default")
    persist_dir = _resolve(args, "persist_dir", "AGENTMEMORY_PERSIST_DIR", None)
    output = args.output

    store = _get_store(agent_id, persist_dir)  # type: ignore[arg-type]
    data = store.export_json(path=output)

    count = len(data.get("episodic", []))
    if output:
        print(f"Exported {count} episodic memories to {output}")
    else:
        print(json.dumps(data, indent=2))


def cmd_import(args: argparse.Namespace) -> None:
    """Import episodic memories from a JSON file."""
    agent_id = _resolve(args, "agent_id", "AGENTMEMORY_AGENT_ID", "default")
    persist_dir = _resolve(args, "persist_dir", "AGENTMEMORY_PERSIST_DIR", None)
    source = args.file
    merge = getattr(args, "merge", False)

    if not os.path.exists(source):
        print(f"ERROR: file not found: {source}", file=sys.stderr)
        sys.exit(1)

    store = _get_store(agent_id, persist_dir)  # type: ignore[arg-type]
    count = store.import_json(source, merge=merge)
    action = "merged into" if merge else "imported into"
    print(f"Imported {count} memories ({action} agent: {agent_id})")


# ── arg parser ────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentmemory",
        description="Inspect and manage agentmemory persistent memories.",
    )
    parser.add_argument("--version", action="version", version="agentmemory 0.1.2")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── inspect ──────────────────────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Display stored memories in a table")
    p_inspect.add_argument("--agent-id", dest="agent_id", default=None,
                           help="Agent namespace (default: $AGENTMEMORY_AGENT_ID or 'default')")
    p_inspect.add_argument("--persist-dir", dest="persist_dir", default=None,
                           help="Storage directory (default: $AGENTMEMORY_PERSIST_DIR or ~/.agentmemory)")
    p_inspect.add_argument("--limit", type=int, default=50,
                           help="Max episodic memories to display (default: 50)")

    # ── export ───────────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export memories to a JSON file")
    p_export.add_argument("--agent-id", dest="agent_id", default=None)
    p_export.add_argument("--persist-dir", dest="persist_dir", default=None)
    p_export.add_argument("--output", "-o", default=None,
                          help="Output file path. If omitted, prints JSON to stdout.")

    # ── import ───────────────────────────────────────────────────────────────
    p_import = sub.add_parser("import", help="Import memories from a JSON file")
    p_import.add_argument("file", help="JSON file to import from")
    p_import.add_argument("--agent-id", dest="agent_id", default=None)
    p_import.add_argument("--persist-dir", dest="persist_dir", default=None)
    p_import.add_argument("--merge", action="store_true", default=False,
                          help="Merge with existing memories instead of replacing them")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "inspect": cmd_inspect,
        "export": cmd_export,
        "import": cmd_import,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
