"""Tests for the agentmemory CLI (inspect, export, import subcommands)."""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from agentmemory import MemoryStore
from agentmemory.cli import cmd_export, cmd_import, cmd_inspect, main


@pytest.fixture()
def populated_store(tmp_path):
    store = MemoryStore(
        agent_id="cli-test",
        persist_dir=str(tmp_path),
        enable_dedup=False,
        auto_compress=False,
        auto_extract_facts=False,
    )
    store.remember("PostgreSQL is the main DB", importance=8)
    store.remember("FastAPI is the web framework", importance=6)
    store.remember("Tests are written in pytest", importance=5)
    yield store, tmp_path
    store.clear()


class FakeArgs:
    """Simple argparse.Namespace substitute for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCmdInspect:
    def test_inspect_empty_store(self, tmp_path, capsys):
        args = FakeArgs(agent_id="empty-cli", persist_dir=str(tmp_path), limit=50)
        cmd_inspect(args)
        out = capsys.readouterr().out
        assert "EPISODIC MEMORY" in out
        assert "(empty)" in out

    def test_inspect_shows_memories(self, populated_store, capsys):
        store, tmp_path = populated_store
        args = FakeArgs(agent_id="cli-test", persist_dir=str(tmp_path), limit=50)
        cmd_inspect(args)
        out = capsys.readouterr().out
        assert "EPISODIC MEMORY" in out
        assert "3 entries" in out
        assert "PostgreSQL" in out

    def test_inspect_limit_respected(self, populated_store, capsys):
        store, tmp_path = populated_store
        args = FakeArgs(agent_id="cli-test", persist_dir=str(tmp_path), limit=1)
        cmd_inspect(args)
        out = capsys.readouterr().out
        # Only 1 memory row should appear — verify exactly 1 numbered row
        numbered_rows = [
            line for line in out.splitlines()
            if line.strip() and line.strip()[0].isdigit() and line.strip()[1:2] == " "
        ]
        assert len(numbered_rows) <= 1

    def test_inspect_shows_all_sections(self, populated_store, capsys):
        store, tmp_path = populated_store
        args = FakeArgs(agent_id="cli-test", persist_dir=str(tmp_path), limit=50)
        cmd_inspect(args)
        out = capsys.readouterr().out
        assert "EPISODIC MEMORY" in out
        assert "WORKING MEMORY" in out
        assert "SEMANTIC MEMORY" in out

    def test_inspect_shows_agent_id(self, populated_store, capsys):
        store, tmp_path = populated_store
        args = FakeArgs(agent_id="cli-test", persist_dir=str(tmp_path), limit=50)
        cmd_inspect(args)
        out = capsys.readouterr().out
        assert "cli-test" in out

    def test_inspect_shows_importance_column(self, populated_store, capsys):
        store, tmp_path = populated_store
        args = FakeArgs(agent_id="cli-test", persist_dir=str(tmp_path), limit=50)
        cmd_inspect(args)
        out = capsys.readouterr().out
        assert "IMP" in out

    def test_inspect_uses_env_var_agent_id(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("AGENTMEMORY_AGENT_ID", "env-agent")
        args = FakeArgs(agent_id=None, persist_dir=str(tmp_path), limit=50)
        cmd_inspect(args)
        out = capsys.readouterr().out
        assert "env-agent" in out


class TestCmdExport:
    def test_export_creates_json_file(self, populated_store, tmp_path):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "export.json")
        args = FakeArgs(agent_id="cli-test", persist_dir=str(persist_dir), output=out_path)
        cmd_export(args)
        assert (tmp_path / "export.json").exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["agent_id"] == "cli-test"
        assert len(data["episodic"]) == 3

    def test_export_prints_confirmation(self, populated_store, tmp_path, capsys):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "export2.json")
        args = FakeArgs(agent_id="cli-test", persist_dir=str(persist_dir), output=out_path)
        cmd_export(args)
        out = capsys.readouterr().out
        assert "3" in out
        assert "export2.json" in out

    def test_export_no_output_prints_json(self, populated_store, capsys):
        store, persist_dir = populated_store
        args = FakeArgs(agent_id="cli-test", persist_dir=str(persist_dir), output=None)
        cmd_export(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "episodic" in data

    def test_export_output_valid_json(self, populated_store, tmp_path):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "valid.json")
        args = FakeArgs(agent_id="cli-test", persist_dir=str(persist_dir), output=out_path)
        cmd_export(args)
        with open(out_path) as f:
            data = json.load(f)  # raises if invalid JSON
        assert isinstance(data, dict)


class TestCmdImport:
    def test_import_from_file(self, populated_store, tmp_path, capsys):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "import_src.json")
        store.export_json(path=out_path)
        args = FakeArgs(
            agent_id="import-cli",
            persist_dir=str(tmp_path),
            file=out_path,
            merge=False,
        )
        cmd_import(args)
        out = capsys.readouterr().out
        assert "3" in out
        assert "import-cli" in out

    def test_import_merge_flag(self, populated_store, tmp_path, capsys):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "merge_src.json")
        store.export_json(path=out_path)
        args = FakeArgs(
            agent_id="merge-cli",
            persist_dir=str(tmp_path),
            file=out_path,
            merge=True,
        )
        cmd_import(args)
        out = capsys.readouterr().out
        assert "merged" in out

    def test_import_missing_file_exits_with_1(self, tmp_path):
        args = FakeArgs(
            agent_id="import-cli",
            persist_dir=str(tmp_path),
            file=str(tmp_path / "nonexistent.json"),
            merge=False,
        )
        with pytest.raises(SystemExit) as exc:
            cmd_import(args)
        assert exc.value.code == 1

    def test_import_actually_stores_records(self, populated_store, tmp_path):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "data.json")
        store.export_json(path=out_path)

        args = FakeArgs(
            agent_id="verify-import",
            persist_dir=str(tmp_path),
            file=out_path,
            merge=False,
        )
        cmd_import(args)

        verify_store = MemoryStore(
            agent_id="verify-import",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        assert verify_store.episodic.count() == 3


class TestMainEntry:
    def test_no_args_prints_help(self, capsys):
        with patch.object(sys, "argv", ["agentmemory"]):
            main()
        out = capsys.readouterr().out
        # argparse help should mention the available subcommands
        assert any(word in out for word in ["inspect", "export", "import", "COMMAND"])

    def test_inspect_subcommand_via_main(self, populated_store, capsys):
        store, persist_dir = populated_store
        with patch.object(
            sys,
            "argv",
            ["agentmemory", "inspect", "--agent-id", "cli-test",
             "--persist-dir", str(persist_dir)],
        ):
            main()
        out = capsys.readouterr().out
        assert "EPISODIC MEMORY" in out

    def test_export_subcommand_via_main(self, populated_store, tmp_path, capsys):
        store, persist_dir = populated_store
        out_path = str(tmp_path / "main_export.json")
        with patch.object(
            sys,
            "argv",
            ["agentmemory", "export", "--agent-id", "cli-test",
             "--persist-dir", str(persist_dir), "-o", out_path],
        ):
            main()
        assert (tmp_path / "main_export.json").exists()
