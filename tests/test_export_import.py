"""Tests for MemoryStore.export_json() and import_json()."""
from __future__ import annotations

import json

import pytest

from agentmemory import MemoryStore


@pytest.fixture()
def memory(tmp_path):
    store = MemoryStore(
        agent_id="export-test",
        persist_dir=str(tmp_path),
        enable_dedup=False,
        auto_compress=False,
        auto_extract_facts=False,
    )
    store.remember("The project uses microservices architecture", importance=8)
    store.remember("PostgreSQL is the main database", importance=7)
    store.remember("Users prefer concise answers", importance=5)
    yield store
    store.clear()


class TestExportJson:
    def test_returns_dict_with_correct_structure(self, memory):
        data = memory.export_json()
        assert "version" in data
        assert "agent_id" in data
        assert "exported_at" in data
        assert "episodic" in data
        assert data["version"] == "1.0"
        assert data["agent_id"] == "export-test"

    def test_episodic_records_have_required_fields(self, memory):
        data = memory.export_json()
        for r in data["episodic"]:
            assert "content" in r
            assert "metadata" in r
            assert "created_at" in r
            assert "importance" in r

    def test_export_count_matches_stored(self, memory):
        data = memory.export_json()
        assert len(data["episodic"]) == 3

    def test_export_writes_json_file(self, memory, tmp_path):
        out_path = str(tmp_path / "backup.json")
        memory.export_json(path=out_path)
        assert (tmp_path / "backup.json").exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["agent_id"] == "export-test"
        assert len(loaded["episodic"]) == 3

    def test_export_returns_dict_even_when_file_written(self, memory, tmp_path):
        out_path = str(tmp_path / "out.json")
        data = memory.export_json(path=out_path)
        assert isinstance(data, dict)
        assert "episodic" in data

    def test_export_no_path_does_not_create_extra_file(self, memory, tmp_path):
        memory.export_json()  # no path — should not write any file
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 0

    def test_export_empty_store(self, tmp_path):
        empty_store = MemoryStore(
            agent_id="empty-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        data = empty_store.export_json()
        assert data["episodic"] == []

    def test_exported_at_is_float(self, memory):
        data = memory.export_json()
        assert isinstance(data["exported_at"], float)
        assert data["exported_at"] > 0


class TestImportJson:
    def test_import_from_file(self, memory, tmp_path):
        out_path = str(tmp_path / "backup.json")
        memory.export_json(path=out_path)
        new_store = MemoryStore(
            agent_id="new-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        count = new_store.import_json(out_path)
        assert count == 3

    def test_import_from_dict(self, memory, tmp_path):
        data = memory.export_json()
        new_store = MemoryStore(
            agent_id="dict-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        count = new_store.import_json(data)
        assert count == 3

    def test_import_merge_false_clears_first(self, memory, tmp_path):
        data = memory.export_json()
        target = MemoryStore(
            agent_id="target-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        target.remember("Pre-existing memory", importance=5)
        assert target.episodic.count() == 1

        target.import_json(data, merge=False)
        assert target.episodic.count() == 3

    def test_import_merge_true_adds_alongside(self, memory, tmp_path):
        data = memory.export_json()
        target = MemoryStore(
            agent_id="merge-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        target.remember("Pre-existing memory", importance=5)
        assert target.episodic.count() == 1

        target.import_json(data, merge=True)
        assert target.episodic.count() == 4

    def test_import_returns_count(self, memory, tmp_path):
        data = memory.export_json()
        new_store = MemoryStore(
            agent_id="count-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        count = new_store.import_json(data)
        assert count == 3

    def test_roundtrip_preserves_content(self, memory, tmp_path):
        data = memory.export_json()
        new_store = MemoryStore(
            agent_id="roundtrip-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        new_store.import_json(data)
        rows = new_store.episodic.recall_recent(n=10)
        contents = {r["content"] for r in rows}
        assert "The project uses microservices architecture" in contents
        assert "PostgreSQL is the main database" in contents

    def test_roundtrip_preserves_importance(self, memory, tmp_path):
        data = memory.export_json()
        new_store = MemoryStore(
            agent_id="importance-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        new_store.import_json(data)
        rows = new_store.episodic.recall_recent(n=10)
        importances = {r["importance"] for r in rows}
        assert 8 in importances
        assert 7 in importances

    def test_import_default_merge_is_false(self, memory, tmp_path):
        data = memory.export_json()
        target = MemoryStore(
            agent_id="default-merge-agent",
            persist_dir=str(tmp_path),
            enable_dedup=False,
        )
        target.remember("Will be cleared", importance=5)
        target.import_json(data)  # merge=False by default
        assert target.episodic.count() == 3
