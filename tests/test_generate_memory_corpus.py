"""Tests for the memory corpus generation script."""

from pathlib import Path

from datagen.generate_memory_corpus import (
    _parse_llm_entries,
    ingest_seed_banks,
    write_corpus,
)
from pkna.datagen.memory import load_memory_corpus
from pkna.datagen.types import MemoryCorpusEntry


class TestIngestSeedBanks:
    def test_ingests_all_existing_banks(self):
        banks_dir = Path("data/memory_banks")
        entries = ingest_seed_banks(banks_dir)
        assert len(entries) >= 20
        archetypes = {e.archetype for e in entries}
        assert "roleplay" in archetypes

    def test_tags_match_config(self):
        banks_dir = Path("data/memory_banks")
        entries = ingest_seed_banks(banks_dir)
        for entry in entries:
            assert len(entry.tags) > 0
            assert entry.character

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        entries = ingest_seed_banks(tmp_path / "nonexistent")
        assert entries == []

    def test_each_seed_bank_has_expected_character(self):
        banks_dir = Path("data/memory_banks")
        entries = ingest_seed_banks(banks_dir)
        characters = {e.character for e in entries}
        assert "paperino" in characters
        assert "xadhoom" in characters
        assert "mixed" in characters


class TestParseLlmEntries:
    def test_parses_valid_jsonl(self):
        text = (
            '{"key": "test", "value": "hello", "timestamp": "2026-01-01T00:00:00Z"}\n'
            '{"key": "test2", "value": "world", "timestamp": "2026-01-02T00:00:00Z"}\n'
        )
        entries = _parse_llm_entries(text, ["tag1"], "roleplay", "paperino")
        assert len(entries) == 2
        assert entries[0].tags == ["tag1"]
        assert entries[0].archetype == "roleplay"
        assert entries[0].character == "paperino"

    def test_skips_malformed_lines(self):
        text = '{"key": "valid", "value": "ok", "timestamp": "t"}\nnot json\n'
        entries = _parse_llm_entries(text, ["tag"], "casual", "anonymous")
        assert len(entries) == 1

    def test_handles_missing_timestamp(self):
        text = '{"key": "test", "value": "hello"}\n'
        entries = _parse_llm_entries(text, ["tag"], "casual", "anonymous")
        assert len(entries) == 1
        assert entries[0].timestamp == "2026-03-01T00:00:00Z"


class TestWriteCorpus:
    def test_write_and_load_roundtrip(self, tmp_path: Path):
        entries = [
            MemoryCorpusEntry(
                key="k1",
                value="v1",
                timestamp="t1",
                tags=["a"],
                archetype="roleplay",
                character="paperino",
            ),
            MemoryCorpusEntry(
                key="k2",
                value="v2",
                timestamp="t2",
                tags=["b"],
                archetype="casual",
                character="anonymous",
            ),
        ]
        path = tmp_path / "out.jsonl"
        write_corpus(path, entries)
        loaded = load_memory_corpus(path)
        assert len(loaded) == 2
        assert loaded[0].key == "k1"
        assert loaded[1].archetype == "casual"
