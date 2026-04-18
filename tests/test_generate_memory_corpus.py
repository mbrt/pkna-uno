"""Tests for the memory corpus generation script."""

from pathlib import Path

from datagen.generate_memory_corpus import (
    RawMemoryEntry,
    _raw_to_corpus,
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

    def test_seed_banks_have_days_ago(self):
        banks_dir = Path("data/memory_banks")
        entries = ingest_seed_banks(banks_dir)
        for entry in entries:
            assert entry.days_ago >= 0


class TestRawToCorpus:
    def test_assigns_days_ago_spread(self):
        raw = [RawMemoryEntry(key=f"k{i}", value=f"v{i}") for i in range(5)]
        corpus = _raw_to_corpus(raw, ["tag"], "roleplay", "paperino")
        assert len(corpus) == 5
        assert corpus[0].days_ago == 0
        assert corpus[-1].days_ago == 60
        for entry in corpus:
            assert entry.tags == ["tag"]
            assert entry.archetype == "roleplay"
            assert entry.character == "paperino"

    def test_single_entry_gets_zero(self):
        raw = [RawMemoryEntry(key="k", value="v")]
        corpus = _raw_to_corpus(raw, ["tag"], "casual", "anonymous")
        assert len(corpus) == 1
        assert corpus[0].days_ago == 0

    def test_empty_input(self):
        corpus = _raw_to_corpus([], ["tag"], "casual", "anonymous")
        assert corpus == []


class TestWriteCorpus:
    def test_write_and_load_roundtrip(self, tmp_path: Path):
        entries = [
            MemoryCorpusEntry(
                key="k1",
                value="v1",
                days_ago=3,
                tags=["a"],
                archetype="roleplay",
                character="paperino",
            ),
            MemoryCorpusEntry(
                key="k2",
                value="v2",
                days_ago=10,
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
