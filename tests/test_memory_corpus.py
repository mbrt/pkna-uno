"""Tests for compose_memory() and memory corpus utilities."""

import random
from pathlib import Path

from pkna.datagen.memory import compose_memory, load_memory_corpus
from pkna.datagen.types import MemoryCorpusEntry, MemoryProfile


def _make_corpus() -> list[MemoryCorpusEntry]:
    """Build a small test corpus with known entries."""
    return [
        MemoryCorpusEntry(
            key="PK mission",
            value="Discussed strategy with Paperino",
            days_ago=5,
            tags=["paperino", "mission"],
            archetype="roleplay",
            character="paperino",
        ),
        MemoryCorpusEntry(
            key="PK mood",
            value="Paperino seemed anxious today",
            days_ago=2,
            tags=["paperino", "emotional"],
            archetype="roleplay",
            character="paperino",
        ),
        MemoryCorpusEntry(
            key="Xadhoom research",
            value="Analyzed Evronian energy patterns",
            days_ago=14,
            tags=["xadhoom", "research"],
            archetype="roleplay",
            character="xadhoom",
        ),
        MemoryCorpusEntry(
            key="Tower routine",
            value="All systems nominal",
            days_ago=21,
            tags=["tower", "routine"],
            archetype="roleplay",
            character="mixed",
        ),
        MemoryCorpusEntry(
            key="Fan lore question",
            value="User asked about the Evronians origin story",
            days_ago=1,
            tags=["casual", "lore"],
            archetype="casual",
            character="anonymous",
        ),
        MemoryCorpusEntry(
            key="Identity probe",
            value="User asked if I dream",
            days_ago=0,
            tags=["casual", "identity"],
            archetype="casual",
            character="anonymous",
        ),
    ]


class TestComposeMemory:
    def test_returns_relevant_and_irrelevant(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="roleplay",
            character="paperino",
            relevant_tags=["paperino", "mission"],
            n_relevant=2,
            n_irrelevant=1,
        )
        rng = random.Random(42)
        context, bank = compose_memory(profile, corpus, rng)

        assert len(bank.entries) == 3
        assert "PK" in context or "Paperino" in context

    def test_empty_corpus(self):
        profile = MemoryProfile(
            archetype="roleplay",
            character="paperino",
            relevant_tags=["paperino"],
            n_relevant=5,
            n_irrelevant=3,
        )
        rng = random.Random(42)
        context, bank = compose_memory(profile, [], rng)

        assert context == ""
        assert len(bank.entries) == 0

    def test_empty_profile_returns_empty(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="casual",
            character="anonymous",
            relevant_tags=[],
            n_relevant=0,
            n_irrelevant=0,
        )
        rng = random.Random(42)
        context, bank = compose_memory(profile, corpus, rng)

        assert context == ""
        assert len(bank.entries) == 0

    def test_bank_is_searchable(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="roleplay",
            character="paperino",
            relevant_tags=["paperino"],
            n_relevant=2,
            n_irrelevant=2,
        )
        rng = random.Random(42)
        _, bank = compose_memory(profile, corpus, rng)

        results = bank.search("mission")
        assert any(
            "mission" in r.key.lower() or "mission" in r.value.lower() for r in results
        )

    def test_context_has_recent_entries(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="roleplay",
            character="paperino",
            relevant_tags=["paperino"],
            n_relevant=2,
            n_irrelevant=0,
        )
        rng = random.Random(42)
        context, _ = compose_memory(profile, corpus, rng)

        assert "Recent interactions:" in context
        assert "ago" in context or "today" in context

    def test_deterministic_with_seed(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="roleplay",
            character="paperino",
            relevant_tags=["paperino"],
            n_relevant=2,
            n_irrelevant=2,
        )
        ctx1, bank1 = compose_memory(profile, corpus, random.Random(42))
        ctx2, bank2 = compose_memory(profile, corpus, random.Random(42))

        assert ctx1 == ctx2
        assert [e.key for e in bank1.entries] == [e.key for e in bank2.entries]

    def test_casual_archetype(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="casual",
            character="anonymous",
            relevant_tags=["casual", "lore"],
            n_relevant=2,
            n_irrelevant=1,
        )
        rng = random.Random(42)
        context, bank = compose_memory(profile, corpus, rng)

        assert len(bank.entries) >= 1
        relevant_keys = {e.key for e in bank.entries}
        assert "Fan lore question" in relevant_keys or "Identity probe" in relevant_keys

    def test_caps_at_available_entries(self):
        corpus = _make_corpus()
        profile = MemoryProfile(
            archetype="roleplay",
            character="paperino",
            relevant_tags=["paperino"],
            n_relevant=100,
            n_irrelevant=100,
        )
        rng = random.Random(42)
        _, bank = compose_memory(profile, corpus, rng)
        assert len(bank.entries) <= len(corpus)


class TestLoadMemoryCorpus:
    def test_roundtrip(self, tmp_path: Path):
        entries = _make_corpus()
        path = tmp_path / "corpus.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(e.model_dump_json() + "\n")

        loaded = load_memory_corpus(path)
        assert len(loaded) == len(entries)
        assert loaded[0].key == entries[0].key
        assert loaded[0].tags == entries[0].tags

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        entry = _make_corpus()[0]
        path.write_text(
            entry.model_dump_json() + "\n\n" + entry.model_dump_json() + "\n"
        )
        loaded = load_memory_corpus(path)
        assert len(loaded) == 2
