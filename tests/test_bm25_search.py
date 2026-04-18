"""Tests for BM25-based search in MemoryBank and WikiIndex."""

from pkna.inference.memory import MemoryBank, MemoryEntry
from pkna.inference.tools import WikiIndex


class TestMemoryBankBM25:
    def test_ranks_relevant_higher(self):
        bank = MemoryBank(
            [
                MemoryEntry(
                    key="weather report", value="sunny day in Duckburg", days_ago=1
                ),
                MemoryEntry(
                    key="PK mission debrief",
                    value="Evronian patrol detected near tower",
                    days_ago=2,
                ),
                MemoryEntry(
                    key="routine check", value="all systems nominal", days_ago=3
                ),
                MemoryEntry(
                    key="Evronian fleet analysis",
                    value="Evronian ships spotted in sector 7",
                    days_ago=4,
                ),
                MemoryEntry(
                    key="soap opera",
                    value="Episode 42 was entertaining",
                    days_ago=5,
                ),
            ]
        )
        results = bank.search("Evronian")
        assert len(results) >= 2
        keys = [r.key for r in results]
        assert "Evronian fleet analysis" in keys[:2]
        assert "PK mission debrief" in keys[:3]

    def test_multi_keyword_ranking(self):
        bank = MemoryBank(
            [
                MemoryEntry(key="mission", value="routine patrol", days_ago=1),
                MemoryEntry(
                    key="mission debrief",
                    value="Evronian mission complete",
                    days_ago=2,
                ),
                MemoryEntry(key="weather", value="rain all day", days_ago=3),
            ]
        )
        results = bank.search("mission Evronian")
        assert len(results) >= 1
        assert results[0].key == "mission debrief"

    def test_empty_bank_returns_empty(self):
        bank = MemoryBank()
        assert bank.search("anything") == []

    def test_index_rebuilt_after_append(self):
        bank = MemoryBank()
        assert bank.search("patrol") == []
        bank.append("patrol", "routine night patrol")
        results = bank.search("patrol")
        assert len(results) == 1
        assert results[0].key == "patrol"


class TestWikiIndexBM25:
    def _make_index(self) -> WikiIndex:
        index = WikiIndex()
        index._create_segment(
            content="Uno is an artificial intelligence created by Everett Ducklair.",
            file_path="characters.md",
            section_stack=[(2, "Uno")],
        )
        index._create_segment(
            content="The Evronians are an alien race that feeds on emotions.",
            file_path="villains.md",
            section_stack=[(2, "Evroniani")],
        )
        index._create_segment(
            content="The Ducklair Tower is located in Duckburg and serves as Uno's home.",
            file_path="locations.md",
            section_stack=[(2, "Ducklair Tower")],
        )
        index._create_segment(
            content="Paperinik is the superhero alter ego of Paperino.",
            file_path="characters.md",
            section_stack=[(2, "Paperinik")],
        )
        return index

    def test_finds_relevant_segment(self):
        index = self._make_index()
        results = index.search("Evronians alien race")
        assert len(results) >= 1
        assert results[0].file_path == "villains.md"

    def test_section_path_boosts_relevance(self):
        index = self._make_index()
        results = index.search("Uno")
        assert len(results) >= 1
        assert results[0].section_path == ["Uno"]

    def test_no_results_for_unknown_query(self):
        index = self._make_index()
        results = index.search("ZZZnonexistent123")
        assert results == []

    def test_respects_max_results(self):
        index = self._make_index()
        results = index.search("Ducklair", max_results=1)
        assert len(results) == 1

    def test_empty_index_returns_empty(self):
        index = WikiIndex()
        assert index.search("anything") == []
