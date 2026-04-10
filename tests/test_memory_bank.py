"""Unit tests for memory bank storage and retrieval."""

from pathlib import Path

from pkna.memory_bank import MemoryBank, MemoryEntry, make_recall, make_remember


class TestMemoryBank:
    def test_load_from_file(self, tmp_path: Path):
        path = tmp_path / "bank.jsonl"
        path.write_text(
            '{"key": "greeting", "value": "Said hello to PK", "timestamp": "2026-04-01T10:00:00Z"}\n'
            '{"key": "mission", "value": "Evronian patrol detected", "timestamp": "2026-04-02T12:00:00Z"}\n'
        )
        bank = MemoryBank.load(path)
        assert len(bank.entries) == 2
        assert bank.entries[0].key == "greeting"
        assert bank.entries[1].value == "Evronian patrol detected"

    def test_load_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "bank.jsonl"
        path.write_text(
            '{"key": "a", "value": "b", "timestamp": "2026-01-01T00:00:00Z"}\n'
            "\n"
            '{"key": "c", "value": "d", "timestamp": "2026-01-02T00:00:00Z"}\n'
        )
        bank = MemoryBank.load(path)
        assert len(bank.entries) == 2

    def test_search_by_key(self):
        bank = MemoryBank(
            [
                MemoryEntry(key="PK mission", value="routine patrol", timestamp="t1"),
                MemoryEntry(key="weather", value="sunny day", timestamp="t2"),
            ]
        )
        results = bank.search("mission")
        assert len(results) == 1
        assert results[0].key == "PK mission"

    def test_search_by_value(self):
        bank = MemoryBank(
            [
                MemoryEntry(key="note", value="Evronian fleet spotted", timestamp="t1"),
                MemoryEntry(key="note", value="sunny day in Duckburg", timestamp="t2"),
            ]
        )
        results = bank.search("Evronian")
        assert len(results) == 1
        assert "fleet" in results[0].value

    def test_search_respects_max_results(self):
        entries = [
            MemoryEntry(key=f"note {i}", value="Uno test", timestamp=f"t{i}")
            for i in range(10)
        ]
        bank = MemoryBank(entries)
        results = bank.search("Uno", max_results=3)
        assert len(results) == 3

    def test_search_no_matches(self):
        bank = MemoryBank([MemoryEntry(key="a", value="b", timestamp="t1")])
        results = bank.search("nonexistent")
        assert results == []

    def test_append(self):
        bank = MemoryBank()
        entry = bank.append("test key", "test value")
        assert entry.key == "test key"
        assert entry.value == "test value"
        assert entry.timestamp  # non-empty
        assert len(bank.entries) == 1

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        bank = MemoryBank()
        bank.append("k1", "v1")
        bank.append("k2", "v2")

        path = tmp_path / "out.jsonl"
        bank.save(path)

        loaded = MemoryBank.load(path)
        assert len(loaded.entries) == 2
        assert loaded.entries[0].key == "k1"
        assert loaded.entries[1].value == "v2"

    def test_entries_returns_copy(self):
        bank = MemoryBank([MemoryEntry(key="a", value="b", timestamp="t")])
        entries = bank.entries
        entries.clear()
        assert len(bank.entries) == 1


class TestRecallTool:
    def test_recall_returns_formatted_results(self):
        bank = MemoryBank(
            [
                MemoryEntry(
                    key="PK debrief",
                    value="He was tired",
                    timestamp="2026-04-01T10:00:00Z",
                ),
            ]
        )
        recall = make_recall(bank)
        result = recall("debrief")
        assert "PK debrief" in result
        assert "He was tired" in result
        assert "2026-04-01" in result

    def test_recall_no_matches(self):
        bank = MemoryBank()
        recall = make_recall(bank)
        result = recall("anything")
        assert "No matching memories" in result

    def test_recall_empty_query(self):
        bank = MemoryBank()
        recall = make_recall(bank)
        result = recall("")
        assert "Error" in result


class TestRememberTool:
    def test_remember_eval_mode_does_not_append(self):
        bank = MemoryBank()
        remember = make_remember(bank, eval_mode=True)
        result = remember("key", "value")
        assert result == "Memory stored."
        assert len(bank.entries) == 0

    def test_remember_trace_mode_appends(self):
        bank = MemoryBank()
        remember = make_remember(bank, eval_mode=False)
        result = remember("key", "value")
        assert result == "Memory stored."
        assert len(bank.entries) == 1
        assert bank.entries[0].key == "key"

    def test_remember_empty_key(self):
        bank = MemoryBank()
        remember = make_remember(bank)
        result = remember("", "value")
        assert "Error" in result

    def test_remember_empty_value(self):
        bank = MemoryBank()
        remember = make_remember(bank)
        result = remember("key", "")
        assert "Error" in result


class TestHandWrittenBanks:
    """Verify the hand-written memory bank files parse correctly."""

    BANKS_DIR = Path(__file__).parent.parent / "data" / "memory_banks"

    def test_paperino_recent_loads(self):
        bank = MemoryBank.load(self.BANKS_DIR / "paperino_recent.jsonl")
        assert len(bank.entries) >= 5
        for entry in bank.entries:
            assert entry.key
            assert entry.value
            assert entry.timestamp

    def test_xadhoom_research_loads(self):
        bank = MemoryBank.load(self.BANKS_DIR / "xadhoom_research.jsonl")
        assert len(bank.entries) >= 5

    def test_mixed_irrelevant_loads(self):
        bank = MemoryBank.load(self.BANKS_DIR / "mixed_irrelevant.jsonl")
        assert len(bank.entries) >= 5

    def test_paperino_bank_searchable(self):
        bank = MemoryBank.load(self.BANKS_DIR / "paperino_recent.jsonl")
        results = bank.search("mission")
        assert len(results) >= 1
