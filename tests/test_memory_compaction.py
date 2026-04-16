"""Unit tests for memory compaction (with mocked LLM backend)."""

from unittest.mock import MagicMock

from pkna.inference.memory import MemoryBank, MemoryEntry
from pkna.inference.memory_compaction import compact_memories
from pkna.llm.backends import GenerateResult


def _make_backend(response_text: str) -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = GenerateResult(
        text=response_text, model_name="mock", usage={}
    )
    return backend


class TestCompactMemories:
    def test_returns_summary(self):
        bank = MemoryBank(
            [
                MemoryEntry(
                    key="mission",
                    value="PK was tired after patrol",
                    timestamp="2026-04-01T10:00:00Z",
                ),
            ]
        )
        backend = _make_backend("Paperinik was tired after a recent patrol.")
        result = compact_memories(bank, "PK's wellbeing", backend)
        assert "tired" in result
        assert "patrol" in result

    def test_empty_bank_returns_empty(self):
        bank = MemoryBank()
        backend = _make_backend("should not be called")
        result = compact_memories(bank, "anything", backend)
        assert result == ""
        backend.generate.assert_not_called()

    def test_passes_topic_to_prompt(self):
        bank = MemoryBank([MemoryEntry(key="k", value="v", timestamp="t")])
        backend = _make_backend("summary")
        compact_memories(bank, "Evronian threat", backend)

        call_args = backend.generate.call_args
        user_message = call_args.kwargs.get("messages") or call_args[1].get(
            "messages", call_args[0][1] if len(call_args[0]) > 1 else None
        )
        prompt_text = user_message[0]["content"]
        assert "Evronian threat" in prompt_text

    def test_includes_all_entries(self):
        entries = [
            MemoryEntry(key=f"entry{i}", value=f"value{i}", timestamp=f"t{i}")
            for i in range(5)
        ]
        bank = MemoryBank(entries)
        backend = _make_backend("summary of 5 entries")
        compact_memories(bank, "general", backend)

        call_args = backend.generate.call_args
        user_message = call_args.kwargs.get("messages") or call_args[1].get(
            "messages", call_args[0][1] if len(call_args[0]) > 1 else None
        )
        prompt_text = user_message[0]["content"]
        for i in range(5):
            assert f"entry{i}" in prompt_text

    def test_handles_none_response(self):
        bank = MemoryBank([MemoryEntry(key="k", value="v", timestamp="t")])
        backend = MagicMock()
        backend.generate.return_value = None
        result = compact_memories(bank, "topic", backend)
        assert result == ""
