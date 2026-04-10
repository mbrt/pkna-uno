"""Unit tests for the eval tool factory."""

import pytest

from pkna.eval_tools import TOOL_NAMES, delegate, make_eval_tools
from pkna.memory_bank import MemoryBank, MemoryEntry


class TestDelegate:
    def test_returns_acknowledgement(self):
        result = delegate("solve x^2 = 4")
        assert "delegated" in result.lower()

    def test_includes_task_description(self):
        result = delegate("write a Python script", context="parse logs")
        assert "write a Python script" in result


class TestMakeEvalTools:
    def test_empty_list(self):
        tools = make_eval_tools([])
        assert tools == []

    def test_single_tool(self):
        tools = make_eval_tools(["delegate"])
        assert len(tools) == 1
        assert getattr(tools[0], "__name__", None) == "delegate"

    def test_knowledge_tools(self):
        tools = make_eval_tools(["search_knowledge", "read_knowledge"])
        assert len(tools) == 2
        names = {getattr(t, "__name__", "") for t in tools}
        assert names == {"search_knowledge", "read_knowledge"}

    def test_all_tools(self):
        bank = MemoryBank([MemoryEntry(key="k", value="v", timestamp="t")])
        tools = make_eval_tools(list(TOOL_NAMES), memory_bank=bank)
        assert len(tools) == len(TOOL_NAMES)

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool 'nonexistent'"):
            make_eval_tools(["nonexistent"])

    def test_recall_tool_searches_bank(self):
        bank = MemoryBank(
            [MemoryEntry(key="mission", value="PK was tired", timestamp="t1")]
        )
        tools = make_eval_tools(["recall"], memory_bank=bank)
        recall = tools[0]
        result = recall(query="mission")
        assert "PK was tired" in result

    def test_remember_eval_mode_is_noop(self):
        bank = MemoryBank()
        tools = make_eval_tools(["remember"], memory_bank=bank, eval_mode=True)
        remember = tools[0]
        result = remember(key="test", value="data")
        assert result == "Memory stored."
        assert len(bank.entries) == 0

    def test_remember_trace_mode_appends(self):
        bank = MemoryBank()
        tools = make_eval_tools(["remember"], memory_bank=bank, eval_mode=False)
        remember = tools[0]
        remember(key="test", value="data")
        assert len(bank.entries) == 1

    def test_preserves_order(self):
        names = ["delegate", "search_knowledge", "recall"]
        tools = make_eval_tools(names, memory_bank=MemoryBank())
        assert [getattr(t, "__name__", "") for t in tools] == [
            "delegate",
            "search_knowledge",
            "recall",
        ]

    def test_default_bank_created_if_none(self):
        tools = make_eval_tools(["recall", "remember"])
        assert len(tools) == 2
