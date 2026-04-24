"""Unit tests for the local inference backend.

Tests the pure parsing functions (tool-call extraction, thinking
extraction, tool schema conversion) without requiring a GPU.
"""

from pkna.llm.local_backend import (
    _callable_to_tool_dict,
    extract_thinking_and_content,
    parse_tool_calls,
    strip_tool_call_text,
)


class TestParseToolCalls:
    def test_single_call_single_param(self):
        text = (
            "<tool_call>\n"
            "<function=search_knowledge>\n"
            "<parameter=keywords>\nEvroniani\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert calls == [
            {"name": "search_knowledge", "arguments": {"keywords": "Evroniani"}},
        ]

    def test_single_call_multiple_params(self):
        text = (
            "<tool_call>\n"
            "<function=search_knowledge>\n"
            "<parameter=keywords>Xadhoom origins</parameter>\n"
            "<parameter=max_results>3</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_knowledge"
        assert calls[0]["arguments"]["keywords"] == "Xadhoom origins"
        assert calls[0]["arguments"]["max_results"] == 3

    def test_multiple_calls(self):
        text = (
            "Let me search for that.\n\n"
            "<tool_call>\n"
            "<function=search_knowledge>\n"
            "<parameter=keywords>Ducklair Tower</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<tool_call>\n"
            "<function=read_knowledge>\n"
            "<parameter=segment_id>locations.md::Ducklair Tower</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "search_knowledge"
        assert calls[1]["name"] == "read_knowledge"
        assert calls[1]["arguments"]["segment_id"] == "locations.md::Ducklair Tower"

    def test_no_tool_calls(self):
        assert parse_tool_calls("Just a normal response.") == []

    def test_empty_string(self):
        assert parse_tool_calls("") == []

    def test_json_param_value(self):
        text = (
            "<tool_call>\n<function=delegate>\n"
            '<parameter=task>"solve equation x^2 + 1 = 0"</parameter>\n'
            "</function>\n</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert calls[0]["arguments"]["task"] == "solve equation x^2 + 1 = 0"

    def test_multiline_param_value(self):
        text = (
            "<tool_call>\n<function=delegate>\n"
            "<parameter=task>Write a Python script\n"
            "that reads CSV files\n"
            "and outputs JSON</parameter>\n"
            "</function>\n</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert "Python script" in calls[0]["arguments"]["task"]
        assert "JSON" in calls[0]["arguments"]["task"]


class TestExtractThinkingAndContent:
    def test_thinking_and_content(self):
        text = "<think>\nThis user seems upset. Be gentle.\n</think>\n\nHey socio, what's wrong?"
        thinking, content = extract_thinking_and_content(text)
        assert thinking == "This user seems upset. Be gentle."
        assert content == "Hey socio, what's wrong?"

    def test_no_thinking(self):
        text = "Just a normal response without thinking."
        thinking, content = extract_thinking_and_content(text)
        assert thinking is None
        assert content == "Just a normal response without thinking."

    def test_empty_thinking(self):
        text = "<think>\n\n</think>\n\nSome content."
        thinking, content = extract_thinking_and_content(text)
        assert thinking is None
        assert content == "Some content."

    def test_thinking_only(self):
        text = "<think>\nReasoning here.\n</think>"
        thinking, content = extract_thinking_and_content(text)
        assert thinking == "Reasoning here."
        assert content == ""

    def test_whitespace_handling(self):
        text = "  <think>  Padded thinking  </think>  Padded content  "
        thinking, content = extract_thinking_and_content(text)
        assert thinking == "Padded thinking"
        assert content == "Padded content"


class TestStripToolCallText:
    def test_strips_tool_calls(self):
        text = (
            "Let me look that up.\n\n"
            "<tool_call>\n<function=search_knowledge>\n"
            "<parameter=keywords>test</parameter>\n"
            "</function>\n</tool_call>"
        )
        assert strip_tool_call_text(text) == "Let me look that up."

    def test_no_tool_calls(self):
        assert strip_tool_call_text("Normal text.") == "Normal text."

    def test_only_tool_calls(self):
        text = (
            "<tool_call>\n<function=search_knowledge>\n"
            "<parameter=keywords>test</parameter>\n"
            "</function>\n</tool_call>"
        )
        assert strip_tool_call_text(text) == ""


class TestCallableToToolDict:
    def test_simple_function(self):
        def search_knowledge(keywords: str, max_results: int = 5) -> str:
            """Search the knowledge base for keywords."""
            return ""

        result = _callable_to_tool_dict(search_knowledge)
        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "search_knowledge"
        assert "Search the knowledge base" in fn["description"]
        assert fn["parameters"]["properties"]["keywords"] == {"type": "string"}
        assert fn["parameters"]["properties"]["max_results"] == {"type": "integer"}
        assert fn["parameters"]["required"] == ["keywords"]

    def test_no_required_params(self):
        def noop(x: str = "", y: int = 0) -> str:
            """Does nothing."""
            return ""

        result = _callable_to_tool_dict(noop)
        assert "required" not in result["function"]["parameters"]

    def test_multiline_docstring(self):
        def delegate(task: str) -> str:
            """Delegate a technical task to a specialist sub-agent.

            Use this for tasks outside your core competency.

            Args:
                task: Description of the task to delegate
            """
            return ""

        result = _callable_to_tool_dict(delegate)
        assert result["function"]["description"] == (
            "Delegate a technical task to a specialist sub-agent."
        )
