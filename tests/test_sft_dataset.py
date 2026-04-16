"""Unit tests for SFT dataset conversion logic."""

from pkna.datagen.types import DatagenTrace
from pkna.training.sft_dataset import (
    _convert_message,
    _convert_tool_calls,
    trace_to_messages,
)


def _make_trace(
    id: str = "t-001",
    system_prompt: str = "You are Uno.",
    messages: list[dict] | None = None,
) -> DatagenTrace:
    return DatagenTrace(
        id=id,
        metadata={},
        system_prompt=system_prompt,
        memory_context="",
        user_summary="Paperino",
        messages=messages or [],
    )


class TestConvertToolCalls:
    def test_single_call(self):
        raw = [{"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}]
        result = _convert_tool_calls(raw)
        assert result == [
            {"function": {"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}}
        ]

    def test_multiple_calls(self):
        raw = [
            {"name": "search_wiki", "arguments": {"keywords": "Ducklair"}},
            {"name": "delegate", "arguments": {"task": "solve equation"}},
        ]
        result = _convert_tool_calls(raw)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "search_wiki"
        assert result[1]["function"]["name"] == "delegate"

    def test_empty_arguments(self):
        raw = [{"name": "recall", "arguments": {}}]
        result = _convert_tool_calls(raw)
        assert result == [{"function": {"name": "recall", "arguments": {}}}]

    def test_missing_fields_default(self):
        raw = [{}]
        result = _convert_tool_calls(raw)
        assert result == [{"function": {"name": "", "arguments": {}}}]


class TestConvertMessage:
    def test_user_message(self):
        msg = {"role": "user", "content": "Ciao, Uno!"}
        assert _convert_message(msg) == {"role": "user", "content": "Ciao, Uno!"}

    def test_assistant_with_thinking(self):
        msg = {
            "role": "assistant",
            "content": "Ciao, socio!",
            "thinking": "Paperino is greeting me. Light tone.",
        }
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Ciao, socio!"
        assert result["reasoning_content"] == "Paperino is greeting me. Light tone."
        assert "tool_calls" not in result

    def test_assistant_without_thinking(self):
        msg = {"role": "assistant", "content": "Sure thing."}
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Sure thing."
        assert "reasoning_content" not in result

    def test_assistant_empty_thinking(self):
        msg = {"role": "assistant", "content": "Ok.", "thinking": ""}
        result = _convert_message(msg)
        assert "reasoning_content" not in result

    def test_assistant_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "search_wiki", "arguments": {"keywords": "Evroniani"}}
            ],
        }
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search_wiki"

    def test_assistant_with_thinking_and_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Let me look that up.",
            "thinking": "Need to search the wiki.",
            "tool_calls": [
                {"name": "search_wiki", "arguments": {"keywords": "Ducklair"}}
            ],
        }
        result = _convert_message(msg)
        assert result["reasoning_content"] == "Need to search the wiki."
        assert result["content"] == "Let me look that up."
        assert len(result["tool_calls"]) == 1

    def test_assistant_empty_tool_calls(self):
        msg = {"role": "assistant", "content": "Hello", "tool_calls": []}
        result = _convert_message(msg)
        assert "tool_calls" not in result

    def test_tool_message(self):
        msg = {"role": "tool", "content": "Xadhoom is a Xerbian scientist."}
        assert _convert_message(msg) == {
            "role": "tool",
            "content": "Xadhoom is a Xerbian scientist.",
        }

    def test_unknown_role_passthrough(self):
        msg = {"role": "developer", "content": "debug info"}
        assert _convert_message(msg) == {"role": "developer", "content": "debug info"}


class TestTraceToMessages:
    def test_single_turn(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Ciao, socio!",
                    "thinking": "A casual greeting.",
                },
            ]
        )
        messages = trace_to_messages(trace)
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are Uno."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2]["role"] == "assistant"
        assert messages[2]["reasoning_content"] == "A casual greeting."
        assert messages[2]["content"] == "Ciao, socio!"

    def test_multi_turn_with_tools(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Who is Xadhoom?"},
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "Factual question. Search the wiki.",
                    "tool_calls": [
                        {
                            "name": "search_wiki",
                            "arguments": {"keywords": "Xadhoom"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_wiki",
                    "content": "Xadhoom is a Xerbian scientist.",
                },
                {
                    "role": "assistant",
                    "content": "Xadhoom is a powerful Xerbian.",
                    "thinking": "Got wiki result, synthesize.",
                },
                {"role": "user", "content": "Thanks!"},
                {
                    "role": "assistant",
                    "content": "Prego, socio!",
                    "thinking": "Casual farewell.",
                },
            ]
        )
        messages = trace_to_messages(trace)
        assert len(messages) == 7  # system + 6 from trace

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        assert messages[2]["role"] == "assistant"
        assert messages[2]["reasoning_content"] == "Factual question. Search the wiki."
        assert messages[2]["tool_calls"][0]["function"]["name"] == "search_wiki"

        assert messages[3]["role"] == "tool"
        assert "Xerbian scientist" in messages[3]["content"]

        assert messages[4]["role"] == "assistant"
        assert messages[4]["reasoning_content"] == "Got wiki result, synthesize."

        assert messages[5]["role"] == "user"
        assert messages[6]["role"] == "assistant"
        assert messages[6]["content"] == "Prego, socio!"

    def test_system_prompt_preserved(self):
        trace = _make_trace(
            system_prompt="Custom system prompt with personality details.",
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )
        messages = trace_to_messages(trace)
        assert (
            messages[0]["content"] == "Custom system prompt with personality details."
        )

    def test_empty_messages(self):
        trace = _make_trace(messages=[])
        messages = trace_to_messages(trace)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_tool_name_not_in_output(self):
        """The 'name' field on tool messages is datagen metadata, not part of
        the Qwen3.5 chat format (tool identity comes from the preceding
        tool_call)."""
        trace = _make_trace(
            messages=[
                {
                    "role": "tool",
                    "name": "search_wiki",
                    "content": "result text",
                },
            ]
        )
        messages = trace_to_messages(trace)
        tool_msg = messages[1]
        assert tool_msg == {"role": "tool", "content": "result text"}
        assert "name" not in tool_msg
