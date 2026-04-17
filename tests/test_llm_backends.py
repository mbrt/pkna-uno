"""Unit tests for llm_backends helpers."""

from pydantic import BaseModel, TypeAdapter

from pkna.llm.backends import (
    GenerateResult,
    GeminiBackend,
    _add_additional_properties_false,
)


class TestGenerateResult:
    def test_defaults(self):
        r = GenerateResult(text="hello", model_name="test-model")
        assert r.text == "hello"
        assert r.model_name == "test-model"
        assert r.usage == {}
        assert r.thinking is None
        assert r.tool_calls == []
        assert r.messages == []

    def test_with_thinking_and_tool_calls(self):
        tc = [{"name": "search_knowledge", "arguments": {"query": "x"}, "result": "y"}]
        msgs = [
            {"role": "assistant", "thinking": "hmm", "content": "ok", "tool_calls": []},
            {"role": "tool", "name": "search_knowledge", "content": "y"},
            {"role": "assistant", "content": "done"},
        ]
        r = GenerateResult(
            text="done",
            model_name="m",
            thinking="hmm",
            tool_calls=tc,
            messages=msgs,
        )
        assert r.thinking == "hmm"
        assert r.tool_calls == tc
        assert len(r.messages) == 3


class TestAddAdditionalPropertiesFalse:
    def test_flat_model(self):
        class Flat(BaseModel):
            name: str
            age: int

        schema = TypeAdapter(list[Flat]).json_schema()
        _add_additional_properties_false(schema)

        obj_def = schema["$defs"]["Flat"]
        assert obj_def["additionalProperties"] is False

    def test_nested_model(self):
        class Inner(BaseModel):
            x: str

        class Outer(BaseModel):
            inner: Inner
            label: str

        schema = TypeAdapter(list[Outer]).json_schema()
        _add_additional_properties_false(schema)

        assert schema["$defs"]["Outer"]["additionalProperties"] is False
        assert schema["$defs"]["Inner"]["additionalProperties"] is False

    def test_no_mutation_on_non_objects(self):
        schema = {"type": "array", "items": {"type": "string"}}
        _add_additional_properties_false(schema)
        assert "additionalProperties" not in schema

    def test_already_present(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        _add_additional_properties_false(schema)
        assert schema["additionalProperties"] is False


class TestToGeminiContents:
    def test_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        contents = GeminiBackend._to_gemini_contents(messages)
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts is not None
        assert contents[0].parts[0].text == "Hello"
        assert contents[1].role == "model"
        assert contents[1].parts is not None
        assert contents[1].parts[0].text == "Hi there"

    def test_assistant_with_tool_calls(self):
        messages = [
            {"role": "user", "content": "Search for X"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "search", "arguments": {"q": "X"}}],
            },
        ]
        contents = GeminiBackend._to_gemini_contents(messages)
        assert len(contents) == 2
        assert contents[1].role == "model"
        assert contents[1].parts is not None
        fc = contents[1].parts[0].function_call
        assert fc is not None
        assert fc.name == "search"
        assert fc.args == {"q": "X"}

    def test_tool_messages_grouped(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "fn_a", "arguments": {}},
                    {"name": "fn_b", "arguments": {}},
                ],
            },
            {"role": "tool", "name": "fn_a", "content": "result_a"},
            {"role": "tool", "name": "fn_b", "content": "result_b"},
            {"role": "assistant", "content": "Done"},
        ]
        contents = GeminiBackend._to_gemini_contents(messages)
        assert len(contents) == 4
        # user, model (tool calls), user (tool results grouped), model (final)
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"
        assert contents[2].parts is not None
        assert len(contents[2].parts) == 2
        fr_a = contents[2].parts[0].function_response
        fr_b = contents[2].parts[1].function_response
        assert fr_a is not None and fr_a.name == "fn_a"
        assert fr_b is not None and fr_b.name == "fn_b"
        assert contents[3].role == "model"
        assert contents[3].parts is not None
        assert contents[3].parts[0].text == "Done"

    def test_empty_messages(self):
        assert GeminiBackend._to_gemini_contents([]) == []

    def test_full_tool_roundtrip(self):
        """Simulates the message sequence from _generate_with_tools result."""
        messages = [
            {"role": "user", "content": "Look up info"},
            {
                "role": "assistant",
                "content": "Let me search",
                "thinking": "need to use tool",
                "tool_calls": [
                    {"name": "search_knowledge", "arguments": {"query": "info"}}
                ],
            },
            {"role": "tool", "name": "search_knowledge", "content": "found it"},
            {"role": "assistant", "content": "Here is what I found"},
            {"role": "user", "content": "Thanks"},
        ]
        contents = GeminiBackend._to_gemini_contents(messages)
        assert len(contents) == 5
        roles = [c.role for c in contents]
        assert roles == ["user", "model", "user", "model", "user"]
        assert contents[2].parts is not None
        fr = contents[2].parts[0].function_response
        assert fr is not None and fr.name == "search_knowledge"
