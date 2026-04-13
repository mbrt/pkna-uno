"""Unit tests for llm_backends helpers."""

from pydantic import BaseModel, TypeAdapter

from pkna.llm_backends import GenerateResult, _add_additional_properties_false


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
