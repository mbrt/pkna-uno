"""Unit tests for llm_backends helpers."""

from pydantic import BaseModel, TypeAdapter

from llm_backends import _add_additional_properties_false


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
