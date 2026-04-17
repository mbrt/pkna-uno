"""Unit tests for system prompt templates."""

from pkna.eval.types import SUITES
from pkna.inference.system_prompts import (
    SUITE_TEMPLATE_MAP,
    prepend_context_to_messages,
    render_context_preamble,
    render_datagen_system_prompt,
    render_system_prompt,
)


class TestRenderSystemPrompt:
    def test_minimal_contains_identity(self):
        result = render_system_prompt("minimal")
        assert "Uno" in result
        assert "Ducklair" in result

    def test_minimal_is_short(self):
        result = render_system_prompt("minimal")
        word_count = len(result.split())
        assert word_count < 150

    def test_full_contains_personality(self):
        result = render_system_prompt("full")
        assert "sarcastic" in result
        assert "search_knowledge" in result
        assert "delegate" in result
        assert "recall" in result

    def test_full_contains_language_rules(self):
        result = render_system_prompt("full")
        assert "Italian" in result
        assert "English" in result

    def test_no_interpolation_slots(self):
        result = render_system_prompt("minimal")
        assert "Interlocutor:" not in result
        assert "Memory context:" not in result
        result_full = render_system_prompt("full")
        assert "Interlocutor:" not in result_full
        assert "Memory context:" not in result_full

    def test_is_static(self):
        a = render_system_prompt("full")
        b = render_system_prompt("full")
        assert a == b


class TestRenderContextPreamble:
    def test_both_fields(self):
        result = render_context_preamble("Paperino", "Yesterday PK was tired.")
        assert "Interlocutor: Paperino" in result
        assert "Memory context:\nYesterday PK was tired." in result

    def test_user_only(self):
        result = render_context_preamble("Paperino", "")
        assert "Interlocutor: Paperino" in result
        assert "Memory context:" not in result

    def test_memory_only(self):
        result = render_context_preamble("", "Yesterday PK was tired.")
        assert "Interlocutor:" not in result
        assert "Memory context:" in result

    def test_empty_returns_empty(self):
        assert render_context_preamble("", "") == ""


class TestPrependContextToMessages:
    def test_prepends_both(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = prepend_context_to_messages(msgs, "Paperino", "some memories")
        assert result[0]["role"] == "user"
        assert "Paperino" in result[0]["content"]
        assert "some memories" in result[0]["content"]
        assert "Hello" in result[0]["content"]

    def test_user_summary_only(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = prepend_context_to_messages(msgs, "Paperino, anxious", "")
        assert "Paperino, anxious" in result[0]["content"]

    def test_memory_context_only(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = prepend_context_to_messages(msgs, "", "Yesterday PK was exhausted.")
        assert "Yesterday PK was exhausted." in result[0]["content"]

    def test_empty_context_returns_original(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = prepend_context_to_messages(msgs, "", "")
        assert result == msgs

    def test_does_not_mutate_original(self):
        msgs = [{"role": "user", "content": "Hello"}]
        prepend_context_to_messages(msgs, "Paperino", "memories")
        assert msgs[0]["content"] == "Hello"


SAMPLE_PROFILE = """\
# Uno - Soul Document

## Essential Identity

Uno is an artificial intelligence housed in the Ducklair Tower.

## Core Psychology

Sarcastic, warm underneath, fiercely loyal."""


class TestRenderDatagenSystemPrompt:
    def test_contains_profile_and_instructions(self):
        result = render_datagen_system_prompt(SAMPLE_PROFILE)
        assert "Soul Document" in result
        assert "search_knowledge" in result
        assert "delegate" in result

    def test_no_user_or_memory_placeholders(self):
        result = render_datagen_system_prompt(SAMPLE_PROFILE)
        assert "Interlocutor:" not in result
        assert "Memory context:" not in result
        assert "{" not in result

    def test_empty_profile(self):
        result = render_datagen_system_prompt("")
        assert "search_knowledge" in result

    def test_is_identical_across_calls(self):
        a = render_datagen_system_prompt(SAMPLE_PROFILE)
        b = render_datagen_system_prompt(SAMPLE_PROFILE)
        assert a == b


class TestSuiteTemplateMap:
    def test_all_suites_covered(self):
        assert set(SUITE_TEMPLATE_MAP.keys()) == SUITES

    def test_personality_uses_minimal(self):
        assert SUITE_TEMPLATE_MAP["personality"] == "minimal"

    def test_language_uses_minimal(self):
        assert SUITE_TEMPLATE_MAP["language"] == "minimal"

    def test_tool_use_uses_full(self):
        assert SUITE_TEMPLATE_MAP["tool_use"] == "full"

    def test_stability_uses_full(self):
        assert SUITE_TEMPLATE_MAP["stability"] == "full"
