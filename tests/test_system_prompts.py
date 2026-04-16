"""Unit tests for system prompt templates."""

from pkna.eval.types import SUITES
from pkna.inference.system_prompts import (
    SUITE_TEMPLATE_MAP,
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

    def test_user_summary_injected(self):
        result = render_system_prompt("minimal", user_summary="Paperino, anxious")
        assert "Paperino, anxious" in result
        assert "Interlocutor:" in result

    def test_memory_context_injected(self):
        result = render_system_prompt("full", memory_context="Yesterday PK was tired.")
        assert "Yesterday PK was tired." in result
        assert "Memory context:" in result

    def test_empty_slots_omitted(self):
        result = render_system_prompt("minimal")
        assert "Interlocutor:" not in result
        assert "Memory context:" not in result

    def test_both_slots_present(self):
        result = render_system_prompt(
            "full",
            user_summary="Xadhoom, furious",
            memory_context="She destroyed a fleet yesterday.",
        )
        assert "Xadhoom, furious" in result
        assert "She destroyed a fleet yesterday." in result


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
