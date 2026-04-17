"""Unit tests for system prompt templates."""

from pkna.eval.types import SUITES
from pkna.inference.system_prompts import (
    SUITE_TEMPLATE_MAP,
    render_datagen_prompt,
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


SAMPLE_PROFILE = """\
# Uno - Soul Document

## Essential Identity

Uno is an artificial intelligence housed in the Ducklair Tower.

## Core Psychology

Sarcastic, warm underneath, fiercely loyal."""


class TestRenderDatagenPrompt:
    def test_contains_profile_content(self):
        result = render_datagen_prompt(SAMPLE_PROFILE)
        assert "Soul Document" in result
        assert "Sarcastic, warm underneath" in result

    def test_contains_operational_instructions(self):
        result = render_datagen_prompt(SAMPLE_PROFILE)
        assert "search_knowledge" in result
        assert "delegate" in result
        assert "recall" in result
        assert "Italian" in result
        assert "English" in result

    def test_user_summary_injected(self):
        result = render_datagen_prompt(SAMPLE_PROFILE, user_summary="Paperino, anxious")
        assert "Paperino, anxious" in result
        assert "Interlocutor:" in result

    def test_memory_context_injected(self):
        result = render_datagen_prompt(
            SAMPLE_PROFILE, memory_context="Yesterday PK was tired."
        )
        assert "Yesterday PK was tired." in result
        assert "Memory context:" in result

    def test_empty_slots_omitted(self):
        result = render_datagen_prompt(SAMPLE_PROFILE)
        assert "Interlocutor:" not in result
        assert "Memory context:" not in result

    def test_empty_profile(self):
        result = render_datagen_prompt("")
        assert "search_knowledge" in result


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
