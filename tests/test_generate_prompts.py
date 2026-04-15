"""Unit tests for the training prompt bank generator."""

from pathlib import Path

from datagen.generate_prompts import (
    GENERATION_SCENARIOS,
    generate_llm_prompts,
    generate_manual_prompts,
    generate_scene_prompts,
    load_prompts,
    write_prompts,
    _scenario_to_tools,
    _scenario_to_user_summary,
    _scene_to_prompts,
)
from pkna.datagen_types import DatagenPrompt
from pkna.llm_backends import GenerateResult
from pkna.pkna_scenes import AnnotatedDialogue, Panel, Scene
from pkna.testing import SequentialBackend


class TestGenerateManualPrompts:
    def test_produces_prompts(self):
        prompts = generate_manual_prompts()
        assert len(prompts) > 50

    def test_unique_ids(self):
        prompts = generate_manual_prompts()
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids))

    def test_all_have_required_metadata(self):
        prompts = generate_manual_prompts()
        for p in prompts:
            assert "prompt_source" in p.metadata
            assert p.metadata["prompt_source"] == "manual"
            assert "category" in p.metadata
            assert len(p.messages) >= 1
            assert p.messages[0]["role"] == "user"

    def test_multi_turn_prompts_have_directives(self):
        prompts = generate_manual_prompts()
        multi = [p for p in prompts if p.metadata.get("multi_turn")]
        assert len(multi) > 0
        for p in multi:
            assert "directives" in p.metadata
            assert "turn_count" in p.metadata
            assert p.metadata["turn_count"] > 1


class TestSceneToPrompts:
    def _make_scene(self) -> Scene:
        return Scene(
            issue="pkna-1",
            page_numbers=[5, 6],
            panels=[
                Panel(
                    description="Paperino enters the tower",
                    dialogues=[
                        AnnotatedDialogue(
                            character="Paperino",
                            line="Ciao Uno, come stai?",
                            tone="friendly",
                        ),
                        AnnotatedDialogue(
                            character="Uno",
                            line="Bene, socio!",
                            tone="cheerful",
                        ),
                    ],
                ),
                Panel(
                    description="They discuss mission",
                    dialogues=[
                        AnnotatedDialogue(
                            character="Paperino",
                            line="Abbiamo un problema.",
                            tone="worried",
                        ),
                    ],
                ),
            ],
            other_characters={"Paperino"},
        )

    def test_extracts_non_uno_lines(self):
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        assert len(prompts) == 2
        assert prompts[0].messages[0]["content"] == "Ciao Uno, come stai?"
        assert prompts[1].messages[0]["content"] == "Abbiamo un problema."

    def test_assigns_correct_user_summary(self):
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        for p in prompts:
            assert "Paperino" in p.user_summary

    def test_scene_metadata(self):
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        for p in prompts:
            assert p.metadata["prompt_source"] == "scene"
            assert p.metadata["issue"] == "pkna-1"

    def test_empty_scene(self):
        scene = Scene(
            issue="pkna-1",
            page_numbers=[1],
            panels=[
                Panel(
                    description="Uno alone",
                    dialogues=[
                        AnnotatedDialogue(
                            character="Uno", line="Hmm...", tone="neutral"
                        ),
                    ],
                ),
            ],
            other_characters=set(),
        )
        prompts = _scene_to_prompts(scene)
        assert prompts == []


class TestGenerateScenePrompts:
    def test_missing_directory(self, tmp_path: Path):
        prompts = generate_scene_prompts(tmp_path / "nonexistent")
        assert prompts == []


class TestScenarioHelpers:
    def test_scenario_to_tools_wiki(self):
        assert "search_knowledge" in _scenario_to_tools("wiki lookup")

    def test_scenario_to_tools_delegation(self):
        tools = _scenario_to_tools("delegation to specialist")
        assert "delegate" in tools

    def test_scenario_to_tools_default(self):
        assert _scenario_to_tools("casual chat") == []

    def test_scenario_to_user_summary(self):
        assert "Paperino" in _scenario_to_user_summary("Paperino")
        assert "Xadhoom" in _scenario_to_user_summary("Xadhoom")
        assert "stranger" in _scenario_to_user_summary("Unknown Person").lower()


class TestGenerateLlmPrompts:
    def test_generates_from_scenarios(self):
        results = [
            GenerateResult(text=f"Message {i}", model_name="test")
            for i in range(len(GENERATION_SCENARIOS))
        ]
        backend = SequentialBackend(results)
        prompts = generate_llm_prompts(backend)
        assert len(prompts) == len(GENERATION_SCENARIOS)
        for p in prompts:
            assert p.metadata["prompt_source"] == "generated"

    def test_skips_failed_generations(self):
        results: list[GenerateResult | None] = [
            None,
            GenerateResult(text="Valid message", model_name="test"),
        ]
        results.extend([None] * (len(GENERATION_SCENARIOS) - 2))
        backend = SequentialBackend(results)
        prompts = generate_llm_prompts(backend)
        assert len(prompts) == 1

    def test_skips_empty_results(self):
        results = [
            GenerateResult(text="  ", model_name="test"),
            GenerateResult(text="Valid message", model_name="test"),
        ]
        results.extend(
            [GenerateResult(text="", model_name="test")]
            * (len(GENERATION_SCENARIOS) - 2)
        )
        backend = SequentialBackend(results)
        prompts = generate_llm_prompts(backend)
        assert len(prompts) == 1


class TestWriteAndLoadPrompts:
    def test_roundtrip(self, tmp_path: Path):
        prompts = [
            DatagenPrompt(
                id="test-001",
                messages=[{"role": "user", "content": "Hello"}],
                user_summary="Test user",
                memory_context="",
                tools=[],
                metadata={"prompt_source": "test"},
            ),
            DatagenPrompt(
                id="test-002",
                messages=[{"role": "user", "content": "Ciao"}],
                user_summary="Another user",
                memory_context="some context",
                tools=["search_knowledge"],
                metadata={"prompt_source": "test", "language": "italian"},
            ),
        ]
        path = tmp_path / "prompts.jsonl"
        write_prompts(path, prompts)
        loaded = load_prompts(path)
        assert len(loaded) == 2
        assert loaded[0] == prompts[0]
        assert loaded[1] == prompts[1]
