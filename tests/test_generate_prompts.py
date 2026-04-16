"""Unit tests for the training prompt bank generator."""

from pathlib import Path

from datagen.generate_prompts import (
    GENERATION_SCENARIOS,
    GenerationScenario,
    generate_llm_prompts,
    generate_manual_prompts,
    generate_scene_prompts,
    load_prompts,
    write_prompts,
    _scenario_to_tools,
    _scenario_to_user_summary,
    _scene_to_prompts,
)
from pkna.datagen.types import DatagenPrompt
from pkna.extract.scenes import AnnotatedDialogue, Panel, Scene
from pkna.llm.backends import GenerateResult
from pkna.llm.testing import SequentialBackend


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
                            line="Abbiamo un problema serio con la missione di domani sera, dobbiamo parlarne subito.",
                            tone="worried",
                        ),
                    ],
                ),
            ],
            other_characters={"Paperino"},
        )

    def test_returns_one_prompt_per_scene(self):
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        assert len(prompts) == 1

    def test_prefers_conversational_lines(self):
        """Lines with 10+ words are preferred over short interjections."""
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        assert len(prompts[0].messages[0]["content"].split()) >= 10

    def test_falls_back_to_longest_when_all_short(self):
        scene = Scene(
            issue="pkna-2",
            page_numbers=[1],
            panels=[
                Panel(
                    description="Quick exchange",
                    dialogues=[
                        AnnotatedDialogue(
                            character="Paperino", line="Ouch!", tone="pained"
                        ),
                        AnnotatedDialogue(
                            character="Paperino",
                            line="Dove sono capitato?",
                            tone="confused",
                        ),
                    ],
                ),
            ],
            other_characters={"Paperino"},
        )
        prompts = _scene_to_prompts(scene)
        assert len(prompts) == 1

    def test_assigns_correct_user_summary(self):
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        assert "Paperino" in prompts[0].user_summary

    def test_scene_metadata(self):
        scene = self._make_scene()
        prompts = _scene_to_prompts(scene)
        assert prompts[0].metadata["prompt_source"] == "scene"
        assert prompts[0].metadata["issue"] == "pkna-1"

    def test_deterministic_selection(self):
        """Same scene always produces the same prompt."""
        scene = self._make_scene()
        prompts_a = _scene_to_prompts(scene)
        prompts_b = _scene_to_prompts(scene)
        assert prompts_a[0].messages == prompts_b[0].messages

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


class TestGenerationScenarios:
    def test_target_counts(self):
        single = [s for s in GENERATION_SCENARIOS if not s.multi_turn]
        multi = [s for s in GENERATION_SCENARIOS if s.multi_turn]
        assert len(single) == 500
        assert len(multi) == 150

    def test_multi_turn_have_directives(self):
        multi = [s for s in GENERATION_SCENARIOS if s.multi_turn]
        for s in multi:
            assert s.turn_count >= 3
            assert len(s.directives) == s.turn_count - 1

    def test_deterministic(self):
        from datagen.generate_prompts import _build_generation_scenarios

        a = _build_generation_scenarios(seed=42)
        b = _build_generation_scenarios(seed=42)
        assert a == b

    def test_no_invalid_combos(self):
        from datagen.generate_prompts import _INVALID_COMBOS

        for s in GENERATION_SCENARIOS:
            assert (s.interlocutor, s.interaction_type) not in _INVALID_COMBOS

    def test_key_characters_overrepresented(self):
        from collections import Counter

        counts = Counter(s.interlocutor for s in GENERATION_SCENARIOS)
        baseline = min(counts[c] for c in ("Xadhoom", "Lyla", "Stranger"))
        assert counts["Paperino"] > baseline
        assert counts["Everett Ducklair"] > baseline


class TestGenerateLlmPrompts:
    def test_generates_from_scenarios(self):
        scenarios = GENERATION_SCENARIOS[:5]
        results = [
            GenerateResult(text=f"Message {i}", model_name="test")
            for i in range(len(scenarios))
        ]
        backend = SequentialBackend(results)
        import datagen.generate_prompts as mod

        original = mod.GENERATION_SCENARIOS
        mod.GENERATION_SCENARIOS = scenarios
        try:
            prompts = generate_llm_prompts(backend)
        finally:
            mod.GENERATION_SCENARIOS = original
        assert len(prompts) == len(scenarios)
        for p in prompts:
            assert p.metadata["prompt_source"] == "generated"

    def test_multi_turn_metadata_propagated(self):
        scenarios = [
            GenerationScenario(
                "Paperino",
                "worried",
                "a mission",
                "emotional support",
                "italian",
                True,
                4,
                ["continue", "escalate", "continue"],
            ),
        ]
        results = [GenerateResult(text="Ciao Uno!", model_name="test")]
        backend = SequentialBackend(results)
        import datagen.generate_prompts as mod

        original = mod.GENERATION_SCENARIOS
        mod.GENERATION_SCENARIOS = scenarios
        try:
            prompts = generate_llm_prompts(backend)
        finally:
            mod.GENERATION_SCENARIOS = original
        assert len(prompts) == 1
        assert prompts[0].metadata["multi_turn"] is True
        assert prompts[0].metadata["turn_count"] == 4
        assert prompts[0].metadata["directives"] == [
            "continue",
            "escalate",
            "continue",
        ]

    def test_skips_failed_generations(self):
        scenarios = GENERATION_SCENARIOS[:3]
        results: list[GenerateResult | None] = [
            None,
            GenerateResult(text="Valid message", model_name="test"),
            None,
        ]
        backend = SequentialBackend(results)
        import datagen.generate_prompts as mod

        original = mod.GENERATION_SCENARIOS
        mod.GENERATION_SCENARIOS = scenarios
        try:
            prompts = generate_llm_prompts(backend)
        finally:
            mod.GENERATION_SCENARIOS = original
        assert len(prompts) == 1

    def test_skips_empty_results(self):
        scenarios = GENERATION_SCENARIOS[:3]
        results = [
            GenerateResult(text="  ", model_name="test"),
            GenerateResult(text="Valid message", model_name="test"),
            GenerateResult(text="", model_name="test"),
        ]
        backend = SequentialBackend(results)
        import datagen.generate_prompts as mod

        original = mod.GENERATION_SCENARIOS
        mod.GENERATION_SCENARIOS = scenarios
        try:
            prompts = generate_llm_prompts(backend)
        finally:
            mod.GENERATION_SCENARIOS = original
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
