"""Unit tests for reflect_scenes.py."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from llm_backends import LLMBackend
from pkna_scenes import AnnotatedDialogue, Panel, Scene
from reflect_scenes import (
    SceneReflection,
    SceneReflector,
    build_story_context,
    format_prior_issue_summary,
    get_scene_event_index,
    load_reflections,
)


class MockBackend(LLMBackend):
    def __init__(self, responses: list[str | None] | str | None = None):
        if isinstance(responses, list):
            self._responses = list(responses)
        else:
            self._responses = [responses] if responses is not None else [""]
        self._call_count = 0
        self.last_system: str = ""
        self.last_messages: list[dict[str, str]] = []

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> str | None:
        self.last_system = system
        self.last_messages = messages
        if self._call_count < len(self._responses):
            result = self._responses[self._call_count]
        else:
            result = self._responses[-1] if self._responses else None
        self._call_count += 1
        return result


def _make_scene(issue: str = "pkna-3", pages: list[int] | None = None) -> Scene:
    return Scene(
        issue=issue,
        page_numbers=pages or [45, 46],
        panels=[
            Panel(
                description="Uno appears on screen",
                dialogues=[
                    AnnotatedDialogue(
                        character="Paperinik",
                        line="Uno, aiutami!",
                        tone="urgent",
                        speech_act="commanding",
                    ),
                    AnnotatedDialogue(
                        character="Uno",
                        line="Ciao, socio!",
                        tone="playful",
                        speech_act="informing",
                    ),
                ],
                visual_cues=["Uno's hologram flickers"],
            ),
        ],
        other_characters={"Paperinik"},
    )


# ============================================================================
# SceneReflection Model
# ============================================================================


class TestSceneReflection:
    def test_model_fields(self):
        r = SceneReflection(
            scene_id="pkna-0_28",
            emotional_state="calm but alert",
            emotional_shifts=["calm -> tense when alarm sounds"],
            behavioral_drivers="Protective instinct toward Paperinik",
            relationship_dynamics="Uno sees Paperinik as partner, not subordinate",
            subtext="Hiding concern about tower systems failing",
        )
        assert r.scene_id == "pkna-0_28"
        assert len(r.emotional_shifts) == 1
        assert "Protective" in r.behavioral_drivers

    def test_serialization_roundtrip(self):
        r = SceneReflection(
            scene_id="pkna-1_10",
            emotional_state="anxious",
            emotional_shifts=[],
            behavioral_drivers="Fear of being shut down",
            relationship_dynamics="Distant from Everett",
            subtext="None apparent",
        )
        data = r.model_dump()
        restored = SceneReflection.model_validate(data)
        assert restored == r


# ============================================================================
# Story Context Building
# ============================================================================


class TestBuildStoryContext:
    def test_no_context(self):
        ctx = build_story_context([], "pkna-0", [], -1)
        assert ctx.strip() == ""

    def test_only_prior_issues(self):
        prior = [
            format_prior_issue_summary("pkna-0", ["Event A", "Event B"]),
        ]
        ctx = build_story_context(prior, "pkna-1", ["Event C"], -1)
        assert "Previously in the series" in ctx
        assert "pkna-0" in ctx
        assert "Event A" in ctx
        assert "Event B" in ctx
        assert "Earlier in this issue" not in ctx

    def test_only_current_issue(self):
        events = ["First thing", "Second thing", "Third thing"]
        ctx = build_story_context([], "pkna-3", events, 1)
        assert "Previously in the series" not in ctx
        assert "Earlier in this issue (pkna-3)" in ctx
        assert "First thing" in ctx
        assert "Second thing" in ctx
        assert "Third thing" not in ctx

    def test_both_layers(self):
        prior = [
            format_prior_issue_summary("pkna-0", ["Intro event"]),
        ]
        events = ["Current event 1", "Current event 2"]
        ctx = build_story_context(prior, "pkna-1", events, 0)
        assert "Previously" in ctx
        assert "Intro event" in ctx
        assert "Earlier in this issue" in ctx
        assert "Current event 1" in ctx
        assert "Current event 2" not in ctx

    def test_event_index_zero_includes_first_event(self):
        events = ["Only event"]
        ctx = build_story_context([], "pkna-0", events, 0)
        assert "Only event" in ctx

    def test_multiple_prior_issues(self):
        prior = [
            format_prior_issue_summary("pkna-0", ["A"]),
            format_prior_issue_summary("pkna-1", ["B", "C"]),
        ]
        ctx = build_story_context(prior, "pkna-2", [], -1)
        assert "pkna-0" in ctx
        assert "pkna-1" in ctx
        assert "- A" in ctx
        assert "- B" in ctx
        assert "- C" in ctx


class TestFormatPriorIssueSummary:
    def test_basic_format(self):
        result = format_prior_issue_summary("pkna-5", ["Event X", "Event Y"])
        assert result.startswith("### pkna-5")
        assert "- Event X" in result
        assert "- Event Y" in result


class TestGetSceneEventIndex:
    def test_with_matching_event(self, tmp_path: Path):
        page_data = {"last_event": '"Event B"', "panels": []}
        page_path = tmp_path / "page_010.json"
        with open(page_path, "w") as f:
            json.dump(page_data, f)

        key_events = ["Event A", "Event B", "Event C"]
        idx = get_scene_event_index(tmp_path, 10, key_events)
        assert idx == 1

    def test_no_quotes_in_last_event(self, tmp_path: Path):
        page_data = {"last_event": "Event A", "panels": []}
        page_path = tmp_path / "page_005.json"
        with open(page_path, "w") as f:
            json.dump(page_data, f)

        key_events = ["Event A", "Event B"]
        idx = get_scene_event_index(tmp_path, 5, key_events)
        assert idx == 0

    def test_no_match(self, tmp_path: Path):
        page_data = {"last_event": "Unknown event", "panels": []}
        page_path = tmp_path / "page_003.json"
        with open(page_path, "w") as f:
            json.dump(page_data, f)

        key_events = ["Event A", "Event B"]
        idx = get_scene_event_index(tmp_path, 3, key_events)
        assert idx == -1

    def test_missing_page_file(self, tmp_path: Path):
        idx = get_scene_event_index(tmp_path, 99, ["Event A"])
        assert idx == -1

    def test_empty_last_event(self, tmp_path: Path):
        page_data = {"last_event": "", "panels": []}
        page_path = tmp_path / "page_001.json"
        with open(page_path, "w") as f:
            json.dump(page_data, f)

        idx = get_scene_event_index(tmp_path, 1, ["Event A"])
        assert idx == -1


# ============================================================================
# SceneReflector
# ============================================================================


class TestSceneReflector:
    def _make_reflection_json(self, **overrides: str) -> str:
        base = {
            "scene_id": "pkna-3_45",
            "emotional_state": "playful but alert",
            "emotional_shifts": ["relaxed -> engaged"],
            "behavioral_drivers": "Partnership instinct",
            "relationship_dynamics": "Sees Paperinik as equal",
            "subtext": "Masking concern with humor",
        }
        base.update(overrides)
        return json.dumps([base])

    def test_successful_reflection(self):
        response = self._make_reflection_json()
        backend = MockBackend(response)
        reflector = SceneReflector(backend)
        scene = _make_scene()

        result = reflector.reflect_on_scene(scene, "Some context")
        assert result is not None
        assert result.scene_id == "pkna-3_45"
        assert "playful" in result.emotional_state
        assert len(result.emotional_shifts) == 1

    def test_prompt_includes_context(self):
        response = self._make_reflection_json()
        backend = MockBackend(response)
        reflector = SceneReflector(backend)
        scene = _make_scene()

        reflector.reflect_on_scene(scene, "## Previously\n- Something happened")
        user_msg = backend.last_messages[0]["content"]
        assert "Previously" in user_msg
        assert "Something happened" in user_msg

    def test_prompt_includes_scene(self):
        response = self._make_reflection_json()
        backend = MockBackend(response)
        reflector = SceneReflector(backend)
        scene = _make_scene()

        reflector.reflect_on_scene(scene, "")
        user_msg = backend.last_messages[0]["content"]
        assert "Ciao, socio!" in user_msg
        assert "playful" in user_msg

    def test_api_failure_returns_none(self):
        backend = MockBackend(None)
        reflector = SceneReflector(backend)
        scene = _make_scene()

        result = reflector.reflect_on_scene(scene, "")
        assert result is None

    def test_invalid_json_returns_none(self):
        backend = MockBackend("not valid json at all")
        reflector = SceneReflector(backend)
        scene = _make_scene()

        result = reflector.reflect_on_scene(scene, "")
        assert result is None

    def test_empty_response_returns_none(self):
        backend = MockBackend("")
        reflector = SceneReflector(backend)
        scene = _make_scene()

        result = reflector.reflect_on_scene(scene, "")
        assert result is None


# ============================================================================
# load_reflections
# ============================================================================


class TestLoadReflections:
    def test_load_from_empty_dir(self, tmp_path: Path):
        result = load_reflections(tmp_path)
        assert result == {}

    def test_load_from_nonexistent_dir(self, tmp_path: Path):
        result = load_reflections(tmp_path / "does_not_exist")
        assert result == {}

    def test_load_valid_reflections(self, tmp_path: Path):
        issue_dir = tmp_path / "pkna-0"
        issue_dir.mkdir()

        r1 = SceneReflection(
            scene_id="pkna-0_28",
            emotional_state="calm",
            emotional_shifts=[],
            behavioral_drivers="duty",
            relationship_dynamics="partner",
            subtext="none",
        )
        with open(issue_dir / "pkna-0_28.json", "w") as f:
            json.dump(r1.model_dump(), f)

        r2 = SceneReflection(
            scene_id="pkna-0_33",
            emotional_state="tense",
            emotional_shifts=["calm -> tense"],
            behavioral_drivers="threat",
            relationship_dynamics="protector",
            subtext="fear",
        )
        with open(issue_dir / "pkna-0_33.json", "w") as f:
            json.dump(r2.model_dump(), f)

        result = load_reflections(tmp_path)
        assert len(result) == 2
        assert "pkna-0_28" in result
        assert "pkna-0_33" in result
        assert result["pkna-0_28"].emotional_state == "calm"

    def test_skips_invalid_json(self, tmp_path: Path):
        issue_dir = tmp_path / "pkna-1"
        issue_dir.mkdir()

        with open(issue_dir / "bad.json", "w") as f:
            f.write("not json")

        r = SceneReflection(
            scene_id="pkna-1_5",
            emotional_state="ok",
            emotional_shifts=[],
            behavioral_drivers="x",
            relationship_dynamics="y",
            subtext="z",
        )
        with open(issue_dir / "pkna-1_5.json", "w") as f:
            json.dump(r.model_dump(), f)

        result = load_reflections(tmp_path)
        assert len(result) == 1
        assert "pkna-1_5" in result
