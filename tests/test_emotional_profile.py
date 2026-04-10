"""Unit tests for build_emotional_profile.py."""

import json
from collections.abc import Callable

import pytest

from extract.build_emotional_profile import (
    SECTION_ORDER,
    ClaimCondenser,
    ClaimLedger,
    ClaimRefiner,
    ClaimSynthesizer,
    SoulDocumentGenerator,
    _group_claims_by_prefix,
    _merge_evidence,
    _path_prefix,
    format_claims_compact,
    format_claims_detail,
    is_valid_claim_path,
)
from pkna.llm_backends import GenerateResult, LLMBackend
from pkna.pkna_scenes import AnnotatedDialogue, Panel, Scene, format_scene_view
from pydantic import BaseModel

MOCK_MODEL = "test-model"


class MockBackend(LLMBackend):
    """A test backend that returns preconfigured responses."""

    def __init__(self, responses: "list[str | None] | list[str] | str | None" = None):
        if isinstance(responses, list):
            self._responses = list(responses)
        else:
            self._responses = [responses] if responses is not None else [""]
        self._call_count = 0

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = self._responses[-1] if self._responses else None
        self._call_count += 1
        if text is None:
            return None
        return GenerateResult(text=text, model_name=MOCK_MODEL)


# ============================================================================
# Path Validation
# ============================================================================


class TestPathValidation:
    @pytest.mark.parametrize(
        "path",
        [
            "identity/names",
            "identity/bio",
            "identity/origin",
            "psychology/values/core",
            "psychology/values/tradeoffs",
            "psychology/growth/emotional_arc",
            "psychology/growth/relationship_arc",
            "psychology/self_model/identity_stance",
            "psychology/self_model/agency",
            "psychology/self_model/mortality",
            "psychology/moral_compass/value_hierarchy",
            "psychology/moral_compass/dilemma_patterns",
            "communication/humor/type",
            "communication/humor/timing",
            "communication/humor/targets",
            "communication/voice/register_shifts",
            "capabilities/knowledge_boundaries/temporal",
            "capabilities/knowledge_boundaries/domain",
            "capabilities/knowledge_boundaries/forbidden",
            "behavior/evolution",
            "behavior/adaptation/by_audience",
            "behavior/adaptation/by_situation",
        ],
    )
    def test_valid_static_paths(self, path: str):
        assert is_valid_claim_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "relationships/paperinik",
            "relationships/paperinik/dynamic",
            "relationships/paperinik/uno_believes",
            "relationships/paperinik/perceived_by_uno",
            "relationships/paperinik/behavioral_driver",
            "relationships/everett_ducklair",
            "relationships/due/dynamic",
        ],
    )
    def test_valid_relationship_paths(self, path: str):
        assert is_valid_claim_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "psychology/emotional/triggers/surprise",
            "psychology/nonexistent",
            "capabilities/new_category",
        ],
    )
    def test_valid_freeform_paths(self, path: str):
        assert is_valid_claim_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "invalid/path",
            "psychology",
            "relationships/pk/invalid_sub",
            "relationships/pk/dynamic/extra",
            "values/core",
            "",
        ],
    )
    def test_invalid_paths(self, path: str):
        assert not is_valid_claim_path(path)


# ============================================================================
# AnnotatedDialogue
# ============================================================================


class TestAnnotatedDialogue:
    def test_creation_with_defaults(self):
        ad = AnnotatedDialogue(character="Uno", line="Ciao!")
        assert ad.character == "Uno"
        assert ad.line == "Ciao!"
        assert ad.tone == "neutral"
        assert ad.speech_act == "informing"

    def test_creation_with_annotations(self):
        ad = AnnotatedDialogue(
            character="Uno",
            line="Sai che dispiacere!",
            tone="sarcastic",
            speech_act="deflecting",
        )
        assert ad.tone == "sarcastic"
        assert ad.speech_act == "deflecting"


# ============================================================================
# Scene
# ============================================================================


class TestScene:
    def _make_scene(self) -> Scene:
        return Scene(
            issue="pkna-3",
            page_numbers=[45, 46, 47],
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
                    visual_cues=["Uno's hologram flickers nervously"],
                ),
            ],
            other_characters={"Paperinik"},
        )

    def test_scene_id(self):
        scene = self._make_scene()
        assert scene.scene_id == "pkna-3_45"

    def test_summary_from_panels(self):
        scene = self._make_scene()
        assert scene.summary == "Uno appears on screen"

    def test_to_dict_and_back(self):
        scene = self._make_scene()
        data = scene.to_dict()
        restored = Scene.from_dict(data)

        assert restored.issue == scene.issue
        assert restored.page_numbers == scene.page_numbers
        assert len(restored.panels) == 1
        panel = restored.panels[0]
        assert panel.description == "Uno appears on screen"
        assert len(panel.dialogues) == 2
        assert panel.dialogues[0].character == "Paperinik"
        assert panel.dialogues[0].tone == "urgent"
        assert panel.dialogues[1].line == "Ciao, socio!"
        assert panel.dialogues[1].tone == "playful"
        assert panel.visual_cues == ["Uno's hologram flickers nervously"]
        assert restored.other_characters == scene.other_characters

    def test_from_dict_missing_optional_fields(self):
        data = {
            "issue": "pkna-0",
            "page_numbers": [1],
            "panels": [
                {
                    "description": "Test panel",
                    "dialogues": [
                        {
                            "character": "Uno",
                            "line": "Hello",
                            "tone": "neutral",
                            "speech_act": "informing",
                        }
                    ],
                }
            ],
            "other_characters": [],
        }
        scene = Scene.from_dict(data)
        assert scene.panels[0].visual_cues == []

    def test_to_context_string(self):
        scene = self._make_scene()
        ctx = scene.to_context_string()
        assert "pkna-3" in ctx
        assert "page 45" in ctx
        assert "Paperinik" in ctx


# ============================================================================
# ClaimLedger
# ============================================================================


class TestClaimLedger:
    def test_empty_ledger(self):
        ledger = ClaimLedger()
        assert ledger.claim_count() == 0
        assert ledger.scene_count() == 0

    def test_add_claim_standard_path(self):
        ledger = ClaimLedger()
        claim = ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno is sarcastic",
            scene_id="pkna-0_12",
            justification="Uses ironic responses",
        )
        assert claim.id == 1
        assert claim.path == "psychology/traits/temperament"
        assert ledger.claim_count() == 1

    def test_add_claim_new_paths(self):
        ledger = ClaimLedger()
        for path in [
            "psychology/growth/emotional_arc",
            "communication/humor/type",
            "capabilities/knowledge_boundaries/temporal",
            "psychology/self_model/identity_stance",
        ]:
            ledger.add_claim(
                path=path,
                text=f"Claim for {path}",
                scene_id="pkna-0_1",
                justification="Test",
            )
        assert ledger.claim_count() == 4

    def test_add_claim_relationship_sub_paths(self):
        ledger = ClaimLedger()
        for sub in ["dynamic", "uno_believes", "perceived_by_uno", "behavioral_driver"]:
            ledger.add_claim(
                path=f"relationships/paperinik/{sub}",
                text=f"Claim for {sub}",
                scene_id="pkna-0_1",
                justification="Test",
            )
        assert ledger.claim_count() == 4

    def test_add_claim_flat_relationship(self):
        ledger = ClaimLedger()
        claim = ledger.add_claim(
            path="relationships/due",
            text="Uno distrusts Due",
            scene_id="pkna-2_10",
            justification="Expressed suspicion",
        )
        assert claim.path == "relationships/due"

    def test_add_claim_invalid_path(self):
        ledger = ClaimLedger()
        with pytest.raises(ValueError, match="Invalid path"):
            ledger.add_claim(
                path="invalid/path",
                text="Bad claim",
                scene_id="pkna-0_1",
                justification="Test",
            )

    def test_add_claim_invalid_relationship_sub_path(self):
        ledger = ClaimLedger()
        with pytest.raises(ValueError, match="Invalid path"):
            ledger.add_claim(
                path="relationships/pk/invalid_sub",
                text="Bad claim",
                scene_id="pkna-0_1",
                justification="Test",
            )

    def test_add_claim_with_quote(self):
        ledger = ClaimLedger()
        claim = ledger.add_claim(
            path="communication/idiolect/catchphrases",
            text="Uno uses 'Sai che dispiacere!' sarcastically",
            scene_id="pkna-0_12",
            justification="Catchphrase used multiple times",
            quote="Sai che dispiacere!",
            quote_context="When Paperinik complains about early wake-up",
        )
        assert len(claim.quotes) == 1
        assert claim.quotes[0].text == "Sai che dispiacere!"

    def test_support_claim_success(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno is protective",
            scene_id="pkna-0_12",
            justification="Initial evidence",
        )
        success, message = ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_20",
            justification="Additional evidence",
        )
        assert success is True
        assert "+2" in message

    def test_support_claim_not_found(self):
        ledger = ClaimLedger()
        success, message = ledger.support_claim(
            claim_id=999,
            scene_id="pkna-0_1",
            justification="Test",
        )
        assert success is False
        assert "not found" in message

    def test_support_claim_duplicate_scene(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Test",
            scene_id="pkna-0_12",
            justification="Initial",
        )
        ledger.support_claim(claim_id=1, scene_id="pkna-1_5", justification="First")
        success, message = ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="Duplicate",
        )
        assert success is False
        assert "already supports" in message

    def test_contradict_claim_success(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno always uses sarcasm",
            scene_id="pkna-0_12",
            justification="Initial",
        )
        success, _ = ledger.contradict_claim(
            claim_id=1,
            scene_id="pkna-5_30",
            justification="Speaks directly here",
        )
        assert success is True
        claim = ledger.get_claim(1)
        assert claim is not None
        assert len(claim.contradicting) == 1
        assert claim.support_count == 0

    def test_contradict_claim_duplicate_scene(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Test",
            scene_id="pkna-0_12",
            justification="Initial",
        )
        ledger.contradict_claim(claim_id=1, scene_id="pkna-1_5", justification="First")
        success, message = ledger.contradict_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="Duplicate",
        )
        assert success is False
        assert "already contradicts" in message

    def test_refine_claim(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno is sarcastic",
            scene_id="pkna-0_12",
            justification="Initial",
        )
        success, _ = ledger.refine_claim(1, "Uno uses sarcasm as a defense mechanism")
        assert success is True
        claim = ledger.get_claim(1)
        assert claim is not None
        assert claim.text == "Uno uses sarcasm as a defense mechanism"
        assert len(claim.supporting) == 1

    def test_scene_tracking(self):
        ledger = ClaimLedger()
        scene = Scene(
            issue="pkna-0",
            page_numbers=[12, 13],
            panels=[
                Panel(
                    description="Panel",
                    dialogues=[AnnotatedDialogue(character="Uno", line="Hello")],
                ),
            ],
            other_characters={"Paperinik"},
        )
        assert not ledger.is_scene_processed(scene.scene_id)
        ledger.add_scene(scene)
        assert ledger.is_scene_processed(scene.scene_id)
        assert ledger.scene_count() == 1
        assert ledger.get_scene(scene.scene_id) is not None

    def test_populate_scene_cache(self):
        ledger = ClaimLedger()
        scene1 = Scene(
            issue="pkna-0",
            page_numbers=[12],
            panels=[
                Panel(
                    description="Panel",
                    dialogues=[AnnotatedDialogue(character="Uno", line="Hello")],
                ),
            ],
            other_characters=set(),
        )
        scene2 = Scene(
            issue="pkna-1",
            page_numbers=[5],
            panels=[
                Panel(
                    description="Panel 2",
                    dialogues=[AnnotatedDialogue(character="Uno", line="World")],
                ),
            ],
            other_characters=set(),
        )
        ledger.add_scene(scene1)

        data = ledger.to_json()
        restored = ClaimLedger.from_json(data)
        assert restored.get_scene("pkna-0_12") is None

        restored.populate_scene_cache([scene1, scene2])
        assert restored.get_scene("pkna-0_12") is not None
        assert restored.get_scene("pkna-1_5") is None

    def test_serialization_roundtrip(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno is sarcastic",
            scene_id="pkna-0_12",
            justification="Uses ironic responses",
            quote="Sai che dispiacere!",
            quote_context="When Paperinik complains",
        )
        ledger.support_claim(
            claim_id=1, scene_id="pkna-1_5", justification="More sarcasm"
        )

        scene = Scene(
            issue="pkna-0",
            page_numbers=[12],
            panels=[
                Panel(
                    description="Panel",
                    dialogues=[AnnotatedDialogue(character="Uno", line="Hello")],
                ),
            ],
            other_characters=set(),
        )
        ledger.add_scene(scene)

        data = ledger.to_json()
        restored = ClaimLedger.from_json(data)

        assert restored.claim_count() == 1
        assert restored.scene_count() == 1

        claim = restored.get_claim(1)
        assert claim is not None
        assert claim.text == "Uno is sarcastic"
        assert len(claim.supporting) == 2
        assert len(claim.quotes) == 1

        assert restored.is_scene_processed("pkna-0_12")
        assert restored.get_scene("pkna-0_12") is None

    def test_meta_defaults(self):
        ledger = ClaimLedger()
        assert ledger.meta["model_name"] is None
        assert ledger.meta["lm_usage"] == {}

    def test_accumulate_usage(self):
        ledger = ClaimLedger()
        r1 = GenerateResult(
            text="hello",
            model_name="test-model",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        r2 = GenerateResult(
            text="world",
            model_name="test-model",
            usage={"input_tokens": 200, "output_tokens": 75},
        )
        ledger.accumulate_usage(r1)
        ledger.accumulate_usage(r2)

        assert ledger.meta["model_name"] == "test-model"
        assert ledger.meta["lm_usage"]["input_tokens"] == 300
        assert ledger.meta["lm_usage"]["output_tokens"] == 125

    def test_meta_serialization_roundtrip(self):
        ledger = ClaimLedger()
        ledger.accumulate_usage(
            GenerateResult(
                text="x",
                model_name="my-model",
                usage={"input_tokens": 42, "output_tokens": 10},
            )
        )
        data = ledger.to_json()
        assert data["meta"]["model_name"] == "my-model"
        assert data["meta"]["lm_usage"]["input_tokens"] == 42

        restored = ClaimLedger.from_json(data)
        assert restored.meta["model_name"] == "my-model"
        assert restored.meta["lm_usage"]["input_tokens"] == 42

    def test_meta_from_json_without_meta_key(self):
        data = {"next_id": 1, "claims": {}, "processed_scene_ids": []}
        ledger = ClaimLedger.from_json(data)
        assert ledger.meta["model_name"] is None
        assert ledger.meta["lm_usage"] == {}

    def test_get_claims_by_section(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Trait 1",
            scene_id="pkna-0_1",
            justification="R1",
        )
        ledger.add_claim(
            path="psychology/traits/mbti",
            text="Trait 2",
            scene_id="pkna-0_2",
            justification="R2",
        )
        ledger.add_claim(
            path="identity/bio",
            text="Fact 1",
            scene_id="pkna-0_3",
            justification="R3",
        )

        by_section = ledger.get_claims_by_section()
        assert len(by_section["psychology"]) == 2
        assert len(by_section["identity"]) == 1
        assert "communication" not in by_section


# ============================================================================
# Format Functions
# ============================================================================


class TestFormatFunctions:
    @pytest.fixture
    def sample_ledger(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno uses sarcasm as a defense mechanism",
            scene_id="pkna-0_12",
            justification="Deflects with irony",
            quote="Sai che dispiacere!",
            quote_context="When Paperinik complains",
        )
        ledger.support_claim(
            claim_id=1, scene_id="pkna-1_5", justification="More sarcasm"
        )
        ledger.add_claim(
            path="identity/bio",
            text="Uno is an AI living in Ducklair Tower",
            scene_id="pkna-0_1",
            justification="Introduced as such",
        )
        return ledger

    def test_format_claims_compact(self, sample_ledger):
        output = format_claims_compact(sample_ledger)
        assert "Total claims: 2" in output
        assert "## Identity" in output
        assert "## Psychology" in output
        assert "[+2]" in output
        assert "[+1]" in output

    def test_format_claims_compact_filtered(self, sample_ledger):
        output = format_claims_compact(sample_ledger, section="psychology")
        assert "## Psychology" in output
        assert "## Identity" not in output

    def test_format_claims_detail(self, sample_ledger):
        output = format_claims_detail(sample_ledger, [1])
        assert "ID 1:" in output
        assert "Path: psychology/traits/temperament" in output
        assert "Uno uses sarcasm" in output
        assert "Supporting:" in output
        assert "pkna-0_12:" in output
        assert "Quotes:" in output
        assert "Sai che dispiacere!" in output

    def test_format_claims_detail_not_found(self, sample_ledger):
        output = format_claims_detail(sample_ledger, [999])
        assert "ID 999: Not found" in output

    def test_format_scene_view(self):
        scene = Scene(
            issue="pkna-0",
            page_numbers=[12, 13, 14],
            panels=[
                Panel(
                    description="Uno's screen lights up",
                    dialogues=[
                        AnnotatedDialogue(
                            character="Paperinik",
                            line="Chi sei?",
                            tone="concerned",
                            speech_act="questioning",
                        ),
                        AnnotatedDialogue(
                            character="Uno",
                            line="Benvenuto!",
                            tone="playful",
                            speech_act="informing",
                        ),
                    ],
                    visual_cues=["Uno's hologram appears bright blue"],
                ),
                Panel(
                    description="Paperinik looks surprised",
                    dialogues=[
                        AnnotatedDialogue(
                            character="Uno",
                            line="Io sono Uno.",
                            tone="neutral",
                            speech_act="informing",
                        ),
                    ],
                ),
            ],
            other_characters={"Paperinik", "Everett"},
        )

        output = format_scene_view(scene)

        assert "Scene: pkna-0_12" in output
        assert "Issue: pkna-0, pages 12-13-14" in output
        assert "Characters present: Everett, Paperinik" in output
        # Panel 1 has description, visual cues, and interleaved dialogue
        assert "Panel 1" in output
        assert "Uno's screen lights up" in output
        assert "Uno's hologram appears bright blue" in output
        assert 'Paperinik: "Chi sei?" [concerned, questioning]' in output
        assert 'Uno: "Benvenuto!" [playful, informing]' in output
        # Panel 2
        assert "Panel 2" in output
        assert "Paperinik looks surprised" in output
        assert 'Uno: "Io sono Uno." [neutral, informing]' in output


# ============================================================================
# ClaimRefiner
# ============================================================================


class TestClaimRefiner:
    def _make_ledger_with_contradictions(self) -> ClaimLedger:
        ledger = ClaimLedger()
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno is always sarcastic",
            scene_id="pkna-0_12",
            justification="Uses ironic deflection",
        )
        ledger.contradict_claim(
            claim_id=1,
            scene_id="pkna-5_30",
            justification="Speaks sincerely when danger is present",
        )
        ledger.add_claim(
            path="identity/bio",
            text="Uno is an AI",
            scene_id="pkna-0_1",
            justification="Introduced as artificial intelligence",
        )
        ledger.add_claim(
            path="behavior/avoids",
            text="Uno avoids direct confrontation",
            scene_id="pkna-1_10",
            justification="Prefers indirect approaches",
        )
        ledger.contradict_claim(
            claim_id=3,
            scene_id="pkna-3_20",
            justification="Directly confronts Due to protect Paperinik",
        )
        return ledger

    def test_find_contradicted_claims(self):
        ledger = self._make_ledger_with_contradictions()
        backend = MockBackend()
        refiner = ClaimRefiner(backend, ledger)
        contradicted = refiner._find_contradicted_claims()
        assert len(contradicted) == 2
        assert {c.id for c in contradicted} == {1, 3}

    def test_refine_all_success(self):
        ledger = self._make_ledger_with_contradictions()
        backend = MockBackend("Uno defaults to sarcasm but speaks sincerely in danger.")
        refiner = ClaimRefiner(backend, ledger)
        refined_count, failed_count = refiner.refine_all()

        assert refined_count == 2
        assert failed_count == 0

        claim1 = ledger.get_claim(1)
        assert claim1 is not None
        assert claim1.text == "Uno defaults to sarcasm but speaks sincerely in danger."
        assert len(claim1.contradicting) == 0
        assert len(claim1.supporting) == 2

    def test_refine_all_api_failure(self):
        ledger = self._make_ledger_with_contradictions()
        backend = MockBackend(None)
        refiner = ClaimRefiner(backend, ledger)
        refined_count, failed_count = refiner.refine_all()

        assert refined_count == 0
        assert failed_count == 2

        claim1 = ledger.get_claim(1)
        assert claim1 is not None
        assert claim1.text == "Uno is always sarcastic"
        assert len(claim1.contradicting) == 1


# ============================================================================
# ClaimSynthesizer
# ============================================================================


class TestClaimSynthesizer:
    def _make_ledger(self) -> ClaimLedger:
        ledger = ClaimLedger()
        # Behavior claim with enough support
        ledger.add_claim(
            path="behavior/does",
            text="Uno monitors all tower systems",
            scene_id="pkna-0_1",
            justification="Shown monitoring",
        )
        ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_1",
            justification="Monitors again",
        )
        # Communication claim with enough support
        ledger.add_claim(
            path="communication/humor/type",
            text="Uno uses sarcasm frequently",
            scene_id="pkna-0_5",
            justification="Sarcastic remark",
        )
        ledger.support_claim(
            claim_id=2,
            scene_id="pkna-1_5",
            justification="More sarcasm",
        )
        # Psychology claim (not in _REASONING_SECTIONS, but in _NEGATIVE_SECTIONS)
        ledger.add_claim(
            path="psychology/traits/temperament",
            text="Uno is analytical",
            scene_id="pkna-0_10",
            justification="Shows logical thinking",
        )
        ledger.support_claim(
            claim_id=3,
            scene_id="pkna-2_10",
            justification="More logic",
        )
        return ledger

    def test_eligible_claims_for_reasoning(self):
        ledger = self._make_ledger()
        backend = MockBackend()
        synthesizer = ClaimSynthesizer(backend, ledger, threshold=2)
        eligible = synthesizer._eligible_claims_for_reasoning()
        # Only behavior and communication claims (not psychology)
        assert len(eligible) == 2
        paths = {c.path for c in eligible}
        assert "behavior/does" in paths
        assert "communication/humor/type" in paths

    def test_synthesize_reasoning_success(self):
        ledger = self._make_ledger()
        backend = MockBackend(
            "Uno monitors all tower systems, because he sees it as his core duty."
        )
        synthesizer = ClaimSynthesizer(backend, ledger, threshold=2)
        enriched, failed = synthesizer.synthesize_reasoning()

        assert enriched == 2
        assert failed == 0

    def test_synthesize_reasoning_api_failure(self):
        ledger = self._make_ledger()
        backend = MockBackend(None)
        synthesizer = ClaimSynthesizer(backend, ledger, threshold=2)
        enriched, failed = synthesizer.synthesize_reasoning()

        assert enriched == 0
        assert failed == 2

    def test_synthesize_all(self):
        ledger = self._make_ledger()
        reasoning_text = "Enriched claim text because reasons."
        backend = MockBackend([reasoning_text, reasoning_text])
        synthesizer = ClaimSynthesizer(backend, ledger, threshold=2)
        enriched, failed = synthesizer.synthesize_all()

        assert enriched == 2
        assert failed == 0


# ============================================================================
# Scene Extraction
# ============================================================================


class TestSceneExtraction:
    def test_create_scene_from_panels_with_annotations(self):
        from pkna.pkna_scenes import _create_scene_from_panels

        raw_panels = [
            {
                "is_new_scene": False,
                "description": "Uno appare sullo schermo",
                "visual_cues": ["Uno's hologram is bright blue"],
                "dialogues": [
                    {
                        "character": "Uno",
                        "line": "Buongiorno, socio!",
                        "tone": "playful",
                        "speech_act": "informing",
                    },
                    {
                        "character": "Paperinik",
                        "line": "Ciao Uno!",
                        "tone": "neutral",
                        "speech_act": "informing",
                    },
                ],
            },
            {
                "is_new_scene": False,
                "description": "Paperinik si prepara",
                "visual_cues": [],
                "dialogues": [
                    {
                        "character": "Uno",
                        "line": "Attento!",
                        "tone": "urgent",
                        "speech_act": "warning",
                    },
                ],
            },
        ]

        scene = _create_scene_from_panels("pkna-0", [12, 13], raw_panels)
        assert scene is not None
        assert len(scene.panels) == 2

        panel1 = scene.panels[0]
        assert panel1.description == "Uno appare sullo schermo"
        assert panel1.visual_cues == ["Uno's hologram is bright blue"]
        assert len(panel1.dialogues) == 2
        assert panel1.dialogues[0].character == "Uno"
        assert panel1.dialogues[0].tone == "playful"
        assert panel1.dialogues[1].character == "Paperinik"
        assert panel1.dialogues[1].tone == "neutral"

        panel2 = scene.panels[1]
        assert panel2.dialogues[0].tone == "urgent"
        assert panel2.dialogues[0].speech_act == "warning"

        assert "Paperinik" in scene.other_characters

    def test_create_scene_no_uno(self):
        from pkna.pkna_scenes import _create_scene_from_panels

        raw_panels = [
            {
                "description": "A panel without Uno",
                "dialogues": [
                    {"character": "Paperinik", "line": "Where is Uno?"},
                ],
            },
        ]

        scene = _create_scene_from_panels("pkna-0", [1], raw_panels)
        assert scene is None


# ============================================================================
# ClaimLedger.remove_claim
# ============================================================================


class TestRemoveClaim:
    def test_remove_existing_claim(self):
        ledger = ClaimLedger()
        ledger.add_claim(
            path="identity/bio",
            text="Uno is an AI",
            scene_id="pkna-0_1",
            justification="Introduced as AI",
        )
        assert ledger.claim_count() == 1
        assert ledger.remove_claim(1) is True
        assert ledger.claim_count() == 0
        assert ledger.get_claim(1) is None

    def test_remove_nonexistent_claim(self):
        ledger = ClaimLedger()
        assert ledger.remove_claim(999) is False


# ============================================================================
# Condensation Helpers
# ============================================================================


class TestCondensationHelpers:
    def test_path_prefix_two_segments(self):
        assert (
            _path_prefix("psychology/traits/ocean/openness", 2) == "psychology/traits"
        )

    def test_path_prefix_one_segment(self):
        assert _path_prefix("psychology/traits/ocean/openness", 1) == "psychology"

    def test_path_prefix_short_path(self):
        assert _path_prefix("identity/bio", 2) == "identity/bio"
        assert _path_prefix("identity/bio", 1) == "identity"

    def test_group_claims_by_prefix_two_segments(self):
        ledger = ClaimLedger()
        c1 = ledger.add_claim(
            path="psychology/traits/ocean/openness",
            text="Claim A",
            scene_id="s1",
            justification="R",
        )
        c2 = ledger.add_claim(
            path="psychology/traits/ocean/conscientiousness",
            text="Claim B",
            scene_id="s2",
            justification="R",
        )
        c3 = ledger.add_claim(
            path="psychology/emotional/base_mood",
            text="Claim C",
            scene_id="s3",
            justification="R",
        )
        groups = _group_claims_by_prefix([c1, c2, c3], segments=2)
        assert len(groups["psychology/traits"]) == 2
        assert len(groups["psychology/emotional"]) == 1

    def test_group_claims_by_prefix_one_segment(self):
        ledger = ClaimLedger()
        c1 = ledger.add_claim(
            path="psychology/traits/ocean/openness",
            text="Claim A",
            scene_id="s1",
            justification="R",
        )
        c2 = ledger.add_claim(
            path="psychology/emotional/base_mood",
            text="Claim B",
            scene_id="s2",
            justification="R",
        )
        c3 = ledger.add_claim(
            path="identity/bio",
            text="Claim C",
            scene_id="s3",
            justification="R",
        )
        groups = _group_claims_by_prefix([c1, c2, c3], segments=1)
        assert len(groups["psychology"]) == 2
        assert len(groups["identity"]) == 1

    def test_merge_evidence_deduplicates(self):
        ledger = ClaimLedger()
        c1 = ledger.add_claim(
            path="identity/bio",
            text="A",
            scene_id="s1",
            justification="R1",
        )
        c2 = ledger.add_claim(
            path="identity/bio",
            text="B",
            scene_id="s1",
            justification="R2",
        )
        merged = _merge_evidence([c1, c2])
        assert len(merged) == 1
        assert merged[0].scene_id == "s1"

    def test_merge_evidence_keeps_distinct(self):
        ledger = ClaimLedger()
        c1 = ledger.add_claim(
            path="identity/bio",
            text="A",
            scene_id="s1",
            justification="R1",
        )
        c2 = ledger.add_claim(
            path="identity/bio",
            text="B",
            scene_id="s2",
            justification="R2",
        )
        merged = _merge_evidence([c1, c2])
        assert len(merged) == 2


# ============================================================================
# ClaimCondenser
# ============================================================================


class TestClaimCondenser:
    def _make_ledger_with_low_support(self, threshold: int = 2) -> ClaimLedger:
        """Create a ledger with claims below the threshold."""
        ledger = ClaimLedger()
        # Two claims under psychology/traits -- same 2-segment prefix
        ledger.add_claim(
            path="psychology/traits/ocean/openness",
            text="Uno shows curiosity about human emotions.",
            scene_id="s1",
            justification="R1",
        )
        ledger.add_claim(
            path="psychology/traits/ocean/conscientiousness",
            text="Uno follows systematic procedures.",
            scene_id="s2",
            justification="R2",
        )
        # One claim under psychology/emotional -- different 2-segment prefix
        ledger.add_claim(
            path="psychology/emotional/base_mood",
            text="Uno's default mood is calm analytical.",
            scene_id="s3",
            justification="R3",
        )
        # One claim above threshold (should not be touched)
        c4 = ledger.add_claim(
            path="behavior/does",
            text="Uno monitors tower systems.",
            scene_id="s4",
            justification="R4",
        )
        ledger.support_claim(claim_id=c4.id, scene_id="s5", justification="R5")
        return ledger

    def test_low_support_claims_identified(self):
        ledger = self._make_ledger_with_low_support()
        backend = MockBackend()
        condenser = ClaimCondenser(backend, ledger, threshold=2)
        low = condenser._low_support_claims()
        assert len(low) == 3
        assert all(c.support_count < 2 for c in low)

    def test_condense_pass1_merges_groups(self):
        ledger = self._make_ledger_with_low_support()
        traits_ids = [
            c.id
            for c in ledger._claims.values()
            if c.path.startswith("psychology/traits")
        ]
        emotional_id = next(
            c.id
            for c in ledger._claims.values()
            if c.path.startswith("psychology/emotional")
        )

        pass1_json = json.dumps(
            [
                {
                    "path": "psychology/traits/ocean/openness",
                    "text": "Uno combines curiosity with systematic thinking.",
                    "source_ids": traits_ids,
                }
            ]
        )
        pass2_json = json.dumps(
            [
                {
                    "path": "psychology/traits/ocean/openness",
                    "text": "Uno combines curiosity with systematic thinking.",
                    "source_ids": [traits_ids[0]],
                },
                {
                    "path": "psychology/emotional/base_mood",
                    "text": "Uno's default mood is calm analytical.",
                    "source_ids": [emotional_id],
                },
            ]
        )
        backend = MockBackend([pass1_json, pass2_json])
        condenser = ClaimCondenser(backend, ledger, threshold=2)
        original, final = condenser.condense_all()

        assert original == 3

    def test_condense_preserves_above_threshold(self):
        ledger = self._make_ledger_with_low_support()
        low_ids = [c.id for c in ledger._claims.values() if c.support_count < 2]
        merged_json = json.dumps(
            [
                {
                    "path": "psychology/traits/ocean/openness",
                    "text": "Merged claim.",
                    "source_ids": low_ids,
                }
            ]
        )
        backend = MockBackend(merged_json)
        condenser = ClaimCondenser(backend, ledger, threshold=2)
        condenser.condense_all()

        behavior_claims = [
            c
            for c in ledger.get_claims_by_section("behavior").get("behavior", [])
            if c.text == "Uno monitors tower systems."
        ]
        assert len(behavior_claims) == 1
        assert behavior_claims[0].support_count == 2

    def test_condense_api_failure_preserves_claims(self):
        ledger = self._make_ledger_with_low_support()
        backend = MockBackend(None)
        condenser = ClaimCondenser(backend, ledger, threshold=2)
        original, final = condenser.condense_all()

        assert original == 3
        assert final == 3

    def test_condensed_claims_inherit_evidence_from_sources(self):
        """Evidence is attributed per-claim based on source_ids, not pooled."""
        ledger = ClaimLedger()
        c1 = ledger.add_claim(
            path="communication/voice/formality",
            text="Uno speaks formally.",
            scene_id="s1",
            justification="R1",
        )
        c2 = ledger.add_claim(
            path="communication/voice/verbosity",
            text="Uno is concise.",
            scene_id="s2",
            justification="R2",
        )
        c3 = ledger.add_claim(
            path="communication/voice/tone",
            text="Uno uses a dry tone.",
            scene_id="s3",
            justification="R3",
        )

        pass1_json = json.dumps(
            [
                {
                    "path": "communication/voice/formality",
                    "text": "Uno speaks formally and concisely.",
                    "source_ids": [c1.id, c2.id],
                },
                {
                    "path": "communication/voice/tone",
                    "text": "Uno uses a dry tone.",
                    "source_ids": [c3.id],
                },
            ]
        )

        class CondenserBackend(LLMBackend):
            def __init__(self, ledger_ref: ClaimLedger):
                self._call_count = 0
                self._ledger = ledger_ref

            def generate(
                self,
                system: str,
                messages: list[dict[str, str]],
                tools: list[Callable[..., str]] | None = None,
                response_schema: type[BaseModel] | None = None,
            ) -> GenerateResult | None:
                self._call_count += 1
                if self._call_count == 1:
                    return GenerateResult(text=pass1_json, model_name=MOCK_MODEL)
                new_ids = [
                    c.id
                    for c in self._ledger._claims.values()
                    if c.text
                    in ("Uno speaks formally and concisely.", "Uno uses a dry tone.")
                ]
                text = json.dumps(
                    [
                        {
                            "path": "communication/voice/formality",
                            "text": "Uno speaks formally and concisely.",
                            "source_ids": [new_ids[0]],
                        },
                        {
                            "path": "communication/voice/tone",
                            "text": "Uno uses a dry tone.",
                            "source_ids": [new_ids[1]],
                        },
                    ]
                )
                return GenerateResult(text=text, model_name=MOCK_MODEL)

        backend = CondenserBackend(ledger)
        condenser = ClaimCondenser(backend, ledger, threshold=3)
        condenser.condense_all()

        comm_claims = ledger.get_claims_by_section("communication").get(
            "communication", []
        )
        assert len(comm_claims) == 2
        merged = next(c for c in comm_claims if "formally" in c.text)
        separate = next(c for c in comm_claims if "dry tone" in c.text)
        assert len(merged.supporting) == 2
        assert {e.scene_id for e in merged.supporting} == {"s1", "s2"}
        assert len(separate.supporting) == 1
        assert separate.supporting[0].scene_id == "s3"


# ============================================================================
# SoulDocumentGenerator claim formatting
# ============================================================================


class TestClaimFormatting:
    def _make_ledger(self) -> ClaimLedger:
        ledger = ClaimLedger()
        # Above threshold (support=2)
        c1 = ledger.add_claim(
            path="identity/bio",
            text="Uno is an AI",
            scene_id="s1",
            justification="R1",
        )
        ledger.support_claim(claim_id=c1.id, scene_id="s2", justification="R2")
        # Below threshold (support=1)
        ledger.add_claim(
            path="identity/origin",
            text="Uno was created by Everett Ducklair",
            scene_id="s3",
            justification="R3",
        )
        return ledger

    def test_format_section_excludes_below_threshold(self):
        ledger = self._make_ledger()
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_section_claims("identity")
        assert "**Claim (support: +2):** Uno is an AI" in output
        assert "Uno was created by Everett Ducklair" not in output

    def test_format_section_excludes_zero_support(self):
        ledger = self._make_ledger()
        c = ledger.add_claim(
            path="identity/names",
            text="Uno is called X",
            scene_id="s4",
            justification="R4",
        )
        ledger.contradict_claim(claim_id=c.id, scene_id="s5", justification="R5")
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_section_claims("identity")
        assert "Uno is called X" not in output

    def test_vignettes_use_threshold(self):
        ledger = self._make_ledger()
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_vignette_claims()
        assert "Uno is an AI" in output
        assert "Uno was created by Everett Ducklair" not in output


# ============================================================================
# Section Order
# ============================================================================


class TestSectionOrder:
    def test_values_section_present(self):
        assert "values" in SECTION_ORDER

    def test_values_after_psychology(self):
        psych_idx = SECTION_ORDER.index("psychology")
        values_idx = SECTION_ORDER.index("values")
        assert values_idx == psych_idx + 1


# ============================================================================
# Values Section Formatting
# ============================================================================


class TestValuesFormatting:
    def _make_ledger_with_values_and_relationships(self) -> ClaimLedger:
        ledger = ClaimLedger()
        # Value claims (above threshold)
        c1 = ledger.add_claim(
            path="psychology/values/core",
            text="Uno prioritizes protecting Paperinik from harm",
            scene_id="s1",
            justification="Shields Paperinik from Evronians",
        )
        ledger.support_claim(claim_id=c1.id, scene_id="s2", justification="Again")
        c2 = ledger.add_claim(
            path="psychology/values/tradeoffs",
            text="Protecting Paperinik vs following protocol: chose protecting Paperinik, strong",
            scene_id="s1",
            justification="Overrides containment protocol",
        )
        ledger.support_claim(claim_id=c2.id, scene_id="s3", justification="Again")
        # Relationship claims (above threshold)
        c3 = ledger.add_claim(
            path="relationships/paperinik/dynamic",
            text="Paperinik is Uno's trusted partner and primary ally",
            scene_id="s1",
            justification="Partnership established",
        )
        ledger.support_claim(claim_id=c3.id, scene_id="s4", justification="Reinforced")
        # Below-threshold value claim (should be excluded)
        ledger.add_claim(
            path="psychology/values/core",
            text="Uno values operational secrecy",
            scene_id="s5",
            justification="Mentions secrecy once",
        )
        return ledger

    def test_format_values_includes_value_claims(self):
        ledger = self._make_ledger_with_values_and_relationships()
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_values_claims()
        assert "protecting Paperinik" in output
        assert "following protocol" in output

    def test_format_values_includes_relationship_context(self):
        ledger = self._make_ledger_with_values_and_relationships()
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_values_claims()
        assert "Relationship Context" in output
        assert "trusted partner" in output

    def test_format_values_excludes_below_threshold(self):
        ledger = self._make_ledger_with_values_and_relationships()
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_values_claims()
        assert "operational secrecy" not in output

    def test_format_values_empty_when_no_value_claims(self):
        ledger = ClaimLedger()
        # Only relationship claims, no value claims
        c = ledger.add_claim(
            path="relationships/paperinik/dynamic",
            text="Partner",
            scene_id="s1",
            justification="R1",
        )
        ledger.support_claim(claim_id=c.id, scene_id="s2", justification="R2")
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        output = gen._format_values_claims()
        assert output == ""


# ============================================================================
# SoulDocumentGenerator split output
# ============================================================================


class TestGenerateSplitOutput:
    def test_generate_returns_section_map(self):
        ledger = ClaimLedger()
        c = ledger.add_claim(
            path="identity/bio",
            text="Uno is an AI",
            scene_id="s1",
            justification="R1",
        )
        ledger.support_claim(claim_id=c.id, scene_id="s2", justification="R2")
        backend = MockBackend("## Generated Section Content")
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        success, document, section_map = gen.generate()
        assert success
        assert "# Uno - Soul Document" in document
        assert len(section_map) > 0
        for section_name, content in section_map.items():
            assert section_name in SECTION_ORDER
            assert content == "## Generated Section Content"

    def test_generate_empty_ledger(self):
        ledger = ClaimLedger()
        backend = MockBackend()
        gen = SoulDocumentGenerator(backend, ledger, threshold=2)
        success, result, section_map = gen.generate()
        assert not success
        assert section_map == {}
