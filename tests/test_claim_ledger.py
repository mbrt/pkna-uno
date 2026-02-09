"""Unit tests for ClaimLedger and related classes."""

from unittest.mock import MagicMock, patch

import pytest

from build_claim_ledger_profile import (
    Claim,
    ClaimLedger,
    ClaimRefiner,
    Quote,
    Scene,
    SceneEvidence,
    format_claims_compact,
    format_claims_detail,
    format_scene_view,
)


class TestQuote:
    """Tests for Quote model."""

    def test_quote_creation(self):
        """Test creating a quote."""
        quote = Quote(
            text="Sai che dispiacere!",
            context="Responding to Paperinik's complaint",
            scene_id="pkna-0_12",
        )
        assert quote.text == "Sai che dispiacere!"
        assert quote.context == "Responding to Paperinik's complaint"
        assert quote.scene_id == "pkna-0_12"


class TestSceneEvidence:
    """Tests for SceneEvidence model."""

    def test_scene_evidence_creation(self):
        """Test creating scene evidence."""
        evidence = SceneEvidence(
            scene_id="pkna-1_45",
            justification="Shows protective behavior toward Paperinik",
        )
        assert evidence.scene_id == "pkna-1_45"
        assert evidence.justification == "Shows protective behavior toward Paperinik"


class TestClaim:
    """Tests for Claim model."""

    def test_claim_creation(self):
        """Test creating a claim."""
        claim = Claim(
            id=1,
            text="Uno uses sarcasm as a defense mechanism",
            section="personality",
        )
        assert claim.id == 1
        assert claim.text == "Uno uses sarcasm as a defense mechanism"
        assert claim.section == "personality"
        assert claim.supporting == []
        assert claim.contradicting == []
        assert claim.quotes == []

    def test_support_count_no_evidence(self):
        """Test support count with no evidence."""
        claim = Claim(id=1, text="Test claim", section="personality")
        assert claim.support_count == 0

    def test_support_count_only_supporting(self):
        """Test support count with only supporting evidence."""
        claim = Claim(
            id=1,
            text="Test claim",
            section="personality",
            supporting=[
                SceneEvidence(scene_id="pkna-0_1", justification="Reason 1"),
                SceneEvidence(scene_id="pkna-0_2", justification="Reason 2"),
                SceneEvidence(scene_id="pkna-1_5", justification="Reason 3"),
            ],
        )
        assert claim.support_count == 3

    def test_support_count_only_contradicting(self):
        """Test support count with only contradicting evidence."""
        claim = Claim(
            id=1,
            text="Test claim",
            section="personality",
            contradicting=[
                SceneEvidence(scene_id="pkna-0_1", justification="Contradiction 1"),
            ],
        )
        assert claim.support_count == -1

    def test_support_count_mixed(self):
        """Test support count with mixed evidence."""
        claim = Claim(
            id=1,
            text="Test claim",
            section="personality",
            supporting=[
                SceneEvidence(scene_id="pkna-0_1", justification="Support 1"),
                SceneEvidence(scene_id="pkna-0_2", justification="Support 2"),
                SceneEvidence(scene_id="pkna-1_5", justification="Support 3"),
            ],
            contradicting=[
                SceneEvidence(scene_id="pkna-2_10", justification="Contradiction 1"),
            ],
        )
        assert claim.support_count == 2  # 3 - 1 = 2

    def test_absorb_contradictions(self):
        """Test absorbing contradictions moves them to supporting."""
        claim = Claim(
            id=1,
            text="Test claim",
            section="personality",
            supporting=[
                SceneEvidence(scene_id="pkna-0_1", justification="Support 1"),
            ],
            contradicting=[
                SceneEvidence(scene_id="pkna-1_5", justification="Contradiction 1"),
                SceneEvidence(scene_id="pkna-2_10", justification="Contradiction 2"),
            ],
        )

        moved = claim.absorb_contradictions()

        assert moved == 2
        assert len(claim.supporting) == 3
        assert len(claim.contradicting) == 0
        assert claim.support_count == 3
        # Verify the moved entries are present
        scene_ids = [ev.scene_id for ev in claim.supporting]
        assert "pkna-1_5" in scene_ids
        assert "pkna-2_10" in scene_ids

    def test_absorb_contradictions_empty(self):
        """Test absorbing contradictions is a no-op when there are none."""
        claim = Claim(
            id=1,
            text="Test claim",
            section="personality",
            supporting=[
                SceneEvidence(scene_id="pkna-0_1", justification="Support 1"),
            ],
        )

        moved = claim.absorb_contradictions()

        assert moved == 0
        assert len(claim.supporting) == 1
        assert len(claim.contradicting) == 0


class TestScene:
    """Tests for Scene dataclass."""

    def test_scene_id(self):
        """Test scene_id property."""
        scene = Scene(
            issue="pkna-3",
            page_numbers=[45, 46, 47],
            summary="Test summary",
            uno_dialogues=["Ciao!"],
            panel_descriptions=["Panel 1"],
            other_characters={"Paperinik"},
        )
        assert scene.scene_id == "pkna-3_45"

    def test_scene_to_dict_and_back(self):
        """Test serialization round-trip."""
        scene = Scene(
            issue="pkna-2",
            page_numbers=[10, 11],
            summary="Summary here",
            uno_dialogues=["Dialogue 1", "Dialogue 2"],
            panel_descriptions=["Panel A", "Panel B"],
            other_characters={"Paperinik", "Everett"},
        )

        data = scene.to_dict()
        restored = Scene.from_dict(data)

        assert restored.issue == scene.issue
        assert restored.page_numbers == scene.page_numbers
        assert restored.summary == scene.summary
        assert restored.uno_dialogues == scene.uno_dialogues
        assert restored.panel_descriptions == scene.panel_descriptions
        assert restored.other_characters == scene.other_characters


class TestClaimLedger:
    """Tests for ClaimLedger class."""

    def test_empty_ledger(self):
        """Test empty ledger initialization."""
        ledger = ClaimLedger()
        assert ledger.claim_count() == 0
        assert ledger.scene_count() == 0

    def test_add_claim(self):
        """Test adding a claim."""
        ledger = ClaimLedger()
        claim = ledger.add_claim(
            section="personality",
            text="Uno is sarcastic",
            scene_id="pkna-0_12",
            justification="Uses ironic responses",
        )

        assert claim.id == 1
        assert claim.text == "Uno is sarcastic"
        assert claim.section == "personality"
        assert len(claim.supporting) == 1
        assert claim.supporting[0].scene_id == "pkna-0_12"
        assert ledger.claim_count() == 1

    def test_add_claim_with_quote(self):
        """Test adding a claim with a quote."""
        ledger = ClaimLedger()
        claim = ledger.add_claim(
            section="communication",
            text="Uno uses 'Sai che dispiacere!' sarcastically",
            scene_id="pkna-0_12",
            justification="Catchphrase used multiple times",
            quote="Sai che dispiacere!",
            quote_context="When Paperinik complains about early wake-up",
        )

        assert len(claim.quotes) == 1
        assert claim.quotes[0].text == "Sai che dispiacere!"

    def test_add_multiple_claims(self):
        """Test ID increments correctly."""
        ledger = ClaimLedger()
        claim1 = ledger.add_claim(
            section="identity",
            text="Claim 1",
            scene_id="pkna-0_1",
            justification="Reason 1",
        )
        claim2 = ledger.add_claim(
            section="personality",
            text="Claim 2",
            scene_id="pkna-0_2",
            justification="Reason 2",
        )

        assert claim1.id == 1
        assert claim2.id == 2
        assert ledger.claim_count() == 2

    def test_support_claim_success(self):
        """Test supporting an existing claim."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
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
        claim = ledger.get_claim(1)
        assert claim is not None
        assert len(claim.supporting) == 2

    def test_support_claim_with_quote(self):
        """Test supporting a claim adds all quotes."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Test claim",
            scene_id="pkna-0_1",
            justification="Initial",
        )

        # Add quotes through support - all should be kept
        for i in range(4):
            ledger.support_claim(
                claim_id=1,
                scene_id=f"pkna-{i}_10",
                justification=f"Support {i}",
                quote=f"Quote {i}",
                quote_context=f"Context {i}",
            )

        claim = ledger.get_claim(1)
        assert claim is not None
        assert len(claim.quotes) == 4  # All quotes kept

    def test_support_claim_not_found(self):
        """Test supporting non-existent claim fails."""
        ledger = ClaimLedger()
        success, message = ledger.support_claim(
            claim_id=999,
            scene_id="pkna-0_1",
            justification="Test",
        )
        assert success is False
        assert "not found" in message

    def test_support_claim_duplicate_scene(self):
        """Test same scene cannot support twice."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Test",
            scene_id="pkna-0_12",
            justification="Initial",
        )

        # First support succeeds
        success1, _ = ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="First",
        )
        assert success1 is True

        # Same scene fails
        success2, message = ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="Duplicate",
        )
        assert success2 is False
        assert "already supports" in message

    def test_contradict_claim_success(self):
        """Test contradicting a claim."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
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
        assert claim.support_count == 0  # 1 - 1

    def test_contradict_claim_duplicate_scene(self):
        """Test same scene cannot contradict twice."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Test",
            scene_id="pkna-0_12",
            justification="Initial",
        )

        success1, _ = ledger.contradict_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="First",
        )
        assert success1 is True

        success2, message = ledger.contradict_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="Duplicate",
        )
        assert success2 is False
        assert "already contradicts" in message

    def test_refine_claim_success(self):
        """Test refining a claim."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Uno is sarcastic",
            scene_id="pkna-0_12",
            justification="Initial",
        )

        success, _ = ledger.refine_claim(
            claim_id=1,
            new_text="Uno uses sarcasm as a defense mechanism",
        )

        assert success is True
        claim = ledger.get_claim(1)
        assert claim is not None
        assert claim.text == "Uno uses sarcasm as a defense mechanism"
        # Evidence preserved
        assert len(claim.supporting) == 1

    def test_refine_claim_not_found(self):
        """Test refining non-existent claim fails."""
        ledger = ClaimLedger()
        success, message = ledger.refine_claim(claim_id=999, new_text="New text")
        assert success is False
        assert "not found" in message

    def test_scene_tracking(self):
        """Test scene processing tracking."""
        ledger = ClaimLedger()
        scene = Scene(
            issue="pkna-0",
            page_numbers=[12, 13],
            summary="Test scene",
            uno_dialogues=["Hello"],
            panel_descriptions=["Panel"],
            other_characters={"Paperinik"},
        )

        assert not ledger.is_scene_processed(scene.scene_id)
        ledger.add_scene(scene)
        assert ledger.is_scene_processed(scene.scene_id)
        assert ledger.scene_count() == 1

        retrieved = ledger.get_scene(scene.scene_id)
        assert retrieved is not None
        assert retrieved.issue == "pkna-0"

    def test_populate_scene_cache(self):
        """Test populating scene cache after loading from checkpoint."""
        # Create ledger and process a scene
        ledger = ClaimLedger()
        scene1 = Scene(
            issue="pkna-0",
            page_numbers=[12],
            summary="Scene 1",
            uno_dialogues=["Hello"],
            panel_descriptions=["Panel"],
            other_characters=set(),
        )
        scene2 = Scene(
            issue="pkna-1",
            page_numbers=[5],
            summary="Scene 2",
            uno_dialogues=["World"],
            panel_descriptions=["Panel 2"],
            other_characters=set(),
        )
        ledger.add_scene(scene1)

        # Simulate checkpoint save/load (only IDs preserved)
        data = ledger.to_json()
        restored = ClaimLedger.from_json(data)

        # Cache is empty after load
        assert restored.get_scene("pkna-0_12") is None

        # Populate cache with all scenes
        restored.populate_scene_cache([scene1, scene2])

        # Now processed scene is available
        retrieved = restored.get_scene("pkna-0_12")
        assert retrieved is not None
        assert retrieved.summary == "Scene 1"

        # Unprocessed scene is not cached
        assert restored.get_scene("pkna-1_5") is None

    def test_get_claims_by_section(self):
        """Test grouping claims by section."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Trait 1",
            scene_id="pkna-0_1",
            justification="R1",
        )
        ledger.add_claim(
            section="personality",
            text="Trait 2",
            scene_id="pkna-0_2",
            justification="R2",
        )
        ledger.add_claim(
            section="identity",
            text="Fact 1",
            scene_id="pkna-0_3",
            justification="R3",
        )

        by_section = ledger.get_claims_by_section()
        assert len(by_section["personality"]) == 2
        assert len(by_section["identity"]) == 1
        assert "communication" not in by_section  # Empty sections not included

    def test_get_claims_by_section_filtered(self):
        """Test filtering claims by section."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Trait",
            scene_id="pkna-0_1",
            justification="R1",
        )
        ledger.add_claim(
            section="identity",
            text="Fact",
            scene_id="pkna-0_2",
            justification="R2",
        )

        by_section = ledger.get_claims_by_section("personality")
        assert len(by_section["personality"]) == 1
        assert "identity" not in by_section  # Filtered out

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Uno is sarcastic",
            scene_id="pkna-0_12",
            justification="Uses ironic responses",
            quote="Sai che dispiacere!",
            quote_context="When Paperinik complains",
        )
        ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="More sarcasm",
        )

        scene = Scene(
            issue="pkna-0",
            page_numbers=[12],
            summary="Test",
            uno_dialogues=["Hello"],
            panel_descriptions=["Panel"],
            other_characters=set(),
        )
        ledger.add_scene(scene)

        # Serialize
        data = ledger.to_json()

        # Deserialize
        restored = ClaimLedger.from_json(data)

        assert restored.claim_count() == 1
        assert restored.scene_count() == 1  # Scene IDs are preserved

        claim = restored.get_claim(1)
        assert claim is not None
        assert claim.text == "Uno is sarcastic"
        assert len(claim.supporting) == 2
        assert len(claim.quotes) == 1

        # Scene cache is NOT serialized - only IDs are preserved for resume
        assert restored.is_scene_processed("pkna-0_12")
        assert (
            restored.get_scene("pkna-0_12") is None
        )  # Cache is empty after deserialize


class TestFormatFunctions:
    """Tests for formatting functions."""

    @pytest.fixture
    def sample_ledger(self):
        """Create a sample ledger with claims."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Uno uses sarcasm as a defense mechanism",
            scene_id="pkna-0_12",
            justification="Deflects with irony",
            quote="Sai che dispiacere!",
            quote_context="When Paperinik complains",
        )
        ledger.support_claim(
            claim_id=1,
            scene_id="pkna-1_5",
            justification="More sarcasm",
        )
        ledger.add_claim(
            section="identity",
            text="Uno is an AI living in Ducklair Tower",
            scene_id="pkna-0_1",
            justification="Introduced as such",
        )
        return ledger

    def test_format_claims_compact(self, sample_ledger):
        """Test compact formatting."""
        output = format_claims_compact(sample_ledger)

        assert "Total claims: 2" in output
        assert "## Identity" in output
        assert "## Personality" in output
        assert "[+2]" in output  # Claim 1 has support 2
        assert "[+1]" in output  # Claim 2 has support 1

    def test_format_claims_compact_filtered(self, sample_ledger):
        """Test compact formatting with section filter."""
        output = format_claims_compact(sample_ledger, section="personality")

        assert "## Personality" in output
        assert "## Identity" not in output

    def test_format_claims_detail(self, sample_ledger):
        """Test detailed formatting."""
        output = format_claims_detail(sample_ledger, [1])

        assert "ID 1:" in output
        assert "Section: personality" in output
        assert "Uno uses sarcasm" in output
        assert "Supporting:" in output
        assert "pkna-0_12:" in output
        assert "Quotes:" in output
        assert "Sai che dispiacere!" in output

    def test_format_claims_detail_not_found(self, sample_ledger):
        """Test detailed formatting with invalid ID."""
        output = format_claims_detail(sample_ledger, [999])
        assert "ID 999: Not found" in output

    def test_format_scene_view(self):
        """Test scene view formatting."""
        scene = Scene(
            issue="pkna-0",
            page_numbers=[12, 13, 14],
            summary="Uno meets Paperinik for the first time",
            uno_dialogues=["Benvenuto!", "Io sono Uno."],
            panel_descriptions=[
                "Panel showing Uno's screen",
                "Paperinik looks surprised",
            ],
            other_characters={"Paperinik", "Everett"},
        )

        output = format_scene_view(scene)

        assert "Scene: pkna-0_12" in output
        assert "Issue: pkna-0, pages 12-13-14" in output
        assert "Characters present: Everett, Paperinik" in output
        assert "Uno meets Paperinik" in output
        assert "Benvenuto!" in output
        assert "Io sono Uno." in output
        assert "Panel showing" in output


class TestClaimRefiner:
    """Tests for ClaimRefiner class."""

    def _make_ledger_with_contradictions(self) -> ClaimLedger:
        """Create a ledger with some contradicted claims."""
        ledger = ClaimLedger()
        # Claim 1: has contradictions
        ledger.add_claim(
            section="personality",
            text="Uno is always sarcastic",
            scene_id="pkna-0_12",
            justification="Uses ironic deflection",
        )
        ledger.contradict_claim(
            claim_id=1,
            scene_id="pkna-5_30",
            justification="Speaks sincerely when danger is present",
        )
        # Claim 2: no contradictions
        ledger.add_claim(
            section="identity",
            text="Uno is an AI",
            scene_id="pkna-0_1",
            justification="Introduced as artificial intelligence",
        )
        # Claim 3: has contradictions
        ledger.add_claim(
            section="behavior",
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
        """Test finding claims with contradictions."""
        ledger = self._make_ledger_with_contradictions()
        client = MagicMock()
        refiner = ClaimRefiner(client, ledger)

        contradicted = refiner._find_contradicted_claims()

        assert len(contradicted) == 2
        ids = {c.id for c in contradicted}
        assert ids == {1, 3}

    def test_find_contradicted_claims_none(self):
        """Test finding contradicted claims when there are none."""
        ledger = ClaimLedger()
        ledger.add_claim(
            section="personality",
            text="Claim without contradictions",
            scene_id="pkna-0_1",
            justification="Reason",
        )
        client = MagicMock()
        refiner = ClaimRefiner(client, ledger)

        contradicted = refiner._find_contradicted_claims()

        assert contradicted == []

    def test_refine_all_success(self):
        """Test refining all contradicted claims successfully."""
        ledger = self._make_ledger_with_contradictions()
        client = MagicMock()

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Uno defaults to sarcasm but speaks sincerely in danger."

        with patch(
            "build_claim_ledger_profile.generate_with_retry",
            return_value=mock_response,
        ):
            refiner = ClaimRefiner(client, ledger)
            refined_count, failed_count = refiner.refine_all()

        assert refined_count == 2
        assert failed_count == 0

        # Check claim 1 was refined and contradictions absorbed
        claim1 = ledger.get_claim(1)
        assert claim1 is not None
        assert claim1.text == "Uno defaults to sarcasm but speaks sincerely in danger."
        assert len(claim1.contradicting) == 0
        assert len(claim1.supporting) == 2  # original + absorbed contradiction

        # Claim 2 (no contradictions) should be unchanged
        claim2 = ledger.get_claim(2)
        assert claim2 is not None
        assert claim2.text == "Uno is an AI"

    def test_refine_all_api_failure(self):
        """Test refining when API calls fail."""
        ledger = self._make_ledger_with_contradictions()
        client = MagicMock()

        with patch(
            "build_claim_ledger_profile.generate_with_retry",
            return_value=None,
        ):
            refiner = ClaimRefiner(client, ledger)
            refined_count, failed_count = refiner.refine_all()

        assert refined_count == 0
        assert failed_count == 2

        # Claims should be unchanged
        claim1 = ledger.get_claim(1)
        assert claim1 is not None
        assert claim1.text == "Uno is always sarcastic"
        assert len(claim1.contradicting) == 1

    def test_refine_all_empty_response(self):
        """Test refining when LLM returns empty text."""
        ledger = self._make_ledger_with_contradictions()
        client = MagicMock()

        mock_response = MagicMock()
        mock_response.text = ""

        with patch(
            "build_claim_ledger_profile.generate_with_retry",
            return_value=mock_response,
        ):
            refiner = ClaimRefiner(client, ledger)
            refined_count, failed_count = refiner.refine_all()

        assert refined_count == 0
        assert failed_count == 2
