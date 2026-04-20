"""Unit tests for claim-derived prompt generation."""

import json
from pathlib import Path

import pytest

from datagen.generate_claim_prompts import (
    Claim,
    compute_trace_weights,
    generate_claim_prompts,
    load_ledger,
    _generate_emotional_trigger,
    _generate_identity_grounding,
    _generate_register_shift,
    _generate_theory_of_mind,
    _generate_value_priority,
)


@pytest.fixture
def sample_ledger(tmp_path: Path) -> Path:
    """Create a minimal ledger file for testing."""
    ledger = {
        "next_id": 100,
        "claims": {
            "1": {
                "id": 1,
                "text": "Sarcasm vs genuine care: chooses persona maintenance",
                "path": "psychology/values/tradeoffs",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(50)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "2": {
                "id": 2,
                "text": "Composure cracks under threat to PK",
                "path": "psychology/emotional/composure",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(70)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "3": {
                "id": 3,
                "text": "Humor as dominance assertion",
                "path": "communication/humor/dominance",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(45)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "4": {
                "id": 4,
                "text": "Evolution from authority to collaborative peer",
                "path": "relationships/paperinik/core_dynamic",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(192)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "5": {
                "id": 5,
                "text": "Suppressed melancholic self-awareness",
                "path": "psychology/self_model/existential",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(37)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "6": {
                "id": 6,
                "text": "Apologize before acting",
                "path": "psychology/moral_compass/resolution",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(7)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "7": {
                "id": 7,
                "text": "Loyalty mixed with grief at abandonment",
                "path": "relationships/ducklair/core",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(21)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "8": {
                "id": 8,
                "text": "Sibling entity, kinship vs hostility",
                "path": "relationships/due/core",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(14)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "9": {
                "id": 9,
                "text": "Self-presentation as all-seeing intelligence",
                "path": "identity/core",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(5)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "10": {
                "id": 10,
                "text": "Low support tradeoff",
                "path": "psychology/values/tradeoffs",
                "scope": "general",
                "supporting": [{"scene_id": "s1", "justification": "x"}],
                "contradicting": [],
                "quotes": [],
            },
            "11": {
                "id": 11,
                "text": "Mid support tradeoff",
                "path": "psychology/values/tradeoffs",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(15)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "12": {
                "id": 12,
                "text": "Behavior claim (should be excluded)",
                "path": "behavior/adaptation/by_situation",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(30)
                ],
                "contradicting": [],
                "quotes": [],
            },
            "13": {
                "id": 13,
                "text": "Voice register claim",
                "path": "communication/voice/register",
                "scope": "general",
                "supporting": [
                    {"scene_id": f"s{i}", "justification": "x"} for i in range(8)
                ],
                "contradicting": [],
                "quotes": [],
            },
        },
        "processed_scene_ids": [],
        "meta": {},
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path


class TestLoadLedger:
    def test_loads_all_claims(self, sample_ledger: Path):
        claims = load_ledger(sample_ledger)
        assert len(claims) == 13

    def test_computes_support_correctly(self, sample_ledger: Path):
        claims = load_ledger(sample_ledger)
        by_id = {c.id: c for c in claims}
        assert by_id[1].support == 50
        assert by_id[2].support == 70
        assert by_id[4].support == 192

    def test_claim_fields(self, sample_ledger: Path):
        claims = load_ledger(sample_ledger)
        by_id = {c.id: c for c in claims}
        assert by_id[1].path == "psychology/values/tradeoffs"
        assert "Sarcasm" in by_id[1].text


class TestComputeTraceWeights:
    def test_top_claims_get_3(self):
        claims = [Claim(id=i, text="", path="x", support=100 - i) for i in range(100)]
        weights = compute_trace_weights(claims)
        # Top 5% = top 5 claims (ids 0-4)
        for i in range(5):
            assert weights[i] == 3

    def test_mid_claims_get_2(self):
        claims = [Claim(id=i, text="", path="x", support=100 - i) for i in range(100)]
        weights = compute_trace_weights(claims)
        # Next 25% after top 5% = indices 5 to 29
        for i in range(5, 30):
            assert weights[i] == 2

    def test_remaining_claims_get_1(self):
        claims = [Claim(id=i, text="", path="x", support=100 - i) for i in range(100)]
        weights = compute_trace_weights(claims)
        for i in range(30, 100):
            assert weights[i] == 1

    def test_all_claims_assigned(self):
        claims = [Claim(id=i, text="", path="x", support=i) for i in range(50)]
        weights = compute_trace_weights(claims)
        assert len(weights) == 50

    def test_single_claim_gets_3(self):
        claims = [Claim(id=1, text="", path="x", support=10)]
        weights = compute_trace_weights(claims)
        assert weights[1] == 3

    def test_custom_percentiles(self):
        claims = [Claim(id=i, text="", path="x", support=100 - i) for i in range(100)]
        weights = compute_trace_weights(claims, top_pct=0.10, mid_pct=0.40)
        top_3 = [cid for cid, w in weights.items() if w == 3]
        mid_2 = [cid for cid, w in weights.items() if w == 2]
        assert len(top_3) == 10
        assert len(mid_2) == 40


@pytest.fixture
def sample_claims(sample_ledger: Path) -> list[Claim]:
    return load_ledger(sample_ledger)


@pytest.fixture
def sample_weights(sample_claims: list[Claim]) -> dict[int, int]:
    return compute_trace_weights(sample_claims)


class TestValuePriority:
    def test_includes_tradeoff_claims(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_value_priority(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 1 in claim_ids  # tradeoff claim
        assert 10 in claim_ids  # low-support tradeoff
        assert 11 in claim_ids  # mid-support tradeoff

    def test_includes_moral_compass(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_value_priority(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 6 in claim_ids

    def test_excludes_non_tradeoff(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_value_priority(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 2 not in claim_ids  # emotional
        assert 12 not in claim_ids  # behavior

    def test_high_support_gets_more_traces(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_value_priority(sample_claims, sample_weights, seed=42)
        claim1_prompts = [p for p in prompts if p.metadata["claim_id"] == 1]
        claim10_prompts = [p for p in prompts if p.metadata["claim_id"] == 10]
        assert len(claim1_prompts) > len(claim10_prompts)

    def test_metadata_format(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_value_priority(sample_claims, sample_weights, seed=42)
        for p in prompts:
            assert p.metadata["prompt_source"] == "claim_derived"
            assert p.metadata["category"] == "value_priority"
            assert "claim_text" in p.metadata
            assert "trace_guidance" in p.metadata


class TestEmotionalTrigger:
    def test_includes_emotional_claims(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_emotional_trigger(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 2 in claim_ids
        assert len(prompts) >= 1

    def test_excludes_non_emotional(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_emotional_trigger(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 1 not in claim_ids  # tradeoff


class TestRegisterShift:
    def test_produces_pairs(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_register_shift(sample_claims, sample_weights, seed=42)
        registers = {p.metadata["register_context"] for p in prompts}
        assert registers == {"calm", "crisis"}

    def test_each_variant_has_both_registers(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_register_shift(sample_claims, sample_weights, seed=42)
        assert len(prompts) % 2 == 0  # always paired

    def test_caps_per_subpath(self):
        """Test that each sub-path is capped at 5 claims."""
        claims = [
            Claim(
                id=i,
                text=f"humor claim {i}",
                path="communication/humor/test",
                support=100 - i,
            )
            for i in range(10)
        ]
        weights = compute_trace_weights(claims)
        prompts = _generate_register_shift(claims, weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert len(claim_ids) == 5


class TestTheoryOfMind:
    def test_includes_only_allowed_characters(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_theory_of_mind(sample_claims, sample_weights, seed=42)
        paths = {p.metadata["claim_path"] for p in prompts}
        for path in paths:
            char = path.split("/")[1]
            assert char in {
                "paperinik",
                "paperino",
                "ducklair",
                "due",
                "lyla",
                "xadhoom",
            }

    def test_pk_merging(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_theory_of_mind(sample_claims, sample_weights, seed=42)
        pk_prompts = [p for p in prompts if "paperinik" in p.metadata["claim_path"]]
        assert len(pk_prompts) >= 1

    def test_user_summary_matches_character(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_theory_of_mind(sample_claims, sample_weights, seed=42)
        for p in prompts:
            path = p.metadata["claim_path"]
            if "ducklair" in path:
                assert "Ducklair" in p.user_summary
            elif "due" in path:
                assert "Due" in p.user_summary
            elif "paperinik" in path or "paperino" in path:
                assert "Paperino" in p.user_summary


class TestIdentityGrounding:
    def test_includes_identity_and_self_model(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_identity_grounding(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 5 in claim_ids  # self_model
        assert 9 in claim_ids  # identity

    def test_excludes_other_types(
        self, sample_claims: list[Claim], sample_weights: dict[int, int]
    ):
        prompts = _generate_identity_grounding(sample_claims, sample_weights, seed=42)
        claim_ids = {p.metadata["claim_id"] for p in prompts}
        assert 1 not in claim_ids  # tradeoff
        assert 4 not in claim_ids  # relationship


class TestGenerateClaimPrompts:
    def test_produces_all_categories(self, sample_ledger: Path):
        prompts = generate_claim_prompts(sample_ledger)
        categories = {p.metadata["category"] for p in prompts}
        assert categories == {
            "value_priority",
            "emotional_trigger",
            "register_shift",
            "theory_of_mind",
            "identity_grounding",
        }

    def test_all_ids_unique(self, sample_ledger: Path):
        prompts = generate_claim_prompts(sample_ledger)
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids))

    def test_all_prompts_have_claim_metadata(self, sample_ledger: Path):
        prompts = generate_claim_prompts(sample_ledger)
        for p in prompts:
            assert p.metadata["prompt_source"] == "claim_derived"
            assert "claim_id" in p.metadata
            assert "claim_text" in p.metadata
            assert "claim_support" in p.metadata
            assert "trace_guidance" in p.metadata

    def test_deterministic(self, sample_ledger: Path):
        a = generate_claim_prompts(sample_ledger, seed=42)
        b = generate_claim_prompts(sample_ledger, seed=42)
        assert [p.id for p in a] == [p.id for p in b]

    def test_excludes_behavior_and_capabilities(self, sample_ledger: Path):
        prompts = generate_claim_prompts(sample_ledger)
        for p in prompts:
            path = p.metadata["claim_path"]
            assert not path.startswith("behavior/")
            assert not path.startswith("capabilities/")
