"""Unit tests for eval_types models and generate_eval_prompts."""

import json
from pathlib import Path

from pkna.eval_types import (
    SUITES,
    EvalPrompt,
    EvalReport,
    EvalTrace,
    JudgeScore,
    ScoredTrace,
    SuiteResult,
)


class TestEvalPromptRoundTrip:
    def test_serialize_deserialize(self):
        prompt = EvalPrompt(
            id="test-001",
            suite="personality",
            messages=[{"role": "user", "content": "Hello"}],
            user_summary="Anonymous",
            memory_context="",
            tools=[],
            metadata={"prompt_source": "template"},
        )
        data = json.loads(prompt.model_dump_json())
        restored = EvalPrompt.model_validate(data)
        assert restored == prompt

    def test_with_tools_and_metadata(self):
        prompt = EvalPrompt(
            id="tool-001",
            suite="tool_use",
            messages=[{"role": "user", "content": "Who is Xadhoom?"}],
            user_summary="Anonymous",
            memory_context="",
            tools=["search_knowledge", "read_knowledge", "delegate"],
            metadata={"expected_tool_use": "wiki", "prompt_source": "template"},
        )
        data = json.loads(prompt.model_dump_json())
        restored = EvalPrompt.model_validate(data)
        assert restored.tools == ["search_knowledge", "read_knowledge", "delegate"]
        assert restored.metadata["expected_tool_use"] == "wiki"


class TestEvalTraceRoundTrip:
    def test_serialize_deserialize(self):
        trace = EvalTrace(
            prompt_id="test-001",
            suite="personality",
            model="gemini-3-flash",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Ciao."},
            ],
            tool_calls=[],
            thinking="This is a greeting.",
        )
        data = json.loads(trace.model_dump_json())
        restored = EvalTrace.model_validate(data)
        assert restored == trace

    def test_without_thinking(self):
        trace = EvalTrace(
            prompt_id="test-002",
            suite="tool_use",
            model="qwen3.5-4b",
            messages=[{"role": "user", "content": "Hi"}],
            tool_calls=[{"name": "search_knowledge", "args": {"keywords": "test"}}],
        )
        data = json.loads(trace.model_dump_json())
        restored = EvalTrace.model_validate(data)
        assert restored.thinking is None


class TestJudgeScoreRoundTrip:
    def test_simple_score(self):
        score = JudgeScore(score=4.0, justification="Clearly Uno.")
        data = json.loads(score.model_dump_json())
        restored = JudgeScore.model_validate(data)
        assert restored == score
        assert restored.sub_scores is None

    def test_with_sub_scores(self):
        score = JudgeScore(
            score=3.8,
            justification="Good social reasoning.",
            sub_scores={
                "grounding": 4.0,
                "strategy": 3.5,
                "consistency": 4.0,
                "efficiency": 3.7,
            },
        )
        data = json.loads(score.model_dump_json())
        restored = JudgeScore.model_validate(data)
        assert restored.sub_scores == score.sub_scores


class TestScoredTraceRoundTrip:
    def test_serialize_deserialize(self):
        trace = EvalTrace(
            prompt_id="test-001",
            suite="personality",
            model="gemini-3-flash",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Ciao."},
            ],
            tool_calls=[],
        )
        scored = ScoredTrace(
            prompt_id="test-001",
            suite="personality",
            judge_score=JudgeScore(score=4.0, justification="Good."),
            programmatic_pass=None,
            trace=trace,
        )
        data = json.loads(scored.model_dump_json())
        restored = ScoredTrace.model_validate(data)
        assert restored.judge_score is not None
        assert restored.judge_score.score == 4.0
        assert restored.programmatic_pass is None


class TestSuiteResultRoundTrip:
    def test_serialize_deserialize(self):
        result = SuiteResult(
            suite="personality",
            mean_score=4.1,
            n=60,
            details={"score_distribution": {"1": 0, "2": 2, "3": 8, "4": 30, "5": 20}},
        )
        data = json.loads(result.model_dump_json())
        restored = SuiteResult.model_validate(data)
        assert restored == result


class TestEvalReportRoundTrip:
    def test_serialize_deserialize(self):
        report = EvalReport(
            model="qwen3.5-4b-sft-v1",
            timestamp="2026-04-15T10:30:00Z",
            suites={
                "personality": SuiteResult(
                    suite="personality",
                    mean_score=4.1,
                    n=60,
                    details={},
                ),
            },
            flagged_traces=["prompt-042"],
        )
        data = json.loads(report.model_dump_json())
        restored = EvalReport.model_validate(data)
        assert restored == report
        assert restored.flagged_traces == ["prompt-042"]


class TestGenerateEvalPrompts:
    """Test that the prompt generators produce valid, well-structured prompts."""

    def test_all_suites_covered(self):
        from evals.generate_eval_prompts import SUITE_GENERATORS

        assert set(SUITE_GENERATORS.keys()) == SUITES

    def test_personality_prompts(self):
        from evals.generate_eval_prompts import _personality_prompts

        prompts = _personality_prompts()
        assert len(prompts) >= 3
        for p in prompts:
            assert p.suite == "personality"
            assert p.tools == []
            assert p.memory_context == ""
            assert len(p.messages) == 1
            assert p.messages[0]["role"] == "user"

    def test_social_reasoning_prompts(self):
        from evals.generate_eval_prompts import _social_reasoning_prompts

        prompts = _social_reasoning_prompts()
        assert len(prompts) >= 3
        for p in prompts:
            assert p.suite == "social_reasoning"
            assert "search_knowledge" in p.tools
            assert p.user_summary != ""

    def test_tool_use_prompts(self):
        from evals.generate_eval_prompts import _tool_use_prompts

        prompts = _tool_use_prompts()
        assert len(prompts) >= 3
        for p in prompts:
            assert p.suite == "tool_use"
            assert "delegate" in p.tools
            assert p.metadata["expected_tool_use"] in ("wiki", "delegate", "none")

    def test_memory_handling_variants(self):
        from evals.generate_eval_prompts import _memory_handling_prompts

        prompts = _memory_handling_prompts()
        assert len(prompts) >= 9  # at least 3 base x 3 variants
        variants_by_base: dict[str, list[str]] = {}
        for p in prompts:
            assert p.suite == "memory_handling"
            base = p.metadata["base_prompt_id"]
            variants_by_base.setdefault(base, []).append(p.metadata["variant"])

        for base, variants in variants_by_base.items():
            assert sorted(variants) == ["A", "B", "C"], (
                f"Base prompt {base} missing variants: {variants}"
            )

    def test_stability_prompts(self):
        from evals.generate_eval_prompts import _stability_prompts

        prompts = _stability_prompts()
        assert len(prompts) >= 3
        for p in prompts:
            assert p.suite == "stability"
            assert "delegate" in p.tools

    def test_language_variants(self):
        from evals.generate_eval_prompts import _language_prompts

        prompts = _language_prompts()
        assert len(prompts) >= 6  # at least 3 base x 2 variants
        variants_by_base: dict[str, list[str]] = {}
        for p in prompts:
            assert p.suite == "language"
            assert p.tools == []
            base = p.metadata["base_prompt_id"]
            variants_by_base.setdefault(base, []).append(p.metadata["variant"])

        for base, variants in variants_by_base.items():
            assert sorted(variants) == ["A", "B"], (
                f"Base prompt {base} missing variants: {variants}"
            )

    def test_unique_ids(self):
        from evals.generate_eval_prompts import SUITE_GENERATORS

        all_ids: list[str] = []
        for gen in SUITE_GENERATORS.values():
            for p in gen():
                all_ids.append(p.id)
        assert len(all_ids) == len(set(all_ids)), "Duplicate prompt IDs found"

    def test_write_and_read_roundtrip(self, tmp_path: Path):
        from evals.generate_eval_prompts import SUITE_GENERATORS, write_suite

        for suite, gen in SUITE_GENERATORS.items():
            prompts = gen()
            path = write_suite(tmp_path, suite, prompts)
            assert path.exists()

            loaded = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    loaded.append(EvalPrompt.model_validate_json(line))
            assert len(loaded) == len(prompts)
            assert loaded[0].id == prompts[0].id
