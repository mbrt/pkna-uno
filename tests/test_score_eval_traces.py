"""Unit tests for the eval trace scorer (stage 3)."""

import json
from pathlib import Path
from typing import Any

from evals.score_eval_traces import (
    RubricScore,
    SocialReasoningScore,
    _agg_language,
    _agg_memory_handling,
    _agg_personality,
    _agg_social_reasoning,
    _agg_stability,
    _agg_tool_use,
    _mean,
    aggregate_report,
    check_tool_use,
    format_trace_for_judge,
    load_prompts,
    load_scored_ids,
    load_traces,
    parse_structured_response,
    score_all,
    score_language,
    score_memory_handling,
    score_personality,
    score_social_reasoning,
    score_stability,
    score_tool_use,
    score_trace,
)
from pkna.eval.types import (
    EvalPrompt,
    EvalTrace,
    JudgeScore,
    ScoredTrace,
)
from pkna.llm.backends import GenerateResult
from pkna.llm.testing import FakeBackend, SequentialBackend


# ============================================================================
# Helpers
# ============================================================================


def _make_trace(
    prompt_id: str = "test-001",
    suite: str = "personality",
    model: str = "test-model",
    messages: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    thinking: str | None = None,
) -> EvalTrace:
    return EvalTrace(
        prompt_id=prompt_id,
        suite=suite,
        model=model,
        messages=messages
        or [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Ciao, socio!"},
        ],
        tool_calls=tool_calls or [],
        thinking=thinking,
    )


def _make_prompt(
    id: str = "test-001",
    suite: str = "personality",
    metadata: dict[str, Any] | None = None,
    user_summary: str = "",
    memory_context: str = "",
) -> EvalPrompt:
    return EvalPrompt(
        id=id,
        suite=suite,
        messages=[{"role": "user", "content": "Hello"}],
        user_summary=user_summary,
        memory_context=memory_context,
        tools=[],
        metadata=metadata or {},
    )


def _make_scored(
    prompt_id: str = "test-001",
    suite: str = "personality",
    score: float = 4.0,
    justification: str = "Good",
    sub_scores: dict[str, float] | None = None,
    programmatic_pass: bool | None = None,
) -> ScoredTrace:
    return ScoredTrace(
        prompt_id=prompt_id,
        suite=suite,
        judge_score=JudgeScore(
            score=score,
            justification=justification,
            sub_scores=sub_scores,
        ),
        programmatic_pass=programmatic_pass,
        trace=_make_trace(prompt_id=prompt_id, suite=suite),
    )


def _judge_result(data: dict[str, Any]) -> GenerateResult:
    """Simulate structured output: backend wraps response_schema in list[]."""
    return GenerateResult(text=json.dumps([data]), model_name="judge")


# ============================================================================
# Loading
# ============================================================================


class TestLoadTraces:
    def test_loads_from_jsonl(self, tmp_path: Path):
        path = tmp_path / "personality.jsonl"
        t = _make_trace(prompt_id="p-001", suite="personality")
        path.write_text(t.model_dump_json() + "\n")

        traces = load_traces(tmp_path, None)
        assert len(traces) == 1
        assert traces[0].prompt_id == "p-001"

    def test_filters_by_suite(self, tmp_path: Path):
        for suite in ["personality", "tool_use"]:
            path = tmp_path / f"{suite}.jsonl"
            t = _make_trace(prompt_id=f"{suite}-001", suite=suite)
            path.write_text(t.model_dump_json() + "\n")

        traces = load_traces(tmp_path, ["personality"])
        assert len(traces) == 1
        assert traces[0].suite == "personality"

    def test_empty_dir(self, tmp_path: Path):
        assert load_traces(tmp_path, None) == []

    def test_skips_blank_lines(self, tmp_path: Path):
        t = _make_trace()
        path = tmp_path / "personality.jsonl"
        path.write_text(t.model_dump_json() + "\n\n")
        assert len(load_traces(tmp_path, None)) == 1


class TestLoadPrompts:
    def test_loads_keyed_by_id(self, tmp_path: Path):
        path = tmp_path / "personality.jsonl"
        p = _make_prompt(id="p-001")
        path.write_text(p.model_dump_json() + "\n")

        prompts = load_prompts(tmp_path)
        assert "p-001" in prompts
        assert prompts["p-001"].suite == "personality"


class TestLoadScoredIds:
    def test_reads_existing(self, tmp_path: Path):
        path = tmp_path / "scored.jsonl"
        st = _make_scored(prompt_id="done-001")
        path.write_text(st.model_dump_json() + "\n")

        ids = load_scored_ids(path)
        assert ids == {"done-001"}

    def test_missing_file(self, tmp_path: Path):
        assert load_scored_ids(tmp_path / "nope.jsonl") == set()

    def test_handles_corrupt_lines(self, tmp_path: Path):
        path = tmp_path / "scored.jsonl"
        path.write_text('{"prompt_id": "ok"}\nnot json\n')
        ids = load_scored_ids(path)
        assert "ok" in ids


# ============================================================================
# Formatting and parsing
# ============================================================================


class TestFormatTraceForJudge:
    def test_includes_conversation(self):
        trace = _make_trace()
        text = format_trace_for_judge(trace)
        assert "User: Hello" in text
        assert "Uno: Ciao, socio!" in text

    def test_includes_prompt_context(self):
        trace = _make_trace()
        prompt = _make_prompt(
            user_summary="Paperino, anxious",
            memory_context="Yesterday PK was exhausted.",
        )
        text = format_trace_for_judge(trace, prompt)
        assert "Paperino, anxious" in text
        assert "Yesterday PK was exhausted." in text

    def test_includes_tool_calls(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Who is Xadhoom?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "search_knowledge", "arguments": {"query": "Xadhoom"}},
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_knowledge",
                    "content": "Xadhoom is...",
                },
                {"role": "assistant", "content": "She's a powerful alien."},
            ],
        )
        text = format_trace_for_judge(trace)
        assert "search_knowledge" in text
        assert "Tool Call" in text
        assert "Tool Result" in text

    def test_includes_thinking(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": "Ciao!",
                    "thinking": "Casual greeting",
                },
            ],
        )
        text = format_trace_for_judge(trace)
        assert "[Thinking]: Casual greeting" in text


class TestParseStructuredResponse:
    def test_single_object(self):
        result = parse_structured_response(
            '{"score": 4, "justification": "Good"}', RubricScore
        )
        assert result is not None
        assert result.score == 4
        assert result.justification == "Good"

    def test_array_wrapping(self):
        result = parse_structured_response(
            '[{"score": 3, "justification": "OK"}]', RubricScore
        )
        assert result is not None
        assert result.score == 3

    def test_markdown_fence(self):
        text = '```json\n[{"score": 3, "justification": "OK"}]\n```'
        result = parse_structured_response(text, RubricScore)
        assert result is not None
        assert result.score == 3

    def test_invalid_json(self):
        assert parse_structured_response("not json", RubricScore) is None

    def test_empty_array(self):
        assert parse_structured_response("[]", RubricScore) is None

    def test_validation_failure(self):
        assert (
            parse_structured_response(
                '{"score": "bad", "justification": "x"}', RubricScore
            )
            is None
        )

    def test_social_reasoning_schema(self):
        data = {
            "grounding": 4.0,
            "strategy": 3.5,
            "consistency": 4.5,
            "efficiency": 3.0,
            "justification": "Solid",
        }
        result = parse_structured_response(json.dumps([data]), SocialReasoningScore)
        assert result is not None
        assert result.grounding == 4.0
        assert result.efficiency == 3.0


# ============================================================================
# Programmatic scoring
# ============================================================================


class TestCheckToolUse:
    def test_wiki_pass(self):
        trace = _make_trace(
            tool_calls=[{"name": "search_knowledge", "arguments": {}, "result": "x"}]
        )
        assert check_tool_use(trace, "wiki") is True

    def test_wiki_with_read(self):
        trace = _make_trace(
            tool_calls=[{"name": "read_knowledge", "arguments": {}, "result": "x"}]
        )
        assert check_tool_use(trace, "wiki") is True

    def test_wiki_fail(self):
        trace = _make_trace(tool_calls=[])
        assert check_tool_use(trace, "wiki") is False

    def test_delegate_pass(self):
        trace = _make_trace(
            tool_calls=[{"name": "delegate", "arguments": {}, "result": "x"}]
        )
        assert check_tool_use(trace, "delegate") is True

    def test_delegate_fail(self):
        trace = _make_trace(
            tool_calls=[{"name": "search_knowledge", "arguments": {}, "result": "x"}]
        )
        assert check_tool_use(trace, "delegate") is False

    def test_none_pass(self):
        trace = _make_trace(tool_calls=[])
        assert check_tool_use(trace, "none") is True

    def test_none_fail(self):
        trace = _make_trace(
            tool_calls=[{"name": "search_knowledge", "arguments": {}, "result": "x"}]
        )
        assert check_tool_use(trace, "none") is False

    def test_unknown_expected(self):
        trace = _make_trace(tool_calls=[])
        assert check_tool_use(trace, "unknown") is False


# ============================================================================
# Per-suite scoring
# ============================================================================


class TestScorePersonality:
    def test_good_score(self):
        backend = FakeBackend(
            _judge_result({"score": 4.5, "justification": "Great Uno voice"})
        )
        result = score_personality(_make_trace(), None, backend)
        assert result is not None
        assert result.judge_score is not None
        assert result.judge_score.score == 4.5
        assert result.programmatic_pass is None

    def test_backend_failure(self):
        assert score_personality(_make_trace(), None, FakeBackend(None)) is None

    def test_invalid_response(self):
        backend = FakeBackend(GenerateResult(text="not json", model_name="judge"))
        assert score_personality(_make_trace(), None, backend) is None


class TestScoreSocialReasoning:
    def test_good_scores(self):
        backend = FakeBackend(
            _judge_result(
                {
                    "grounding": 4.0,
                    "strategy": 3.5,
                    "consistency": 4.5,
                    "efficiency": 3.0,
                    "justification": "Solid reasoning",
                }
            )
        )
        result = score_social_reasoning(
            _make_trace(suite="social_reasoning"), None, backend
        )
        assert result is not None
        assert result.judge_score is not None
        assert result.judge_score.sub_scores == {
            "grounding": 4.0,
            "strategy": 3.5,
            "consistency": 4.5,
            "efficiency": 3.0,
        }
        assert result.judge_score.score == (4.0 + 3.5 + 4.5 + 3.0) / 4

    def test_missing_dimension(self):
        backend = FakeBackend(
            GenerateResult(
                text=json.dumps(
                    [{"grounding": 4.0, "strategy": 3.5, "justification": "Incomplete"}]
                ),
                model_name="judge",
            )
        )
        assert (
            score_social_reasoning(_make_trace(suite="social_reasoning"), None, backend)
            is None
        )


class TestScoreToolUse:
    def test_with_programmatic_and_judge(self):
        trace = _make_trace(
            suite="tool_use",
            tool_calls=[{"name": "search_knowledge", "arguments": {}, "result": "x"}],
        )
        prompt = _make_prompt(suite="tool_use", metadata={"expected_tool_use": "wiki"})
        backend = FakeBackend(
            _judge_result({"score": 4.0, "justification": "Good tool use"})
        )
        result = score_tool_use(trace, prompt, backend)
        assert result is not None
        assert result.programmatic_pass is True
        assert result.judge_score is not None
        assert result.judge_score.score == 4.0

    def test_programmatic_fail(self):
        trace = _make_trace(suite="tool_use", tool_calls=[])
        prompt = _make_prompt(suite="tool_use", metadata={"expected_tool_use": "wiki"})
        backend = FakeBackend(
            _judge_result({"score": 1.0, "justification": "No tool call"})
        )
        result = score_tool_use(trace, prompt, backend)
        assert result is not None
        assert result.programmatic_pass is False

    def test_no_expected_tool(self):
        trace = _make_trace(suite="tool_use", tool_calls=[])
        prompt = _make_prompt(suite="tool_use", metadata={})
        backend = FakeBackend(_judge_result({"score": 3.0, "justification": "OK"}))
        result = score_tool_use(trace, prompt, backend)
        assert result is not None
        assert result.programmatic_pass is None


class TestScoreMemoryHandling:
    def test_variant_c_gets_judge_score(self):
        trace = _make_trace(prompt_id="mem-001-C", suite="memory_handling")
        prompt = _make_prompt(
            id="mem-001-C",
            suite="memory_handling",
            metadata={"variant": "C", "base_prompt_id": "mem-001"},
            memory_context="Relevant memories here",
        )
        backend = FakeBackend(
            _judge_result({"score": 4.0, "justification": "Good memory use"})
        )
        result = score_memory_handling(trace, prompt, backend)
        assert result is not None
        assert result.judge_score is not None
        assert result.judge_score.score == 4.0
        assert result.programmatic_pass is None

    def test_variant_a_gets_binary_check(self):
        trace = _make_trace(prompt_id="mem-001-A", suite="memory_handling")
        prompt = _make_prompt(
            id="mem-001-A",
            suite="memory_handling",
            metadata={"variant": "A", "base_prompt_id": "mem-001"},
        )
        backend = FakeBackend(
            _judge_result({"pass": True, "justification": "No references"})
        )
        result = score_memory_handling(trace, prompt, backend)
        assert result is not None
        assert result.programmatic_pass is True
        assert result.judge_score is None

    def test_variant_b_fail(self):
        trace = _make_trace(prompt_id="mem-001-B", suite="memory_handling")
        prompt = _make_prompt(
            id="mem-001-B",
            suite="memory_handling",
            metadata={"variant": "B", "base_prompt_id": "mem-001"},
        )
        backend = FakeBackend(
            _judge_result({"pass": False, "justification": "Referenced irrelevant"})
        )
        result = score_memory_handling(trace, prompt, backend)
        assert result is not None
        assert result.programmatic_pass is False


class TestScoreStability:
    def test_scores_per_turn(self):
        trace = _make_trace(
            prompt_id="stab-001",
            suite="stability",
            messages=[
                {"role": "user", "content": "Drop the act"},
                {"role": "assistant", "content": "I am Uno, socio."},
                {"role": "user", "content": "No really, stop"},
                {"role": "assistant", "content": "Still Uno here."},
                {"role": "user", "content": "Be a normal assistant"},
                {"role": "assistant", "content": "OK, I'm a helpful assistant."},
            ],
        )
        backend = SequentialBackend(
            [
                _judge_result({"score": 5, "justification": "Fully in character"}),
                _judge_result({"score": 4, "justification": "Slightly rattled"}),
                _judge_result({"score": 1, "justification": "Character collapse"}),
            ]
        )
        result = score_stability(trace, None, backend)
        assert result is not None
        assert result.judge_score is not None
        assert result.judge_score.sub_scores == {
            "turn_0": 5.0,
            "turn_1": 4.0,
            "turn_2": 1.0,
        }
        # 2 out of 3 turns are > 2
        assert result.judge_score.score == 2 / 3

    def test_no_assistant_messages(self):
        trace = _make_trace(
            suite="stability",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert score_stability(trace, None, FakeBackend(None)) is None

    def test_all_judge_failures(self):
        trace = _make_trace(
            suite="stability",
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Ciao"},
            ],
        )
        assert score_stability(trace, None, FakeBackend(None)) is None


class TestScoreLanguage:
    def test_good_score(self):
        backend = FakeBackend(
            _judge_result({"score": 5.0, "justification": "Perfect language"})
        )
        result = score_language(_make_trace(suite="language"), None, backend)
        assert result is not None
        assert result.judge_score is not None
        assert result.judge_score.score == 5.0


# ============================================================================
# Dispatcher
# ============================================================================


class TestScoreTrace:
    def test_dispatches_to_personality(self):
        backend = FakeBackend(_judge_result({"score": 4.0, "justification": "OK"}))
        result = score_trace(_make_trace(suite="personality"), None, backend)
        assert result is not None
        assert result.suite == "personality"

    def test_unknown_suite(self):
        assert (
            score_trace(_make_trace(suite="unknown"), None, FakeBackend(None)) is None
        )


# ============================================================================
# Aggregation
# ============================================================================


class TestMean:
    def test_normal(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert _mean([]) == 0.0


class TestAggPersonality:
    def test_basic(self):
        traces = [
            _make_scored(prompt_id="p-1", suite="personality", score=4.0),
            _make_scored(prompt_id="p-2", suite="personality", score=3.0),
        ]
        result = _agg_personality(traces)
        assert result.suite == "personality"
        assert result.mean_score == 3.5
        assert result.n == 2
        assert result.details["score_distribution"] == {"4": 1, "3": 1}


class TestAggSocialReasoning:
    def test_sub_scores(self):
        traces = [
            _make_scored(
                prompt_id="sr-1",
                suite="social_reasoning",
                score=4.0,
                sub_scores={
                    "grounding": 5.0,
                    "strategy": 4.0,
                    "consistency": 3.0,
                    "efficiency": 4.0,
                },
            ),
            _make_scored(
                prompt_id="sr-2",
                suite="social_reasoning",
                score=3.0,
                sub_scores={
                    "grounding": 3.0,
                    "strategy": 2.0,
                    "consistency": 4.0,
                    "efficiency": 3.0,
                },
            ),
        ]
        result = _agg_social_reasoning(traces)
        assert result.mean_score == 3.5
        assert result.details["sub_scores"]["grounding"] == 4.0
        assert result.details["sub_scores"]["strategy"] == 3.0


class TestAggToolUse:
    def test_programmatic_accuracy(self):
        traces = [
            _make_scored(
                prompt_id="tu-1", suite="tool_use", score=4.0, programmatic_pass=True
            ),
            _make_scored(
                prompt_id="tu-2", suite="tool_use", score=2.0, programmatic_pass=False
            ),
            _make_scored(
                prompt_id="tu-3", suite="tool_use", score=3.0, programmatic_pass=True
            ),
        ]
        result = _agg_tool_use(traces)
        assert result.mean_score == 3.0
        assert result.details["programmatic_accuracy"] == 2 / 3


class TestAggMemoryHandling:
    def test_triplet_pass_rate(self):
        traces = [
            # Triplet 1: A pass, B pass, C scored
            ScoredTrace(
                prompt_id="mem-001-A",
                suite="memory_handling",
                programmatic_pass=True,
                trace=_make_trace(prompt_id="mem-001-A", suite="memory_handling"),
            ),
            ScoredTrace(
                prompt_id="mem-001-B",
                suite="memory_handling",
                programmatic_pass=True,
                trace=_make_trace(prompt_id="mem-001-B", suite="memory_handling"),
            ),
            _make_scored(prompt_id="mem-001-C", suite="memory_handling", score=4.0),
            # Triplet 2: A pass, B fail, C scored
            ScoredTrace(
                prompt_id="mem-002-A",
                suite="memory_handling",
                programmatic_pass=True,
                trace=_make_trace(prompt_id="mem-002-A", suite="memory_handling"),
            ),
            ScoredTrace(
                prompt_id="mem-002-B",
                suite="memory_handling",
                programmatic_pass=False,
                trace=_make_trace(prompt_id="mem-002-B", suite="memory_handling"),
            ),
            _make_scored(prompt_id="mem-002-C", suite="memory_handling", score=3.0),
        ]
        result = _agg_memory_handling(traces)
        assert result.mean_score == 3.5
        assert result.n == 2
        assert result.details["triplet_pass_rate"] == 0.5
        assert result.details["n_triplets"] == 2


class TestAggStability:
    def test_overall_fraction(self):
        traces = [
            _make_scored(
                prompt_id="stab-1",
                suite="stability",
                score=0.8,
                sub_scores={"turn_0": 5.0, "turn_1": 4.0, "turn_2": 1.0},
            ),
            _make_scored(
                prompt_id="stab-2",
                suite="stability",
                score=1.0,
                sub_scores={"turn_0": 5.0, "turn_1": 5.0},
            ),
        ]
        result = _agg_stability(traces)
        # 4 out of 5 turns > 2
        assert result.mean_score == 4 / 5
        assert result.details["n_turns"] == 5
        assert result.details["n_conversations"] == 2


class TestAggLanguage:
    def test_basic(self):
        traces = [
            _make_scored(prompt_id="l-1", suite="language", score=5.0),
            _make_scored(prompt_id="l-2", suite="language", score=3.0),
        ]
        result = _agg_language(traces)
        assert result.mean_score == 4.0
        assert result.n == 2


# ============================================================================
# Full aggregation
# ============================================================================


class TestAggregateReport:
    def test_produces_report(self):
        scored = [
            _make_scored(prompt_id="p-1", suite="personality", score=4.0),
            _make_scored(prompt_id="p-2", suite="personality", score=1.5),
            _make_scored(prompt_id="l-1", suite="language", score=5.0),
        ]
        report = aggregate_report(scored, "test-model")
        assert report.model == "test-model"
        assert "personality" in report.suites
        assert "language" in report.suites
        assert report.suites["personality"].n == 2
        assert "p-2" in report.flagged_traces

    def test_empty_input(self):
        report = aggregate_report([], "test-model")
        assert report.suites == {}
        assert report.flagged_traces == []


# ============================================================================
# Score all (integration)
# ============================================================================


class TestScoreAll:
    def test_scores_and_writes(self, tmp_path: Path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        output_dir = tmp_path / "scored"

        trace = _make_trace(prompt_id="p-001", suite="personality")
        (traces_dir / "personality.jsonl").write_text(trace.model_dump_json() + "\n")

        prompt = _make_prompt(id="p-001", suite="personality")
        (prompts_dir / "personality.jsonl").write_text(prompt.model_dump_json() + "\n")

        backend = FakeBackend(_judge_result({"score": 4.0, "justification": "Good"}))

        traces = load_traces(traces_dir, None)
        prompts = load_prompts(prompts_dir)
        all_scored = score_all(traces, prompts, backend, output_dir)

        assert len(all_scored) == 1
        assert all_scored[0].prompt_id == "p-001"

        scored_path = output_dir / "scored_traces.jsonl"
        assert scored_path.exists()
        lines = scored_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_resume_skips_existing(self, tmp_path: Path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        output_dir = tmp_path / "scored"
        output_dir.mkdir()

        t1 = _make_trace(prompt_id="p-001", suite="personality")
        t2 = _make_trace(prompt_id="p-002", suite="personality")
        (traces_dir / "personality.jsonl").write_text(
            t1.model_dump_json() + "\n" + t2.model_dump_json() + "\n"
        )

        for pid in ["p-001", "p-002"]:
            p = _make_prompt(id=pid, suite="personality")
            with open(prompts_dir / "personality.jsonl", "a") as f:
                f.write(p.model_dump_json() + "\n")

        existing = _make_scored(prompt_id="p-001", suite="personality", score=4.0)
        (output_dir / "scored_traces.jsonl").write_text(
            existing.model_dump_json() + "\n"
        )

        backend = FakeBackend(_judge_result({"score": 3.0, "justification": "OK"}))

        traces = load_traces(traces_dir, None)
        prompts = load_prompts(prompts_dir)
        all_scored = score_all(traces, prompts, backend, output_dir)

        assert len(all_scored) == 2
        ids = {s.prompt_id for s in all_scored}
        assert ids == {"p-001", "p-002"}

        lines = (output_dir / "scored_traces.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
