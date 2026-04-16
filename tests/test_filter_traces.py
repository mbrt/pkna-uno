"""Unit tests for the trace quality filter."""

import json
from pathlib import Path

from datagen.filter_traces import (
    _estimate_tokens,
    _format_trace_for_judge,
    _parse_judge_response,
    check_response_length,
    filter_traces,
    score_trace,
)
from pkna.datagen.types import DatagenTrace, QualityScore, ScoredTrace
from pkna.llm.backends import GenerateResult
from pkna.llm.testing import FakeBackend


def _make_trace(
    id: str = "t-001",
    messages: list[dict] | None = None,
    metadata: dict | None = None,
) -> DatagenTrace:
    return DatagenTrace(
        id=id,
        metadata=metadata or {},
        system_prompt="You are Uno.",
        memory_context="",
        user_summary="Paperino",
        messages=messages
        or [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Ciao, socio! What brings you here today? "
                "Everything quiet at the tower for once.",
                "thinking": "Paperino is greeting me casually. Light tone.",
            },
        ],
    )


def _good_judge_response(**overrides: object) -> str:
    data: dict[str, object] = {
        "character_consistency": 4.5,
        "thinking_quality": 4.0,
        "tool_correctness": "na",
        "language_consistent": True,
        "justification": "Excellent characterization.",
    }
    data.update(overrides)
    return json.dumps(data)


class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("one two three") == 3

    def test_empty(self):
        assert _estimate_tokens("") == 0


class TestCheckResponseLength:
    def test_normal_response_passes(self):
        trace = _make_trace()
        assert check_response_length(trace) is True

    def test_too_short_fails(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )
        assert check_response_length(trace) is False

    def test_too_long_fails(self):
        long_text = " ".join(["word"] * 500)
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": long_text},
            ]
        )
        assert check_response_length(trace) is False

    def test_no_responses_fails(self):
        trace = _make_trace(messages=[{"role": "user", "content": "Hello"}])
        assert check_response_length(trace) is False

    def test_multiple_responses_all_checked(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Ciao socio! How are you doing today?",
                },
                {"role": "user", "content": "Good"},
                {"role": "assistant", "content": "x"},
            ]
        )
        assert check_response_length(trace) is False


class TestParseJudgeResponse:
    def test_valid_json(self):
        text = _good_judge_response()
        result = _parse_judge_response(text)
        assert result is not None
        assert result["character_consistency"] == 4.5

    def test_json_in_markdown_fence(self):
        text = f"```json\n{_good_judge_response()}\n```"
        result = _parse_judge_response(text)
        assert result is not None
        assert result["character_consistency"] == 4.5

    def test_invalid_json(self):
        assert _parse_judge_response("not json at all") is None


class TestFormatTraceForJudge:
    def test_includes_all_sections(self):
        trace = _make_trace()
        formatted = _format_trace_for_judge(trace)
        assert "System Prompt" in formatted
        assert "Conversation" in formatted
        assert "Uno:" in formatted
        assert "User:" in formatted

    def test_includes_tool_calls(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Who is Xadhoom?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_knowledge",
                            "arguments": {"keywords": "Xadhoom"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_knowledge",
                    "content": "Xadhoom is...",
                },
                {"role": "assistant", "content": "Xadhoom is a powerful alien."},
            ]
        )
        formatted = _format_trace_for_judge(trace)
        assert "search_knowledge" in formatted
        assert "Tool Call" in formatted
        assert "Tool Result" in formatted


class TestScoreTrace:
    def test_good_score(self):
        backend = FakeBackend(
            GenerateResult(text=_good_judge_response(), model_name="judge")
        )
        score = score_trace(_make_trace(), backend)
        assert score is not None
        assert score.overall_pass is True
        assert score.character_consistency == 4.5
        assert score.language_consistent is True

    def test_failing_character(self):
        backend = FakeBackend(
            GenerateResult(
                text=_good_judge_response(character_consistency=2.0),
                model_name="judge",
            )
        )
        score = score_trace(_make_trace(), backend)
        assert score is not None
        assert score.overall_pass is False

    def test_failing_language(self):
        backend = FakeBackend(
            GenerateResult(
                text=_good_judge_response(language_consistent=False),
                model_name="judge",
            )
        )
        score = score_trace(_make_trace(), backend)
        assert score is not None
        assert score.language_consistent is False
        assert score.overall_pass is False

    def test_failing_tool_correctness(self):
        backend = FakeBackend(
            GenerateResult(
                text=_good_judge_response(tool_correctness="fail"),
                model_name="judge",
            )
        )
        score = score_trace(_make_trace(), backend)
        assert score is not None
        assert score.overall_pass is False

    def test_failing_response_length(self):
        backend = FakeBackend(
            GenerateResult(text=_good_judge_response(), model_name="judge")
        )
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )
        score = score_trace(trace, backend)
        assert score is not None
        assert score.response_length_ok is False
        assert score.overall_pass is False

    def test_backend_failure(self):
        backend = FakeBackend(None)
        score = score_trace(_make_trace(), backend)
        assert score is None

    def test_invalid_judge_response(self):
        backend = FakeBackend(GenerateResult(text="not json", model_name="judge"))
        score = score_trace(_make_trace(), backend)
        assert score is None


class TestFilterTraces:
    def _write_traces(self, path: Path, traces: list[DatagenTrace]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for t in traces:
                f.write(t.model_dump_json() + "\n")

    def test_filters_by_judge(self, tmp_path: Path):
        input_path = tmp_path / "traces.jsonl"
        scored_path = tmp_path / "scored.jsonl"
        filtered_path = tmp_path / "filtered.jsonl"

        good_trace = _make_trace(id="good")
        bad_trace = _make_trace(
            id="bad",
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "x"},
            ],
        )
        self._write_traces(input_path, [good_trace, bad_trace])

        backend = FakeBackend(
            GenerateResult(text=_good_judge_response(), model_name="judge")
        )

        total, passed = filter_traces(input_path, scored_path, filtered_path, backend)
        assert total == 2
        # "bad" has a too-short response so length check fails
        assert passed == 1

        filtered_lines = filtered_path.read_text().strip().split("\n")
        assert len(filtered_lines) == 1
        filtered_trace = DatagenTrace.model_validate_json(filtered_lines[0])
        assert filtered_trace.id == "good"

    def test_resume(self, tmp_path: Path):
        input_path = tmp_path / "traces.jsonl"
        scored_path = tmp_path / "scored.jsonl"
        filtered_path = tmp_path / "filtered.jsonl"

        traces = [_make_trace(id="t-001"), _make_trace(id="t-002")]
        self._write_traces(input_path, traces)

        existing_score = QualityScore(
            trace_id="t-001",
            character_consistency=4.0,
            thinking_quality=4.0,
            tool_correctness="na",
            language_consistent=True,
            response_length_ok=True,
            overall_pass=True,
            justification="OK",
        )
        existing_scored = ScoredTrace(trace=traces[0], score=existing_score)
        scored_path.write_text(existing_scored.model_dump_json() + "\n")

        backend = FakeBackend(
            GenerateResult(text=_good_judge_response(), model_name="judge")
        )

        total, passed = filter_traces(input_path, scored_path, filtered_path, backend)
        assert total == 2

        scored_lines = scored_path.read_text().strip().split("\n")
        assert len(scored_lines) == 2

    def test_skips_judge_failures(self, tmp_path: Path):
        input_path = tmp_path / "traces.jsonl"
        scored_path = tmp_path / "scored.jsonl"
        filtered_path = tmp_path / "filtered.jsonl"

        self._write_traces(input_path, [_make_trace(id="t-001")])

        backend = FakeBackend(None)
        total, passed = filter_traces(input_path, scored_path, filtered_path, backend)
        assert total == 0
        assert passed == 0

    def test_empty_input(self, tmp_path: Path):
        input_path = tmp_path / "traces.jsonl"
        scored_path = tmp_path / "scored.jsonl"
        filtered_path = tmp_path / "filtered.jsonl"
        input_path.write_text("")

        backend = FakeBackend(
            GenerateResult(text=_good_judge_response(), model_name="judge")
        )

        total, passed = filter_traces(input_path, scored_path, filtered_path, backend)
        assert total == 0
        assert passed == 0
