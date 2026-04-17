"""Unit tests for datagen data types."""

from pkna.datagen.types import (
    DatagenPrompt,
    DatagenTrace,
    QualityScore,
    ScoredTrace,
)


class TestDatagenPrompt:
    def test_roundtrip_json(self):
        prompt = DatagenPrompt(
            id="p-001",
            messages=[{"role": "user", "content": "Ciao, Uno!"}],
            user_summary="Paperino, anxious",
            memory_context="Yesterday PK was exhausted.",
            memory_bank_id="paperino_recent",
            tools=["search_knowledge", "read_knowledge"],
            metadata={"prompt_source": "manual", "language": "italian"},
        )
        restored = DatagenPrompt.model_validate_json(prompt.model_dump_json())
        assert restored == prompt

    def test_defaults(self):
        prompt = DatagenPrompt(
            id="p-002",
            messages=[{"role": "user", "content": "Hello"}],
            user_summary="",
            memory_context="",
            tools=[],
            metadata={},
        )
        assert prompt.memory_bank_id == ""


class TestDatagenTrace:
    def test_roundtrip_json(self):
        trace = DatagenTrace(
            id="trace-001",
            metadata={"prompt_source": "manual", "turns": 1},
            memory_context="",
            user_summary="Stranger",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Ciao!", "thinking": "greeting"},
            ],
        )
        restored = DatagenTrace.model_validate_json(trace.model_dump_json())
        assert restored == trace

    def test_messages_with_tool_calls(self):
        trace = DatagenTrace(
            id="trace-002",
            metadata={},
            memory_context="",
            user_summary="",
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
                {"role": "tool", "name": "search_knowledge", "content": "results..."},
                {"role": "assistant", "content": "Xadhoom is..."},
            ],
        )
        assert len(trace.messages) == 4


class TestQualityScore:
    def test_passing_score(self):
        score = QualityScore(
            trace_id="trace-001",
            character_consistency=4.0,
            thinking_quality=3.5,
            tool_correctness="pass",
            language_consistent=True,
            response_length_ok=True,
            overall_pass=True,
            justification="Good trace.",
        )
        assert score.overall_pass

    def test_failing_score(self):
        score = QualityScore(
            trace_id="trace-002",
            character_consistency=2.0,
            thinking_quality=1.5,
            tool_correctness="fail",
            language_consistent=False,
            response_length_ok=False,
            overall_pass=False,
            justification="Character breaks, too short.",
        )
        assert not score.overall_pass

    def test_tool_correctness_na(self):
        score = QualityScore(
            trace_id="trace-003",
            character_consistency=4.0,
            thinking_quality=4.0,
            tool_correctness="na",
            language_consistent=True,
            response_length_ok=True,
            overall_pass=True,
            justification="No tools expected.",
        )
        assert score.tool_correctness == "na"


class TestScoredTrace:
    def test_roundtrip(self):
        trace = DatagenTrace(
            id="trace-001",
            metadata={},
            memory_context="",
            user_summary="",
            messages=[{"role": "user", "content": "Hi"}],
        )
        score = QualityScore(
            trace_id="trace-001",
            character_consistency=4.0,
            thinking_quality=4.0,
            tool_correctness="na",
            language_consistent=True,
            response_length_ok=True,
            overall_pass=True,
            justification="OK",
        )
        scored = ScoredTrace(trace=trace, score=score)
        restored = ScoredTrace.model_validate_json(scored.model_dump_json())
        assert restored == scored
