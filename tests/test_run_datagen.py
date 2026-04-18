"""Unit tests for the datagen execution loop."""

from pathlib import Path
from typing import Any

from datagen.run_datagen import (
    _get_directive,
    _visible_messages,
    load_completed_ids,
    run_datagen,
    run_multi_turn,
    run_single_prompt,
    run_single_turn,
)
from pkna.datagen.types import DatagenPrompt, DatagenTrace
from pkna.llm.backends import GenerateResult
from pkna.llm.testing import FakeBackend, SequentialBackend


SAMPLE_PROFILE = "# Uno - Character Profile\n\nUno is a sarcastic AI."


class TestLoadCompletedIds:
    def test_reads_existing_traces(self, tmp_path: Path):
        path = tmp_path / "traces.jsonl"
        trace = DatagenTrace(
            id="done-001",
            metadata={},
            memory_context="",
            user_summary="",
            messages=[],
        )
        path.write_text(trace.model_dump_json() + "\n")
        ids = load_completed_ids(path)
        assert ids == {"done-001"}

    def test_missing_file(self, tmp_path: Path):
        ids = load_completed_ids(tmp_path / "nonexistent.jsonl")
        assert ids == set()

    def test_handles_corrupt_lines(self, tmp_path: Path):
        path = tmp_path / "traces.jsonl"
        path.write_text('{"id": "ok"}\nnot json\n')
        ids = load_completed_ids(path)
        assert "ok" in ids


class TestVisibleMessages:
    def test_filters_tool_messages(self):
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking": "greet"},
            {"role": "tool", "name": "search_knowledge", "content": "result"},
            {"role": "assistant", "content": "Found it."},
        ]
        visible = _visible_messages(messages)
        assert visible == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "assistant", "content": "Found it."},
        ]

    def test_skips_empty_content(self):
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Hey"},
        ]
        visible = _visible_messages(messages)
        assert visible == [{"role": "user", "content": "Hey"}]


class TestGetDirective:
    def test_returns_by_index(self):
        assert _get_directive(["a", "b", "c"], 1) == "b"

    def test_cycles(self):
        assert _get_directive(["a", "b"], 3) == "b"

    def test_empty_returns_continue(self):
        assert _get_directive([], 0) == "continue"


class TestRunSingleTurn:
    def test_records_trace(self):
        backend = FakeBackend(
            GenerateResult(
                text="Ciao!",
                model_name="test",
                thinking="greeting",
                messages=[
                    {"role": "assistant", "content": "Ciao!", "thinking": "greeting"}
                ],
            )
        )
        result = run_single_turn(
            prompt_id="t-001",
            system_prompt="You are Uno.",
            user_summary="Paperino",
            memory_context="",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"prompt_source": "manual"},
            backend=backend,
            tools=None,
        )
        assert result is not None
        assert result.trace.id == "t-001"
        assert result.trace.user_summary == "Paperino"
        assert result.trace.messages[0] == {"role": "user", "content": "Hello"}
        assert result.trace.messages[1]["content"] == "Ciao!"
        assert result.trace.messages[1]["thinking"] == "greeting"

    def test_returns_none_on_failure(self):
        backend = FakeBackend(None)
        result = run_single_turn(
            prompt_id="t-fail",
            system_prompt="sys",
            user_summary="",
            memory_context="",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={},
            backend=backend,
            tools=None,
        )
        assert result is None

    def test_fallback_when_messages_empty(self):
        backend = FakeBackend(GenerateResult(text="fallback text", model_name="test"))
        result = run_single_turn(
            prompt_id="t-fb",
            system_prompt="sys",
            user_summary="",
            memory_context="",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={},
            backend=backend,
            tools=None,
        )
        assert result is not None
        assert result.trace.messages[-1] == {
            "role": "assistant",
            "content": "fallback text",
        }


class TestRunMultiTurn:
    def test_runs_multiple_turns(self):
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="R1",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "R1"}],
                ),
                GenerateResult(
                    text="R2",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "R2"}],
                ),
                GenerateResult(
                    text="R3",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "R3"}],
                ),
            ]
        )
        sim_backend = SequentialBackend(
            [
                GenerateResult(text="User 2", model_name="sim"),
                GenerateResult(text="User 3", model_name="sim"),
            ]
        )
        result = run_multi_turn(
            prompt_id="mt-001",
            system_prompt="sys",
            user_summary="Paperino",
            memory_context="",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"turn_count": 3, "directives": ["continue", "escalate"]},
            backend=model_backend,
            tools=None,
            simulator_backend=sim_backend,
        )
        assert result is not None
        user_msgs = [m for m in result.trace.messages if m["role"] == "user"]
        assistant_msgs = [m for m in result.trace.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3
        assert len(user_msgs) == 3

    def test_stops_on_model_failure(self):
        backend = SequentialBackend(
            [
                GenerateResult(
                    text="R1",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "R1"}],
                ),
                None,
            ]
        )
        sim_backend = SequentialBackend(
            [GenerateResult(text="Follow up", model_name="sim")]
        )
        result = run_multi_turn(
            prompt_id="mt-fail",
            system_prompt="sys",
            user_summary="",
            memory_context="",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"turn_count": 5, "directives": ["continue"]},
            backend=backend,
            tools=None,
            simulator_backend=sim_backend,
        )
        assert result is not None
        assert len([m for m in result.trace.messages if m["role"] == "assistant"]) == 1

    def test_preserves_tool_messages_across_turns(self):
        """Tool messages from turn 1 should be present in the trace and
        passed correctly to the backend on turn 2."""
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="Found it",
                    model_name="test",
                    messages=[
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"name": "search", "arguments": {"q": "test"}}
                            ],
                        },
                        {"role": "tool", "name": "search", "content": "result"},
                        {"role": "assistant", "content": "Found it"},
                    ],
                ),
                GenerateResult(
                    text="Glad to help",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "Glad to help"}],
                ),
            ]
        )
        sim_backend = SequentialBackend(
            [GenerateResult(text="Thanks!", model_name="sim")]
        )
        result = run_multi_turn(
            prompt_id="mt-tools",
            system_prompt="sys",
            user_summary="Paperino",
            memory_context="",
            messages=[{"role": "user", "content": "Search for info"}],
            metadata={"turn_count": 2, "directives": ["continue"]},
            backend=model_backend,
            tools=None,
            simulator_backend=sim_backend,
        )
        assert result is not None
        tool_msgs = [m for m in result.trace.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["name"] == "search"
        assistant_msgs = [m for m in result.trace.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3

    def test_returns_none_when_no_response(self):
        backend = SequentialBackend([None])
        result = run_multi_turn(
            prompt_id="mt-none",
            system_prompt="sys",
            user_summary="",
            memory_context="",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"turn_count": 3},
            backend=backend,
            tools=None,
        )
        assert result is None


class TestRunSinglePromptDispatch:
    def test_dispatches_multi_turn(self):
        backend = SequentialBackend(
            [
                GenerateResult(
                    text="R1",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "R1"}],
                ),
            ]
        )
        result = run_single_prompt(
            prompt_id="dispatch-mt",
            system_prompt="sys",
            user_summary="",
            memory_context="",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"multi_turn": True, "turn_count": 1},
            backend=backend,
            tools=None,
        )
        assert result is not None

    def test_dispatches_single_turn(self):
        backend = FakeBackend(
            GenerateResult(
                text="Single",
                model_name="test",
                messages=[{"role": "assistant", "content": "Single"}],
            )
        )
        result = run_single_prompt(
            prompt_id="dispatch-st",
            system_prompt="sys",
            user_summary="",
            memory_context="",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={},
            backend=backend,
            tools=None,
        )
        assert result is not None


class TestRunDatagen:
    def _write_prompts(self, path: Path, prompts: list[DatagenPrompt]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(p.model_dump_json() + "\n")

    def test_generates_traces(self, tmp_path: Path):
        prompts_path = tmp_path / "prompts.jsonl"
        output_path = tmp_path / "traces.jsonl"

        self._write_prompts(
            prompts_path,
            [
                DatagenPrompt(
                    id="p-001",
                    messages=[{"role": "user", "content": "Hello"}],
                    user_summary="Test",
                    tools=[],
                    metadata={"prompt_source": "test"},
                ),
                DatagenPrompt(
                    id="p-002",
                    messages=[{"role": "user", "content": "Ciao"}],
                    user_summary="Test",
                    tools=[],
                    metadata={"prompt_source": "test"},
                ),
            ],
        )

        backend = FakeBackend(
            GenerateResult(
                text="Response",
                model_name="test",
                messages=[{"role": "assistant", "content": "Response"}],
            )
        )

        written = run_datagen(prompts_path, output_path, backend)
        assert written == 2

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        trace = DatagenTrace.model_validate_json(lines[0])
        assert trace.id == "p-001"

    def test_resumes_from_existing(self, tmp_path: Path):
        prompts_path = tmp_path / "prompts.jsonl"
        output_path = tmp_path / "traces.jsonl"

        self._write_prompts(
            prompts_path,
            [
                DatagenPrompt(
                    id="p-001",
                    messages=[{"role": "user", "content": "Hello"}],
                    user_summary="",
                    tools=[],
                    metadata={},
                ),
                DatagenPrompt(
                    id="p-002",
                    messages=[{"role": "user", "content": "Ciao"}],
                    user_summary="",
                    tools=[],
                    metadata={},
                ),
            ],
        )

        existing_trace = DatagenTrace(
            id="p-001",
            metadata={},
            memory_context="",
            user_summary="",
            messages=[{"role": "user", "content": "Hello"}],
        )
        output_path.write_text(existing_trace.model_dump_json() + "\n")

        backend = FakeBackend(
            GenerateResult(
                text="Response",
                model_name="test",
                messages=[{"role": "assistant", "content": "Response"}],
            )
        )

        written = run_datagen(prompts_path, output_path, backend)
        assert written == 1

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_skips_failed_inference(self, tmp_path: Path):
        prompts_path = tmp_path / "prompts.jsonl"
        output_path = tmp_path / "traces.jsonl"

        self._write_prompts(
            prompts_path,
            [
                DatagenPrompt(
                    id="p-001",
                    messages=[{"role": "user", "content": "Hello"}],
                    user_summary="",
                    tools=[],
                    metadata={},
                ),
            ],
        )

        backend = FakeBackend(None)
        written = run_datagen(prompts_path, output_path, backend)
        assert written == 0
