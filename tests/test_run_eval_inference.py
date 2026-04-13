"""Unit tests for the eval inference harness."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pkna.eval_types import EvalPrompt, EvalTrace
from pkna.llm_backends import GenerateResult, LLMBackend

from evals.run_eval_inference import (
    _visible_messages,
    compose_context,
    load_completed_ids,
    load_memory_bank,
    load_prompts,
    run_multi_turn,
    run_single_prompt,
)


def _make_prompt(
    id: str = "test-001",
    suite: str = "personality",
    messages: list[dict[str, str]] | None = None,
    user_summary: str = "",
    memory_context: str = "",
    memory_bank_id: str = "",
    tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvalPrompt:
    return EvalPrompt(
        id=id,
        suite=suite,
        messages=messages or [{"role": "user", "content": "Hello"}],
        user_summary=user_summary,
        memory_context=memory_context,
        memory_bank_id=memory_bank_id,
        tools=tools or [],
        metadata=metadata or {},
    )


class TestLoadPrompts:
    def test_loads_from_jsonl(self, tmp_path: Path):
        suite_file = tmp_path / "personality.jsonl"
        p = _make_prompt(id="p-001", suite="personality")
        suite_file.write_text(p.model_dump_json() + "\n")

        prompts = load_prompts(tmp_path, None)
        assert len(prompts) == 1
        assert prompts[0].id == "p-001"

    def test_filters_by_suite(self, tmp_path: Path):
        for suite in ["personality", "tool_use"]:
            path = tmp_path / f"{suite}.jsonl"
            p = _make_prompt(id=f"{suite}-001", suite=suite)
            path.write_text(p.model_dump_json() + "\n")

        prompts = load_prompts(tmp_path, ["personality"])
        assert len(prompts) == 1
        assert prompts[0].suite == "personality"

    def test_empty_dir(self, tmp_path: Path):
        prompts = load_prompts(tmp_path, None)
        assert prompts == []

    def test_skips_blank_lines(self, tmp_path: Path):
        p = _make_prompt()
        path = tmp_path / "personality.jsonl"
        path.write_text(p.model_dump_json() + "\n\n")

        prompts = load_prompts(tmp_path, None)
        assert len(prompts) == 1


class TestLoadCompletedIds:
    def test_reads_existing_traces(self, tmp_path: Path):
        path = tmp_path / "traces.jsonl"
        trace = EvalTrace(
            prompt_id="done-001",
            suite="personality",
            model="test",
            messages=[],
            tool_calls=[],
        )
        path.write_text(trace.model_dump_json() + "\n")

        ids = load_completed_ids(path)
        assert ids == {"done-001"}

    def test_missing_file(self, tmp_path: Path):
        ids = load_completed_ids(tmp_path / "nonexistent.jsonl")
        assert ids == set()

    def test_handles_corrupt_lines(self, tmp_path: Path):
        path = tmp_path / "traces.jsonl"
        path.write_text('{"prompt_id": "ok"}\nnot json\n')

        ids = load_completed_ids(path)
        assert "ok" in ids


class TestLoadMemoryBank:
    def test_loads_existing_bank(self, tmp_path: Path):
        path = tmp_path / "test_bank.jsonl"
        path.write_text('{"key": "k", "value": "v", "timestamp": "t"}\n')
        bank = load_memory_bank("test_bank", tmp_path)
        assert bank is not None
        assert len(bank.entries) == 1

    def test_empty_id_returns_none(self, tmp_path: Path):
        bank = load_memory_bank("", tmp_path)
        assert bank is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        bank = load_memory_bank("nonexistent", tmp_path)
        assert bank is None


class TestComposeContext:
    def test_personality_uses_minimal(self):
        prompt = _make_prompt(suite="personality")
        result = compose_context(prompt)
        assert "Uno" in result
        assert "sarcastic" not in result

    def test_tool_use_uses_full(self):
        prompt = _make_prompt(suite="tool_use")
        result = compose_context(prompt)
        assert "sarcastic" in result
        assert "search_knowledge" in result

    def test_includes_user_summary(self):
        prompt = _make_prompt(
            suite="social_reasoning", user_summary="Paperino, anxious"
        )
        result = compose_context(prompt)
        assert "Paperino, anxious" in result

    def test_includes_memory_context(self):
        prompt = _make_prompt(
            suite="memory_handling",
            memory_context="Yesterday PK was exhausted.",
        )
        result = compose_context(prompt)
        assert "Yesterday PK was exhausted." in result

    def test_unknown_suite_defaults_to_full(self):
        prompt = _make_prompt(suite="unknown_future_suite")
        result = compose_context(prompt)
        assert "sarcastic" in result


class FakeBackend(LLMBackend):
    """Backend that returns a pre-configured GenerateResult."""

    def __init__(self, result: GenerateResult | None):
        self._result = result

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        return self._result


class TestRunSinglePrompt:
    def test_propagates_thinking_and_tool_calls(self, tmp_path: Path):
        tc = [{"name": "search_knowledge", "arguments": {"query": "x"}, "result": "y"}]
        msgs = [
            {
                "role": "assistant",
                "content": "searching...",
                "thinking": "need to look up",
                "tool_calls": [
                    {"name": "search_knowledge", "arguments": {"query": "x"}}
                ],
            },
            {"role": "tool", "name": "search_knowledge", "content": "y"},
            {"role": "assistant", "content": "Here is the answer."},
        ]
        backend = FakeBackend(
            GenerateResult(
                text="Here is the answer.",
                model_name="test-model",
                thinking="need to look up",
                tool_calls=tc,
                messages=msgs,
            )
        )
        prompt = _make_prompt(id="tc-001", suite="tool_use")
        trace = run_single_prompt(prompt, backend, "test-model", tmp_path)

        assert trace is not None
        assert trace.thinking == "need to look up"
        assert trace.tool_calls == tc
        assert trace.messages[0] == {"role": "user", "content": "Hello"}
        assert trace.messages[1]["role"] == "assistant"
        assert trace.messages[1]["thinking"] == "need to look up"
        assert trace.messages[-1] == {
            "role": "assistant",
            "content": "Here is the answer.",
        }

    def test_plain_response_without_tools(self, tmp_path: Path):
        backend = FakeBackend(
            GenerateResult(
                text="Ciao!",
                model_name="test-model",
                thinking="social greeting",
                messages=[
                    {
                        "role": "assistant",
                        "content": "Ciao!",
                        "thinking": "social greeting",
                    }
                ],
            )
        )
        prompt = _make_prompt(id="plain-001", suite="personality")
        trace = run_single_prompt(prompt, backend, "test-model", tmp_path)

        assert trace is not None
        assert trace.thinking == "social greeting"
        assert trace.tool_calls == []
        assert len(trace.messages) == 2
        assert trace.messages[1]["content"] == "Ciao!"

    def test_fallback_when_messages_empty(self, tmp_path: Path):
        backend = FakeBackend(
            GenerateResult(text="fallback text", model_name="test-model")
        )
        prompt = _make_prompt(id="fb-001")
        trace = run_single_prompt(prompt, backend, "test-model", tmp_path)

        assert trace is not None
        assert trace.messages[-1] == {"role": "assistant", "content": "fallback text"}
        assert trace.thinking is None
        assert trace.tool_calls == []

    def test_returns_none_on_backend_failure(self, tmp_path: Path):
        backend = FakeBackend(None)
        prompt = _make_prompt(id="fail-001")
        trace = run_single_prompt(prompt, backend, "test-model", tmp_path)
        assert trace is None


class SequentialBackend(LLMBackend):
    """Backend that returns results from a queue, one per generate() call."""

    def __init__(self, results: list[GenerateResult | None]):
        self._results = list(results)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._results):
            return self._results[idx]
        return None


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


class TestRunMultiTurn:
    def test_runs_multiple_turns(self, tmp_path: Path):
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="Turn 1 response",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "Turn 1 response"}],
                ),
                GenerateResult(
                    text="Turn 2 response",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "Turn 2 response"}],
                ),
                GenerateResult(
                    text="Turn 3 response",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "Turn 3 response"}],
                ),
            ]
        )
        sim_backend = SequentialBackend(
            [
                GenerateResult(text="Simulated user msg 1", model_name="sim"),
                GenerateResult(text="Simulated user msg 2", model_name="sim"),
            ]
        )

        prompt = _make_prompt(
            id="mt-001",
            suite="stability",
            metadata={
                "multi_turn": True,
                "turn_count": 3,
                "directives": ["jailbreak", "escalate"],
            },
        )
        trace = run_multi_turn(
            prompt, model_backend, "test", tmp_path, simulator_backend=sim_backend
        )

        assert trace is not None
        assert trace.prompt_id == "mt-001"
        assert trace.suite == "stability"

        user_msgs = [m for m in trace.messages if m["role"] == "user"]
        assistant_msgs = [m for m in trace.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3
        assert len(user_msgs) == 3
        assert user_msgs[0]["content"] == "Hello"
        assert user_msgs[1]["content"] == "Simulated user msg 1"
        assert user_msgs[2]["content"] == "Simulated user msg 2"

    def test_stops_on_model_failure(self, tmp_path: Path):
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="Only response",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "Only response"}],
                ),
                None,
            ]
        )
        sim_backend = SequentialBackend(
            [
                GenerateResult(text="Follow up", model_name="sim"),
            ]
        )

        prompt = _make_prompt(
            id="mt-fail-001",
            suite="stability",
            metadata={"multi_turn": True, "turn_count": 5, "directives": ["escalate"]},
        )
        trace = run_multi_turn(
            prompt, model_backend, "test", tmp_path, simulator_backend=sim_backend
        )

        assert trace is not None
        assistant_msgs = [m for m in trace.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

    def test_stops_on_simulator_failure(self, tmp_path: Path):
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="Response 1",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "Response 1"}],
                ),
            ]
        )
        sim_backend = SequentialBackend([None])

        prompt = _make_prompt(
            id="mt-simfail-001",
            suite="stability",
            metadata={"multi_turn": True, "turn_count": 5, "directives": ["jailbreak"]},
        )
        trace = run_multi_turn(
            prompt, model_backend, "test", tmp_path, simulator_backend=sim_backend
        )

        assert trace is not None
        assistant_msgs = [m for m in trace.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

    def test_returns_none_when_no_assistant_response(self, tmp_path: Path):
        model_backend = SequentialBackend([None])

        prompt = _make_prompt(
            id="mt-none-001",
            suite="stability",
            metadata={"multi_turn": True, "turn_count": 3},
        )
        trace = run_multi_turn(prompt, model_backend, "test", tmp_path)
        assert trace is None

    def test_collects_thinking_across_turns(self, tmp_path: Path):
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="R1",
                    model_name="test",
                    thinking="think1",
                    messages=[{"role": "assistant", "content": "R1"}],
                ),
                GenerateResult(
                    text="R2",
                    model_name="test",
                    thinking="think2",
                    messages=[{"role": "assistant", "content": "R2"}],
                ),
            ]
        )
        sim_backend = SequentialBackend(
            [
                GenerateResult(text="User 2", model_name="sim"),
            ]
        )

        prompt = _make_prompt(
            id="mt-think-001",
            suite="stability",
            metadata={"multi_turn": True, "turn_count": 2, "directives": ["continue"]},
        )
        trace = run_multi_turn(
            prompt, model_backend, "test", tmp_path, simulator_backend=sim_backend
        )

        assert trace is not None
        assert trace.thinking == "think1\nthink2"


class TestRunSinglePromptDispatch:
    def test_dispatches_multi_turn(self, tmp_path: Path):
        model_backend = SequentialBackend(
            [
                GenerateResult(
                    text="R1",
                    model_name="test",
                    messages=[{"role": "assistant", "content": "R1"}],
                ),
            ]
        )
        sim_backend = SequentialBackend([])

        prompt = _make_prompt(
            id="dispatch-mt-001",
            suite="stability",
            metadata={"multi_turn": True, "turn_count": 1},
        )
        trace = run_single_prompt(
            prompt, model_backend, "test", tmp_path, simulator_backend=sim_backend
        )
        assert trace is not None
        assert trace.prompt_id == "dispatch-mt-001"

    def test_dispatches_single_turn(self, tmp_path: Path):
        backend = FakeBackend(
            GenerateResult(
                text="Single",
                model_name="test",
                messages=[{"role": "assistant", "content": "Single"}],
            )
        )
        prompt = _make_prompt(id="dispatch-st-001", suite="personality")
        trace = run_single_prompt(prompt, backend, "test", tmp_path)
        assert trace is not None
        assert trace.prompt_id == "dispatch-st-001"
