"""Unit tests for the eval inference harness."""

from pathlib import Path

from pkna.eval_types import EvalPrompt, EvalTrace

from evals.run_eval_inference import (
    compose_context,
    load_completed_ids,
    load_memory_bank,
    load_prompts,
)


def _make_prompt(
    id: str = "test-001",
    suite: str = "personality",
    messages: list[dict[str, str]] | None = None,
    user_summary: str = "",
    memory_context: str = "",
    memory_bank_id: str = "",
    tools: list[str] | None = None,
) -> EvalPrompt:
    return EvalPrompt(
        id=id,
        suite=suite,
        messages=messages or [{"role": "user", "content": "Hello"}],
        user_summary=user_summary,
        memory_context=memory_context,
        memory_bank_id=memory_bank_id,
        tools=tools or [],
        metadata={},
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
