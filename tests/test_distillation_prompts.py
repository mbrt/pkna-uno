"""Unit tests for distillation prompt sampling."""

from datasets import Dataset

from distillation.generate_prompts import sample_prompts, to_prompt_only


def _make_tulu_dataset(rows: list[dict]) -> Dataset:
    """Build a minimal dataset that looks like tulu-3-sft-mixture."""
    messages = [r["messages"] for r in rows]
    sources = [r["source"] for r in rows]
    return Dataset.from_dict({"messages": messages, "source": sources})


class TestToPromptOnly:
    def test_extracts_system_and_first_user(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Follow-up"},
        ]
        result = to_prompt_only(messages)
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    def test_no_system_prompt(self):
        messages = [
            {"role": "user", "content": "Just a question"},
            {"role": "assistant", "content": "An answer"},
        ]
        result = to_prompt_only(messages)
        assert result == [{"role": "user", "content": "Just a question"}]

    def test_system_only_no_user(self):
        messages = [{"role": "system", "content": "System msg"}]
        result = to_prompt_only(messages)
        assert result == [{"role": "system", "content": "System msg"}]

    def test_empty_messages(self):
        assert to_prompt_only([]) == []

    def test_assistant_turns_stripped(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        result = to_prompt_only(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Q1"


class TestSamplePrompts:
    def _make_dataset(self, n: int = 50, n_sources: int = 3) -> Dataset:
        rows = []
        sources = [f"source_{i}" for i in range(n_sources)]
        for i in range(n):
            src = sources[i % n_sources]
            msgs = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
            rows.append({"messages": msgs, "source": src})
        return _make_tulu_dataset(rows)

    def test_returns_correct_count(self):
        ds = self._make_dataset(n=100, n_sources=5)
        result = sample_prompts(n_prompts=20, seed=42, dataset=ds)
        assert len(result) == 20

    def test_output_has_messages_column(self):
        ds = self._make_dataset(n=50)
        result = sample_prompts(n_prompts=10, seed=42, dataset=ds)
        assert "messages" in result.column_names

    def test_messages_are_prompt_only(self):
        ds = self._make_dataset(n=50)
        result = sample_prompts(n_prompts=10, seed=42, dataset=ds)
        for msgs in result["messages"]:
            roles = [m["role"] for m in msgs]
            assert "assistant" not in roles
            assert "user" in roles

    def test_preserves_system_prompts(self):
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
                "source": "with_system",
            }
            for i in range(30)
        ]
        ds = _make_tulu_dataset(rows)
        result = sample_prompts(n_prompts=10, seed=42, dataset=ds)
        for msgs in result["messages"]:
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == "Be helpful."

    def test_deterministic_with_same_seed(self):
        ds = self._make_dataset(n=100, n_sources=5)
        r1 = sample_prompts(n_prompts=20, seed=123, dataset=ds)
        r2 = sample_prompts(n_prompts=20, seed=123, dataset=ds)
        assert r1["messages"] == r2["messages"]

    def test_different_seeds_give_different_results(self):
        ds = self._make_dataset(n=100, n_sources=5)
        r1 = sample_prompts(n_prompts=20, seed=1, dataset=ds)
        r2 = sample_prompts(n_prompts=20, seed=2, dataset=ds)
        assert r1["messages"] != r2["messages"]

    def test_stratified_across_sources(self):
        rows = []
        for i in range(80):
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": f"Big Q{i}"},
                        {"role": "assistant", "content": f"Big A{i}"},
                    ],
                    "source": "big_source",
                }
            )
        for i in range(20):
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": f"Small Q{i}"},
                        {"role": "assistant", "content": f"Small A{i}"},
                    ],
                    "source": "small_source",
                }
            )
        ds = _make_tulu_dataset(rows)
        result = sample_prompts(n_prompts=50, seed=42, dataset=ds)

        contents = [msgs[0]["content"] for msgs in result["messages"]]
        big_count = sum(1 for c in contents if c.startswith("Big"))
        small_count = sum(1 for c in contents if c.startswith("Small"))
        assert big_count > 0
        assert small_count > 0
        assert big_count > small_count

    def test_skips_examples_without_user_message(self):
        rows = [
            {
                "messages": [{"role": "system", "content": "Only system"}],
                "source": "sys_only",
            }
        ] * 5 + [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
                "source": "normal",
            }
            for i in range(30)
        ]
        ds = _make_tulu_dataset(rows)
        result = sample_prompts(n_prompts=20, seed=42, dataset=ds)
        for msgs in result["messages"]:
            assert any(m["role"] == "user" for m in msgs)
