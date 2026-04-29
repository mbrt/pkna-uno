"""Unit tests for SFT dataset conversion logic."""

from pkna.datagen.types import DatagenTrace
from pkna.inference.system_prompts import TRACE_GUIDANCE_CLOSE, TRACE_GUIDANCE_OPEN
from pkna.training.sft_dataset import (
    _convert_message,
    _convert_tool_calls,
    patch_chat_template_for_sft,
    trace_to_chatml_text,
    trace_to_messages,
)


def _make_trace(
    id: str = "t-001",
    messages: list[dict] | None = None,
) -> DatagenTrace:
    return DatagenTrace(
        id=id,
        metadata={},
        memory_context="",
        user_summary="Paperino",
        messages=messages or [],
    )


class TestConvertToolCalls:
    def test_single_call(self):
        raw = [{"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}]
        result = _convert_tool_calls(raw)
        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "search_wiki",
                    "arguments": {"keywords": "Xadhoom"},
                },
            }
        ]

    def test_multiple_calls(self):
        raw = [
            {"name": "search_wiki", "arguments": {"keywords": "Ducklair"}},
            {"name": "delegate", "arguments": {"task": "solve equation"}},
        ]
        result = _convert_tool_calls(raw)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "search_wiki"
        assert result[1]["function"]["name"] == "delegate"
        assert all(tc["type"] == "function" for tc in result)

    def test_empty_arguments(self):
        raw = [{"name": "recall", "arguments": {}}]
        result = _convert_tool_calls(raw)
        assert result == [
            {"type": "function", "function": {"name": "recall", "arguments": {}}}
        ]

    def test_missing_fields_default(self):
        raw = [{}]
        result = _convert_tool_calls(raw)
        assert result == [
            {"type": "function", "function": {"name": "", "arguments": {}}}
        ]


class TestConvertMessage:
    def test_user_message(self):
        msg = {"role": "user", "content": "Ciao, Uno!"}
        assert _convert_message(msg) == {"role": "user", "content": "Ciao, Uno!"}

    def test_assistant_with_thinking(self):
        msg = {
            "role": "assistant",
            "content": "Ciao, socio!",
            "thinking": "Paperino is greeting me. Light tone.",
        }
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Ciao, socio!"
        assert result["reasoning_content"] == "Paperino is greeting me. Light tone."
        assert "tool_calls" not in result

    def test_assistant_without_thinking(self):
        msg = {"role": "assistant", "content": "Sure thing."}
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Sure thing."
        assert "reasoning_content" not in result

    def test_assistant_empty_thinking(self):
        msg = {"role": "assistant", "content": "Ok.", "thinking": ""}
        result = _convert_message(msg)
        assert "reasoning_content" not in result

    def test_assistant_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "search_wiki", "arguments": {"keywords": "Evroniani"}}
            ],
        }
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "search_wiki"

    def test_assistant_with_thinking_and_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Let me look that up.",
            "thinking": "Need to search the wiki.",
            "tool_calls": [
                {"name": "search_wiki", "arguments": {"keywords": "Ducklair"}}
            ],
        }
        result = _convert_message(msg)
        assert result["reasoning_content"] == "Need to search the wiki."
        assert result["content"] == "Let me look that up."
        assert len(result["tool_calls"]) == 1

    def test_assistant_empty_tool_calls(self):
        msg = {"role": "assistant", "content": "Hello", "tool_calls": []}
        result = _convert_message(msg)
        assert "tool_calls" not in result

    def test_tool_message(self):
        msg = {"role": "tool", "content": "Xadhoom is a Xerbian scientist."}
        assert _convert_message(msg) == {
            "role": "tool",
            "content": "Xadhoom is a Xerbian scientist.",
        }

    def test_unknown_role_passthrough(self):
        msg = {"role": "developer", "content": "debug info"}
        assert _convert_message(msg) == {"role": "developer", "content": "debug info"}


SYSTEM_PROMPT = "You are Uno."


class TestTraceToMessages:
    def test_single_turn(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Ciao, socio!",
                    "thinking": "A casual greeting.",
                },
            ]
        )
        messages = trace_to_messages(trace, SYSTEM_PROMPT)
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are Uno."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2]["role"] == "assistant"
        assert messages[2]["reasoning_content"] == "A casual greeting."
        assert messages[2]["content"] == "Ciao, socio!"

    def test_multi_turn_with_tools(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Who is Xadhoom?"},
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "Factual question. Search the wiki.",
                    "tool_calls": [
                        {
                            "name": "search_wiki",
                            "arguments": {"keywords": "Xadhoom"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_wiki",
                    "content": "Xadhoom is a Xerbian scientist.",
                },
                {
                    "role": "assistant",
                    "content": "Xadhoom is a powerful Xerbian.",
                    "thinking": "Got wiki result, synthesize.",
                },
                {"role": "user", "content": "Thanks!"},
                {
                    "role": "assistant",
                    "content": "Prego, socio!",
                    "thinking": "Casual farewell.",
                },
            ]
        )
        messages = trace_to_messages(trace, SYSTEM_PROMPT)
        assert len(messages) == 7  # system + 6 from trace

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        assert messages[2]["role"] == "assistant"
        assert messages[2]["reasoning_content"] == "Factual question. Search the wiki."
        assert messages[2]["tool_calls"][0]["type"] == "function"
        assert messages[2]["tool_calls"][0]["function"]["name"] == "search_wiki"

        assert messages[3]["role"] == "tool"
        assert "Xerbian scientist" in messages[3]["content"]

        assert messages[4]["role"] == "assistant"
        assert messages[4]["reasoning_content"] == "Got wiki result, synthesize."

        assert messages[5]["role"] == "user"
        assert messages[6]["role"] == "assistant"
        assert messages[6]["content"] == "Prego, socio!"

    def test_system_prompt_preserved(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )
        custom = "Custom system prompt with personality details."
        messages = trace_to_messages(trace, custom)
        assert messages[0]["content"] == custom

    def test_empty_messages(self):
        trace = _make_trace(messages=[])
        messages = trace_to_messages(trace, SYSTEM_PROMPT)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_tool_name_not_in_output(self):
        """The 'name' field on tool messages is datagen metadata, not part of
        the standard chat format (tool identity comes from the preceding
        tool_call)."""
        trace = _make_trace(
            messages=[
                {
                    "role": "tool",
                    "name": "search_wiki",
                    "content": "result text",
                },
            ]
        )
        messages = trace_to_messages(trace, SYSTEM_PROMPT)
        tool_msg = messages[1]
        assert tool_msg == {"role": "tool", "content": "result text"}
        assert "name" not in tool_msg


class TestTraceGuidanceStripping:
    def test_user_message_with_guidance_stripped(self):
        content = (
            "<context>\nInterlocutor: Paperino\n</context>\n\n"
            f"{TRACE_GUIDANCE_OPEN}\nShow tradeoff analysis.\n"
            f"{TRACE_GUIDANCE_CLOSE}\n\n"
            "<message>\nCiao, Uno!\n</message>"
        )
        msg = {"role": "user", "content": content}
        result = _convert_message(msg)
        assert TRACE_GUIDANCE_OPEN not in result["content"]
        assert "Show tradeoff analysis" not in result["content"]
        assert "Ciao, Uno!" in result["content"]

    def test_user_message_without_guidance_unchanged(self):
        content = "<context>\nInterlocutor: Paperino\n</context>\n\n<message>\nCiao!\n</message>"
        msg = {"role": "user", "content": content}
        result = _convert_message(msg)
        assert result["content"] == content

    def test_plain_user_message_unchanged(self):
        msg = {"role": "user", "content": "Hello, Uno!"}
        result = _convert_message(msg)
        assert result["content"] == "Hello, Uno!"

    def test_assistant_message_not_affected(self):
        msg = {
            "role": "assistant",
            "content": f"Some text with {TRACE_GUIDANCE_OPEN} in it.",
        }
        result = _convert_message(msg)
        assert TRACE_GUIDANCE_OPEN in result["content"]


class _FakeTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer.

    Records the messages it was called with and emits a predictable text
    form that mirrors ChatML closely enough for assertions, without
    pulling in ``transformers``. Argument rendering stays faithful: only
    the keys actually present in the call are emitted, so tests fail if a
    future refactor reintroduces ``None``-padded keys.

    The ``chat_template`` attribute contains the exact clause that
    ``patch_chat_template_for_sft`` looks for, so callers can exercise
    the patching path even against the fake.
    """

    chat_template = (
        "{# fake qwen-ish template #}\n"
        "        {%- if loop.index0 > ns.last_query_index %}\n"
    )

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        chat_template=None,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.calls.append(
            {
                "messages": messages,
                "chat_template": chat_template,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            parts.append(f"<|im_start|>{role}")
            if role == "assistant":
                parts.append(f"<think>\n{m.get('reasoning_content', '')}\n</think>")
                if m.get("content"):
                    parts.append(m["content"])
                for tc in m.get("tool_calls", []) or []:
                    fn = tc["function"]
                    parts.append(f"<tool_call>\n<function={fn['name']}>")
                    for k, v in fn["arguments"].items():
                        parts.append(f"<parameter={k}>\n{v}\n</parameter>")
                    parts.append("</function>\n</tool_call>")
            elif role == "tool":
                parts.append(f"<tool_response>\n{m['content']}\n</tool_response>")
            else:
                parts.append(m.get("content", ""))
            parts.append("<|im_end|>")
        return "\n".join(parts)


class TestPatchChatTemplateForSft:
    def test_rewrites_position_check_to_content_check(self):
        src = (
            "... preamble ...\n"
            "        {%- if loop.index0 > ns.last_query_index %}\n"
            "            render with think\n"
            "        {%- endif %}\n"
        )
        patched = patch_chat_template_for_sft(src)
        assert "reasoning_content or loop.index0 > ns.last_query_index" in patched
        assert "{%- if loop.index0 > ns.last_query_index %}" not in patched

    def test_raises_when_clause_missing(self):
        import pytest

        with pytest.raises(ValueError, match="Qwen think-position clause"):
            patch_chat_template_for_sft("{# some other template #}")


class TestTraceToChatmlText:
    def test_renders_and_invokes_template(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Ciao"},
                {
                    "role": "assistant",
                    "content": "Ciao, socio.",
                    "thinking": "Casual reply.",
                },
            ]
        )
        tok = _FakeTokenizer()
        text = trace_to_chatml_text(trace, SYSTEM_PROMPT, tok)
        assert "<|im_start|>system" in text
        assert "Casual reply." in text
        assert "Ciao, socio." in text
        assert len(tok.calls) == 1
        kwargs = tok.calls[0]
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is False
        assert kwargs["enable_thinking"] is True

    def test_tool_call_arguments_have_no_none(self):
        """Regression guard: rendered tool calls must only carry the argument
        keys that were actually set on the trace (no struct-unification
        leakage from HF Datasets)."""
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Search that up"},
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
            ]
        )
        tok = _FakeTokenizer()
        text = trace_to_chatml_text(trace, SYSTEM_PROMPT, tok)
        assert "<parameter=keywords>" in text
        assert "Xadhoom" in text
        assert "None" not in text
        for forbidden in ("query", "segment_id", "task", "value", "key"):
            assert f"<parameter={forbidden}>" not in text

    def test_passes_patched_template_to_tokenizer(self):
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello", "thinking": "Short reply."},
            ]
        )
        tok = _FakeTokenizer()
        trace_to_chatml_text(trace, SYSTEM_PROMPT, tok)
        assert len(tok.calls) == 1
        passed = tok.calls[0]["chat_template"]
        assert passed is not None
        assert "reasoning_content or loop.index0 > ns.last_query_index" in passed


class TestTraceToChatmlTextWithRealTokenizer:
    """Integration tests against a real Qwen3 tokenizer.

    Skipped when the local merged-model tokenizer isn't available.
    """

    @staticmethod
    def _load_tokenizer():
        import pytest

        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        from pathlib import Path

        path = Path("output/sft/qwen3-5-4b-merged")
        if not path.exists():
            pytest.skip(f"Qwen3 tokenizer not available at {path}")
        return AutoTokenizer.from_pretrained(path)

    def test_multi_turn_preserves_thinking_on_intermediate_turn(self):
        """Regression guard: the patched template must render <think> on an
        intermediate assistant turn that has real thinking, even when another
        user turn follows. This is exactly the case Qwen's stock template
        drops."""
        tok = self._load_tokenizer()
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "First question."},
                {
                    "role": "assistant",
                    "content": "Intermediate answer.",
                    "thinking": "INTERMEDIATE_REASONING_MARKER",
                },
                {"role": "user", "content": "Follow-up question."},
                {
                    "role": "assistant",
                    "content": "Final answer.",
                    "thinking": "FINAL_REASONING_MARKER",
                },
            ]
        )
        text = trace_to_chatml_text(trace, SYSTEM_PROMPT, tok)
        assert "INTERMEDIATE_REASONING_MARKER" in text
        assert "FINAL_REASONING_MARKER" in text
        # Both should appear inside proper <think>...</think> blocks.
        assert text.count("<think>") >= 2
        assert text.count("</think>") >= 2

    def test_multi_turn_tool_call_thinking_precedes_tool_call(self):
        """Patched template renders <think> before the <tool_call> block
        on a tool-calling intermediate turn."""
        tok = self._load_tokenizer()
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "Who is Xadhoom?"},
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "TOOL_REASONING_MARKER",
                    "tool_calls": [
                        {
                            "name": "search_knowledge",
                            "arguments": {"keywords": "Xadhoom"},
                        }
                    ],
                },
                {"role": "tool", "content": "Xadhoom is a Xerbian scientist."},
                {
                    "role": "assistant",
                    "content": "She's a Xerbian scientist.",
                    "thinking": "FINAL_REASONING_MARKER",
                },
                {"role": "user", "content": "Thanks!"},
                {
                    "role": "assistant",
                    "content": "Prego, socio!",
                    "thinking": "CLOSING_REASONING_MARKER",
                },
            ]
        )
        text = trace_to_chatml_text(trace, SYSTEM_PROMPT, tok)
        tool_idx = text.index("<tool_call>")
        tool_reason_idx = text.index("TOOL_REASONING_MARKER")
        assert tool_reason_idx < tool_idx, (
            "thinking must appear before the tool_call block it motivates"
        )
        # All three reasoning markers should survive the render, not just the last.
        assert "TOOL_REASONING_MARKER" in text
        assert "FINAL_REASONING_MARKER" in text
        assert "CLOSING_REASONING_MARKER" in text

    def test_no_spurious_empty_think_on_intermediate_turn_without_thinking(self):
        """When an intermediate assistant turn has no thinking, the patched
        template must not inject an empty <think></think> block (that would
        teach the model to always emit empty reasoning). Qwen's default
        emits empty <think> only on the last turn; we preserve that."""
        tok = self._load_tokenizer()
        trace = _make_trace(
            messages=[
                {"role": "user", "content": "First."},
                {"role": "assistant", "content": "No reasoning here."},
                {"role": "user", "content": "Second."},
                {
                    "role": "assistant",
                    "content": "Final.",
                    "thinking": "FINAL_REASONING_MARKER",
                },
            ]
        )
        text = trace_to_chatml_text(trace, SYSTEM_PROMPT, tok)
        # Split on the first assistant block and check it has no think tags.
        first_asst = text.split("<|im_start|>assistant\n", 1)[1]
        first_asst = first_asst.split("<|im_end|>", 1)[0]
        assert "<think>" not in first_asst
        assert "</think>" not in first_asst
