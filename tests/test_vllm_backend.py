"""Unit tests for the vLLM backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pkna.llm.backends import create_backend
from pkna.llm.vllm_backend import VllmBackend


def _fake_response(
    content: str = "hello",
    reasoning: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Build a fake OpenAI ChatCompletion response."""
    msg = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_tool_call(name: str, arguments: dict, call_id: str = "call_0"):
    fn = SimpleNamespace(name=name, arguments='{"query": "test"}')
    if arguments:
        import json

        fn.arguments = json.dumps(arguments)
    return SimpleNamespace(id=call_id, type="function", function=fn)


class TestVllmBackendNoTools:
    def test_plain_text(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")
        response = _fake_response(content="Hello world")
        backend._client = MagicMock()
        backend._client.chat.completions.create.return_value = response

        result = backend.generate(
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result is not None
        assert result.text == "Hello world"
        assert result.model_name == "test-model"
        assert result.thinking is None
        assert result.tool_calls == []
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "assistant"
        assert result.messages[0]["content"] == "Hello world"

    def test_with_thinking(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")
        response = _fake_response(content="Answer", reasoning="Let me think...")
        backend._client = MagicMock()
        backend._client.chat.completions.create.return_value = response

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Q"}],
        )

        assert result is not None
        assert result.text == "Answer"
        assert result.thinking == "Let me think..."
        assert result.messages[0]["thinking"] == "Let me think..."

    def test_empty_response_returns_none(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")
        response = _fake_response(content="", reasoning=None)
        backend._client = MagicMock()
        backend._client.chat.completions.create.return_value = response

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Q"}],
        )

        assert result is None

    def test_usage_extracted(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")
        response = _fake_response(content="ok", prompt_tokens=100, completion_tokens=50)
        backend._client = MagicMock()
        backend._client.chat.completions.create.return_value = response

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Q"}],
        )

        assert result is not None
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    def test_response_schema_raises(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")
        with pytest.raises(NotImplementedError, match="response_schema"):
            backend.generate(
                system="sys",
                messages=[{"role": "user", "content": "Q"}],
                response_schema=dict,
            )


class TestVllmBackendWithTools:
    def test_single_tool_call(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")

        tc = _fake_tool_call("search_knowledge", {"query": "test"})
        response_with_tool = _fake_response(
            content="", tool_calls=[tc], finish_reason="tool_calls"
        )
        response_final = _fake_response(content="Found it")

        backend._client = MagicMock()
        backend._client.chat.completions.create.side_effect = [
            response_with_tool,
            response_final,
        ]

        def search_knowledge(query: str) -> str:
            """Search the knowledge base."""
            return f"Result for: {query}"

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Search for test"}],
            tools=[search_knowledge],
        )

        assert result is not None
        assert result.text == "Found it"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search_knowledge"
        assert result.tool_calls[0]["result"] == "Result for: test"
        assert any(m["role"] == "tool" for m in result.messages)

    def test_tool_error_handled(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")

        tc = _fake_tool_call("bad_tool", {"x": 1})
        response_with_tool = _fake_response(
            content="", tool_calls=[tc], finish_reason="tool_calls"
        )
        response_final = _fake_response(content="Handled error")

        backend._client = MagicMock()
        backend._client.chat.completions.create.side_effect = [
            response_with_tool,
            response_final,
        ]

        def bad_tool(x: int) -> str:
            """A tool that fails."""
            raise ValueError("boom")

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Do it"}],
            tools=[bad_tool],
        )

        assert result is not None
        assert result.tool_calls[0]["result"] == "Error: boom"

    def test_unknown_tool_handled(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")

        tc = _fake_tool_call("nonexistent", {})
        response_with_tool = _fake_response(
            content="", tool_calls=[tc], finish_reason="tool_calls"
        )
        response_final = _fake_response(content="Done")

        backend._client = MagicMock()
        backend._client.chat.completions.create.side_effect = [
            response_with_tool,
            response_final,
        ]

        def real_tool() -> str:
            """A real tool."""
            return "ok"

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Call it"}],
            tools=[real_tool],
        )

        assert result is not None
        assert "unknown tool" in result.tool_calls[0]["result"]

    def test_thinking_accumulated_across_tool_turns(self):
        backend = VllmBackend(model="test-model", base_url="http://fake:8000/v1")

        tc = _fake_tool_call("my_tool", {"q": "x"})
        r1 = _fake_response(content="", tool_calls=[tc], reasoning="Think step 1")
        r2 = _fake_response(content="Final answer", reasoning="Think step 2")

        backend._client = MagicMock()
        backend._client.chat.completions.create.side_effect = [r1, r2]

        def my_tool(q: str) -> str:
            """Tool."""
            return "result"

        result = backend.generate(
            system="sys",
            messages=[{"role": "user", "content": "Q"}],
            tools=[my_tool],
        )

        assert result is not None
        assert result.thinking == "Think step 1\nThink step 2"


class TestVllmBackendMessageConversion:
    def test_user_and_assistant(self):
        msgs = VllmBackend._to_api_messages(
            "system prompt",
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        )
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "system prompt"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_tool_messages(self):
        msgs = VllmBackend._to_api_messages(
            "sys",
            [
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
                },
                {"role": "tool", "name": "search", "content": "found"},
            ],
        )
        assert len(msgs) == 4
        assert msgs[2]["role"] == "assistant"
        assert "tool_calls" in msgs[2]
        assert msgs[3]["role"] == "tool"


class TestCreateBackendVllm:
    def test_vllm_requires_model(self):
        with pytest.raises(ValueError, match="requires a model name"):
            create_backend("vllm")

    @patch.dict("os.environ", {"VLLM_BASE_URL": "http://custom:9000/v1"})
    def test_vllm_uses_env_base_url(self):
        backend = create_backend("vllm", model="test-model")
        assert isinstance(backend, VllmBackend)
        assert backend._client.base_url.host == "custom"

    def test_vllm_default_base_url(self):
        backend = create_backend("vllm", model="test-model")
        assert isinstance(backend, VllmBackend)
        assert "localhost" in str(backend._client.base_url)
