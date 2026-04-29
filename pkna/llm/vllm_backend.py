"""vLLM inference backend using the OpenAI-compatible API.

Connects to an external vLLM server over HTTP. The server handles KV cache
management, PagedAttention, and continuous batching -- making this backend
much faster than LocalBackend for multi-turn evaluation.

Start the server separately before running evals::

    vllm serve /path/to/merged-model \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --dtype auto

This module only imports the lightweight ``openai`` client package, not
``vllm`` itself, so it works on machines without GPU drivers.
"""

import json
import logging
from collections.abc import Callable
from typing import Any, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from pkna.llm.backends import (
    MAX_TOOL_ITERATIONS,
    GenerateResult,
    LLMBackend,
    OutputTruncatedError,
    _retry_with_backoff,
)
from pkna.llm.local_backend import _callable_to_tool_dict

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096


class VllmBackend(LLMBackend):
    """Inference backend that talks to a vLLM OpenAI-compatible server.

    Supports tool calling (server must be started with
    ``--enable-auto-tool-choice --tool-call-parser qwen3_xml``) and Qwen3
    thinking mode (``--reasoning-parser qwen3``).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        if isinstance(e, openai.RateLimitError | openai.APITimeoutError):
            return True
        if isinstance(e, openai.APIStatusError) and e.status_code in (429, 503):
            return True
        return False

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any]:
        if not response.usage:
            return {}
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    @staticmethod
    def _to_api_messages(
        system: str, messages: list[dict[str, Any]]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal message format to OpenAI API messages."""
        api_msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            role = m["role"]
            if role == "user":
                api_msgs.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.get("content", ""),
                }
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for i, tc in enumerate(m["tool_calls"])
                    ]
                api_msgs.append(msg)
            elif role == "tool":
                api_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{0}",
                        "content": m["content"],
                    }
                )
        return cast(list[ChatCompletionMessageParam], api_msgs)

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type | None = None,
    ) -> GenerateResult | None:
        if response_schema is not None:
            raise NotImplementedError(
                "VllmBackend does not support response_schema. "
                "Use a cloud backend (gemini/anthropic) for structured output."
            )
        if tools:
            return self._generate_with_tools(system, messages, tools)
        return self._generate_no_tools(system, messages)

    def _call_api(
        self,
        api_messages: list[ChatCompletionMessageParam],
        tool_defs: list[dict[str, Any]] | None = None,
    ) -> Any | None:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
        if tool_defs:
            kwargs["tools"] = tool_defs
            kwargs["tool_choice"] = "auto"

        def _call():
            return self._client.chat.completions.create(**kwargs)

        try:
            return _retry_with_backoff(_call, self._is_retryable)
        except openai.BadRequestError as e:
            if "maximum context length" in str(e) or "max_model_len" in str(e):
                raise OutputTruncatedError(
                    f"Prompt exceeded model context window: {e}"
                ) from e
            raise

    @staticmethod
    def _extract_response(
        response: Any,
    ) -> tuple[str | None, str, list[dict[str, Any]]]:
        """Extract thinking, content, and tool calls from a response."""
        msg = response.choices[0].message
        thinking = getattr(msg, "reasoning", None) or None
        content = msg.content or ""
        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append({"name": tc.function.name, "arguments": args})
        return thinking, content, tool_calls

    def _generate_no_tools(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> GenerateResult | None:
        api_messages = self._to_api_messages(system, messages)
        response = self._call_api(api_messages)
        if response is None:
            return None

        thinking, content, _ = self._extract_response(response)
        if not content and not thinking:
            log.warning("vLLM returned empty response")
            return None

        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if thinking:
            msg["thinking"] = thinking

        return GenerateResult(
            text=content,
            model_name=self._model,
            usage=self._extract_usage(response),
            thinking=thinking,
            messages=[msg],
        )

    def _generate_with_tools(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Any],
    ) -> GenerateResult | None:
        tool_defs = [_callable_to_tool_dict(fn) for fn in tools]
        tool_map: dict[str, Any] = {fn.__name__: fn for fn in tools}

        conversation: list[dict[str, Any]] = list(messages)
        all_thinking: list[str] = []
        all_tool_calls: list[dict[str, Any]] = []
        result_messages: list[dict[str, Any]] = []
        cumulative_usage: dict[str, int] = {}

        for _ in range(MAX_TOOL_ITERATIONS):
            api_messages = self._to_api_messages(system, conversation)
            response = self._call_api(api_messages, tool_defs)
            if response is None:
                return None

            usage = self._extract_usage(response)
            for k, v in usage.items():
                if isinstance(v, int):
                    cumulative_usage[k] = cumulative_usage.get(k, 0) + v

            thinking, content, parsed_calls = self._extract_response(response)
            if thinking:
                all_thinking.append(thinking)

            if not parsed_calls:
                msg: dict[str, Any] = {"role": "assistant", "content": content}
                if thinking:
                    msg["thinking"] = thinking
                result_messages.append(msg)
                break

            tc_records = [
                {"name": tc["name"], "arguments": tc["arguments"]}
                for tc in parsed_calls
            ]
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": tc_records,
            }
            if thinking:
                assistant_msg["thinking"] = thinking
            result_messages.append(assistant_msg)
            conversation.append(assistant_msg)

            for tc in parsed_calls:
                fn = tool_map.get(tc["name"])
                if fn is None:
                    result_text = f"Error: unknown tool '{tc['name']}'"
                else:
                    try:
                        result_text = fn(**tc["arguments"])
                    except Exception as e:
                        result_text = f"Error: {e}"

                all_tool_calls.append({**tc, "result": result_text})
                tool_msg = {
                    "role": "tool",
                    "name": tc["name"],
                    "content": result_text,
                }
                result_messages.append(tool_msg)
                conversation.append(tool_msg)
        else:
            log.warning("vLLM: max tool iterations reached")
            return None

        combined_thinking = "\n".join(all_thinking) if all_thinking else None
        final_text = result_messages[-1].get("content", "") if result_messages else ""

        return GenerateResult(
            text=final_text,
            model_name=self._model,
            usage=cumulative_usage,
            thinking=combined_thinking,
            tool_calls=all_tool_calls,
            messages=result_messages,
        )
