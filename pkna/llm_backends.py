"""LLM backend abstraction for Gemini and Anthropic (Bedrock).

Provides a unified interface for generating text from different LLM providers,
with retry logic, tool-use support (Anthropic), and structured output.
"""

import inspect
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic import AnthropicBedrock
from anthropic.types import MessageParam, ToolParam
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    GenerateContentConfig,
    HttpOptions,
    Part,
    ThinkingConfig,
)
from pydantic import BaseModel, TypeAdapter

load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_ANTHROPIC_MODEL = "eu.anthropic.claude-sonnet-4-6"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0
API_TIMEOUT_SECONDS = 300
MAX_TOOL_ITERATIONS = 64


def _retry_with_backoff(
    fn: Callable[[], Any],
    is_retryable: Callable[[Exception], bool],
) -> Any | None:
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e):
                raise
            if attempt == MAX_RETRIES - 1:
                log.error(f"Max retries ({MAX_RETRIES}) exceeded: {e}")
                return None
            jitter = backoff * 0.25 * (2 * random.random() - 1)
            sleep_time = min(backoff + jitter, MAX_BACKOFF_SECONDS)
            log.warning(
                f"Retryable error (attempt {attempt + 1}/{MAX_RETRIES}), "
                f"retrying in {sleep_time:.1f}s: {e}"
            )
            time.sleep(sleep_time)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)
    return None


# ============================================================================
# LLM Backend Abstraction
# ============================================================================


@dataclass
class GenerateResult:
    text: str
    model_name: str
    usage: dict[str, Any] = field(default_factory=dict)
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class LLMBackend(ABC):
    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        """Generate a response from the LLM.

        Args:
            system: System prompt.
            messages: Conversation as [{"role": "user"|"assistant", "content": "..."}].
            tools: Python callables the model may invoke. The backend runs the
                tool-call loop internally and returns the final text.
            response_schema: If set, constrain output to JSON matching this
                Pydantic model's schema.
        """
        ...


# ============================================================================
# Gemini Backend
# ============================================================================


class GeminiBackend(LLMBackend):
    def __init__(self, model: str):
        self._model = model
        self._client = genai.Client(
            http_options=HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000)
        )

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        s = str(e).lower()
        is_timeout = "timeout" in s or "timed out" in s
        is_rate = ("resource" in s and "exhausted" in s) or "429" in s
        return is_timeout or is_rate

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any]:
        if not response.usage_metadata:
            return {}
        um = response.usage_metadata
        return {
            "prompt_tokens": um.prompt_token_count,
            "completion_tokens": um.candidates_token_count,
            "total_tokens": um.total_token_count,
        }

    @staticmethod
    def _extract_parts(response: Any) -> tuple[str | None, str]:
        """Extract thinking summary and visible text from a Gemini response."""
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_parts.append(part.text)
            else:
                text_parts.append(part.text)
        thinking = "\n".join(thinking_parts) if thinking_parts else None
        text = "\n".join(text_parts)
        return thinking, text

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": 0.7,
            "top_p": 0.95,
            "thinking_config": ThinkingConfig(include_thoughts=True),
        }
        if tools:
            config_kwargs["tools"] = tools
            config_kwargs["automatic_function_calling"] = (
                AutomaticFunctionCallingConfig(disable=True)
            )
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = list[response_schema]

        config = GenerateContentConfig(**config_kwargs)
        conversation: list[Content] = [
            Content(role=m["role"], parts=[Part.from_text(text=m["content"])])
            for m in messages
        ]

        if not tools:
            return self._generate_no_tools(config, conversation)
        return self._generate_with_tools(config, conversation, tools)

    def _generate_no_tools(
        self,
        config: GenerateContentConfig,
        conversation: list[Content],
    ) -> GenerateResult | None:
        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=conversation,  # type: ignore[arg-type]
                config=config,
            )

        response = _retry_with_backoff(_call, self._is_retryable)
        if response is None:
            return None

        thinking, text = self._extract_parts(response)
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if thinking:
            msg["thinking"] = thinking

        return GenerateResult(
            text=text,
            model_name=self._model,
            usage=self._extract_usage(response),
            thinking=thinking,
            messages=[msg],
        )

    def _generate_with_tools(
        self,
        config: GenerateContentConfig,
        conversation: list[Content],
        tools: list[Any],
    ) -> GenerateResult | None:
        tool_map: dict[str, Any] = {fn.__name__: fn for fn in tools}
        all_thinking: list[str] = []
        all_tool_calls: list[dict[str, Any]] = []
        result_messages: list[dict[str, Any]] = []
        cumulative_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _ in range(MAX_TOOL_ITERATIONS):

            def _call():
                return self._client.models.generate_content(
                    model=self._model,
                    contents=conversation,  # type: ignore[arg-type]
                    config=config,
                )

            response = _retry_with_backoff(_call, self._is_retryable)
            if response is None:
                return None

            usage = self._extract_usage(response)
            for k in cumulative_usage:
                cumulative_usage[k] += usage.get(k, 0)

            thinking, text = self._extract_parts(response)
            if thinking:
                all_thinking.append(thinking)

            fn_calls = response.function_calls
            if not fn_calls:
                msg: dict[str, Any] = {"role": "assistant", "content": text}
                if thinking:
                    msg["thinking"] = thinking
                result_messages.append(msg)
                break

            # Record assistant turn with tool calls
            tc_records: list[dict[str, Any]] = []
            for fc in fn_calls:
                tc_records.append({"name": fc.name, "arguments": dict(fc.args)})
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "tool_calls": tc_records,
            }
            if thinking:
                assistant_msg["thinking"] = thinking
            result_messages.append(assistant_msg)

            # Append the model's raw response to conversation for the next turn
            conversation.append(response.candidates[0].content)

            # Execute tools and build function response parts
            fn_response_parts: list[Part] = []
            for fc in fn_calls:
                fn = tool_map.get(fc.name)
                if fn is None:
                    result_text = f"Error: unknown tool '{fc.name}'"
                else:
                    try:
                        result_text = fn(**fc.args)
                    except Exception as e:
                        result_text = f"Error: {e}"

                all_tool_calls.append(
                    {"name": fc.name, "arguments": dict(fc.args), "result": result_text}
                )
                result_messages.append(
                    {"role": "tool", "name": fc.name, "content": result_text}
                )
                fn_response_parts.append(
                    Part.from_function_response(
                        name=fc.name, response={"result": result_text}
                    )
                )

            conversation.append(Content(role="tool", parts=fn_response_parts))
        else:
            log.warning("Gemini: max tool iterations reached")
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


# ============================================================================
# Anthropic Tool Schema Helpers
# ============================================================================


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is type(None):
        return {"type": "string"}

    origin = getattr(annotation, "__origin__", None)

    if origin is type(None):
        return {"type": "string"}
    args = getattr(annotation, "__args__", None)
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            base = _python_type_to_json_schema(non_none[0])
            return {**base, "nullable": True}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    if origin is list:
        item_args = getattr(annotation, "__args__", None)
        if item_args:
            return {"type": "array", "items": _python_type_to_json_schema(item_args[0])}
        return {"type": "array"}

    return {"type": "string"}


def _parse_google_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring into description and param descriptions."""
    lines = docstring.strip().split("\n")
    desc_lines: list[str] = []
    param_descs: dict[str, str] = {}
    in_args = False
    current_param: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if stripped.lower().startswith("returns:"):
            in_args = False
            continue
        if in_args:
            match = re.match(r"^(\w+)\s*(?:\(.*?\))?:\s*(.+)", stripped)
            if match:
                current_param = match.group(1)
                param_descs[current_param] = match.group(2).strip()
            elif current_param and stripped:
                param_descs[current_param] += " " + stripped
        elif not in_args and not stripped.startswith("---"):
            desc_lines.append(stripped)

    description = " ".join(desc_lines).strip()
    return description, param_descs


def _callable_to_anthropic_tool(fn: Any) -> ToolParam:
    """Convert a Python callable to an Anthropic tool definition dict."""
    sig = inspect.signature(fn)
    hints = dict(inspect.get_annotations(fn))
    hints.pop("return", None)
    docstring = inspect.getdoc(fn) or ""
    description, param_descs = _parse_google_docstring(docstring)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, param.annotation)
        schema = _python_type_to_json_schema(annotation)
        if name in param_descs:
            schema["description"] = param_descs[name]
        properties[name] = schema

        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", None)
        is_optional = args and type(None) in args
        has_default = param.default is not inspect.Parameter.empty
        if not is_optional and not has_default and origin is not type(None):
            required.append(name)

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    return ToolParam(
        name=fn.__name__,
        description=description,
        input_schema=input_schema,
    )


# ============================================================================
# Anthropic (Bedrock) Backend
# ============================================================================


def _to_anthropic_messages(messages: list[dict[str, str]]) -> list[MessageParam]:
    return [
        MessageParam(role=m["role"], content=m["content"])  # type: ignore[typeddict-item]
        for m in messages
    ]


def _add_additional_properties_false(schema: dict | list) -> None:
    """Anthropic requires 'additionalProperties: false' on all object types."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
        for v in schema.values():
            if isinstance(v, (dict, list)):
                _add_additional_properties_false(v)
    elif isinstance(schema, list):
        for item in schema:
            if isinstance(item, (dict, list)):
                _add_additional_properties_false(item)


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str):
        self._model = model
        self._client = AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )

    def _is_retryable(self, e: Exception) -> bool:
        if isinstance(e, anthropic.RateLimitError | anthropic.APITimeoutError):
            return True
        if isinstance(e, anthropic.APIStatusError) and e.status_code in (429, 503, 529):
            log.debug("Retryable API status error (code %d): %s", e.status_code, e)
            return True
        return False

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        if tools:
            return self._generate_with_tools(system, messages, tools)
        if response_schema:
            return self._generate_with_schema(system, messages, response_schema)
        return self._generate_plain(system, messages)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any]:
        if not (hasattr(response, "usage") and response.usage):
            return {}
        return {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    @staticmethod
    def _extract_content(response: Any) -> tuple[str | None, str]:
        """Extract thinking text and visible text from an Anthropic response."""
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif hasattr(block, "text"):
                text_parts.append(block.text)
        thinking = "\n".join(thinking_parts) if thinking_parts else None
        text = "\n".join(text_parts)
        return thinking, text

    def _make_result(self, response: Any) -> GenerateResult:
        usage = self._extract_usage(response)
        thinking, text = self._extract_content(response)
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if thinking:
            msg["thinking"] = thinking
        return GenerateResult(
            text=text,
            model_name=self._model,
            usage=usage,
            thinking=thinking,
            messages=[msg],
        )

    def _generate_plain(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> GenerateResult | None:
        api_messages = _to_anthropic_messages(messages)

        def _call():
            return self._client.messages.create(
                model=self._model,
                max_tokens=8192,
                temperature=1.0,
                system=system,
                messages=api_messages,
            )

        response = _retry_with_backoff(_call, self._is_retryable)
        if response is None:
            return None
        return self._make_result(response)

    def _generate_with_schema(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> GenerateResult | None:
        json_schema = TypeAdapter(list[schema]).json_schema()  # type: ignore[valid-type]
        _add_additional_properties_false(json_schema)
        api_messages = _to_anthropic_messages(messages)

        def _call():
            return self._client.messages.create(
                model=self._model,
                max_tokens=8192,
                temperature=1.0,
                system=system,
                messages=api_messages,
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema,
                    }
                },
            )

        response = _retry_with_backoff(_call, self._is_retryable)
        if response is None:
            return None
        return self._make_result(response)

    def _generate_with_tools(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Any],
    ) -> GenerateResult | None:
        tool_defs = [_callable_to_anthropic_tool(fn) for fn in tools]
        tool_map: dict[str, Any] = {fn.__name__: fn for fn in tools}

        api_messages: list[MessageParam] = _to_anthropic_messages(messages)
        cumulative_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        all_thinking: list[str] = []
        all_tool_calls: list[dict[str, Any]] = []
        result_messages: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ITERATIONS):

            def _call():
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=8192,
                    temperature=1.0,
                    system=system,
                    messages=api_messages,
                    tools=tool_defs,
                )

            response = _retry_with_backoff(_call, self._is_retryable)
            if response is None:
                return None

            usage = self._extract_usage(response)
            for k in cumulative_usage:
                cumulative_usage[k] += usage.get(k, 0)

            thinking, text = self._extract_content(response)
            if thinking:
                all_thinking.append(thinking)

            if response.stop_reason != "tool_use":
                msg: dict[str, Any] = {"role": "assistant", "content": text}
                if thinking:
                    msg["thinking"] = thinking
                result_messages.append(msg)
                break

            # Build assistant message with tool calls
            tc_records: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    tc_records.append(
                        {"name": block.name, "arguments": dict(block.input)}
                    )
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "tool_calls": tc_records,
            }
            if thinking:
                assistant_msg["thinking"] = thinking
            result_messages.append(assistant_msg)

            api_messages.append(
                MessageParam(role="assistant", content=response.content)
            )

            # Execute tools
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = tool_map.get(block.name)
                if fn is None:
                    result_text = f"Error: unknown tool '{block.name}'"
                else:
                    try:
                        result_text = fn(**block.input)
                    except Exception as e:
                        result_text = f"Error: {e}"

                all_tool_calls.append(
                    {
                        "name": block.name,
                        "arguments": dict(block.input),
                        "result": result_text,
                    }
                )
                result_messages.append(
                    {"role": "tool", "name": block.name, "content": result_text}
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

            api_messages.append(
                MessageParam(role="user", content=tool_results)  # type: ignore[typeddict-item]
            )
        else:
            log.warning("Max tool iterations reached")
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


# ============================================================================
# Factory
# ============================================================================


def create_backend(backend_name: str, model: str | None = None) -> LLMBackend:
    if backend_name == "gemini":
        return GeminiBackend(model=model or DEFAULT_GEMINI_MODEL)
    if backend_name == "anthropic":
        return AnthropicBackend(model=model or DEFAULT_ANTHROPIC_MODEL)
    raise ValueError(f"Unknown backend: {backend_name}")
