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
    Content,
    GenerateContentConfig,
    HttpOptions,
    Part,
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
        }
        if tools:
            config_kwargs["tools"] = tools
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = list[response_schema]

        config = GenerateContentConfig(**config_kwargs)
        conversation = [
            Content(role=m["role"], parts=[Part.from_text(text=m["content"])])
            for m in messages
        ]

        def _is_retryable(e: Exception) -> bool:
            s = str(e).lower()
            is_timeout = "timeout" in s or "timed out" in s
            is_rate = ("resource" in s and "exhausted" in s) or "429" in s
            return is_timeout or is_rate

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=conversation,
                config=config,
            )

        response = _retry_with_backoff(_call, _is_retryable)
        if response is None:
            return None

        usage: dict[str, Any] = {}
        if response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count,
                "total_tokens": um.total_token_count,
            }

        return GenerateResult(
            text=response.text or "",
            model_name=self._model,
            usage=usage,
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
        if isinstance(e, anthropic.APIStatusError) and e.status_code in (429, 529):
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

    def _make_result(self, response: Any) -> GenerateResult:
        usage: dict[str, Any] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        return GenerateResult(
            text=self._extract_text(response),
            model_name=self._model,
            usage=usage,
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

            if hasattr(response, "usage") and response.usage:
                cumulative_usage["input_tokens"] += response.usage.input_tokens
                cumulative_usage["output_tokens"] += response.usage.output_tokens

            if response.stop_reason != "tool_use":
                return GenerateResult(
                    text=self._extract_text(response),
                    model_name=self._model,
                    usage=cumulative_usage,
                )

            api_messages.append(
                MessageParam(role="assistant", content=response.content)
            )

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

        log.warning("Max tool iterations reached")
        return None

    @staticmethod
    def _extract_text(response: Any) -> str:
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)


# ============================================================================
# Factory
# ============================================================================


def create_backend(backend_name: str, model: str | None = None) -> LLMBackend:
    if backend_name == "gemini":
        return GeminiBackend(model=model or DEFAULT_GEMINI_MODEL)
    if backend_name == "anthropic":
        return AnthropicBackend(model=model or DEFAULT_ANTHROPIC_MODEL)
    raise ValueError(f"Unknown backend: {backend_name}")
