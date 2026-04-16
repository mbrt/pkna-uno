"""Shared test doubles for LLM backends.

Provides FakeBackend (fixed single response) and SequentialBackend
(queue of responses) for use in unit tests and pipeline smoke tests.
"""

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from pkna.llm.backends import GenerateResult, LLMBackend


class FakeBackend(LLMBackend):
    """Backend that returns a pre-configured GenerateResult.

    Stores the last system prompt and messages for assertion.
    """

    def __init__(self, result: GenerateResult | None):
        self._result = result
        self.last_system: str = ""
        self.last_messages: list[dict[str, str]] = []

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> GenerateResult | None:
        self.last_system = system
        self.last_messages = messages
        return self._result


class SequentialBackend(LLMBackend):
    """Backend that returns results from a queue, one per generate() call."""

    def __init__(self, results: Sequence[GenerateResult | None]):
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


def make_result(
    text: str = "",
    thinking: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> GenerateResult:
    """Build a GenerateResult with sensible defaults."""
    return GenerateResult(
        text=text,
        model_name="fake",
        thinking=thinking,
        tool_calls=tool_calls or [],
        messages=messages or [],
    )
