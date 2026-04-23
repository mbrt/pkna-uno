"""Convert DatagenTrace objects into standard chat message format for SFT.

Transforms the internal trace representation (with 'thinking', 'tool_calls',
etc.) into a tokenizer-independent message format using widely adopted
conventions:

- reasoning_content: de facto standard for thinking models (Qwen3, DeepSeek)
- OpenAI-style tool_calls: {"type": "function", "function": {"name", "arguments"}}

The output can be passed directly to tokenizer.apply_chat_template() on any
model that supports these conventions, with no additional conversion step.

No GPU or model dependencies -- this is pure data transformation.
"""

from typing import Any

from pkna.datagen.types import DatagenTrace
from pkna.inference.system_prompts import strip_trace_guidance


def _convert_tool_calls(
    raw_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert internal tool_call dicts to OpenAI tool_calls format.

    Input:  {"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}
    Output: {"type": "function", "function": {"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}}
    """
    converted = []
    for tc in raw_calls:
        name = tc.get("name", "")
        arguments = tc.get("arguments", {})
        converted.append(
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        )
    return converted


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a single trace message to standard chat format."""
    role = msg.get("role", "")

    if role == "user":
        content = strip_trace_guidance(msg.get("content", ""))
        return {"role": "user", "content": content}

    if role == "assistant":
        result: dict[str, Any] = {"role": "assistant"}

        thinking = msg.get("thinking", "")
        if thinking:
            result["reasoning_content"] = thinking

        result["content"] = msg.get("content", "")

        raw_tool_calls = msg.get("tool_calls", [])
        if raw_tool_calls:
            result["tool_calls"] = _convert_tool_calls(raw_tool_calls)

        return result

    if role == "tool":
        return {"role": "tool", "content": msg.get("content", "")}

    return {"role": role, "content": msg.get("content", "")}


def trace_to_messages(trace: DatagenTrace, system_prompt: str) -> list[dict[str, Any]]:
    """Convert a DatagenTrace to standard chat messages for SFT.

    Returns a list of message dicts in the standard format (reasoning_content
    + OpenAI tool_calls), suitable for passing directly to
    tokenizer.apply_chat_template().

    Args:
        trace: The recorded conversation trace.
        system_prompt: Rendered system prompt (template + profile). Passed
            explicitly so callers can swap profiles at SFT assembly time.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    for msg in trace.messages:
        messages.append(_convert_message(msg))
    return messages
