"""Convert DatagenTrace objects into Qwen3.5 chat message format for SFT.

Transforms the internal trace representation (with 'thinking', 'tool_calls',
etc.) into the message format expected by Qwen3.5's tokenizer.apply_chat_template()
with thinking mode enabled.

Key mappings:
- system_prompt (passed explicitly) -> {"role": "system", "content": ...}
- assistant 'thinking' field -> 'reasoning_content' (Qwen3.5's field name)
- assistant 'tool_calls' -> {"function": {"name": ..., "arguments": ...}}
- tool results -> {"role": "tool", "content": ...}

No GPU or model dependencies -- this is pure data transformation.
"""

from typing import Any

from pkna.datagen.types import DatagenTrace


def _convert_tool_calls(
    raw_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert internal tool_call dicts to Qwen3.5 format.

    Input:  {"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}
    Output: {"function": {"name": "search_wiki", "arguments": {"keywords": "Xadhoom"}}}
    """
    converted = []
    for tc in raw_calls:
        name = tc.get("name", "")
        arguments = tc.get("arguments", {})
        converted.append({"function": {"name": name, "arguments": arguments}})
    return converted


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a single trace message to Qwen3.5 chat format."""
    role = msg.get("role", "")

    if role == "user":
        return {"role": "user", "content": msg.get("content", "")}

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
    """Convert a DatagenTrace to Qwen3.5 chat messages.

    Returns a list of message dicts suitable for passing to
    tokenizer.apply_chat_template(messages, enable_thinking=True).

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
