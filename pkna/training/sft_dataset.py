"""Convert DatagenTrace objects into standard chat message format for SFT.

Transforms the internal trace representation (with 'thinking', 'tool_calls',
etc.) into a tokenizer-independent message format using widely adopted
conventions:

- reasoning_content: de facto standard for thinking models (Qwen3, DeepSeek)
- OpenAI-style tool_calls: {"type": "function", "function": {"name", "arguments"}}

The output can be passed directly to tokenizer.apply_chat_template() on any
model that supports these conventions, with no additional conversion step.

For publishing the SFT dataset, prefer ``trace_to_chatml_text`` which returns
a fully-rendered ChatML string: storing a single ``text`` column avoids the
PyArrow struct-unification that pollutes ``tool_calls.arguments`` with
``None`` entries when the dataset is round-tripped through
``Dataset.from_dict`` / ``save_to_disk``.

No GPU dependencies -- text rendering needs a tokenizer but not the model.
"""

from typing import Any, cast

from pkna.datagen.types import DatagenTrace
from pkna.inference.system_prompts import strip_trace_guidance

# Qwen3's stock chat template only renders <think>...</think> on the final
# assistant turn (``loop.index0 > ns.last_query_index``) and strips reasoning
# from every earlier turn. That's intentional for inference (saves context),
# but during SFT it discards the "reason → tool call / reason → response"
# supervision signal on every multi-turn intermediate step. We patch the
# clause to *also* render thinking whenever ``reasoning_content`` is
# non-empty, so real reasoning is preserved on every turn while Qwen's
# default behavior for empty-thinking turns is untouched.
_QWEN_THINK_CLAUSE = "        {%- if loop.index0 > ns.last_query_index %}"
_QWEN_THINK_CLAUSE_PATCHED = (
    "        {%- if reasoning_content or loop.index0 > ns.last_query_index %}"
)


def patch_chat_template_for_sft(template: str) -> str:
    """Patch Qwen3's chat template to preserve thinking on all turns.

    Raises ValueError if the expected clause is missing -- catches upstream
    template changes before they silently drop supervision signal.
    """
    if _QWEN_THINK_CLAUSE not in template:
        raise ValueError(
            "Could not find Qwen think-position clause in chat template. "
            "The tokenizer may not be Qwen3, or upstream changed the template."
        )
    return template.replace(_QWEN_THINK_CLAUSE, _QWEN_THINK_CLAUSE_PATCHED)


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


def trace_to_chatml_text(
    trace: DatagenTrace,
    system_prompt: str,
    tokenizer: Any,
) -> str:
    """Render a DatagenTrace to a ChatML text string for SFT.

    Applies the tokenizer's chat template to the converted messages. The
    result is the canonical ChatML string that the model will see during
    training (e.g. ``<|im_start|>assistant\\n<think>\\n...``), with tool
    calls rendered from *actual* argument dicts so no ``None`` parameters
    leak through.

    Storing this string in a ``text`` column lets the SFT dataset round-trip
    cleanly through Arrow/Parquet without struct-schema unification.
    """
    messages = trace_to_messages(trace, system_prompt)
    patched_template = patch_chat_template_for_sft(tokenizer.chat_template)
    return cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            chat_template=patched_template,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        ),
    )
