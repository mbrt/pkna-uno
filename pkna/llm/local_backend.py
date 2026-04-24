"""Local inference backend using Unsloth/transformers.

Loads a model or LoRA adapter via FastLanguageModel and runs inference
locally on GPU. Supports Qwen3.5's thinking mode and XML-based tool-call
format.

This module is kept separate from backends.py to avoid importing torch
and unsloth in CPU-only contexts (e.g. eval scoring with Gemini judge).
"""

import inspect
import json
import logging
import re
from collections.abc import Callable
from typing import Any, cast

from pkna.llm.backends import (
    MAX_TOOL_ITERATIONS,
    GenerateResult,
    LLMBackend,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 4096

# Regex for Qwen3.5 XML tool-call blocks.
# Matches: <tool_call>\n<function=name>\n<parameter=p1>\nval\n</parameter>\n</function>\n</tool_call>
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)

# Thinking tags used by Qwen3.5 in thinking mode.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen3.5 XML-format tool calls from generated text.

    Returns a list of dicts with "name" and "arguments" keys.
    """
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        fn_name = match.group(1)
        body = match.group(2)
        args: dict[str, Any] = {}
        for pm in _PARAMETER_RE.finditer(body):
            param_name = pm.group(1)
            raw_value = pm.group(2)
            try:
                args[param_name] = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                args[param_name] = raw_value
        calls.append({"name": fn_name, "arguments": args})
    return calls


def extract_thinking_and_content(text: str) -> tuple[str | None, str]:
    """Split thinking traces and visible content from generated text.

    Qwen3.5 wraps reasoning in <think>...</think> tags at the start of
    the assistant turn.
    """
    think_match = _THINK_RE.search(text)
    if think_match:
        thinking = think_match.group(1).strip()
        content = text[: think_match.start()] + text[think_match.end() :]
        content = content.strip()
        return thinking if thinking else None, content
    return None, text.strip()


def strip_tool_call_text(text: str) -> str:
    """Remove tool-call XML blocks from visible text."""
    return _TOOL_CALL_RE.sub("", text).strip()


def _callable_to_tool_dict(fn: Any) -> dict[str, Any]:
    """Convert a Python callable to Qwen3.5 tool JSON format.

    The chat template expects a list of tool dicts with "type": "function"
    and a nested "function" object containing name, description, and
    parameters in JSON Schema format.
    """
    sig = inspect.signature(fn)
    hints = dict(inspect.get_annotations(fn))
    hints.pop("return", None)
    docstring = inspect.getdoc(fn) or ""
    description = docstring.split("\n\n")[0].strip() if docstring else fn.__name__

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, param.annotation)
        schema = _type_to_json_schema(annotation)
        properties[name] = schema

        has_default = param.default is not inspect.Parameter.empty
        if not has_default:
            required.append(name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": parameters,
        },
    }


def _type_to_json_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is type(None):
        return {"type": "string"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    return {"type": "string"}


class LocalBackend(LLMBackend):
    """Inference backend using a locally loaded Unsloth/transformers model.

    Loads the model once at construction time and reuses it for all
    generate() calls. Supports LoRA adapters (pass the adapter path as
    model_name).
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_seq_length: int = 8192,
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LocalBackend requires a CUDA GPU. No CUDA device detected. "
                "Use --backend gemini or --backend anthropic for CPU-only machines."
            )

        from unsloth import FastLanguageModel

        from training import select_device_map

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens

        device_map = select_device_map()
        log.info("Loading model %s (device_map=%s)", model_name, device_map)

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            device_map=device_map,
        )
        FastLanguageModel.for_inference(self._model)

        # PEFT-wrapped models may report .device as cpu even when
        # parameters live on GPU. Use the first parameter's device instead.
        self._device = next(self._model.parameters()).device

        # Unwrap processor to get a real tokenizer for encode/decode.
        self._encoder = (
            self._tokenizer.tokenizer
            if hasattr(self._tokenizer, "tokenizer")
            else self._tokenizer
        )
        if self._encoder.pad_token_id is None:
            self._encoder.pad_token_id = self._encoder.eos_token_id

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[Callable[..., str]] | None = None,
        response_schema: type | None = None,
    ) -> GenerateResult | None:
        if response_schema is not None:
            raise NotImplementedError(
                "LocalBackend does not support response_schema. "
                "Use a cloud backend (gemini/anthropic) for judge scoring."
            )

        if tools:
            return self._generate_with_tools(system, messages, tools)
        return self._generate_no_tools(system, messages)

    def _build_prompt(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the tokenized prompt string via the chat template."""
        chat_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *messages,
        ]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        if tools:
            kwargs["tools"] = tools
        return cast(
            str,
            self._tokenizer.apply_chat_template(chat_messages, **kwargs),
        )

    def _run_generation(self, prompt_text: str) -> str:
        """Tokenize, generate, decode, and return only the new tokens."""
        import torch

        encoded = self._encoder(
            prompt_text, return_tensors="pt", return_attention_mask=True
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._max_new_tokens,
                pad_token_id=self._encoder.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

        new_tokens = output_ids[0, prompt_len:]
        return self._encoder.decode(new_tokens, skip_special_tokens=True)

    def _generate_no_tools(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> GenerateResult | None:
        prompt_text = self._build_prompt(system, messages)
        raw_output = self._run_generation(prompt_text)

        thinking, content = extract_thinking_and_content(raw_output)
        if not content and not thinking:
            log.warning("Local model returned empty response")
            return None

        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if thinking:
            msg["thinking"] = thinking

        return GenerateResult(
            text=content,
            model_name=self._model_name,
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

        for _ in range(MAX_TOOL_ITERATIONS):
            prompt_text = self._build_prompt(system, conversation, tool_defs)
            raw_output = self._run_generation(prompt_text)

            thinking, content = extract_thinking_and_content(raw_output)
            if thinking:
                all_thinking.append(thinking)

            parsed_calls = parse_tool_calls(content)
            if not parsed_calls:
                visible_text = strip_tool_call_text(content)
                msg: dict[str, Any] = {"role": "assistant", "content": visible_text}
                if thinking:
                    msg["thinking"] = thinking
                result_messages.append(msg)
                break

            visible_text = strip_tool_call_text(content)
            tc_records = [
                {"name": tc["name"], "arguments": tc["arguments"]}
                for tc in parsed_calls
            ]
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": visible_text,
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
            log.warning("Local: max tool iterations reached")
            return None

        combined_thinking = "\n".join(all_thinking) if all_thinking else None
        final_text = result_messages[-1].get("content", "") if result_messages else ""

        return GenerateResult(
            text=final_text,
            model_name=self._model_name,
            thinking=combined_thinking,
            tool_calls=all_tool_calls,
            messages=result_messages,
        )
