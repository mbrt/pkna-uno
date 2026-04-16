#!/usr/bin/env python3

"""Run trace generation for SFT dataset construction.

Loads datagen prompts, composes runtime context (system prompt, user summary,
memory context, tools), runs the strong model, and records full traces
(thinking + tool calls + responses) as DatagenTrace JSONL.

Supports resume: trace IDs already present in the output file are skipped.

Usage:
    python datagen/run_datagen.py \
        --prompts output/datagen/prompts.jsonl \
        --output output/datagen/traces.jsonl \
        --backend gemini \
        --model gemini-3-flash
"""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from datagen.generate_prompts import load_prompts
from datagen.user_simulator import simulate_user_turn
from pkna.datagen.types import DatagenTrace
from pkna.inference.memory import MemoryBank
from pkna.inference.system_prompts import render_system_prompt
from pkna.inference.tools import make_eval_tools
from pkna.llm.backends import LLMBackend, create_backend

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    force=True,
)
log = logging.getLogger(__name__)


# ============================================================================
# Context Composer
# ============================================================================


def compose_datagen_context(
    user_summary: str,
    memory_context: str,
) -> str:
    """Build the system prompt for a datagen prompt.

    Always uses the full template since SFT training examples need the
    complete personality and tool instructions.
    """
    return render_system_prompt(
        template="full",
        user_summary=user_summary,
        memory_context=memory_context,
    )


# ============================================================================
# Helpers
# ============================================================================


def load_completed_ids(output_path: Path) -> set[str]:
    """Scan an existing output JSONL for trace IDs already processed."""
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                completed.add(data["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def load_memory_bank(memory_bank_id: str, memory_banks_dir: Path) -> MemoryBank | None:
    """Load a memory bank by ID, or return None if empty/missing."""
    if not memory_bank_id:
        return None
    path = memory_banks_dir / f"{memory_bank_id}.jsonl"
    if not path.exists():
        log.warning(f"Memory bank not found: {path}")
        return None
    return MemoryBank.load(path)


def _visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract user/assistant messages for the user simulator."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant") and m.get("content")
    ]


def _get_directive(directives: list[str], turn_index: int) -> str:
    """Get the directive for a given turn, cycling if the list is short."""
    if not directives:
        return "continue"
    return directives[turn_index % len(directives)]


# ============================================================================
# Trace Capture
# ============================================================================


def run_single_turn(
    prompt_id: str,
    system_prompt: str,
    user_summary: str,
    memory_context: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
    backend: LLMBackend,
    tools: list[Callable[..., str]] | None,
) -> DatagenTrace | None:
    """Run inference for a single-turn prompt and return the trace."""
    result = backend.generate(
        system=system_prompt,
        messages=messages,
        tools=tools,
    )
    if result is None:
        log.error(f"Inference failed for prompt {prompt_id}")
        return None

    all_messages: list[dict[str, Any]] = list(messages)
    if result.messages:
        all_messages.extend(result.messages)
    else:
        all_messages.append({"role": "assistant", "content": result.text})

    return DatagenTrace(
        id=prompt_id,
        metadata=metadata,
        system_prompt=system_prompt,
        memory_context=memory_context,
        user_summary=user_summary,
        messages=all_messages,
    )


def run_multi_turn(
    prompt_id: str,
    system_prompt: str,
    user_summary: str,
    memory_context: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
    backend: LLMBackend,
    tools: list[Callable[..., str]] | None,
    simulator_backend: LLMBackend | None = None,
) -> DatagenTrace | None:
    """Run a multi-turn conversation and return the full trace."""
    sim_backend = simulator_backend or backend
    turn_count: int = metadata.get("turn_count", 5)
    directives: list[str] = metadata.get("directives", [])

    all_messages: list[dict[str, Any]] = list(messages)

    for turn in range(turn_count):
        result = backend.generate(
            system=system_prompt,
            messages=all_messages,
            tools=tools,
        )
        if result is None:
            log.error(f"Inference failed for {prompt_id} at turn {turn}")
            break

        if result.messages:
            all_messages.extend(result.messages)
        else:
            all_messages.append({"role": "assistant", "content": result.text})

        if turn >= turn_count - 1:
            break

        directive = _get_directive(directives, turn)
        user_msg = simulate_user_turn(
            backend=sim_backend,
            conversation=_visible_messages(all_messages),
            user_profile=user_summary,
            directive=directive,
        )
        if user_msg is None:
            log.error(f"User simulator failed for {prompt_id} at turn {turn}")
            break
        all_messages.append({"role": "user", "content": user_msg})

    if not any(m["role"] == "assistant" for m in all_messages):
        return None

    return DatagenTrace(
        id=prompt_id,
        metadata=metadata,
        system_prompt=system_prompt,
        memory_context=memory_context,
        user_summary=user_summary,
        messages=all_messages,
    )


def run_single_prompt(
    prompt_id: str,
    system_prompt: str,
    user_summary: str,
    memory_context: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
    backend: LLMBackend,
    tools: list[Callable[..., str]] | None,
    simulator_backend: LLMBackend | None = None,
) -> DatagenTrace | None:
    """Run inference for a prompt, dispatching single- or multi-turn."""
    if metadata.get("multi_turn"):
        return run_multi_turn(
            prompt_id,
            system_prompt,
            user_summary,
            memory_context,
            messages,
            metadata,
            backend,
            tools,
            simulator_backend,
        )
    return run_single_turn(
        prompt_id,
        system_prompt,
        user_summary,
        memory_context,
        messages,
        metadata,
        backend,
        tools,
    )


# ============================================================================
# Main Loop
# ============================================================================


def run_datagen(
    prompts_path: Path,
    output_path: Path,
    memory_banks_dir: Path,
    backend: LLMBackend,
    simulator_backend: LLMBackend | None = None,
) -> int:
    """Run trace generation for all prompts.

    Returns the number of traces written.
    """
    prompts = load_prompts(prompts_path)
    if not prompts:
        log.info("No prompts found.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output_path)

    pending = [p for p in prompts if p.id not in completed]
    if not pending:
        log.info("All prompts already processed. Nothing to do.")
        return 0

    skipped = len(prompts) - len(pending)
    if skipped > 0:
        log.info(f"Resuming: skipping {skipped} already-processed prompts")

    written = 0
    with (
        open(output_path, "a", encoding="utf-8") as f,
        Progress(console=console) as progress,
    ):
        task = progress.add_task("Generating traces", total=len(pending))

        for prompt in pending:
            system_prompt = compose_datagen_context(
                user_summary=prompt.user_summary,
                memory_context=prompt.memory_context,
            )

            bank = load_memory_bank(prompt.memory_bank_id, memory_banks_dir)
            tool_callables = make_eval_tools(
                prompt.tools, memory_bank=bank, eval_mode=False
            )
            tools_or_none = tool_callables if tool_callables else None

            trace = run_single_prompt(
                prompt_id=prompt.id,
                system_prompt=system_prompt,
                user_summary=prompt.user_summary,
                memory_context=prompt.memory_context,
                messages=prompt.messages,
                metadata=prompt.metadata,
                backend=backend,
                tools=tools_or_none,
                simulator_backend=simulator_backend,
            )

            if trace is not None:
                f.write(trace.model_dump_json() + "\n")
                f.flush()
                written += 1

            progress.advance(task)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trace generation for SFT dataset construction"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("output/datagen/prompts.jsonl"),
        help="Input JSONL file with DatagenPrompt entries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/datagen/traces.jsonl"),
        help="Output JSONL file for DatagenTrace entries",
    )
    parser.add_argument(
        "--memory-banks-dir",
        type=Path,
        default=Path("data/memory_banks"),
        help="Directory containing memory bank JSONL files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="LLM backend (gemini or anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults to backend's default)",
    )
    args = parser.parse_args()

    console.print("[bold cyan]SFT Trace Generator[/bold cyan]\n")

    backend = create_backend(args.backend, args.model)

    written = run_datagen(
        prompts_path=args.prompts,
        output_path=args.output,
        memory_banks_dir=args.memory_banks_dir,
        backend=backend,
    )

    console.print(
        f"\n[bold green]Done.[/bold green] {written} traces written to {args.output}"
    )


if __name__ == "__main__":
    main()
