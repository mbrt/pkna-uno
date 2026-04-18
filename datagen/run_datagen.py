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
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.progress import Progress

from datagen.generate_prompts import load_prompts
from datagen.user_simulator import simulate_user_turn
from pkna.datagen.memory import compose_memory, load_memory_corpus
from pkna.datagen.types import DatagenTrace, MemoryCorpusEntry
from pkna.inference.memory import MemoryBank
from pkna.inference.system_prompts import (
    DATAGEN_TEMPLATE,
    prepend_context_to_messages,
    render_datagen_system_prompt,
)
from pkna.inference.tools import make_eval_tools
from pkna.llm.backends import LLMBackend, create_backend
from pkna.logging import setup_logging

console, log = setup_logging()


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


@dataclass
class TraceResult:
    """Bundles a DatagenTrace with cumulative LLM usage for cache tracking."""

    trace: DatagenTrace
    usage: dict[str, Any] = field(default_factory=dict)


def run_single_turn(
    prompt_id: str,
    system_prompt: str,
    user_summary: str,
    memory_context: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
    backend: LLMBackend,
    tools: list[Callable[..., str]] | None,
) -> TraceResult | None:
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

    trace = DatagenTrace(
        id=prompt_id,
        metadata=metadata,
        memory_context=memory_context,
        user_summary=user_summary,
        messages=all_messages,
    )
    return TraceResult(trace=trace, usage=result.usage)


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
) -> TraceResult | None:
    """Run a multi-turn conversation and return the full trace."""
    sim_backend = simulator_backend or backend
    turn_count: int = metadata.get("turn_count", 5)
    directives: list[str] = metadata.get("directives", [])

    all_messages: list[dict[str, Any]] = list(messages)
    cumulative_usage: dict[str, int] = {}

    for turn in range(turn_count):
        result = backend.generate(
            system=system_prompt,
            messages=all_messages,
            tools=tools,
        )
        if result is None:
            log.error(f"Inference failed for {prompt_id} at turn {turn}")
            break

        for k, v in result.usage.items():
            if isinstance(v, int):
                cumulative_usage[k] = cumulative_usage.get(k, 0) + v

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

    trace = DatagenTrace(
        id=prompt_id,
        metadata=metadata,
        memory_context=memory_context,
        user_summary=user_summary,
        messages=all_messages,
    )
    return TraceResult(trace=trace, usage=cumulative_usage)


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
) -> TraceResult | None:
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
# Sidecar Files
# ============================================================================


TEMPLATE_FILENAME = "system_prompt_template.txt"
PROFILE_FILENAME = "character_profile.md"


def write_sidecar_files(output_dir: Path, character_profile: str) -> None:
    """Write the system prompt template and character profile alongside traces.

    These sidecar files let downstream consumers (SFT assembly, filtering)
    reconstruct the system prompt with any profile, enabling profile swaps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TEMPLATE_FILENAME).write_text(DATAGEN_TEMPLATE, encoding="utf-8")
    (output_dir / PROFILE_FILENAME).write_text(character_profile, encoding="utf-8")


def load_system_prompt(output_dir: Path, profile_path: Path | None = None) -> str:
    """Reconstruct the system prompt from sidecar files.

    Args:
        output_dir: Directory containing the sidecar files.
        profile_path: Optional override for the character profile. When None,
            uses the profile stored in the sidecar files.
    """
    template = (output_dir / TEMPLATE_FILENAME).read_text(encoding="utf-8")
    if profile_path is not None:
        profile = profile_path.read_text(encoding="utf-8")
    else:
        profile = (output_dir / PROFILE_FILENAME).read_text(encoding="utf-8")
    return template.format(character_profile=profile)


# ============================================================================
# Main Loop
# ============================================================================


def run_datagen(
    prompts_path: Path,
    output_path: Path,
    memory_banks_dir: Path,
    backend: LLMBackend,
    character_profile: str = "",
    simulator_backend: LLMBackend | None = None,
    corpus: list[MemoryCorpusEntry] | None = None,
    seed: int = 42,
) -> int:
    """Run trace generation for all prompts.

    Args:
        character_profile: Full character profile markdown. When empty,
            the datagen template is used with a blank profile section
            (useful for smoke tests).
        corpus: Tagged memory corpus for dynamic composition. When set,
            prompts with a ``memory_profile`` use ``compose_memory()``
            instead of loading a static bank.
        seed: RNG seed for reproducible memory sampling.

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

    # Group identical system prompts together so Gemini's implicit caching
    # can reuse the cached prefix across consecutive requests.
    pending.sort(key=lambda p: (p.user_summary, p.memory_context))

    skipped = len(prompts) - len(pending)
    if skipped > 0:
        log.info(f"Resuming: skipping {skipped} already-processed prompts")

    system_prompt = render_datagen_system_prompt(character_profile)
    write_sidecar_files(output_path.parent, character_profile)

    rng = random.Random(seed)
    written = 0
    total_prompt_tokens = 0
    total_cached_tokens = 0

    with (
        open(output_path, "a", encoding="utf-8") as f,
        Progress(console=console) as progress,
    ):
        task = progress.add_task("Generating traces", total=len(pending))

        for prompt in pending:
            # Compose memory: prefer dynamic composition from corpus,
            # fall back to static bank for backward compatibility.
            if prompt.memory_profile is not None and corpus:
                memory_context, bank = compose_memory(
                    prompt.memory_profile, corpus, rng
                )
            else:
                memory_context = prompt.memory_context
                bank = load_memory_bank(prompt.memory_bank_id, memory_banks_dir)

            messages = prepend_context_to_messages(
                prompt.messages, prompt.user_summary, memory_context
            )

            tool_callables = make_eval_tools(
                prompt.tools, memory_bank=bank, eval_mode=False
            )
            tools_or_none = tool_callables if tool_callables else None

            result = run_single_prompt(
                prompt_id=prompt.id,
                system_prompt=system_prompt,
                user_summary=prompt.user_summary,
                memory_context=memory_context,
                messages=messages,
                metadata=prompt.metadata,
                backend=backend,
                tools=tools_or_none,
                simulator_backend=simulator_backend,
            )

            if result is not None:
                f.write(result.trace.model_dump_json() + "\n")
                f.flush()
                written += 1

                cached = result.usage.get("cached_content_token_count", 0)
                prompt_toks = result.usage.get("prompt_tokens", 0)
                total_prompt_tokens += prompt_toks
                total_cached_tokens += cached
                if cached:
                    log.info(
                        f"[{prompt.id}] cache hit: "
                        f"{cached}/{prompt_toks} prompt tokens cached"
                    )

            progress.advance(task)

    if total_prompt_tokens > 0:
        pct = total_cached_tokens / total_prompt_tokens * 100
        log.info(
            f"Cache summary: {total_cached_tokens:,}/{total_prompt_tokens:,} "
            f"prompt tokens cached ({pct:.1f}%)"
        )

    return written


DEFAULT_PROFILE = Path("results/uno_soul_document.md")


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
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"Character profile markdown file (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--memory-banks-dir",
        type=Path,
        default=Path("data/memory_banks"),
        help="Directory containing memory bank JSONL files",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("output/datagen/memory_corpus.jsonl"),
        help="Memory corpus JSONL file (for dynamic composition)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible memory sampling",
    )
    args = parser.parse_args()

    console.print("[bold cyan]SFT Trace Generator[/bold cyan]\n")

    profile_path: Path = args.profile
    if not profile_path.exists():
        console.print(f"[bold red]Error:[/bold red] Profile not found: {profile_path}")
        raise SystemExit(1)
    character_profile = profile_path.read_text(encoding="utf-8")
    log.info(f"Loaded character profile from {profile_path}")

    corpus: list[MemoryCorpusEntry] = []
    corpus_path: Path = args.corpus
    if corpus_path.exists():
        corpus = load_memory_corpus(corpus_path)
        log.info(f"Loaded {len(corpus)} corpus entries from {corpus_path}")
    else:
        log.info(f"No corpus file at {corpus_path}, using static memory banks")

    backend = create_backend(args.backend, args.model)

    written = run_datagen(
        prompts_path=args.prompts,
        output_path=args.output,
        memory_banks_dir=args.memory_banks_dir,
        backend=backend,
        character_profile=character_profile,
        corpus=corpus if corpus else None,
        seed=args.seed,
    )

    console.print(
        f"\n[bold green]Done.[/bold green] {written} traces written to {args.output}"
    )


if __name__ == "__main__":
    main()
