#!/usr/bin/env python3

"""Stage 2: Run eval inference.

Loads eval prompts from stage 1, composes runtime context (system prompt,
user summary, memory context, tools), runs the model under test, and
records full traces as EvalTrace JSONL.

Supports resume: prompt IDs already present in the output file are skipped.

Usage:
    python evals/run_eval_inference.py \
        --prompts-dir output/evals/prompts/ \
        --output-dir output/evals/traces/ \
        --backend gemini \
        --model gemini-3-flash
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from pkna.eval_tools import make_eval_tools
from pkna.eval_types import EvalPrompt, EvalTrace
from pkna.llm_backends import LLMBackend, create_backend
from pkna.memory_bank import MemoryBank
from pkna.system_prompts import SUITE_TEMPLATE_MAP, render_system_prompt

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    force=True,
)
log = logging.getLogger(__name__)


def load_prompts(prompts_dir: Path, suites: list[str] | None) -> list[EvalPrompt]:
    """Load eval prompts from JSONL files in the prompts directory."""
    prompts: list[EvalPrompt] = []
    for path in sorted(prompts_dir.glob("*.jsonl")):
        suite_name = path.stem
        if suites and suite_name not in suites:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(EvalPrompt.model_validate_json(line))
    return prompts


def load_completed_ids(output_path: Path) -> set[str]:
    """Scan an existing output JSONL for prompt IDs already processed."""
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
                completed.add(data["prompt_id"])
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


def compose_context(
    prompt: EvalPrompt,
) -> str:
    """Build the system prompt for a given eval prompt."""
    template = SUITE_TEMPLATE_MAP.get(prompt.suite, "full")
    return render_system_prompt(
        template=template,
        user_summary=prompt.user_summary,
        memory_context=prompt.memory_context,
    )


def run_single_prompt(
    prompt: EvalPrompt,
    backend: LLMBackend,
    model_name: str,
    memory_banks_dir: Path,
) -> EvalTrace | None:
    """Run inference for a single eval prompt and return the trace."""
    system_prompt = compose_context(prompt)
    bank = load_memory_bank(prompt.memory_bank_id, memory_banks_dir)
    tools = make_eval_tools(prompt.tools, memory_bank=bank, eval_mode=True)

    result = backend.generate(
        system=system_prompt,
        messages=prompt.messages,
        tools=tools if tools else None,
    )
    if result is None:
        log.error(f"Inference failed for prompt {prompt.id}")
        return None

    tool_calls: list[dict[str, Any]] = []
    thinking: str | None = None

    messages: list[dict[str, Any]] = list(prompt.messages)
    messages.append({"role": "assistant", "content": result.text})

    return EvalTrace(
        prompt_id=prompt.id,
        suite=prompt.suite,
        model=model_name,
        messages=messages,
        tool_calls=tool_calls,
        thinking=thinking,
    )


def run_eval(
    prompts: list[EvalPrompt],
    backend: LLMBackend,
    model_name: str,
    output_dir: Path,
    memory_banks_dir: Path,
) -> int:
    """Run inference for all prompts, writing traces to per-suite JSONL files.

    Returns the number of traces written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    suites_in_run = sorted({p.suite for p in prompts})
    completed_by_suite: dict[str, set[str]] = {}
    for suite in suites_in_run:
        output_path = output_dir / f"{suite}.jsonl"
        completed_by_suite[suite] = load_completed_ids(output_path)

    pending = [p for p in prompts if p.id not in completed_by_suite.get(p.suite, set())]

    if not pending:
        log.info("All prompts already processed. Nothing to do.")
        return 0

    skipped = len(prompts) - len(pending)
    if skipped > 0:
        log.info(f"Resuming: skipping {skipped} already-processed prompts")

    written = 0
    file_handles: dict[str, Any] = {}

    try:
        with Progress(console=console) as progress:
            task = progress.add_task("Running inference", total=len(pending))

            for prompt in pending:
                trace = run_single_prompt(prompt, backend, model_name, memory_banks_dir)
                if trace is None:
                    progress.advance(task)
                    continue

                suite = prompt.suite
                if suite not in file_handles:
                    output_path = output_dir / f"{suite}.jsonl"
                    file_handles[suite] = open(output_path, "a", encoding="utf-8")

                file_handles[suite].write(trace.model_dump_json() + "\n")
                file_handles[suite].flush()
                written += 1
                progress.advance(task)
    finally:
        for fh in file_handles.values():
            fh.close()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval inference (stage 2 of the eval pipeline)"
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("output/evals/prompts"),
        help="Directory containing prompt JSONL files from stage 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/evals/traces"),
        help="Directory to write trace JSONL files",
    )
    parser.add_argument(
        "--memory-banks-dir",
        type=Path,
        default=Path("data/memory_banks"),
        help="Directory containing raw memory bank JSONL files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="LLM backend to use (gemini or anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults to backend's default)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help="Comma-separated list of suites to run (default: all)",
    )
    args = parser.parse_args()

    suite_filter = args.suites.split(",") if args.suites else None

    console.print("[bold cyan]Eval Inference Runner[/bold cyan]\n")

    prompts = load_prompts(args.prompts_dir, suite_filter)
    if not prompts:
        console.print("[yellow]No prompts found.[/yellow]")
        return

    log.info(
        f"Loaded {len(prompts)} prompts across {len({p.suite for p in prompts})} suites"
    )

    backend = create_backend(args.backend, args.model)
    model_name = args.model or args.backend

    written = run_eval(
        prompts, backend, model_name, args.output_dir, args.memory_banks_dir
    )

    console.print(
        f"\n[bold green]Done.[/bold green] {written} traces written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
