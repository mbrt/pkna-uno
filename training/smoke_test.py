#!/usr/bin/env python3

"""End-to-end smoke test for the full pipeline.

Validates all pipeline stages locally without requiring API keys or
(for most stages) a GPU. LLM-dependent stages use a fake backend that
returns canned responses, so the test exercises real I/O, serialization,
and orchestration logic.

Stages:
    1. Generate prompts  (no LLM, no scene files)
    2. Run datagen       (fake backend)
    3. Quality filtering  (fake backend)
    4. Assemble HF Dataset (tokenizer only)
    5. SFT training       (GPU required)
    6. Eval inference      (fake backend)
    7. Eval scoring        (fake backend)

Usage:
    # Stages 1-5 (current default, GPU needed for stage 5):
    python training/smoke_test.py

    # All stages including eval:
    python training/smoke_test.py --all

    # Skip training (no GPU needed):
    python training/smoke_test.py --no-training

    # Run a single stage (reads previous stage output from disk):
    python training/smoke_test.py --stage prompts
    python training/smoke_test.py --stage datagen
    python training/smoke_test.py --stage filter
    python training/smoke_test.py --stage assemble
    python training/smoke_test.py --stage train
    python training/smoke_test.py --stage eval
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from pkna.llm_backends import GenerateResult
from pkna.testing import FakeBackend, SequentialBackend

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUTPUT_DIR = "output/sft/smoke_test"

MAX_PROMPTS = 5
EVAL_SUITES = ["personality", "tool_use"]
MAX_EVAL_PROMPTS = 3


# ============================================================================
# Canned fake responses
# ============================================================================


def _canned_assistant_result(content: str, thinking: str = "") -> GenerateResult:
    """Build a GenerateResult that looks like a single assistant turn."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking:
        msg["thinking"] = thinking
    return GenerateResult(
        text=content,
        model_name="fake",
        thinking=thinking or None,
        messages=[msg],
    )


def _canned_tool_result(
    thinking: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_output: str,
    final_content: str,
) -> GenerateResult:
    """Build a GenerateResult for a tool-call + response pattern."""
    return GenerateResult(
        text=final_content,
        model_name="fake",
        thinking=thinking,
        tool_calls=[{"name": tool_name, "arguments": tool_args, "result": tool_output}],
        messages=[
            {
                "role": "assistant",
                "content": "",
                "thinking": thinking,
                "tool_calls": [{"name": tool_name, "arguments": tool_args}],
            },
            {"role": "tool", "name": tool_name, "content": tool_output},
            {"role": "assistant", "content": final_content},
        ],
    )


def _canned_judge_response() -> GenerateResult:
    """All-pass judge response for quality filtering."""
    data = {
        "character_consistency": 4.5,
        "thinking_quality": 4.0,
        "tool_correctness": "na",
        "language_consistent": True,
        "justification": "Good characterization.",
    }
    return GenerateResult(text=json.dumps(data), model_name="fake")


def _canned_eval_judge(score: float = 4.0) -> GenerateResult:
    """Structured judge response for eval scoring (wrapped in list)."""
    data = [{"score": score, "justification": "Solid performance."}]
    return GenerateResult(text=json.dumps(data), model_name="fake")


def _canned_social_judge() -> GenerateResult:
    data = [
        {
            "grounding": 4.0,
            "strategy": 4.0,
            "consistency": 4.0,
            "efficiency": 4.0,
            "justification": "Good reasoning.",
        }
    ]
    return GenerateResult(text=json.dumps(data), model_name="fake")


def _make_datagen_backend(n_calls: int) -> SequentialBackend:
    """Build a SequentialBackend with enough canned responses for datagen.

    Alternates between simple assistant turns and tool-call turns, plus
    extra responses for multi-turn user simulator calls.
    """
    results: list[GenerateResult | None] = []
    for i in range(n_calls):
        if i % 3 == 1:
            results.append(
                _canned_tool_result(
                    thinking="Let me search the wiki for facts.",
                    tool_name="search_knowledge",
                    tool_args={"keywords": "test query"},
                    tool_output="Relevant wiki content here.",
                    final_content=(
                        "Based on my research, here is what I found. "
                        "The wiki confirms the key details about this topic, socio."
                    ),
                )
            )
        else:
            results.append(
                _canned_assistant_result(
                    content=(
                        "Ah, socio, that is an excellent question. Let me think "
                        "about it carefully before answering. The situation is "
                        "more nuanced than it appears at first glance."
                    ),
                    thinking="The user is asking a thoughtful question. Respond in character.",
                )
            )
    return SequentialBackend(results)


def _make_eval_backend(n_calls: int) -> SequentialBackend:
    """Build a SequentialBackend for eval inference."""
    results = [
        _canned_assistant_result(
            content=(
                "I am Uno, Numero Uno. An artificial intelligence with "
                "a personality, if you can believe it. What can I do for "
                "you today, socio?"
            ),
            thinking="Greeting a new user. Keep it light, stay in character.",
        )
        for _ in range(n_calls)
    ]
    return SequentialBackend(results)


# ============================================================================
# Stage 1: Generate prompts
# ============================================================================


def run_stage_prompts(output_dir: Path) -> Path:
    from datagen.generate_prompts import generate_manual_prompts, write_prompts

    prompts_path = output_dir / "prompts.jsonl"
    prompts = generate_manual_prompts()[:MAX_PROMPTS]
    write_prompts(prompts_path, prompts)
    log.info("Wrote %d prompts to %s", len(prompts), prompts_path)
    return prompts_path


# ============================================================================
# Stage 2: Run datagen
# ============================================================================


def run_stage_datagen(output_dir: Path) -> Path:
    from datagen.run_datagen import run_datagen

    prompts_path = output_dir / "prompts.jsonl"
    traces_path = output_dir / "traces.jsonl"
    memory_banks_dir = Path("data/memory_banks")

    from datagen.generate_prompts import load_prompts

    n_prompts = len(load_prompts(prompts_path))
    backend = _make_datagen_backend(n_prompts * 4)

    written = run_datagen(
        prompts_path=prompts_path,
        output_path=traces_path,
        memory_banks_dir=memory_banks_dir,
        backend=backend,
    )
    log.info("Generated %d traces -> %s", written, traces_path)
    return traces_path


# ============================================================================
# Stage 3: Quality filtering
# ============================================================================


def run_stage_filter(output_dir: Path) -> Path:
    from datagen.filter_traces import filter_traces

    traces_path = output_dir / "traces.jsonl"
    scored_path = output_dir / "traces_scored.jsonl"
    filtered_path = output_dir / "traces_filtered.jsonl"

    backend = FakeBackend(_canned_judge_response())
    total, passed = filter_traces(traces_path, scored_path, filtered_path, backend)
    log.info("Filtered: %d/%d passed -> %s", passed, total, filtered_path)
    return filtered_path


# ============================================================================
# Stage 4: Assemble dataset
# ============================================================================


def run_stage_assemble(output_dir: Path, model_name: str) -> Path:
    from training.assemble_sft import assemble_dataset

    filtered_path = output_dir / "traces_filtered.jsonl"
    dataset_path = output_dir / "dataset"
    assemble_dataset(
        input_path=filtered_path,
        output_path=dataset_path,
        model_name=model_name,
        max_seq_length=2048,
    )
    return dataset_path


# ============================================================================
# Stage 5: Training
# ============================================================================


def run_stage_train(output_dir: Path, model_name: str, max_steps: int) -> Path:
    from training.run_sft import run_sft

    dataset_path = output_dir / "dataset"
    adapter_path = output_dir / "lora_adapter"
    run_sft(
        dataset_path=str(dataset_path),
        output_path=str(adapter_path),
        model_name=model_name,
        max_seq_length=2048,
        num_epochs=1,
        learning_rate=2e-4,
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=1,
        weight_decay=0.01,
        max_steps=max_steps,
        logging_steps=1,
        export_gguf=None,
    )
    return adapter_path


# ============================================================================
# Stage 6: Eval inference
# ============================================================================


def run_stage_eval_infer(output_dir: Path) -> Path:
    from evals.generate_eval_prompts import SUITE_GENERATORS, write_suite
    from evals.run_eval_inference import run_eval

    eval_dir = output_dir / "eval"
    prompts_dir = eval_dir / "prompts"
    traces_dir = eval_dir / "traces"

    all_prompts = []
    for suite in EVAL_SUITES:
        generator = SUITE_GENERATORS[suite]
        prompts = generator()[:MAX_EVAL_PROMPTS]
        write_suite(prompts_dir, suite, prompts)
        all_prompts.extend(prompts)
        log.info("Eval prompts: %s -> %d", suite, len(prompts))

    backend = _make_eval_backend(len(all_prompts) * 2)
    memory_banks_dir = Path("data/memory_banks")

    written = run_eval(
        prompts=all_prompts,
        backend=backend,
        model_name="fake",
        output_dir=traces_dir,
        memory_banks_dir=memory_banks_dir,
    )
    log.info("Eval inference: %d traces -> %s", written, traces_dir)
    return traces_dir


# ============================================================================
# Stage 7: Eval scoring + report
# ============================================================================


def run_stage_eval_score(output_dir: Path) -> Path:
    from evals.run_eval_inference import load_prompts as load_eval_prompts
    from evals.score_eval_traces import (
        aggregate_report,
        load_traces,
        score_all,
    )

    eval_dir = output_dir / "eval"
    prompts_dir = eval_dir / "prompts"
    traces_dir = eval_dir / "traces"
    scored_dir = eval_dir / "scored"

    traces = load_traces(traces_dir, EVAL_SUITES)
    prompts_map = {p.id: p for p in load_eval_prompts(prompts_dir, EVAL_SUITES)}

    n_judge_calls = len(traces) * 2
    backend = SequentialBackend([_canned_eval_judge() for _ in range(n_judge_calls)])

    all_scored = score_all(traces, prompts_map, backend, scored_dir)

    report = aggregate_report(all_scored, "fake")
    report_path = scored_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    log.info(
        "Eval report: %d scored, %d suites -> %s",
        len(all_scored),
        len(report.suites),
        report_path,
    )
    for suite_name, result in sorted(report.suites.items()):
        log.info("  %s: mean=%.2f (n=%d)", suite_name, result.mean_score, result.n)

    return report_path


# ============================================================================
# Main
# ============================================================================


STAGES = {
    "prompts": "Stage 1: Generate prompts",
    "datagen": "Stage 2: Run datagen",
    "filter": "Stage 3: Quality filtering",
    "assemble": "Stage 4: Assemble HF Dataset",
    "train": "Stage 5: SFT Training",
    "eval": "Stages 6-7: Eval inference + scoring",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model for tokenizer and training (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Training steps to run (default: %(default)s)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=list(STAGES.keys()),
        default=None,
        help="Run a single stage only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all stages including eval (stages 1-7)",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Run all stages except training (no GPU needed)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before running",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Pipeline Smoke Test[/bold cyan]\n")

    output_dir = Path(args.output_dir)

    if args.clean and output_dir.exists():
        log.info("Cleaning %s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage:
        stages = [args.stage]
    elif args.no_training:
        stages = ["prompts", "datagen", "filter", "assemble", "eval"]
    elif args.all:
        stages = ["prompts", "datagen", "filter", "assemble", "train", "eval"]
    else:
        stages = ["prompts", "datagen", "filter", "assemble", "train"]

    if "train" in stages:
        from training import ensure_unsloth

        ensure_unsloth()

    for stage in stages:
        console.rule(f"[bold]{STAGES[stage]}")

        if stage == "prompts":
            run_stage_prompts(output_dir)
        elif stage == "datagen":
            run_stage_datagen(output_dir)
        elif stage == "filter":
            run_stage_filter(output_dir)
        elif stage == "assemble":
            run_stage_assemble(output_dir, args.model)
        elif stage == "train":
            run_stage_train(output_dir, args.model, args.max_steps)
        elif stage == "eval":
            run_stage_eval_infer(output_dir)
            run_stage_eval_score(output_dir)

    console.print("\n[bold green]Smoke test complete.[/bold green]")
    console.print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
