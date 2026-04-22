#!/usr/bin/env python3

"""Quality filtering for generated traces.

Scores each trace with an LLM judge on four dimensions plus a
programmatic response-length check, then filters out low-quality examples.

LLM-as-judge scoring:
- Character consistency (1-5)
- Thinking quality (1-5)
- Tool correctness (pass/fail/na)
- Language consistency (yes/no)

Programmatic check:
- Response length (10-500 visible tokens)

Usage:
    python datagen/filter_traces.py \
        --input output/datagen/traces.jsonl \
        --output output/datagen/traces_filtered.jsonl \
        --backend gemini
"""

import argparse
import json
from pathlib import Path

from rich.progress import Progress

from datagen.run_datagen import load_system_prompt
from pkna.datagen.types import (
    DatagenTrace,
    JudgeResponse,
    QualityScore,
    ScoredTrace,
)
from pkna.llm.backends import LLMBackend, create_backend
from pkna.logging import setup_logging

console, log = setup_logging()

MIN_RESPONSE_TOKENS = 10
MAX_RESPONSE_TOKENS = 500


# ============================================================================
# Trace Loading
# ============================================================================


def load_traces(path: Path) -> list[DatagenTrace]:
    """Load traces from a JSONL file."""
    traces: list[DatagenTrace] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(DatagenTrace.model_validate_json(line))
    return traces


# ============================================================================
# Programmatic Checks
# ============================================================================


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: word count * 1.3."""
    return int(len(text.split()) * 1.3)


def _get_visible_responses(trace: DatagenTrace) -> list[str]:
    """Extract visible assistant response texts (no tool-call-only turns)."""
    responses: list[str] = []
    for m in trace.messages:
        if m.get("role") == "assistant" and m.get("content"):
            responses.append(m["content"])
    return responses


def check_response_length(trace: DatagenTrace) -> bool:
    """Check that all visible responses are within token bounds."""
    responses = _get_visible_responses(trace)
    if not responses:
        return False
    for text in responses:
        tokens = _estimate_tokens(text)
        if tokens < MIN_RESPONSE_TOKENS or tokens > MAX_RESPONSE_TOKENS:
            return False
    return True


# ============================================================================
# LLM-as-Judge Scoring
# ============================================================================

JUDGE_SYSTEM = """\
You are a quality evaluator for AI character training data. You score \
conversation traces where an AI character named Uno (from the PKNA comic \
series) responds to users.

You will be given a conversation trace and must score it on multiple \
dimensions.

Scoring rubrics:
- character_consistency (1-5): Does the assistant sound like Uno? Does it \
use appropriate register shifts, humor, sarcasm, Italian expressions? \
5 = perfectly in character, 1 = generic assistant.
- thinking_quality (1-5): Score the thinking trace relative to the \
complexity of the interaction. Thinking must always be present (score 1 \
if absent). For complex interactions (emotional crisis, identity probes, \
adversarial, multi-turn), high-quality thinking references specific \
traits, values, relationships, or behavioral patterns from the profile; \
identifies competing impulses and resolves tradeoffs with explicit \
priorities; and chooses a strategy calibrated to the interlocutor. \
For straightforward interactions (casual greetings, simple factual \
questions, light banter), brief thinking that correctly identifies the \
situation and grounds the response in character is sufficient for a \
high score. Formulaic restating of the prompt or generic reasoning \
not tied to the profile scores low regardless of complexity. \
5 = reasoning well-calibrated to the interaction's complexity, \
grounded in the profile, 3 = adequate but vague on profile specifics, \
1 = absent or formulaic.
- tool_correctness: Did the agent use tools appropriately? "pass" if \
tools were used correctly or correctly avoided, "fail" if it hallucinated \
facts instead of searching or called wrong tools, "na" if no tools were \
expected or available.
- language_consistent: Does the response language match the user's \
language? true if the user writes in Italian and Uno responds in Italian, \
or the user writes in English and Uno responds in English. Short Italian \
expressions like "socio" or "ciao" are acceptable in English responses. \
false if the response is in the wrong language.\
"""


def _format_trace_for_judge(trace: DatagenTrace, system_prompt: str = "") -> str:
    """Format a trace for the judge model."""
    parts = [f"## System Prompt\n{system_prompt}"]

    if trace.user_summary:
        parts.append(f"\n## User Summary\n{trace.user_summary}")
    if trace.memory_context:
        parts.append(f"\n## Memory Context\n{trace.memory_context}")

    parts.append("\n## Conversation")
    for m in trace.messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        thinking = m.get("thinking", "")
        tool_calls = m.get("tool_calls", [])

        if role == "user":
            parts.append(f"\nUser: {content}")
        elif role == "assistant":
            if thinking:
                parts.append(f"\n[Thinking]: {thinking}")
            if tool_calls:
                for tc in tool_calls:
                    parts.append(
                        f"\n[Tool Call]: {tc.get('name', '?')}({tc.get('arguments', {})})"
                    )
            if content:
                parts.append(f"\nUno: {content}")
        elif role == "tool":
            name = m.get("name", "?")
            parts.append(f"\n[Tool Result ({name})]: {content[:200]}")

    tools_available = trace.metadata.get("tools", [])
    if tools_available:
        parts.append(f"\n## Available Tools: {', '.join(tools_available)}")
    else:
        parts.append("\n## Available Tools: none")

    return "\n".join(parts)


def score_trace(
    trace: DatagenTrace,
    backend: LLMBackend,
    system_prompt: str = "",
) -> QualityScore | None:
    """Score a trace using an LLM judge plus a programmatic length check."""
    formatted = _format_trace_for_judge(trace, system_prompt)
    result = backend.generate(
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": formatted}],
        response_schema=JudgeResponse,
    )
    if result is None:
        log.error(f"Judge failed for trace {trace.id}")
        return None

    try:
        judge = JudgeResponse.model_validate_json(result.text)
    except ValueError as e:
        log.warning(f"Invalid judge response for {trace.id}: {e}")
        return None

    length_ok = check_response_length(trace)

    overall = (
        judge.character_consistency >= 3.0
        and judge.thinking_quality >= 3.0
        and judge.language_consistent
        and judge.tool_correctness != "fail"
        and length_ok
    )

    return QualityScore(
        trace_id=trace.id,
        character_consistency=judge.character_consistency,
        thinking_quality=judge.thinking_quality,
        tool_correctness=judge.tool_correctness,
        language_consistent=judge.language_consistent,
        response_length_ok=length_ok,
        overall_pass=overall,
        justification=judge.justification,
    )


# ============================================================================
# Main
# ============================================================================


def load_scored_ids(output_path: Path) -> set[str]:
    """Scan an existing scored JSONL for trace IDs already scored."""
    scored: set[str] = set()
    if not output_path.exists():
        return scored
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                scored.add(data["score"]["trace_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return scored


def filter_traces(
    input_path: Path,
    scored_path: Path,
    filtered_path: Path,
    backend: LLMBackend,
    system_prompt: str = "",
    max_items: int = 0,
) -> tuple[int, int]:
    """Score and filter traces.

    Returns (total_scored, total_passed).
    """
    traces = load_traces(input_path)
    if not traces:
        log.info("No traces found.")
        return 0, 0

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    already_scored = load_scored_ids(scored_path)
    pending = [t for t in traces if t.id not in already_scored]

    if max_items > 0:
        pending = pending[:max_items]

    if not pending:
        log.info("All traces already scored.")
        passed = 0
        with open(scored_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("score", {}).get("overall_pass"):
                        passed += 1
                except json.JSONDecodeError:
                    continue
        return len(traces), passed

    skipped = len(traces) - len(pending)
    if skipped > 0:
        log.info(f"Resuming: skipping {skipped} already-scored traces")

    total_scored = skipped
    total_passed = 0

    with (
        open(scored_path, "a", encoding="utf-8") as scored_f,
        open(filtered_path, "a", encoding="utf-8") as filtered_f,
        Progress(console=console) as progress,
    ):
        task = progress.add_task("Scoring traces", total=len(pending))

        for trace in pending:
            score = score_trace(trace, backend, system_prompt)

            if score is None:
                log.warning(f"Skipping trace {trace.id}: judge returned no score")
                progress.advance(task)
                continue

            scored = ScoredTrace(trace=trace, score=score)
            scored_f.write(scored.model_dump_json() + "\n")
            scored_f.flush()
            total_scored += 1

            if score.overall_pass:
                filtered_f.write(trace.model_dump_json() + "\n")
                filtered_f.flush()
                total_passed += 1

            progress.advance(task)

    return total_scored, total_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score and filter generated traces for SFT quality"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/datagen/traces.jsonl"),
        help="Input JSONL file with DatagenTrace entries",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=Path("output/datagen/traces_scored.jsonl"),
        help="Output JSONL with scored traces",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path("output/datagen/traces_filtered.jsonl"),
        help="Output JSONL with passing traces only",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="LLM backend for judge scoring (e.g. gemini, anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for the judge",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Score at most N traces (0 = unlimited)",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Trace Quality Filter[/bold cyan]\n")

    traces_dir = args.input.parent
    system_prompt = load_system_prompt(traces_dir)
    log.info(f"Loaded system prompt from sidecar files in {traces_dir}")

    backend = create_backend(args.backend, args.model)
    log.info(f"Using LLM judge: {args.backend}")

    total, passed = filter_traces(
        args.input,
        args.scored_output,
        args.filtered_output,
        backend,
        system_prompt,
        max_items=args.max_items,
    )

    if total > 0:
        console.print(
            f"\n[bold green]Done.[/bold green] "
            f"{passed}/{total} traces passed filtering "
            f"({passed / total * 100:.0f}%)"
        )
    else:
        console.print("\n[bold green]Done.[/bold green] No traces to filter.")


if __name__ == "__main__":
    main()
