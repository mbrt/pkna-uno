#!/usr/bin/env python3

"""Pretty-print datagen traces for manual inspection.

Renders DatagenTrace or ScoredTrace JSONL files as readable conversations
using Rich panels and styled text.

Usage:
    python datagen/view_traces.py output/datagen/traces_filtered.jsonl
    python datagen/view_traces.py output/datagen/traces_scored.jsonl
    python datagen/view_traces.py output/datagen/traces.jsonl --id manual-emotional-001
    python datagen/view_traces.py output/datagen/traces_scored.jsonl --failed
"""

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pkna.datagen.types import DatagenTrace, QualityScore, ScoredTrace

console = Console()


# ============================================================================
# Loading
# ============================================================================


def _load_entries(path: Path) -> list[tuple[DatagenTrace, QualityScore | None]]:
    """Load traces from JSONL, auto-detecting raw vs scored format."""
    entries: list[tuple[DatagenTrace, QualityScore | None]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "trace" in data and "score" in data:
                scored = ScoredTrace.model_validate(data)
                entries.append((scored.trace, scored.score))
            else:
                entries.append((DatagenTrace.model_validate(data), None))
    return entries


# ============================================================================
# Rendering
# ============================================================================


def _render_header(trace: DatagenTrace) -> Panel:
    meta = trace.metadata
    lines = Text()
    lines.append("ID: ", style="bold")
    lines.append(trace.id)
    lines.append("\nUser: ", style="bold")
    lines.append(trace.user_summary)

    category = meta.get("category", "")
    if category:
        lines.append("\nCategory: ", style="bold")
        lines.append(category)

    language = meta.get("language", "")
    if language:
        lines.append("\nLanguage: ", style="bold")
        lines.append(language)

    tools = meta.get("tools", [])
    if tools:
        lines.append("\nTools: ", style="bold")
        lines.append(", ".join(tools) if isinstance(tools, list) else str(tools))

    if meta.get("multi_turn"):
        lines.append("\nTurns: ", style="bold")
        lines.append(str(meta.get("turn_count", "?")))

    return Panel(lines, title="Trace", border_style="cyan")


def _render_message(msg: dict[str, Any], *, show_thinking: bool) -> list[Text]:
    """Render a single message as one or more Rich Text objects."""
    role = msg.get("role", "?")
    blocks: list[Text] = []

    if role == "user":
        t = Text()
        t.append("USER: ", style="bold green")
        t.append(msg.get("content", ""))
        blocks.append(t)

    elif role == "assistant":
        thinking = msg.get("thinking", "")
        if thinking and show_thinking:
            t = Text()
            t.append("  [thinking] ", style="dim italic")
            t.append(thinking, style="dim")
            blocks.append(t)

        for tc in msg.get("tool_calls", []):
            t = Text()
            t.append("  [tool call] ", style="bold yellow")
            t.append(f"{tc.get('name', '?')}(")
            t.append(json.dumps(tc.get("arguments", {}), ensure_ascii=False))
            t.append(")")
            blocks.append(t)

        content = msg.get("content", "")
        if content:
            t = Text()
            t.append("UNO: ", style="bold blue")
            t.append(content)
            blocks.append(t)

    elif role == "tool":
        t = Text()
        name = msg.get("name", "?")
        t.append(f"  [tool result: {name}] ", style="yellow")
        t.append(msg.get("content", ""), style="dim")
        blocks.append(t)

    return blocks


def _render_score(score: QualityScore) -> Panel:
    lines = Text()
    passed = score.overall_pass
    status_style = "bold green" if passed else "bold red"
    lines.append("PASS" if passed else "FAIL", style=status_style)

    lines.append("\nCharacter: ", style="bold")
    lines.append(f"{score.character_consistency:.1f}/5")
    lines.append("  Thinking: ", style="bold")
    lines.append(f"{score.thinking_quality:.1f}/5")
    lines.append("  Tools: ", style="bold")
    lines.append(score.tool_correctness)
    lines.append("  Language: ", style="bold")
    lines.append("ok" if score.language_consistent else "FAIL")
    lines.append("  Length: ", style="bold")
    lines.append("ok" if score.response_length_ok else "FAIL")

    if score.justification:
        lines.append(f"\n{score.justification}", style="dim")

    border = "green" if passed else "red"
    return Panel(lines, title="Score", border_style=border)


def render_trace(
    trace: DatagenTrace,
    score: QualityScore | None,
    *,
    show_thinking: bool,
) -> None:
    console.print()
    console.print(_render_header(trace))

    if trace.memory_context:
        mem = Text()
        mem.append(trace.memory_context, style="dim")
        console.print(Panel(mem, title="Memory Context", border_style="dim"))

    for msg in trace.messages:
        for block in _render_message(msg, show_thinking=show_thinking):
            console.print(block)

    if score is not None:
        console.print(_render_score(score))

    console.rule(style="dim")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print datagen traces for manual inspection"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSONL file (traces, filtered traces, or scored traces)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        dest="id_pattern",
        help="Show only traces whose ID contains this substring",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Show only traces that failed quality filtering (scored files)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Hide thinking blocks",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        dest="max_traces",
        help="Show at most N traces (0 = all)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {args.input}")
        raise SystemExit(1)

    entries = _load_entries(args.input)

    if args.id_pattern:
        entries = [(t, s) for t, s in entries if args.id_pattern in t.id]

    if args.failed:
        entries = [(t, s) for t, s in entries if s is not None and not s.overall_pass]

    if args.max_traces > 0:
        entries = entries[: args.max_traces]

    if not entries:
        console.print("[yellow]No matching traces found.[/yellow]")
        return

    console.print(f"[bold]Showing {len(entries)} trace(s)[/bold]")

    for trace, score in entries:
        render_trace(trace, score, show_thinking=not args.no_thinking)


if __name__ == "__main__":
    main()
