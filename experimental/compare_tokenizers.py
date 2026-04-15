#!/usr/bin/env python3
"""Compare tokenizer coverage between Gemini 3 and SmolLM3.

Analyzes how tokens from one vocabulary are represented in the other,
providing statistics on exact matches, splits, and expansion ratios.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TaskID, TimeElapsedColumn
from rich.table import Table
from transformers import AutoTokenizer, PreTrainedTokenizerBase

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)

GEMINI_MODEL = "google/gemma-3-1b-it"
SMOLLM_MODEL = "HuggingFaceTB/SmolLM3-3B"


@dataclass
class TokenMapping:
    """Mapping of a source token to its target representation."""

    source_token: str
    source_id: int
    target_tokens: list[str]
    target_ids: list[int]
    is_exact_match: bool


@dataclass
class CoverageStats:
    """Statistics for tokenizer coverage analysis."""

    source_name: str
    target_name: str
    source_vocab_size: int
    target_vocab_size: int
    exact_matches: int = 0
    one_to_one: int = 0
    one_to_many: int = 0
    total_target_tokens: int = 0
    max_split: int = 0
    max_split_token: str = ""

    @property
    def avg_expansion(self) -> float:
        """Average number of target tokens per source token."""
        total = self.one_to_one + self.one_to_many
        if total == 0:
            return 0.0
        return self.total_target_tokens / total


@dataclass
class AnalysisResult:
    """Complete analysis result for one direction."""

    stats: CoverageStats
    mappings: list[TokenMapping] = field(default_factory=list)
    sample_exact: list[TokenMapping] = field(default_factory=list)
    sample_splits: list[TokenMapping] = field(default_factory=list)


def load_tokenizers() -> tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    """Load both tokenizers from HuggingFace."""
    log.info(f"Loading Gemini tokenizer from {GEMINI_MODEL}")
    gemini = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(GEMINI_MODEL))

    log.info(f"Loading SmolLM3 tokenizer from {SMOLLM_MODEL}")
    smollm = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(SMOLLM_MODEL))

    return gemini, smollm


def get_vocab(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """Extract vocabulary from tokenizer."""
    return tokenizer.get_vocab()


def build_coverage_map(
    target_tokenizer: PreTrainedTokenizerBase,
    source_vocab: dict[str, int],
    limit: int | None = None,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> list[TokenMapping]:
    """Map each source token to its target representation."""
    mappings = []
    items = list(source_vocab.items())
    if limit:
        items = items[:limit]

    for i, (token, token_id) in enumerate(items):
        if progress and task_id is not None:
            progress.update(task_id, completed=i + 1)

        try:
            target_ids: list[int] = target_tokenizer.encode(
                token, add_special_tokens=False
            )
            target_tokens = [str(target_tokenizer.decode([tid])) for tid in target_ids]

            is_exact = len(target_ids) == 1 and target_tokens[0] == token

            mappings.append(
                TokenMapping(
                    source_token=token,
                    source_id=token_id,
                    target_tokens=target_tokens,
                    target_ids=target_ids,
                    is_exact_match=is_exact,
                )
            )
        except Exception as e:
            log.warning(f"Failed to process token {repr(token)}: {e}")

    return mappings


def compute_stats(
    mappings: list[TokenMapping],
    source_name: str,
    target_name: str,
    source_size: int,
    target_size: int,
) -> CoverageStats:
    """Compute coverage statistics from mappings."""
    stats = CoverageStats(
        source_name=source_name,
        target_name=target_name,
        source_vocab_size=source_size,
        target_vocab_size=target_size,
    )

    for m in mappings:
        num_targets = len(m.target_ids)
        stats.total_target_tokens += num_targets

        if m.is_exact_match:
            stats.exact_matches += 1
            stats.one_to_one += 1
        elif num_targets == 1:
            stats.one_to_one += 1
        else:
            stats.one_to_many += 1

        if num_targets > stats.max_split:
            stats.max_split = num_targets
            stats.max_split_token = m.source_token

    return stats


def analyze_direction(
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    source_name: str,
    target_name: str,
    limit: int | None = None,
) -> AnalysisResult:
    """Analyze coverage from source to target tokenizer."""
    source_vocab = get_vocab(source_tokenizer)
    target_vocab = get_vocab(target_tokenizer)
    source_size = len(source_vocab)
    target_size = len(target_vocab)

    log.info(f"Analyzing {source_name} → {target_name}")
    log.info(f"  Source vocab: {source_size:,} tokens")
    log.info(f"  Target vocab: {target_size:,} tokens")

    total = min(limit, source_size) if limit else source_size

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Mapping {source_name} tokens", total=total)
        mappings = build_coverage_map(
            target_tokenizer, source_vocab, limit=limit, progress=progress, task_id=task
        )

    stats = compute_stats(mappings, source_name, target_name, source_size, target_size)

    # Collect samples
    sample_exact = [m for m in mappings if m.is_exact_match][:10]
    sample_splits = sorted(
        [m for m in mappings if len(m.target_ids) > 1], key=lambda x: -len(x.target_ids)
    )[:10]

    return AnalysisResult(
        stats=stats,
        mappings=mappings,
        sample_exact=sample_exact,
        sample_splits=sample_splits,
    )


def print_report(result: AnalysisResult) -> None:
    """Print coverage report to console."""
    stats = result.stats
    analyzed = stats.one_to_one + stats.one_to_many

    console.print()
    console.print(
        f"[bold]{stats.source_name} → {stats.target_name} Coverage Analysis[/bold]"
    )
    console.print("=" * 50)
    console.print()
    console.print(f"Source: {stats.source_name} ({stats.source_vocab_size:,} tokens)")
    console.print(f"Target: {stats.target_name} ({stats.target_vocab_size:,} tokens)")
    console.print(f"Analyzed: {analyzed:,} tokens")
    console.print()

    if analyzed > 0:
        exact_pct = stats.exact_matches / analyzed * 100
        one_to_one_pct = stats.one_to_one / analyzed * 100
        one_to_many_pct = stats.one_to_many / analyzed * 100

        console.print("[bold]Coverage Statistics:[/bold]")
        console.print(
            f"  Exact matches:   {stats.exact_matches:>8,} ({exact_pct:.1f}%)"
        )
        console.print(
            f"  One-to-one:      {stats.one_to_one:>8,} ({one_to_one_pct:.1f}%)"
        )
        console.print(
            f"  One-to-many:     {stats.one_to_many:>8,} ({one_to_many_pct:.1f}%)"
        )
        console.print(f"  Avg expansion:   {stats.avg_expansion:>8.2f} tokens")
        console.print(
            f"  Max expansion:   {stats.max_split:>8} tokens ({repr(stats.max_split_token)})"
        )
        console.print()

    # Sample exact matches table
    if result.sample_exact:
        table = Table(title="Sample Exact Matches")
        table.add_column("Token", style="cyan")
        table.add_column(f"{stats.source_name} ID", justify="right")
        table.add_column(f"{stats.target_name} ID", justify="right")

        for m in result.sample_exact[:5]:
            table.add_row(repr(m.source_token), str(m.source_id), str(m.target_ids[0]))

        console.print(table)
        console.print()

    # Sample splits table
    if result.sample_splits:
        table = Table(title="Sample Splits (one-to-many)")
        table.add_column(f"{stats.source_name} Token", style="cyan")
        table.add_column(f"{stats.target_name} Tokens", style="yellow")
        table.add_column("Count", justify="right")

        for m in result.sample_splits[:5]:
            target_repr = repr(m.target_tokens)
            if len(target_repr) > 50:
                target_repr = target_repr[:47] + "..."
            table.add_row(repr(m.source_token), target_repr, str(len(m.target_ids)))

        console.print(table)
        console.print()


def analyze_text(
    text: str, gemini_tok: PreTrainedTokenizerBase, smollm_tok: PreTrainedTokenizerBase
) -> None:
    """Analyze tokenization of specific text with both tokenizers."""
    console.print()
    console.print("[bold]Text Tokenization Analysis[/bold]")
    console.print("=" * 50)
    console.print()

    # Truncate display if too long
    display_text = text if len(text) <= 100 else text[:97] + "..."
    console.print(f"Text: {repr(display_text)}")
    console.print()

    gemini_ids = gemini_tok.encode(text, add_special_tokens=False)
    smollm_ids = smollm_tok.encode(text, add_special_tokens=False)

    gemini_tokens = [gemini_tok.decode([tid]) for tid in gemini_ids]
    smollm_tokens = [smollm_tok.decode([tid]) for tid in smollm_ids]

    console.print(f"[cyan]Gemini 3[/cyan]: {len(gemini_ids)} tokens")
    console.print(f"  Tokens: {gemini_tokens[:20]}")
    if len(gemini_tokens) > 20:
        console.print(f"  ... and {len(gemini_tokens) - 20} more")
    console.print()

    console.print(f"[yellow]SmolLM3[/yellow]: {len(smollm_ids)} tokens")
    console.print(f"  Tokens: {smollm_tokens[:20]}")
    if len(smollm_tokens) > 20:
        console.print(f"  ... and {len(smollm_tokens) - 20} more")
    console.print()

    ratio = len(gemini_ids) / len(smollm_ids) if smollm_ids else 0
    console.print(f"Ratio (Gemini/SmolLM3): {ratio:.2f}")
    console.print()


def export_json(
    gemini_result: AnalysisResult | None,
    smollm_result: AnalysisResult | None,
    output_path: Path,
) -> None:
    """Export analysis results to JSON."""
    data: dict = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gemini_model": GEMINI_MODEL,
            "smollm_model": SMOLLM_MODEL,
        },
    }

    if gemini_result:
        stats = gemini_result.stats
        analyzed = stats.one_to_one + stats.one_to_many
        data["gemini_to_smollm"] = {
            "statistics": {
                "source_vocab_size": stats.source_vocab_size,
                "target_vocab_size": stats.target_vocab_size,
                "analyzed_tokens": analyzed,
                "exact_matches": stats.exact_matches,
                "one_to_one": stats.one_to_one,
                "one_to_many": stats.one_to_many,
                "avg_expansion": round(stats.avg_expansion, 4),
                "max_expansion": stats.max_split,
                "max_expansion_token": stats.max_split_token,
            },
            "mappings": {
                m.source_token: {
                    "id": m.source_id,
                    "target_ids": m.target_ids,
                    "target_tokens": m.target_tokens,
                    "exact": m.is_exact_match,
                }
                for m in gemini_result.mappings
            },
        }

    if smollm_result:
        stats = smollm_result.stats
        analyzed = stats.one_to_one + stats.one_to_many
        data["smollm_to_gemini"] = {
            "statistics": {
                "source_vocab_size": stats.source_vocab_size,
                "target_vocab_size": stats.target_vocab_size,
                "analyzed_tokens": analyzed,
                "exact_matches": stats.exact_matches,
                "one_to_one": stats.one_to_one,
                "one_to_many": stats.one_to_many,
                "avg_expansion": round(stats.avg_expansion, 4),
                "max_expansion": stats.max_split,
                "max_expansion_token": stats.max_split_token,
            },
            "mappings": {
                m.source_token: {
                    "id": m.source_id,
                    "target_ids": m.target_ids,
                    "target_tokens": m.target_tokens,
                    "exact": m.is_exact_match,
                }
                for m in smollm_result.mappings
            },
        }

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    log.info(f"Exported results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tokenizer coverage between Gemini 3 and SmolLM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--direction",
        choices=["gemini-to-smollm", "smollm-to-gemini", "both"],
        default="both",
        help="Which direction(s) to analyze (default: both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit analysis to first N tokens (for quick testing)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Analyze tokenization of specific text",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Analyze tokenization of text from file",
    )

    args = parser.parse_args()

    # Load tokenizers
    gemini_tok, smollm_tok = load_tokenizers()

    gemini_result = None
    smollm_result = None

    # Analyze specified direction(s)
    if args.direction in ("gemini-to-smollm", "both"):
        gemini_result = analyze_direction(
            gemini_tok, smollm_tok, "Gemini 3", "SmolLM3", limit=args.limit
        )
        print_report(gemini_result)

    if args.direction in ("smollm-to-gemini", "both"):
        smollm_result = analyze_direction(
            smollm_tok, gemini_tok, "SmolLM3", "Gemini 3", limit=args.limit
        )
        print_report(smollm_result)

    # Analyze specific text if provided
    text_to_analyze = None
    if args.text:
        text_to_analyze = args.text
    elif args.file:
        text_to_analyze = args.file.read_text()

    if text_to_analyze:
        analyze_text(text_to_analyze, gemini_tok, smollm_tok)

    # Export to JSON if requested
    if args.output:
        export_json(gemini_result, smollm_result, args.output)


if __name__ == "__main__":
    main()
