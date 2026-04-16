#!/usr/bin/env python3

"""Assemble SFT dataset from filtered traces.

Reads quality-filtered DatagenTrace JSONL, converts each trace to the
Qwen3.5 chat message format, renders via the tokenizer's chat template,
and saves as a HuggingFace Dataset ready for SFTTrainer.

Runs on CPU -- no GPU required.

Usage:
    python training/assemble_sft.py \
        --input output/datagen/traces_filtered.jsonl \
        --output output/sft/dataset \
        --model Qwen/Qwen3.5-4B \
        --max-seq-length 8192
"""

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from datagen.filter_traces import load_traces
from pkna.training.sft_dataset import trace_to_messages

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    force=True,
)
log = logging.getLogger(__name__)


def assemble_dataset(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_seq_length: int,
) -> Dataset:
    """Load traces, convert to chat format, tokenize, and save.

    Returns the assembled Dataset.
    """
    traces = load_traces(input_path)
    if not traces:
        log.warning("No traces found in %s", input_path)
        return Dataset.from_dict({"text": []})

    log.info("Loaded %d filtered traces", len(traces))

    log.info("Loading tokenizer for %s", model_name)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True),
    )

    texts: list[str] = []
    token_lengths: list[int] = []
    skipped = 0

    for trace in traces:
        messages = trace_to_messages(trace)
        text = cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            ),
        )
        n_tokens = len(tokenizer.encode(text))

        if n_tokens > max_seq_length:
            skipped += 1
            continue

        texts.append(text)
        token_lengths.append(n_tokens)

    if skipped > 0:
        log.info("Skipped %d examples exceeding %d tokens", skipped, max_seq_length)

    dataset = Dataset.from_dict({"text": texts})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    log.info("Saved dataset with %d examples to %s", len(dataset), output_path)

    _print_stats(token_lengths)

    return dataset


def _print_stats(token_lengths: list[int]) -> None:
    """Print token length statistics."""
    if not token_lengths:
        console.print("[yellow]No examples to report stats on.[/yellow]")
        return

    arr = np.array(token_lengths)
    table = Table(title="Token Length Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Count", str(len(arr)))
    table.add_row("Min", str(int(arr.min())))
    table.add_row("Max", str(int(arr.max())))
    table.add_row("Mean", f"{arr.mean():.0f}")
    table.add_row("Median", f"{np.median(arr):.0f}")
    table.add_row("P95", f"{np.percentile(arr, 95):.0f}")
    table.add_row("P99", f"{np.percentile(arr, 99):.0f}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble SFT dataset from filtered traces"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/datagen/traces_filtered.jsonl"),
        help="Input JSONL with filtered DatagenTrace entries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sft/dataset"),
        help="Output directory for the HuggingFace Dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Model name for tokenizer (used for chat template + token counting)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length in tokens; longer examples are dropped",
    )
    args = parser.parse_args()

    console.print("[bold cyan]SFT Dataset Assembly[/bold cyan]\n")

    assemble_dataset(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        max_seq_length=args.max_seq_length,
    )

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
