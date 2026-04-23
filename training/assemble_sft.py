#!/usr/bin/env python3

"""Assemble SFT dataset from filtered traces.

Reads quality-filtered DatagenTrace JSONL, converts each trace to standard
chat messages (reasoning_content + OpenAI tool_calls), and saves as a
HuggingFace Dataset.

The output is tokenizer-independent: each row has a ``messages`` column
containing a list of message dicts that can be passed directly to any
tokenizer's apply_chat_template(). Token-length filtering happens at
training time, not here.

Usage:
    python training/assemble_sft.py \
        --input output/datagen/traces_filtered.jsonl \
        --output output/sft/dataset
"""

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset

from datagen.filter_traces import load_traces
from pkna.inference.system_prompts import MINIMAL_TEMPLATE
from pkna.logging import setup_logging
from pkna.training.sft_dataset import trace_to_messages

console, log = setup_logging()


def assemble_dataset(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
) -> Dataset:
    """Load traces, convert to standard chat messages, and save.

    Returns the assembled Dataset.
    """
    traces = load_traces(input_path)
    if not traces:
        log.warning("No traces found in %s", input_path)
        return Dataset.from_dict({"messages": []})

    log.info("Loaded %d filtered traces", len(traces))

    all_messages: list[list[dict[str, Any]]] = []
    for trace in traces:
        messages = trace_to_messages(trace, system_prompt)
        all_messages.append(messages)

    dataset = Dataset.from_dict({"messages": all_messages})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    log.info("Saved dataset with %d examples to %s", len(dataset), output_path)

    return dataset


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
    args = parser.parse_args()

    console.print("[bold cyan]SFT Dataset Assembly[/bold cyan]\n")
    log.info("Using minimal system prompt for SFT")

    assemble_dataset(
        input_path=args.input,
        output_path=args.output,
        system_prompt=MINIMAL_TEMPLATE,
    )

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
