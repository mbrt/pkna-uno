#!/usr/bin/env python3

"""Assemble SFT dataset from filtered traces.

Reads quality-filtered DatagenTrace JSONL, renders each trace into a ChatML
text string using the target tokenizer's chat template, and saves as a
HuggingFace Dataset with a single ``text`` column.

Pre-rendering at assembly time is intentional: PyArrow's struct-schema
unification would otherwise pad every ``tool_call.arguments`` dict with the
union of all argument keys seen across the dataset, polluting the training
text with spurious ``<parameter=...>\\nNone\\n</parameter>`` blocks. Storing
only a string column keeps the on-disk / HF-Hub representation clean and
makes training a straight tokenize-and-filter step.

Usage:
    python training/assemble_sft.py \
        --input output/datagen/traces_filtered.jsonl \
        --output output/sft/dataset \
        --model unsloth/Qwen3.5-4B
"""

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from datagen.filter_traces import load_traces
from pkna.inference.system_prompts import MINIMAL_TEMPLATE
from pkna.logging import setup_logging
from pkna.training.sft_dataset import trace_to_chatml_text

console, log = setup_logging()


def assemble_dataset(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    model_name: str,
) -> Dataset:
    """Load traces, render to ChatML text, and save as a text-only dataset.

    Returns the assembled Dataset.
    """
    traces = load_traces(input_path)
    if not traces:
        log.warning("No traces found in %s", input_path)
        return Dataset.from_dict({"text": []})

    log.info("Loaded %d filtered traces", len(traces))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts: list[str] = []
    for trace in traces:
        texts.append(trace_to_chatml_text(trace, system_prompt, tokenizer))

    dataset = Dataset.from_dict({"text": texts})

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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Tokenizer to render ChatML with "
            "(e.g. unsloth/Qwen3.5-4B or a local merged model path)"
        ),
    )
    args = parser.parse_args()

    console.print("[bold cyan]SFT Dataset Assembly[/bold cyan]\n")
    log.info("Using minimal system prompt for SFT")

    assemble_dataset(
        input_path=args.input,
        output_path=args.output,
        system_prompt=MINIMAL_TEMPLATE,
        model_name=args.model,
    )

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
