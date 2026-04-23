#!/usr/bin/env python3

"""Sample distillation prompts from Tulu3.

Downloads `allenai/tulu-3-sft-mixture` from HuggingFace and samples
N prompts with stratified random sampling across source subsets.
Each example is converted to prompt-only format (system + first user
message only -- no assistant turns).

The output dataset is suitable for TRL's DistillationTrainer with
lmbda=1.0 (fully on-policy): the student generates its own completions.

Usage:
    python distillation/generate_prompts.py \
        --output output/distillation/prompts \
        --max-items 600 \
        --seed 42
"""

import argparse
import math
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from rich.table import Table

from pkna.logging import setup_logging

console, log = setup_logging()

TULU3_DATASET = "allenai/tulu-3-sft-mixture"
DEFAULT_N_PROMPTS = 600
DEFAULT_SEED = 42


def to_prompt_only(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Extract prompt-only messages: system (if present) + first user turn."""
    result: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            result.append({"role": "system", "content": msg["content"]})
        elif role == "user":
            result.append({"role": "user", "content": msg["content"]})
            break
    return result


def sample_prompts(
    n_prompts: int = DEFAULT_N_PROMPTS,
    seed: int = DEFAULT_SEED,
    dataset: Dataset | None = None,
) -> Dataset:
    """Sample prompts from Tulu3 with stratified sampling across sources.

    Args:
        n_prompts: Total number of prompts to sample.
        seed: Random seed for reproducibility.
        dataset: Pre-loaded Tulu3 dataset (for testing). If None, loads
            from HuggingFace.

    Returns:
        HF Dataset with a single ``messages`` column (prompt-only).
    """
    if dataset is None:
        log.info("Loading %s from HuggingFace...", TULU3_DATASET)
        dataset = cast(Dataset, load_dataset(TULU3_DATASET, split="train"))

    sources: dict[str, list[int]] = {}
    for i, source in enumerate(dataset["source"]):
        sources.setdefault(source, []).append(i)

    log.info("Found %d sources with %d total examples", len(sources), len(dataset))

    import random

    rng = random.Random(seed)

    # Proportional allocation: each source gets ceil(proportion * n_prompts),
    # then we trim to exactly n_prompts.
    total = len(dataset)
    selected_indices: list[int] = []

    for source_name, indices in sorted(sources.items()):
        proportion = len(indices) / total
        n_from_source = max(1, math.ceil(proportion * n_prompts))
        n_from_source = min(n_from_source, len(indices))
        sampled = rng.sample(indices, n_from_source)
        selected_indices.extend(sampled)

    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:n_prompts]

    all_messages: list[list[dict[str, str]]] = []
    skipped = 0
    for idx in selected_indices:
        raw_messages = dataset[idx]["messages"]
        prompt = to_prompt_only(raw_messages)
        if not any(m["role"] == "user" for m in prompt):
            skipped += 1
            continue
        all_messages.append(prompt)

    if skipped > 0:
        log.info("Skipped %d examples with no user message", skipped)

    result = Dataset.from_dict({"messages": all_messages})
    log.info("Sampled %d prompts from %d sources", len(result), len(sources))
    return result


def print_stats(dataset: Dataset, full_dataset: Dataset | None = None) -> None:
    """Print sampling statistics."""
    source_counts: dict[str, int] = {}
    if full_dataset is not None:
        for source in full_dataset["source"]:
            source_counts[source] = source_counts.get(source, 0) + 1

    has_system = sum(
        1 for msgs in dataset["messages"] if any(m["role"] == "system" for m in msgs)
    )

    table = Table(title="Distillation Prompt Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total prompts", str(len(dataset)))
    table.add_row("With system prompt", str(has_system))
    table.add_row("Without system prompt", str(len(dataset) - has_system))
    if source_counts:
        table.add_row("Source subsets in Tulu3", str(len(source_counts)))
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample distillation prompts from Tulu3"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/distillation/prompts"),
        help="Output directory for HuggingFace Dataset",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_N_PROMPTS,
        help="Number of prompts to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: %(default)s)",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Distillation Prompt Sampler[/bold cyan]\n")

    log.info("Loading Tulu3 dataset...")
    full_dataset = cast(Dataset, load_dataset(TULU3_DATASET, split="train"))

    result = sample_prompts(
        n_prompts=args.max_items,
        seed=args.seed,
        dataset=full_dataset,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(args.output))
    log.info("Saved dataset to %s", args.output)

    print_stats(result, full_dataset)

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
