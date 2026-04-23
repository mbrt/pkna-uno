#!/usr/bin/env python3

"""Push a local HuggingFace Dataset to the Hub.

Loads a dataset saved with save_to_disk() and pushes it to HuggingFace
Hub with proper Parquet conversion (avoids the _format_kwargs Arrow
error that occurs with raw file uploads).

Usage:
    python scripts/push_dataset.py output/sft/dataset your-user/uno-sft-dataset
    python scripts/push_dataset.py output/distillation/prompts your-user/uno-distill-prompts
    python scripts/push_dataset.py output/sft/dataset your-user/uno-sft-dataset --private
"""

import argparse

from datasets import load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a local HF Dataset to the Hub")
    parser.add_argument("path", help="Local dataset directory (from save_to_disk)")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. user/dataset-name)")
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository"
    )
    args = parser.parse_args()

    dataset = load_from_disk(args.path)
    print(f"Loaded dataset with {len(dataset)} examples from {args.path}")

    dataset.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
