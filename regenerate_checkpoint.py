#!/usr/bin/env python3

"""
Regenerate a checkpoint by applying diffs sequentially.

This script takes a checkpoint number and regenerates it by starting from the
seed document and applying all diffs up to that checkpoint in sequence.

Usage:
    ./regenerate_checkpoint.py 15
    ./regenerate_checkpoint.py 15 --version v6
    ./regenerate_checkpoint.py 15 --output custom.md
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import tiktoken
from rich.console import Console

console = Console()

# Default settings
DEFAULT_VERSION = "v7"
ENCODING_NAME = "cl100k_base"


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


def find_base_dir() -> Path:
    """Find the project base directory."""
    return Path(__file__).parent


def validate_checkpoint_number(checkpoint_num: int) -> None:
    """Validate that checkpoint number is positive."""
    if checkpoint_num < 1:
        console.print("[bold red]Error:[/bold red] Checkpoint number must be >= 1")
        sys.exit(1)


def get_paths(version: str) -> tuple[Path, Path, Path, Path]:
    """Get all required paths for the regeneration process.

    Returns:
        Tuple of (base_dir, seed_path, diffs_dir, output_dir)
    """
    base_dir = find_base_dir()
    profile_dir = base_dir / "output" / "character-profile" / "uno" / version
    seed_path = profile_dir / "seed_document.md"
    diffs_dir = profile_dir / "diffs"
    output_dir = profile_dir / "regenerated"

    return base_dir, seed_path, diffs_dir, output_dir


def validate_paths(seed_path: Path, diffs_dir: Path, checkpoint_num: int) -> list[Path]:
    """Validate that all required files exist.

    Returns:
        List of diff file paths in order.
    """
    if not seed_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Seed document not found: {seed_path}"
        )
        sys.exit(1)

    if not diffs_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Diffs directory not found: {diffs_dir}"
        )
        sys.exit(1)

    # Check all required diffs exist
    diff_files = []
    missing_diffs = []

    for i in range(1, checkpoint_num + 1):
        diff_path = diffs_dir / f"checkpoint_{i:03d}.diff"
        if not diff_path.exists():
            missing_diffs.append(f"checkpoint_{i:03d}.diff")
        else:
            diff_files.append(diff_path)

    if missing_diffs:
        console.print("[bold red]Error:[/bold red] Missing diff files:")
        for missing in missing_diffs:
            console.print(f"  - {missing}")
        sys.exit(1)

    return diff_files


def apply_patch(target_file: Path, diff_file: Path) -> bool:
    """Apply a unified diff to a file using the patch command.

    Args:
        target_file: File to patch
        diff_file: Diff file to apply

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use patch command with:
        # -p0: don't strip any path prefixes
        # --quiet: suppress normal output
        # --input: read patch from file
        # target_file: file to patch
        result = subprocess.run(
            ["patch", "-p0", "--quiet", str(target_file)],
            stdin=open(diff_file, "r"),
            capture_output=True,
            text=True,
        )

        return result.returncode == 0
    except Exception as e:
        console.print(f"[bold red]Error applying patch:[/bold red] {e}")
        return False


def regenerate_checkpoint(
    checkpoint_num: int,
    seed_path: Path,
    diff_files: list[Path],
    output_path: Path,
) -> None:
    """Regenerate a checkpoint by applying diffs sequentially.

    Args:
        checkpoint_num: Target checkpoint number
        seed_path: Path to seed document
        diff_files: List of diff files to apply in order
        output_path: Where to save the regenerated checkpoint
    """
    console.print(
        f"\n[bold cyan]Regenerating checkpoint {checkpoint_num}[/bold cyan]\n"
    )
    console.print(f"Seed document: {seed_path}")
    console.print(f"Applying {len(diff_files)} diffs...\n")

    # Create a temporary file to work with
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as temp_file:
        # Copy seed document to temp file
        seed_content = seed_path.read_text(encoding="utf-8")
        temp_file.write(seed_content)
        temp_path = Path(temp_file.name)

    try:
        # Apply each diff in sequence
        for i, diff_file in enumerate(diff_files, start=1):
            console.print(f"Applying diff {i}/{len(diff_files)}: {diff_file.name}")

            if not apply_patch(temp_path, diff_file):
                console.print(
                    f"[bold red]Error:[/bold red] Failed to apply {diff_file.name}"
                )
                sys.exit(1)

        # Read the final result
        final_content = temp_path.read_text(encoding="utf-8")

        # Save to output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_content, encoding="utf-8")

        # Print summary
        token_count = count_tokens(final_content)
        line_count = len(final_content.splitlines())
        word_count = len(final_content.split())

        console.print(
            f"\n[bold green]✓ Successfully regenerated checkpoint {checkpoint_num}[/bold green]\n"
        )
        console.print(f"Output: {output_path}")
        console.print(f"  Lines: {line_count:,}")
        console.print(f"  Words: {word_count:,}")
        console.print(f"  Tokens: {token_count:,}")

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate a checkpoint by applying diffs sequentially from the seed document."
    )
    parser.add_argument(
        "checkpoint", type=int, help="Checkpoint number to regenerate (e.g., 15)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"Version tag (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output file path (default: output/character-profile/uno/{version}/regenerated/checkpoint_{num:03d}.md)",
    )

    args = parser.parse_args()

    # Validate checkpoint number
    validate_checkpoint_number(args.checkpoint)

    # Get paths
    base_dir, seed_path, diffs_dir, output_dir = get_paths(args.version)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"checkpoint_{args.checkpoint:03d}.md"

    # Validate all required files exist
    diff_files = validate_paths(seed_path, diffs_dir, args.checkpoint)

    # Regenerate the checkpoint
    regenerate_checkpoint(args.checkpoint, seed_path, diff_files, output_path)


if __name__ == "__main__":
    main()
