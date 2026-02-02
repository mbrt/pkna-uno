#!/usr/bin/env python3

"""
Regenerate a checkpoint by applying diffs sequentially, or rollback to a checkpoint.

This script takes a checkpoint number and can either:
1. Regenerate it by starting from the seed document and applying all diffs up to
   that checkpoint in sequence.
2. Rollback the directory state to that checkpoint by deleting all data after it
   (with a timestamped backup created first).

Usage:
    # Regenerate a checkpoint
    ./regenerate_checkpoint.py 15
    ./regenerate_checkpoint.py 15 --version v6
    ./regenerate_checkpoint.py 15 --output custom.md

    # Rollback to a checkpoint (creates backup, deletes data after checkpoint)
    ./regenerate_checkpoint.py 100 --rollback
    ./regenerate_checkpoint.py 50 --version v6 --rollback
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
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


def create_backup(profile_dir: Path, version: str) -> Path:
    """Create a timestamped backup of the profile directory.

    Args:
        profile_dir: The version directory to backup
        version: Version name (e.g., 'v7')

    Returns:
        Path to the backup directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"{version}-bkp-{timestamp}"
    backup_dir = profile_dir.parent / backup_name

    console.print(f"\n[bold cyan]Creating backup:[/bold cyan] {backup_name}")

    try:
        shutil.copytree(profile_dir, backup_dir)
        console.print(f"[bold green]✓ Backup created:[/bold green] {backup_dir}\n")
        return backup_dir
    except Exception as e:
        console.print(f"[bold red]Error creating backup:[/bold red] {e}")
        sys.exit(1)


def rollback_processing_log(log_path: Path, checkpoint_num: int) -> int:
    """Truncate processing log to keep only entries up to checkpoint_num.

    Args:
        log_path: Path to processing_log.jsonl
        checkpoint_num: Checkpoint number to keep (inclusive)

    Returns:
        Number of entries removed
    """
    if not log_path.exists():
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Processing log not found: {log_path}"
        )
        return 0

    # Read all entries
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Filter to keep only entries <= checkpoint_num
    kept_entries = [e for e in entries if e.get("scene_number", 0) <= checkpoint_num]
    removed_count = len(entries) - len(kept_entries)

    # Write back the filtered entries
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in kept_entries:
            f.write(json.dumps(entry) + "\n")

    return removed_count


def delete_diffs_after(diffs_dir: Path, checkpoint_num: int) -> int:
    """Delete all diff files after checkpoint_num.

    Args:
        diffs_dir: Directory containing checkpoint diffs
        checkpoint_num: Checkpoint number to keep (inclusive)

    Returns:
        Number of diff files deleted
    """
    if not diffs_dir.exists():
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Diffs directory not found: {diffs_dir}"
        )
        return 0

    deleted_count = 0
    for diff_file in sorted(diffs_dir.glob("checkpoint_*.diff")):
        # Extract checkpoint number from filename
        try:
            num_str = diff_file.stem.split("_")[1]
            num = int(num_str)
            if num > checkpoint_num:
                diff_file.unlink()
                deleted_count += 1
        except (IndexError, ValueError):
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Skipping invalid diff filename: {diff_file.name}"
            )

    return deleted_count


def delete_checkpoints_after(checkpoints_dir: Path, checkpoint_num: int) -> int:
    """Delete all checkpoint markdown files except the target checkpoint.

    Args:
        checkpoints_dir: Directory containing checkpoint markdown files
        checkpoint_num: Checkpoint number to keep (only this one is preserved)

    Returns:
        Number of checkpoint files deleted
    """
    if not checkpoints_dir.exists():
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Checkpoints directory not found: {checkpoints_dir}"
        )
        return 0

    deleted_count = 0
    for checkpoint_file in sorted(checkpoints_dir.glob("checkpoint_*.md")):
        # Extract checkpoint number from filename
        try:
            num_str = checkpoint_file.stem.split("_")[1]
            num = int(num_str)
            # Delete all checkpoints except the target one
            if num != checkpoint_num:
                checkpoint_file.unlink()
                deleted_count += 1
        except (IndexError, ValueError):
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Skipping invalid checkpoint filename: {checkpoint_file.name}"
            )

    return deleted_count


def delete_regenerated_dir(regenerated_dir: Path) -> bool:
    """Delete the regenerated directory if it exists.

    Args:
        regenerated_dir: Path to regenerated directory

    Returns:
        True if directory was deleted, False if it didn't exist
    """
    if regenerated_dir.exists():
        shutil.rmtree(regenerated_dir)
        return True
    return False


def perform_rollback(checkpoint_num: int, version: str) -> None:
    """Perform a rollback to a specific checkpoint.

    Args:
        checkpoint_num: Target checkpoint number (inclusive)
        version: Version tag (e.g., 'v7')
    """
    console.print(
        f"\n[bold cyan]Rolling back to checkpoint {checkpoint_num}[/bold cyan]\n"
    )

    # Get paths
    base_dir, seed_path, diffs_dir, regenerated_dir = get_paths(version)
    profile_dir = base_dir / "output" / "character-profile" / "uno" / version
    checkpoints_dir = profile_dir / "checkpoints"
    log_path = profile_dir / "processing_log.jsonl"

    # Validate that the checkpoint exists
    target_diff = diffs_dir / f"checkpoint_{checkpoint_num:03d}.diff"
    if not target_diff.exists():
        console.print(
            f"[bold red]Error:[/bold red] Target checkpoint does not exist: {target_diff}"
        )
        sys.exit(1)

    # Create backup first
    backup_dir = create_backup(profile_dir, version)

    # Perform rollback operations
    console.print("[bold cyan]Performing rollback operations...[/bold cyan]\n")

    # 1. Truncate processing log
    removed_logs = rollback_processing_log(log_path, checkpoint_num)
    if removed_logs > 0:
        console.print(
            f"[bold green]✓[/bold green] Removed {removed_logs} log entries after checkpoint {checkpoint_num}"
        )
    else:
        console.print("[bold yellow]•[/bold yellow] No log entries to remove")

    # 2. Delete diff files
    deleted_diffs = delete_diffs_after(diffs_dir, checkpoint_num)
    if deleted_diffs > 0:
        console.print(
            f"[bold green]✓[/bold green] Deleted {deleted_diffs} diff files after checkpoint {checkpoint_num}"
        )
    else:
        console.print("[bold yellow]•[/bold yellow] No diff files to delete")

    # 3. Delete checkpoint markdown files (keep only the target checkpoint)
    deleted_checkpoints = delete_checkpoints_after(checkpoints_dir, checkpoint_num)
    if deleted_checkpoints > 0:
        console.print(
            f"[bold green]✓[/bold green] Deleted {deleted_checkpoints} checkpoint files (kept only checkpoint_{checkpoint_num:03d}.md)"
        )
    else:
        console.print("[bold yellow]•[/bold yellow] No checkpoint files to delete")

    # 4. Delete regenerated directory
    regenerated_deleted = delete_regenerated_dir(regenerated_dir)
    if regenerated_deleted:
        console.print("[bold green]✓[/bold green] Deleted regenerated directory")
    else:
        console.print(
            "[bold yellow]•[/bold yellow] Regenerated directory did not exist"
        )

    # 5. Regenerate the target checkpoint and save to checkpoints directory
    console.print(
        f"\n[bold cyan]Regenerating checkpoint {checkpoint_num}...[/bold cyan]\n"
    )
    checkpoint_output = checkpoints_dir / f"checkpoint_{checkpoint_num:03d}.md"

    # Get diff files for regeneration
    diff_files = []
    for i in range(1, checkpoint_num + 1):
        diff_path = diffs_dir / f"checkpoint_{i:03d}.diff"
        if diff_path.exists():
            diff_files.append(diff_path)

    # Regenerate without the verbose output
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as temp_file:
        seed_content = seed_path.read_text(encoding="utf-8")
        temp_file.write(seed_content)
        temp_path = Path(temp_file.name)

    try:
        # Apply each diff in sequence
        for diff_file in diff_files:
            if not apply_patch(temp_path, diff_file):
                console.print(
                    f"[bold red]Error:[/bold red] Failed to apply {diff_file.name}"
                )
                sys.exit(1)

        # Read the final result and save to checkpoints directory
        final_content = temp_path.read_text(encoding="utf-8")
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_output.write_text(final_content, encoding="utf-8")

        console.print(
            f"[bold green]✓[/bold green] Regenerated checkpoint saved to {checkpoint_output.name}"
        )
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

    # Print summary
    console.print("\n[bold green]✓ Rollback complete![/bold green]\n")
    console.print(f"State preserved up to and including checkpoint {checkpoint_num}")
    console.print(f"Checkpoint file: {checkpoint_output}")
    console.print(f"Backup location: {backup_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate a checkpoint by applying diffs sequentially from the seed document, or rollback to a specific checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=int,
        help="Checkpoint number to regenerate or rollback to (e.g., 15)",
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
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to the specified checkpoint (deletes all data after it, creates backup first)",
    )

    args = parser.parse_args()

    # Validate checkpoint number
    validate_checkpoint_number(args.checkpoint)

    # Handle rollback mode
    if args.rollback:
        perform_rollback(args.checkpoint, args.version)
        return

    # Normal regeneration mode
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
