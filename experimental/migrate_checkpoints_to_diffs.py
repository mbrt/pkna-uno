#!/usr/bin/env python3
"""
Migrate existing checkpoints to diff-based storage.

This script:
1. Generates diffs between consecutive checkpoints
2. Keeps only the last 3 full checkpoints
3. Deletes older full checkpoints (diffs are preserved)
"""

import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
CHECKPOINTS_DIR = BASE_DIR / "output/character-profile/uno/v5/checkpoints"
DIFFS_DIR = BASE_DIR / "output/character-profile/uno/v5/diffs"
SEED_PATH = BASE_DIR / "output/character-profile/uno/v5/seed_document.md"


def generate_diff(
    old_path: Path, new_path: Path, diff_path: Path, version_num: int
) -> None:
    """Generate unified diff between two files.

    Args:
        old_path: Path to old version
        new_path: Path to new version
        diff_path: Path where diff should be saved
        version_num: Current version number
    """
    result = subprocess.run(
        [
            "diff",
            "-u",
            "--label",
            f"a/checkpoint_{version_num - 1:03d}.md",
            "--label",
            f"b/checkpoint_{version_num:03d}.md",
            str(old_path),
            str(new_path),
        ],
        capture_output=True,
        text=True,
    )

    # Save diff
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    print(f"Generated: {diff_path.name}")


def main():
    """Generate diffs and clean up old checkpoints."""
    # Get all checkpoint files
    checkpoints = sorted(CHECKPOINTS_DIR.glob("checkpoint_*.md"))

    if not checkpoints:
        print("No checkpoints found!")
        return

    total = len(checkpoints)
    print(f"Found {total} checkpoints")
    print("Will keep last 3 full checkpoints, generate diffs for all\n")

    # Create diffs directory
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate diff for first checkpoint (from seed)
    print("Generating diffs...")
    if SEED_PATH.exists():
        generate_diff(SEED_PATH, checkpoints[0], DIFFS_DIR / "checkpoint_001.diff", 1)
    else:
        print(f"Warning: Seed document not found at {SEED_PATH}")
        print("Skipping diff for checkpoint_001\n")

    # Generate diffs between consecutive checkpoints
    for i in range(1, len(checkpoints)):
        version_num = i + 1
        generate_diff(
            checkpoints[i - 1],
            checkpoints[i],
            DIFFS_DIR / f"checkpoint_{version_num:03d}.diff",
            version_num,
        )

    print(f"\nGenerated {total} diffs in {DIFFS_DIR}")

    # Delete old checkpoints (keep last 3)
    if total > 3:
        to_delete = checkpoints[:-3]
        print(f"\nDeleting {len(to_delete)} old full checkpoints (keeping last 3)...")

        for checkpoint in to_delete:
            checkpoint.unlink()
            print(f"Deleted: {checkpoint.name}")

        print(f"\nKept: {', '.join(cp.name for cp in checkpoints[-3:])}")
    else:
        print(f"\nOnly {total} checkpoints - keeping all")

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
