#!/usr/bin/env python3

"""Remove claim-derived entries with USER_STRANGER from cache and trace files.

After fixing the user_summary assignment in generate_claim_prompts.py, this
script removes only the affected entries so that the resume logic regenerates
them with the correct interlocutors.

Affected files:
  - output/datagen/claim_prompts_cache.jsonl  (claim prompt cache)
  - output/datagen/traces.jsonl               (generated traces)

Usage:
    python datagen/cleanup_stranger_traces.py [--dry-run]
"""

import argparse
import json
import shutil
from pathlib import Path

USER_STRANGER = (
    "Unknown stranger. First interaction. You have no information about this "
    "person. Use formal register, make no assumptions."
)

CLAIM_PREFIXES = (
    "claim-value-priority-",
    "claim-emotional-",
    "claim-register-",
    "claim-identity-",
)


def is_affected(entry: dict) -> bool:
    entry_id: str = entry.get("id", "")
    if not any(entry_id.startswith(p) for p in CLAIM_PREFIXES):
        return False
    return entry.get("user_summary") == USER_STRANGER


def filter_jsonl(path: Path, dry_run: bool) -> tuple[int, int]:
    """Remove affected lines from a JSONL file. Returns (kept, removed)."""
    if not path.exists():
        print(f"  Skipping {path} (not found)")
        return 0, 0

    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue

        if is_affected(entry):
            removed += 1
        else:
            kept.append(line)

    if dry_run:
        print(f"  {path}: would remove {removed}, keep {len(kept)}")
    else:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"  {path}: removed {removed}, kept {len(kept)} (backup: {backup.name})")

    return len(kept), removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove claim-derived USER_STRANGER entries from datagen outputs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without modifying files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/datagen"),
        help="Directory containing datagen output files",
    )
    args = parser.parse_args()

    print(f"{'DRY RUN - ' if args.dry_run else ''}Cleaning up USER_STRANGER entries:\n")

    total_removed = 0

    for filename in ("claim_prompts_cache.jsonl", "traces.jsonl"):
        path = args.output_dir / filename
        _kept, removed = filter_jsonl(path, args.dry_run)
        total_removed += removed

    print(
        f"\nTotal entries to {'remove' if args.dry_run else 'removed'}: {total_removed}"
    )

    if not args.dry_run and total_removed > 0:
        print(
            "\nNext steps:\n"
            "  1. Re-run generate_prompts.py (with --ledger) to rebuild prompts.jsonl\n"
            "  2. Re-run run_datagen.py — resume logic will regenerate only missing traces"
        )


if __name__ == "__main__":
    main()
