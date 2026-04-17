#!/usr/bin/env python3

"""Migrate v1 traces to the new format (no system_prompt per trace).

Reads old-format traces (with embedded system_prompt), rewrites them:
1. Prepends user_summary/memory_context to the first user message.
2. Drops the system_prompt field.
3. Writes sidecar files (template + profile) for downstream consumers.
4. Renames the old file to traces_v1.jsonl as backup.

Usage:
    python datagen/migrate_traces.py \
        --input output/datagen/traces.jsonl \
        --profile results/uno_soul_document.md
"""

import argparse
import json
from pathlib import Path

from datagen.run_datagen import write_sidecar_files
from pkna.inference.system_prompts import prepend_context_to_messages
from pkna.logging import setup_logging

console, log = setup_logging()


def migrate_traces(
    input_path: Path,
    profile_path: Path,
) -> int:
    """Migrate old-format traces to new format.

    Renames input_path to <stem>_v1.jsonl, writes migrated traces to
    the original path, and creates sidecar files next to it.

    Returns the number of traces migrated.
    """
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        return 0

    profile = profile_path.read_text(encoding="utf-8")
    output_dir = input_path.parent

    backup_path = input_path.with_name(input_path.stem + "_v1" + input_path.suffix)
    if backup_path.exists():
        log.error(f"Backup already exists: {backup_path}")
        raise SystemExit(1)

    old_traces: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                old_traces.append(json.loads(line))

    if not old_traces:
        log.info("No traces to migrate.")
        return 0

    input_path.rename(backup_path)
    log.info(f"Backed up {input_path} -> {backup_path}")

    write_sidecar_files(output_dir, profile)
    log.info(f"Wrote sidecar files to {output_dir}")

    migrated = 0
    with open(input_path, "w", encoding="utf-8") as f:
        for old in old_traces:
            user_summary = old.get("user_summary", "")
            memory_context = old.get("memory_context", "")
            messages = old.get("messages", [])

            new_messages = prepend_context_to_messages(
                messages, user_summary, memory_context
            )

            new_trace = {
                "id": old["id"],
                "metadata": old.get("metadata", {}),
                "memory_context": memory_context,
                "user_summary": user_summary,
                "messages": new_messages,
            }
            f.write(json.dumps(new_trace, ensure_ascii=False) + "\n")
            migrated += 1

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate v1 traces to new format (externalized system prompt)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/datagen/traces.jsonl"),
        help="Input JSONL file with old-format DatagenTrace entries",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("results/uno_soul_document.md"),
        help="Character profile markdown file",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Trace Migration (v1 -> v2)[/bold cyan]\n")

    if not args.profile.exists():
        console.print(f"[bold red]Error:[/bold red] Profile not found: {args.profile}")
        raise SystemExit(1)

    migrated = migrate_traces(args.input, args.profile)
    console.print(f"\n[bold green]Done.[/bold green] {migrated} traces migrated.")


if __name__ == "__main__":
    main()
