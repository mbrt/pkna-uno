#!/usr/bin/env python3

"""Migrate v2 traces from bracket tags to XML tags.

Rewrites user messages from:
    [Context]
    ...preamble...

    [Message]
    ...user text...

To:
    <context>
    ...preamble...
    </context>

    <message>
    ...user text...
    </message>

Also drops claim-derived traces generated with empty user messages
(these need to be regenerated with proper LLM-produced messages).

Usage:
    python datagen/migrate_traces_v3.py --input output/datagen/traces.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from pkna.logging import setup_logging

console, log = setup_logging()

_CONTEXT_MESSAGE_RE = re.compile(
    r"^\[Context\]\n(?P<ctx>.*?)\n\n\[Message\]\n(?P<msg>.*)$",
    re.DOTALL,
)

_MESSAGE_ONLY_RE = re.compile(
    r"^\[Message\]\n(?P<msg>.*)$",
    re.DOTALL,
)


def migrate_content(content: str) -> str:
    """Convert bracket-delimited user message to XML tags."""
    m = _CONTEXT_MESSAGE_RE.match(content)
    if m:
        ctx = m.group("ctx")
        msg = m.group("msg")
        return f"<context>\n{ctx}\n</context>\n\n<message>\n{msg}\n</message>"

    m = _MESSAGE_ONLY_RE.match(content)
    if m:
        msg = m.group("msg")
        return f"<message>\n{msg}\n</message>"

    return content


def _has_empty_user_message(trace: dict) -> bool:
    """Check if the actual user message (after [Context]/[Message] wrapper) is empty."""
    for msg in trace.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg["content"]
        m = _CONTEXT_MESSAGE_RE.match(content)
        if m:
            return not m.group("msg").strip()
        m = _MESSAGE_ONLY_RE.match(content)
        if m:
            return not m.group("msg").strip()
        return not content.strip()
    return True


def migrate_traces(input_path: Path) -> tuple[int, int]:
    """Migrate bracket-tagged traces to XML-tagged format.

    Renames input_path to <stem>_v2.jsonl as backup, writes migrated
    traces to the original path. Drops claim-derived traces with empty
    user messages (they were generated before LLM message filling was
    implemented).

    Returns (migrated_count, dropped_count).
    """
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        return 0, 0

    backup_path = input_path.with_name(input_path.stem + "_v2" + input_path.suffix)
    if backup_path.exists():
        log.error(f"Backup already exists: {backup_path}")
        raise SystemExit(1)

    traces: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    if not traces:
        log.info("No traces to migrate.")
        return 0, 0

    input_path.rename(backup_path)
    log.info(f"Backed up {input_path} -> {backup_path}")

    migrated = 0
    dropped = 0
    with open(input_path, "w", encoding="utf-8") as f:
        for trace in traces:
            is_claim = trace.get("metadata", {}).get("prompt_source") == "claim_derived"
            if is_claim and _has_empty_user_message(trace):
                dropped += 1
                continue

            messages = trace.get("messages", [])
            new_messages = []
            changed = False
            for msg in messages:
                if msg.get("role") == "user":
                    old = msg["content"]
                    new = migrate_content(old)
                    if new != old:
                        changed = True
                    new_messages.append({**msg, "content": new})
                else:
                    new_messages.append(msg)
            trace["messages"] = new_messages
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if changed:
                migrated += 1

    log.info(f"Migrated {migrated} / {len(traces)} traces, dropped {dropped}")
    return migrated, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate v2 traces: bracket tags -> XML tags"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/datagen/traces.jsonl"),
        help="Input JSONL file with v2 DatagenTrace entries",
    )
    args = parser.parse_args()

    console.print(
        "[bold cyan]Trace Migration (v2 -> v3): bracket -> XML tags[/bold cyan]\n"
    )

    migrated, dropped = migrate_traces(args.input)
    console.print(
        f"\n[bold green]Done.[/bold green] {migrated} traces migrated, "
        f"{dropped} empty claim traces dropped."
    )


if __name__ == "__main__":
    main()
