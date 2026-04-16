"""Raw memory bank storage and retrieval.

Provides a simple key-value-timestamp memory store backed by JSONL files.
Used by the eval harness and dataset generation agent to support the
``recall`` and ``remember`` tools.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class MemoryEntry:
    """A single raw memory entry."""

    key: str
    value: str
    timestamp: str


class MemoryBank:
    """In-memory store of raw memory entries with keyword search."""

    def __init__(self, entries: list[MemoryEntry] | None = None):
        self._entries: list[MemoryEntry] = list(entries) if entries else []

    @staticmethod
    def load(path: Path) -> "MemoryBank":
        """Load a memory bank from a JSONL file."""
        entries: list[MemoryEntry] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(
                    MemoryEntry(
                        key=data["key"],
                        value=data["value"],
                        timestamp=data["timestamp"],
                    )
                )
        return MemoryBank(entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def search(self, query: str, max_results: int = 5) -> list[MemoryEntry]:
        """Keyword search over keys and values, returning top matches."""
        query_lower = query.lower()
        keywords = query_lower.split()

        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self._entries:
            key_lower = entry.key.lower()
            value_lower = entry.value.lower()
            score = 0
            for kw in keywords:
                score += key_lower.count(kw) * 3
                score += value_lower.count(kw)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def append(self, key: str, value: str) -> MemoryEntry:
        """Append a new entry with the current timestamp."""
        entry = MemoryEntry(
            key=key,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._entries.append(entry)
        return entry

    def save(self, path: Path) -> None:
        """Write all entries to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                data = {
                    "key": entry.key,
                    "value": entry.value,
                    "timestamp": entry.timestamp,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _format_results(entries: list[MemoryEntry]) -> str:
    """Format memory entries for tool output."""
    if not entries:
        return "No matching memories found."
    lines: list[str] = []
    for entry in entries:
        lines.append(f"[{entry.timestamp}] {entry.key}: {entry.value}")
    return "\n".join(lines)


def make_recall(bank: MemoryBank) -> Callable[..., str]:
    """Create a recall tool function bound to a specific memory bank.

    Args:
        bank: The memory bank to search.

    Returns:
        A callable with the signature ``recall(query, max_results) -> str``.
    """

    def recall(query: str, max_results: int = 5) -> str:
        """Search through stored memories for entries matching the query.

        Args:
            query: Keywords to search for in memory entries
            max_results: Maximum number of results to return (default: 5)

        Returns:
            Formatted list of matching memory entries with timestamps
        """
        if not query or not query.strip():
            return "Error: Please provide a search query"
        results = bank.search(query.strip(), max_results)
        return _format_results(results)

    return recall


def make_remember(bank: MemoryBank, *, eval_mode: bool = True) -> Callable[..., str]:
    """Create a remember tool function bound to a specific memory bank.

    In eval mode the tool acknowledges the write without side effects.
    In trace-generation mode it appends to the bank.

    Args:
        bank: The memory bank to write to.
        eval_mode: If True, writes are no-ops.

    Returns:
        A callable with the signature ``remember(key, value) -> str``.
    """

    def remember(key: str, value: str) -> str:
        """Store a new memory entry for future recall.

        Args:
            key: Short label or topic for this memory
            value: Detailed content to remember

        Returns:
            Confirmation message
        """
        if not key or not key.strip():
            return "Error: Please provide a key"
        if not value or not value.strip():
            return "Error: Please provide a value"
        if not eval_mode:
            bank.append(key.strip(), value.strip())
        return "Memory stored."

    return remember
