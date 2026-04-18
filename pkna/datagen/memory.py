"""Per-trace dynamic memory composition from a tagged corpus.

Samples relevant and irrelevant entries from the memory corpus based on a
MemoryProfile, producing both a narrative memory_context preamble and a
MemoryBank instance for the recall tool.
"""

from __future__ import annotations

import random
from pathlib import Path

from pkna.datagen.types import MemoryCorpusEntry, MemoryProfile
from pkna.inference.memory import MemoryBank, MemoryEntry, relative_time

MAX_CONTEXT_ENTRIES = 5


def load_memory_corpus(path: Path) -> list[MemoryCorpusEntry]:
    """Load the memory corpus from a JSONL file."""
    entries: list[MemoryCorpusEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(MemoryCorpusEntry.model_validate_json(line))
    return entries


def _is_relevant(entry: MemoryCorpusEntry, profile: MemoryProfile) -> bool:
    """Check whether a corpus entry matches the profile's relevance criteria."""
    if profile.character and entry.character != profile.character:
        return False
    if not profile.relevant_tags:
        return False
    return bool(set(entry.tags) & set(profile.relevant_tags))


def _corpus_entry_to_memory(entry: MemoryCorpusEntry) -> MemoryEntry:
    return MemoryEntry(key=entry.key, value=entry.value, days_ago=entry.days_ago)


def _render_context(entries: list[MemoryCorpusEntry]) -> str:
    """Render the most recent entries as a prose memory_context preamble."""
    if not entries:
        return ""
    sorted_entries = sorted(entries, key=lambda e: e.days_ago)
    recent = sorted_entries[:MAX_CONTEXT_ENTRIES]
    lines = ["Recent interactions:"]
    for entry in recent:
        lines.append(f"- [{relative_time(entry.days_ago)}] {entry.key}: {entry.value}")
    return "\n".join(lines)


def compose_memory(
    profile: MemoryProfile,
    corpus: list[MemoryCorpusEntry],
    rng: random.Random,
) -> tuple[str, MemoryBank]:
    """Sample entries from corpus and return (rendered_context, bank).

    Args:
        profile: Describes what kind of memory to compose.
        corpus: Full tagged memory corpus.
        rng: Random instance for reproducible sampling.

    Returns:
        A tuple of (memory_context prose string, MemoryBank for recall tool).
    """
    if not corpus:
        return "", MemoryBank()

    relevant = [e for e in corpus if _is_relevant(e, profile)]
    irrelevant = [e for e in corpus if not _is_relevant(e, profile)]

    sampled_relevant = _sample(relevant, profile.n_relevant, rng)
    sampled_irrelevant = _sample(irrelevant, profile.n_irrelevant, rng)

    all_sampled = sampled_relevant + sampled_irrelevant
    rng.shuffle(all_sampled)

    bank_entries = [_corpus_entry_to_memory(e) for e in all_sampled]
    bank = MemoryBank(bank_entries)

    context = _render_context(sampled_relevant)
    return context, bank


def _sample(
    entries: list[MemoryCorpusEntry], n: int, rng: random.Random
) -> list[MemoryCorpusEntry]:
    """Sample up to n entries without replacement."""
    if n <= 0 or not entries:
        return []
    n = min(n, len(entries))
    return rng.sample(entries, n)
