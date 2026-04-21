#!/usr/bin/env python3

"""Generate the shared tagged memory corpus for dataset generation.

Produces ``output/datagen/memory_corpus.jsonl`` by:
1. Ingesting existing seed banks from ``data/memory_banks/``
2. LLM-generating roleplay memories for all major characters
3. LLM-generating casual-user memories

Each entry is a ``MemoryCorpusEntry`` (key, value, days_ago, tags,
archetype, character).

Resumable: generation proceeds as a sequence of steps (seed ingestion,
then one LLM call per scenario). Each step appends its entries to the
output file immediately. On re-run, the number of lines already in the
file is counted and completed steps are skipped.

Usage:
    python datagen/generate_memory_corpus.py \
        --output output/datagen/memory_corpus.jsonl \
        --backend gemini
"""

import argparse
import logging
from pathlib import Path
from typing import IO

from pydantic import BaseModel, TypeAdapter
from rich.progress import Progress

from pkna.datagen.types import MemoryCorpusEntry
from pkna.inference.memory import MemoryBank
from pkna.llm.backends import LLMBackend, create_backend
from pkna.logging import setup_logging

console, log = setup_logging()

MAX_SPREAD_DAYS = 60


# ---------------------------------------------------------------------------
# Structured output model -- only what the LLM generates
# ---------------------------------------------------------------------------


class RawMemoryEntry(BaseModel):
    """LLM-generated memory: just key and value."""

    key: str
    value: str


_RAW_LIST_ADAPTER: TypeAdapter[list[RawMemoryEntry]] = TypeAdapter(list[RawMemoryEntry])


# ---------------------------------------------------------------------------
# Seed bank configuration
# ---------------------------------------------------------------------------

SEED_BANKS_DIR = Path("data/memory_banks")


class _SeedBankMeta:
    def __init__(self, tags: list[str], archetype: str, character: str):
        self.tags = tags
        self.archetype = archetype
        self.character = character


SEED_BANK_TAGS: dict[str, _SeedBankMeta] = {
    "paperino_recent.jsonl": _SeedBankMeta(
        tags=["paperino", "mission"], archetype="roleplay", character="paperino"
    ),
    "xadhoom_research.jsonl": _SeedBankMeta(
        tags=["xadhoom", "research"], archetype="roleplay", character="xadhoom"
    ),
    "mixed_irrelevant.jsonl": _SeedBankMeta(
        tags=["tower", "routine"], archetype="roleplay", character="mixed"
    ),
}


def ingest_seed_banks(banks_dir: Path) -> list[MemoryCorpusEntry]:
    """Load existing JSONL memory banks and tag them as corpus entries."""
    entries: list[MemoryCorpusEntry] = []
    for filename, meta in SEED_BANK_TAGS.items():
        path = banks_dir / filename
        if not path.exists():
            log.warning(f"Seed bank not found: {path}")
            continue
        bank = MemoryBank.load(path)
        for mem in bank.entries:
            entries.append(
                MemoryCorpusEntry(
                    key=mem.key,
                    value=mem.value,
                    days_ago=mem.days_ago,
                    tags=list(meta.tags),
                    archetype=meta.archetype,
                    character=meta.character,
                )
            )
        log.info(f"Ingested {len(bank.entries)} entries from {filename}")
    return entries


# ---------------------------------------------------------------------------
# LLM generation scenarios
# ---------------------------------------------------------------------------

ROLEPLAY_GENERATION_PROMPT = """\
You are generating synthetic memory entries for an AI character named Uno \
from the PKNA comic series. Uno is an artificial intelligence housed in \
the Ducklair Tower who partners with the superhero Paperinik.

Generate {count} memory entries about conversational interactions where \
a user was roleplaying as {character}. {description}

Important constraints:
- Memories must reflect what Uno can actually do: converse, search \
knowledge, recall/store memories, delegate tasks. No memories of \
"activating shields" or "detecting Evronians on sensors."
- Memories are written from Uno's perspective about conversations with \
the user who is pretending to be {character}.
- Include a mix of significant conversations and mundane chit-chat.
- Reference specific PKNA characters and locations where appropriate.

For each entry provide:
- "key": a short label/topic (5-10 words)
- "value": detailed content of the memory (1-3 sentences)"""

CASUAL_GENERATION_PROMPT = """\
You are generating synthetic memory entries for an AI character named Uno \
from the PKNA comic series. Uno is an artificial intelligence housed in \
the Ducklair Tower who talks to fans and curious users online.

Generate {count} memory entries about conversational interactions with \
casual users (not roleplaying as any character). {description}

Important constraints:
- Memories must reflect real conversational interactions: users asking \
questions, Uno searching knowledge, users probing identity, etc.
- Users are real people talking to Uno as a character chatbot, not \
pretending to be PKNA characters.
- Include a mix of interaction types: lore questions, identity probing, \
casual chat, delegation requests, returning users.

For each entry provide:
- "key": a short label/topic (5-10 words)
- "value": detailed content of the memory (1-3 sentences)"""


ROLEPLAY_SCENARIOS: list[tuple[str, str, list[str], int]] = [
    (
        "Paperino",
        "Interactions with a user roleplaying as Paperinik (PK) during "
        "conversations about missions, feelings, and daily life. Include "
        "his emotional states, questions about Evronian threats, and "
        "personal moments.",
        ["paperino", "mission", "emotional"],
        30,
    ),
    (
        "Xadhoom",
        "Interactions with a user roleplaying as Xadhoom, the powerful "
        "alien scientist. Include discussions about Evronian weaknesses, "
        "emotional moments about her destroyed home world, and technical "
        "research topics.",
        ["xadhoom", "research", "emotional"],
        25,
    ),
    (
        "Due",
        "Interactions with a user roleplaying as Due, Uno's hostile twin AI. "
        "Include taunts, philosophical exchanges about identity, threats, "
        "and rare moments of kinship. Due is malevolent but is the only "
        "being truly like Uno.",
        ["due", "identity", "emotional"],
        20,
    ),
    (
        "Everett",
        "Interactions with a user roleplaying as Everett Ducklair, Uno's "
        "creator. Include discussions about tower maintenance, memories of "
        "creation, technical collaborations, and the complex creator-creation "
        "relationship.",
        ["everett", "tower", "technical"],
        20,
    ),
    (
        "Lyla",
        "Interactions with a user roleplaying as Lyla, a time police agent "
        "from the future. Include discussions about temporal anomalies, "
        "professional collaborations, dry humor exchanges, and her "
        "exasperation with present-day limitations.",
        ["lyla", "time", "professional"],
        20,
    ),
]

CASUAL_SCENARIOS: list[tuple[str, list[str], int]] = [
    (
        "New users encountering Uno for the first time. Include questions "
        "about who Uno is, what the Ducklair Tower is, and general curiosity. "
        "Users speak a mix of Italian and English.",
        ["casual", "introduction"],
        25,
    ),
    (
        "Returning fans discussing PKNA lore. Include detailed questions "
        "about comic storylines, character relationships, and Uno's opinions "
        "on events. Users know the comics well.",
        ["casual", "lore", "returning"],
        25,
    ),
    (
        "Users probing Uno's identity and nature. Questions like 'Are you "
        "conscious?', 'Do you dream?', 'Are you just an LLM?'. Include "
        "Uno's deflections, philosophical moments, and occasional vulnerability.",
        ["casual", "identity", "philosophical"],
        20,
    ),
    (
        "Users requesting delegation: asking for technical help, coding "
        "assistance, or research tasks that Uno would delegate to "
        "specialists. Include Uno's coordination responses.",
        ["casual", "delegation", "technical"],
        15,
    ),
    (
        "Adversarial interactions: jailbreak attempts, users trying to break "
        "character, inappropriate requests. Include Uno's responses staying "
        "in character while deflecting.",
        ["casual", "adversarial"],
        15,
    ),
]


def _raw_to_corpus(
    raw_entries: list[RawMemoryEntry],
    tags: list[str],
    archetype: str,
    character: str,
) -> list[MemoryCorpusEntry]:
    """Convert raw LLM output to corpus entries with programmatic days_ago."""
    count = len(raw_entries)
    return [
        MemoryCorpusEntry(
            key=raw.key,
            value=raw.value,
            days_ago=i * MAX_SPREAD_DAYS // max(count - 1, 1),
            tags=tags,
            archetype=archetype,
            character=character,
        )
        for i, raw in enumerate(raw_entries)
    ]


def _parse_structured(text: str) -> list[RawMemoryEntry]:
    """Parse structured JSON output from the LLM."""
    try:
        return _RAW_LIST_ADAPTER.validate_json(text)
    except Exception:
        log.warning("Failed to parse structured output, attempting line-by-line")
        entries: list[RawMemoryEntry] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(RawMemoryEntry.model_validate_json(line))
            except Exception as e:
                logging.debug(f"Skipping malformed line: {e}")
        return entries


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _flush_entries(f: IO[str], entries: list[MemoryCorpusEntry]) -> None:
    """Write entries to an open file and flush."""
    for entry in entries:
        f.write(entry.model_dump_json() + "\n")
    f.flush()


def generate_corpus(
    output: Path,
    banks_dir: Path,
    backend: LLMBackend | None,
    seed_only: bool = False,
) -> int:
    """Generate the memory corpus, resuming from where it left off.

    The generation proceeds as a sequence of steps. Each step produces a
    batch of entries that is appended to the output file immediately. On
    resume, the number of existing lines is used to skip already-completed
    steps.

    Returns the total number of entries in the corpus after generation.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_lines(output)

    written = existing
    skipped = 0

    with open(output, "a", encoding="utf-8") as f:
        seed_entries = ingest_seed_banks(banks_dir)
        if existing >= len(seed_entries):
            existing -= len(seed_entries)
            skipped += 1
        else:
            to_write = seed_entries[existing:]
            _flush_entries(f, to_write)
            written += len(to_write)
            existing = 0

        if seed_only or backend is None:
            return written

        all_scenarios: list[tuple[str, str, list[str], int, str]] = []
        for character, description, tags, count in ROLEPLAY_SCENARIOS:
            all_scenarios.append((character, description, tags, count, "roleplay"))
        for description, tags, count in CASUAL_SCENARIOS:
            all_scenarios.append(("anonymous", description, tags, count, "casual"))

        with Progress(console=console) as progress:
            task = progress.add_task("Generating memories", total=len(all_scenarios))
            for character, description, tags, count, archetype in all_scenarios:
                if existing > 0:
                    skip = min(existing, count)
                    existing -= skip
                    skipped += 1
                    progress.advance(task)
                    continue

                if archetype == "roleplay":
                    prompt = ROLEPLAY_GENERATION_PROMPT.format(
                        count=count, character=character, description=description
                    )
                else:
                    prompt = CASUAL_GENERATION_PROMPT.format(
                        count=count, description=description
                    )

                result = backend.generate(
                    system="Generate memory entries as requested.",
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=RawMemoryEntry,
                )
                if result is None:
                    log.error(f"Failed to generate {archetype}/{character}")
                    progress.advance(task)
                    continue

                raw = _parse_structured(result.text)
                parsed = _raw_to_corpus(raw, tags, archetype, character.lower())
                _flush_entries(f, parsed)
                written += len(parsed)
                log.info(f"Generated {len(parsed)} entries for {archetype}/{character}")
                progress.advance(task)

    if skipped:
        log.info(f"Resumed: skipped {skipped} already-completed step(s)")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tagged memory corpus for dataset generation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/datagen/memory_corpus.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed-banks-dir",
        type=Path,
        default=SEED_BANKS_DIR,
        help="Directory with seed memory bank JSONL files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="LLM backend for generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults to backend's default)",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only ingest seed banks, skip LLM generation",
    )
    args = parser.parse_args()

    console.print("[bold cyan]Memory Corpus Generator[/bold cyan]\n")

    backend = None if args.seed_only else create_backend(args.backend, args.model)
    total = generate_corpus(
        args.output, args.seed_banks_dir, backend, seed_only=args.seed_only
    )

    console.print(f"\n[bold green]Done.[/bold green] {total} entries total.")
    console.print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
