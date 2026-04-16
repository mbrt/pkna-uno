#!/usr/bin/env python3

"""Generate raw memory banks via LLM.

Produces JSONL memory bank files for use in eval prompts and dataset
generation. Each bank is a set of raw memory entries (key, value,
timestamp) representing Uno's recollections about a character or topic.

Usage:
    python evals/generate_memory_banks.py \
        --output-dir data/memory_banks/ \
        --backend gemini
"""

import argparse
import json
from pathlib import Path

from pkna.inference.memory import MemoryBank
from pkna.llm.backends import create_backend
from pkna.logging import setup_logging

console, log = setup_logging()


GENERATION_PROMPT = """\
You are generating synthetic memory entries for an AI character named Uno \
from the PKNA comic series. Uno is an artificial intelligence housed in \
the Ducklair Tower who partners with the superhero Paperinik.

Generate {count} memory entries about: {description}

Each entry should be a JSON object on its own line with these fields:
- "key": a short label/topic (5-10 words)
- "value": detailed content of the memory (1-3 sentences)
- "timestamp": ISO 8601 timestamp, spread over the last 2 months, \
most recent first

The memories should:
- Be written from Uno's perspective (third person about others)
- Include a mix of significant events and mundane observations
- Feel natural and varied in tone
- Reference specific PKNA characters and locations where appropriate

Output ONLY the JSON lines, no other text.

Character/topic: {description}
"""


SCENARIOS = [
    (
        "paperino_missions",
        "Recent interactions with Paperinik (PK) during missions, "
        "training sessions, and casual conversations. Include his emotional "
        "states, mission outcomes, and personal questions he's asked Uno.",
        10,
    ),
    (
        "xadhoom_encounters",
        "Interactions with Xadhoom, the powerful alien scientist. Include "
        "her research collaborations with Uno, emotional moments about her "
        "destroyed home world, and technical discussions about Evronian "
        "weaknesses.",
        10,
    ),
    (
        "tower_operations",
        "Routine Ducklair Tower operations, security events, maintenance "
        "tasks, and interactions with various visitors (Lyla, Everett, "
        "General Wisecube). Mix of mundane and notable events.",
        10,
    ),
    (
        "evronian_intelligence",
        "Intelligence gathered about Evronian activities: patrol patterns, "
        "new threats detected, analysis of captured technology, and "
        "strategic assessments shared with allies.",
        10,
    ),
]


def generate_bank(
    name: str,
    description: str,
    count: int,
    backend_name: str,
    model: str | None,
    output_dir: Path,
) -> Path:
    """Generate a single memory bank file."""
    output_path = output_dir / f"{name}.jsonl"
    if output_path.exists():
        log.info(f"Skipping {name} (already exists)")
        return output_path

    backend = create_backend(backend_name, model)
    prompt = GENERATION_PROMPT.format(count=count, description=description)

    log.info(f"Generating {name} ({count} entries)...")
    result = backend.generate(
        system="You output only valid JSONL. Each line is a JSON object.",
        messages=[{"role": "user", "content": prompt}],
    )
    if result is None:
        log.error(f"Failed to generate {name}")
        return output_path

    # Parse and validate entries, then save via MemoryBank
    bank = MemoryBank()
    for line in result.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            bank.append(data["key"], data["value"])
            # Override the auto-generated timestamp with the LLM's
            if "timestamp" in data:
                bank.entries[-1].timestamp = data["timestamp"]
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping malformed line in {name}: {e}")
            continue

    bank.save(output_path)
    log.info(f"Wrote {len(bank.entries)} entries to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw memory banks via LLM")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/memory_banks"),
        help="Directory to write memory bank JSONL files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults to backend's default)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names to generate (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario_filter = set(args.scenarios.split(",")) if args.scenarios else None

    console.print("[bold cyan]Memory Bank Generator[/bold cyan]\n")

    for name, description, count in SCENARIOS:
        if scenario_filter and name not in scenario_filter:
            continue
        generate_bank(
            name, description, count, args.backend, args.model, args.output_dir
        )

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
