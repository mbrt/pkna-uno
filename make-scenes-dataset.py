#!/usr/bin/env python3

"""
Create scenes dataset from panel extractions.

Groups panels into scenes based on is_new_scene flag and filters
for scenes containing the character "Uno". Outputs to CSV format.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_FILE = BASE_DIR / "output" / "dataset" / "dataset-2.csv"


# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)


def extract_page_reference(input_page_path: str) -> str:
    """Extract the relative page reference from the full path.

    Example: /home/user/input/pkna/pkna-1/pkna1-06.jpg
             -> pkna-1/pkna1-06.jpg
    """
    parts = Path(input_page_path).parts
    # Find 'pkna' directory and take the last two parts
    try:
        pkna_idx = parts.index("pkna")
        return str(Path(parts[pkna_idx + 1]) / parts[pkna_idx + 2])
    except (ValueError, IndexError):
        # Fallback: just use the last two parts
        return str(Path(parts[-2]) / parts[-1])


def scene_contains_uno(panels: list[dict[str, Any]]) -> bool:
    """Check if any panel in the scene has dialogue from character 'Uno'."""
    for panel in panels:
        dialogues = panel.get("dialogues", [])
        for dialogue in dialogues:
            if dialogue.get("character").lower() == "uno":
                return True
    return False


def format_conversations(panels: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Format panel dialogues as conversation role/content pairs.

    Uses character 'uno' as 'assistant' and any other character as 'human'.
    Consecutive lines with the same role are collated with newline separation.
    """
    conversations = []
    for panel in panels:
        dialogues = panel.get("dialogues", [])
        for dialogue in dialogues:
            character = dialogue.get("character", "").strip()
            line = dialogue.get("line", "").strip()
            if not character or not line:
                continue

            # Determine role based on character name
            role = "assistant" if character.lower() == "uno" else "human"
            # Collate consecutive lines with the same role
            if conversations and conversations[-1]["role"] == role:
                conversations[-1]["content"] += "\n" + line
            else:
                conversations.append({"role": role, "content": line})
    return conversations


def get_unique_characters(panels: list[dict[str, Any]]) -> list[str]:
    """Extract unique character names from scene panels."""
    characters = set()
    for panel in panels:
        dialogues = panel.get("dialogues", [])
        for dialogue in dialogues:
            character = dialogue.get("character", "").strip()
            if character:
                characters.add(character)
    return sorted(list(characters))


def process_issue(issue_dir: Path) -> list[dict[str, Any]]:
    """Process all pages in an issue and return scene data.

    Returns a list of scene dictionaries ready for CSV output.
    """
    issue_name = issue_dir.name
    page_files = sorted(issue_dir.glob("page_*.json"))

    if not page_files:
        log.error(f"No page files found in {issue_dir}")
        return []

    current_scene_panels = []
    current_scene_pages = []
    scenes = []

    for page_file in page_files:
        with open(page_file) as f:
            page_data = json.load(f)

        panels = page_data.get("panels", [])
        if not panels:
            continue

        # Extract page reference from metadata
        page_ref = extract_page_reference(page_data["meta"]["input_page_path"])

        for panel in panels:
            # Check if this starts a new scene
            if panel.get("is_new_scene", False) and current_scene_panels:
                # Save the current scene if it contains Uno
                if scene_contains_uno(current_scene_panels):
                    scenes.append(
                        {
                            "pkna": issue_name,
                            "input_pages": current_scene_pages.copy(),
                            "characters": get_unique_characters(current_scene_panels),
                            "conversations": format_conversations(current_scene_panels),
                        }
                    )

                # Start a new scene
                current_scene_panels = []
                current_scene_pages = []

            # Add panel to current scene
            current_scene_panels.append(panel)
            if page_ref not in current_scene_pages:
                current_scene_pages.append(page_ref)

    # Save the last scene if it contains Uno
    if current_scene_panels and scene_contains_uno(current_scene_panels):
        scenes.append(
            {
                "pkna": issue_name,
                "input_pages": current_scene_pages.copy(),
                "characters": get_unique_characters(current_scene_panels),
                "conversations": format_conversations(current_scene_panels),
            }
        )

    log.info(f"Extracted {len(scenes)} scenes containing 'Uno' from {issue_name}")
    return scenes


def main() -> None:
    """Main function to process all issues and write to CSV."""
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Get all issue directories
    issue_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])

    # Collect all scenes from all issues
    all_scenes = []
    for issue_dir in issue_dirs:
        log.info(f"Processing {issue_dir.name}...")
        scenes = process_issue(issue_dir)
        all_scenes.extend(scenes)

    # Write to CSV
    if all_scenes:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["pkna", "input_pages", "characters", "conversations"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()

            for scene in all_scenes:
                # Convert lists to Python repr format (matching original dataset format)
                writer.writerow(
                    {
                        "pkna": scene["pkna"],
                        "input_pages": repr(scene["input_pages"]),
                        "characters": repr(scene["characters"]),
                        "conversations": repr(scene["conversations"]),
                    }
                )

        log.info(f"Created {len(all_scenes)} scenes containing 'Uno' in {OUTPUT_FILE}")
    else:
        log.warning("No scenes found containing 'Uno'")


if __name__ == "__main__":
    main()
