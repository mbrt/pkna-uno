#!/usr/bin/env python3

"""
Create scenes dataset from panel extractions.

Groups panels into scenes based on is_new_scene flag and filters
for scenes containing the character "Uno".
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


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


def generate_scene_summary(panels: list[dict[str, Any]]) -> str:
    """Generate a combined summary from all panel descriptions."""
    descriptions = []
    for panel in panels:
        desc = panel.get("description")
        if desc:
            descriptions.append(desc)
    return " ".join(descriptions)


def process_issue(issue_dir: Path, output_dir: Path) -> int:
    """Process all pages in an issue and create scene files.

    Returns the number of scenes created.
    """
    issue_name = issue_dir.name
    page_files = sorted(issue_dir.glob("page_*.json"))

    if not page_files:
        log.error(f"No page files found in {issue_dir}")
        return 0

    current_scene_panels = []
    current_scene_pages = []
    scene_number = 1
    scenes_created = 0

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
                if not scene_contains_uno(current_scene_panels):
                    continue

                scene_data = {
                    "issue": issue_name,
                    "scene_number": scene_number,
                    "pages": current_scene_pages,
                    "scene_summary": generate_scene_summary(current_scene_panels),
                    "panels": current_scene_panels,
                }

                output_file = (
                    output_dir / f"{issue_name}_scene_{scene_number:03d}.json"
                )
                with open(output_file, "w") as f:
                    json.dump(scene_data, f, indent=2, ensure_ascii=False)

                scenes_created += 1

                # Start a new scene
                scene_number += 1
                current_scene_panels = []
                current_scene_pages = []

            # Add panel to current scene (remove is_new_scene from output)
            panel_copy = {k: v for k, v in panel.items() if k != "is_new_scene"}
            current_scene_panels.append(panel_copy)
            if page_ref not in current_scene_pages:
                current_scene_pages.append(page_ref)

    # Save the last scene if it contains Uno
    if current_scene_panels and scene_contains_uno(current_scene_panels):
        scene_data = {
            "issue": issue_name,
            "scene_number": scene_number,
            "pages": current_scene_pages,
            "scene_summary": generate_scene_summary(current_scene_panels),
            "panels": current_scene_panels,
        }

        output_file = output_dir / f"{issue_name}_scene_{scene_number:03d}.json"
        with open(output_file, "w") as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)

        scenes_created += 1

    log.info(f"Created {scenes_created} scenes for issue {issue_name}")
    return scenes_created


def main() -> None:
    """Main function to process all issues."""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "output" / "dspy-extract-full" / "v2"
    output_dir = base_dir / "output" / "scenes"

    # Create (or clear) output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all issue directories
    issue_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    total_scenes = 0
    for issue_dir in issue_dirs:
        log.info(f"Processing {issue_dir.name}...")
        scenes = process_issue(issue_dir, output_dir)
        total_scenes += scenes

    log.info(f"Created {total_scenes} scenes containing 'Uno' in {output_dir}")


if __name__ == "__main__":
    main()
