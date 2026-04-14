#!/usr/bin/env python3

"""One-time migration: rename scene-reflection files for collided scene IDs.

Before the scene_index fix, two scenes starting on the same page got the same
scene_id.  The second scene's reflection overwrote the first on disk.  After
the fix, the first scene keeps the original ID (scene_index=0) and the second
gets a new suffix (scene_index=1).

This script:
1. Re-extracts scenes to find which IDs now have scene_index > 0
2. For each collision, renames the existing file to the new disambiguated name
3. Reports what was moved so the pipeline can re-generate the missing first
   scene's reflection.

Usage:
    python -m extract.migrate_scene_reflections [--dry-run]
"""

import argparse
import json
import shutil
from pathlib import Path

from rich.console import Console

from pkna.pkna_scenes import extract_scenes_from_issue, natural_sort_key

console = Console(stderr=True)

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "output" / "extract-emotional" / "v2"
REFLECTIONS_DIR = BASE_DIR / "output" / "scene-reflections" / "v2"


def find_collisions() -> list[tuple[str, str, str]]:
    """Find scene IDs where scene_index > 0 and the base file exists.

    Returns list of (issue, old_scene_id, new_scene_id) tuples.
    """
    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )
    collisions = []
    for issue_dir in issue_dirs:
        scenes = extract_scenes_from_issue(issue_dir)
        for scene in scenes:
            if scene.scene_index == 0:
                continue
            base_id = f"{scene.issue}_{scene.page_numbers[0]}"
            new_id = scene.scene_id
            collisions.append((scene.issue, base_id, new_id))
    return collisions


def _patch_scene_id(path: Path, new_scene_id: str) -> None:
    """Update the scene_id field inside a reflection JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("scene_id") != new_scene_id:
        data["scene_id"] = new_scene_id
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def migrate(dry_run: bool = False) -> None:
    collisions = find_collisions()
    if not collisions:
        console.print("[green]No collisions found — nothing to migrate.[/green]")
        return

    console.print(f"Found {len(collisions)} collided scene IDs to migrate:\n")
    renamed = 0
    skipped = 0

    for issue, old_id, new_id in collisions:
        issue_dir = REFLECTIONS_DIR / issue
        old_path = issue_dir / f"{old_id}.json"
        new_path = issue_dir / f"{new_id}.json"

        if new_path.exists():
            console.print(f"  [yellow]SKIP[/yellow] {new_path.name} already exists")
            skipped += 1
            continue

        if not old_path.exists():
            console.print(
                f"  [yellow]SKIP[/yellow] {old_path.name} not found (issue {issue})"
            )
            skipped += 1
            continue

        if dry_run:
            console.print(
                f"  [cyan]WOULD MIGRATE[/cyan] {old_path.name} -> {new_path.name}"
            )
        else:
            shutil.move(old_path, new_path)
            _patch_scene_id(new_path, new_id)
            console.print(
                f"  [green]MIGRATED[/green] {old_path.name} -> {new_path.name}"
            )
        renamed += 1

    console.print(f"\n{'Would rename' if dry_run else 'Renamed'}: {renamed}")
    console.print(f"Skipped: {skipped}")
    if not dry_run and renamed > 0:
        console.print(
            "\n[bold]Re-run reflect_scenes.py to generate reflections "
            "for the now-missing base scene IDs.[/bold]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate scene-reflection files after scene_index fix"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually moving files",
    )
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
