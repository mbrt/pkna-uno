#!/usr/bin/env python3
"""Tool that event-log agents run from their workspace.

validate-log: check the scene and fact files an agent wrote against the
schemas and the issue's panel references and character names.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from pkna.extract.events import Fact, IssueFacts, IssueIndex, Scene, log_problems


def load_log(out_dir: Path) -> tuple[list[Scene], list[Fact], list[str]]:
    """Scenes in file order and facts, with the files that failed to parse."""
    problems: list[str] = []
    facts: list[Fact] = []
    try:
        facts = IssueFacts.model_validate_json(
            (out_dir / "facts.json").read_text(encoding="utf-8")
        ).facts
    except (OSError, ValidationError) as e:
        problems.append(f"facts.json: {e}")
    scenes: list[Scene] = []
    for path in sorted((out_dir / "scenes").glob("*.json")):
        try:
            scenes.append(Scene.model_validate_json(path.read_text(encoding="utf-8")))
        except ValidationError as e:
            problems.append(f"scenes/{path.name}: {e}")
    return scenes, facts, problems


def validate_log(out_dir: Path, index_path: Path, series_path: Path) -> list[str]:
    """Problems with an agent's log files; empty when they are valid."""
    scenes, facts, problems = load_log(out_dir)
    if problems:
        return problems
    index = IssueIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    series = IssueFacts.model_validate_json(series_path.read_text(encoding="utf-8"))
    return log_problems(scenes, facts, index, {f.id for f in series.facts})


def main() -> None:
    parser = argparse.ArgumentParser(description="Tools for event-log agents")
    sub = parser.add_subparsers(dest="command", required=True)
    log_parser = sub.add_parser("validate-log", help="Check scene and fact files")
    log_parser.add_argument("out_dir", type=Path)
    log_parser.add_argument("--index", type=Path, default=Path("index.json"))
    log_parser.add_argument("--series", type=Path, default=Path("series-facts.json"))
    args = parser.parse_args()

    problems = validate_log(args.out_dir, args.index, args.series)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"OK: {args.out_dir}")


if __name__ == "__main__":
    main()
