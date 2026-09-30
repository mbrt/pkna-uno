#!/usr/bin/env python3
"""Tool that fact-link agents run from their workspace.

validate-links: check the links file an agent wrote against the schema and the
fact ids of the issue and of the facts before it.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from pkna.extract.fact_links import IssueLinks, LinkIndex, link_problems


def load_links(path: Path) -> tuple[IssueLinks | None, list[str]]:
    try:
        return IssueLinks.model_validate_json(path.read_text(encoding="utf-8")), []
    except (OSError, ValidationError) as e:
        return None, [f"{path.name}: {e}"]


def validate_links(path: Path, index_path: Path) -> list[str]:
    """Problems with an agent's links file; empty when it is valid."""
    links, problems = load_links(path)
    if links is None:
        return problems
    index = LinkIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    return link_problems(links, index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tools for fact-link agents")
    sub = parser.add_subparsers(dest="command", required=True)
    links_parser = sub.add_parser("validate-links", help="Check the links file")
    links_parser.add_argument("path", type=Path)
    links_parser.add_argument("--index", type=Path, default=Path("index.json"))
    args = parser.parse_args()

    problems = validate_links(args.path, args.index)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"OK: {args.path}")


if __name__ == "__main__":
    main()
