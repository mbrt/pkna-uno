#!/usr/bin/env python3
"""Merge per-issue character resolutions into the series registry.

Named characters are linked across issues by shared names and persona names,
plus the manual merges and separations in data/registry/overrides.json. Writes the registry and
a review report listing merges and near-duplicates worth checking.
"""

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from pkna.extract.registry import (
    IssueCast,
    Overrides,
    Registry,
    RegistryCharacter,
    merge_casts,
    name_key,
)
from pkna.logging import setup_logging

console, log = setup_logging()

BASE_DIR = Path(__file__).parent.parent
ISSUES_ROOT = BASE_DIR / "output/registry/v1/issues"
OUT_ROOT = BASE_DIR / "output/registry/v1"
SERIES_PATH = BASE_DIR / "data/registry/series.json"
OVERRIDES_PATH = BASE_DIR / "data/registry/overrides.json"

SIMILARITY_THRESHOLD = 0.85
# Shorter words ('Qui', 'Quo') are too often similar by chance.
MIN_WORD_LENGTH = 4


def load_casts(issues_root: Path) -> dict[str, IssueCast]:
    casts: dict[str, IssueCast] = {}
    for path in sorted(issues_root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        casts[data["issue"]] = IssueCast.model_validate(data["cast"])
    return casts


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= SIMILARITY_THRESHOLD


def similar_names(characters: list[RegistryCharacter]) -> list[tuple[str, str]]:
    """Pairs of named characters whose names suggest they may be the same.

    One name's words contained in the other's ('Zondag', 'Generale Zondag'), or
    nearly identical spelling of the names or of one of their longer words
    ('Paperone', "Paperon de' Paperoni").
    """
    named = [c for c in characters if c.kind == "named"]
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(named):
        for b in named[i + 1 :]:
            ka, kb = name_key(a.name), name_key(b.name)
            wa, wb = set(ka.split()), set(kb.split())
            similar_word = any(
                _similar(x, y)
                for x in wa
                for y in wb
                if min(len(x), len(y)) >= MIN_WORD_LENGTH
            )
            if wa <= wb or wb <= wa or _similar(ka, kb) or similar_word:
                pairs.append((a.id, b.id))
    return pairs


def format_review(registry: Registry) -> str:
    named = sorted(
        (c for c in registry.characters if c.kind == "named"),
        key=lambda c: (-len(c.issues), c.name),
    )
    others = len(registry.characters) - len(named)
    lines = [
        "# Character registry review",
        "",
        f"{len(named)} named characters; {others} unnamed characters and groups "
        f"scoped to single issues; {len(registry.labels)} issues.",
        "",
        "Fix wrong links by adding name groups to `merge` or `separate` in "
        "`data/registry/overrides.json` and rerunning the merge.",
        "",
        "## Merged under different names",
        "",
    ]
    lines += [
        f"- `{c.id}`: {', '.join(c.names)} ({len(c.issues)} issues)"
        for c in named
        if len(c.names) > 1
    ]
    lines += ["", "## Shared names kept apart (distinct within an issue)", ""]
    lines += [f"- `{a}` / `{b}`" for a, b in registry.kept_apart]
    lines += ["", "## Similar names kept separate", ""]
    lines += [f"- `{a}` / `{b}`" for a, b in similar_names(named)]
    lines += [
        "",
        "## Named characters",
        "",
        "| id | name | personas | issues |",
        "|---|---|---|---|",
    ]
    for c in named:
        personas = "; ".join(f"{p.name} ({len(p.labels)} labels)" for p in c.personas)
        lines.append(f"| `{c.id}` | {c.name} | {personas} | {len(c.issues)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-issue character resolutions into the registry"
    )
    parser.add_argument("--issues-dir", type=Path, default=ISSUES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    casts = load_casts(args.issues_dir)
    overrides = Overrides()
    if OVERRIDES_PATH.exists():
        overrides = Overrides.model_validate_json(
            OVERRIDES_PATH.read_text(encoding="utf-8")
        )
    series = json.loads(SERIES_PATH.read_text(encoding="utf-8"))
    preferred = {identity["character"] for identity in series["identities"]}

    registry = merge_casts(casts, overrides, preferred)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "registry.json").write_text(
        registry.model_dump_json(indent=1), encoding="utf-8"
    )
    report = format_review(registry)
    (args.out_dir / "review.md").write_text(report, encoding="utf-8")
    console.print(
        f"Merged {len(casts)} issues into {len(registry.characters)} characters. "
        f"Output: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
