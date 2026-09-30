#!/usr/bin/env python3
"""Show what characters believe, and where they were last seen, at a story point.

Replays the event and knowledge logs up to the cutoff (before a panel or line
of an issue, or the end of the issue) with characters resolved through the
registry, and facts joined into propositions by the proposed and manual fact
links.
"""

import argparse
from pathlib import Path

from rich.table import Table

from pkna.extract.events import IssueLog
from pkna.extract.fact_links import load_fact_links
from pkna.extract.registry import Registry
from pkna.extract.world_state import Cutoff, WorldState, world_state
from pkna.logging import setup_logging

console, log = setup_logging()

BASE_DIR = Path(__file__).parent.parent
EVENTS_ROOT = BASE_DIR / "output/events/v1"
REGISTRY_PATH = BASE_DIR / "output/registry/v1/registry.json"
LINKS_ROOT = EVENTS_ROOT / "links"
MANUAL_LINKS_PATH = BASE_DIR / "data/events/fact-links.json"


def load_logs(events_root: Path) -> list[IssueLog]:
    return [
        IssueLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(events_root.glob("*.json"))
    ]


def character_id(registry: Registry, name_or_id: str) -> str:
    if any(c.id == name_or_id for c in registry.characters):
        return name_or_id
    character = registry.find(name_or_id)
    if character is None:
        raise SystemExit(f"No named character or id {name_or_id!r} in the registry")
    return character.id


def beliefs_table(state: WorldState, registry: Registry, cid: str) -> Table:
    names = {c.id: c.name for c in registry.characters}
    seen = state.last_seen.get(cid)
    where = f"{' > '.join(seen.place)} ({seen.issue} {seen.scene})" if seen else "-"
    table = Table(
        title=f"{names[cid]}: last seen at {where}", show_lines=True, expand=True
    )
    table.add_column("fact", ratio=3)
    for column in ("truth", "stance", "how", "where"):
        table.add_column(column, ratio=1)
    for b in state.beliefs.get(cid, {}).values():
        fact = state.facts.get(b.fact)
        how = b.source + (f" ← {names[b.informant]}" if b.informant else "")
        table.add_row(
            fact.statement if fact else b.fact,
            state.truth.get(b.fact, ""),
            b.stance,
            how,
            f"{b.issue} {b.at}",
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show character beliefs and whereabouts at a story point"
    )
    parser.add_argument("--issue", required=True)
    parser.add_argument(
        "--ref", help="Panel or lettering reference; omit for the end of the issue"
    )
    parser.add_argument(
        "--character", nargs="*", default=[], help="Names or registry ids"
    )
    parser.add_argument("--events-dir", type=Path, default=EVENTS_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--links-dir", type=Path, default=LINKS_ROOT)
    parser.add_argument("--manual-links", type=Path, default=MANUAL_LINKS_PATH)
    args = parser.parse_args()

    registry = Registry.model_validate_json(args.registry.read_text(encoding="utf-8"))
    logs = load_logs(args.events_dir)
    links, problems = load_fact_links(args.links_dir, args.manual_links, logs)
    for problem in problems:
        log.warning(f"Fact link ignored: {problem}")
    state = world_state(logs, registry, Cutoff(issue=args.issue, ref=args.ref), links)
    ids = [character_id(registry, c) for c in args.character] or list(state.beliefs)
    for cid in ids:
        console.print(beliefs_table(state, registry, cid))


if __name__ == "__main__":
    main()
