#!/usr/bin/env python3
"""Score speaker attribution against hand-checked labels.

Accepts observation-layer directories and extract-emotional v2 directories, so
every extraction is measured on the same labeled balloons. Lines are matched
by text; a gold line with no matching prediction counts as missing, which is
scored as wrong.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import BaseModel
from rich.table import Table

from extract.compare_observations import (
    Line,
    align_lines,
    load_records,
    new_lines,
    v2_lines,
)
from pkna.extract.observations import load_v2_pages
from pkna.extract.registry import Registry
from pkna.logging import setup_logging

console, log = setup_logging()

BASE_DIR = Path(__file__).parent.parent
DEFAULT_GOLD = BASE_DIR / "data/gold/speakers-pkna-0.json"

# Gold label for speakers who are not named characters (TV voices, extras).
OTHER = "*other*"


class GoldLine(BaseModel):
    image: str
    text: str
    speaker: str
    uncertain: bool = False


class GoldSet(BaseModel):
    aliases: dict[str, str] = {}
    lines: list[GoldLine]


@dataclass
class Tally:
    total: int = 0
    correct: int = 0
    missing: int = 0
    # Descriptive labels (e.g. 'Ostaggio dai capelli rossi') for a named gold
    # speaker: the right balloon binding, with the name left to entity resolution.
    unresolved: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class Score:
    pages: int = 0
    certain: Tally = field(default_factory=Tally)
    uncertain: Tally = field(default_factory=Tally)
    errors: list[dict] = field(default_factory=list)


def load_gold(path: Path) -> GoldSet:
    return GoldSet.model_validate_json(path.read_text(encoding="utf-8"))


def canonical(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name).casefold()


def score_speakers(predicted: dict[str, list[Line]], gold: GoldSet) -> Score:
    """Score predicted lines, keyed by page image name, against the gold set.

    Only pages present in the predictions are scored. A gold '*other*' line is
    correct when the prediction names none of the characters labeled elsewhere
    in the gold set.
    """
    named = {
        canonical(g.speaker, gold.aliases) for g in gold.lines if g.speaker != OTHER
    }
    by_image: dict[str, list[GoldLine]] = defaultdict(list)
    for g in gold.lines:
        by_image[g.image].append(g)

    score = Score()
    for image, gold_lines in by_image.items():
        if image not in predicted:
            continue
        score.pages += 1
        expected = [
            Line(speaker=g.speaker, text=g.text, ref=str(i))
            for i, g in enumerate(gold_lines)
        ]
        pairs, _, missing = align_lines(predicted[image], expected)
        for pred, exp in pairs:
            g = gold_lines[int(exp.ref)]
            tally = score.uncertain if g.uncertain else score.certain
            tally.total += 1
            got = canonical(pred.speaker, gold.aliases)
            if g.speaker == OTHER:
                ok = got not in named
            else:
                ok = got == canonical(g.speaker, gold.aliases)
            if ok:
                tally.correct += 1
            else:
                if g.speaker != OTHER and got not in named:
                    tally.unresolved += 1
                score.errors.append(
                    {
                        "image": image,
                        "text": g.text,
                        "gold": g.speaker,
                        "got": pred.speaker,
                    }
                )
        for exp in missing:
            g = gold_lines[int(exp.ref)]
            tally = score.uncertain if g.uncertain else score.certain
            tally.total += 1
            tally.missing += 1
            score.errors.append(
                {"image": image, "text": g.text, "gold": g.speaker, "got": None}
            )
    return score


def observation_predictions(issue_dir: Path) -> dict[str, list[Line]]:
    return {r.source.image: new_lines(r) for r in load_records(issue_dir)}


def resolve_predictions(
    predicted: dict[str, list[Line]], issue: str, registry: Registry
) -> dict[str, list[Line]]:
    """Replace speaker labels with the registry ids they resolve to."""

    def resolve(label: str) -> str:
        ref = registry.resolve(issue, label)
        return ref.id if ref else label

    return {
        image: [replace(line, speaker=resolve(line.speaker)) for line in lines]
        for image, lines in predicted.items()
    }


def resolve_gold(gold: GoldSet, registry: Registry) -> GoldSet:
    """Replace gold speaker names with the ids of the named characters they match."""

    def resolve(name: str) -> str:
        if name == OTHER:
            return name
        character = registry.find(gold.aliases.get(name, name))
        return character.id if character else name

    return GoldSet(
        lines=[g.model_copy(update={"speaker": resolve(g.speaker)}) for g in gold.lines]
    )


def v2_predictions(v2_issue_dir: Path) -> dict[str, list[Line]]:
    return {
        image: v2_lines(page, image)
        for image, page in load_v2_pages(v2_issue_dir).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score speaker attribution against hand-checked labels"
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--obs",
        type=Path,
        nargs="*",
        default=[],
        help="Observation-layer issue directories to score",
    )
    parser.add_argument(
        "--v2", type=Path, nargs="*", default=[], help="v2 issue directories to score"
    )
    parser.add_argument(
        "--errors", action="store_true", help="List every wrong or missing line"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Also score observation directories after resolving labels "
        "through this registry.json (directory names are the issues)",
    )
    args = parser.parse_args()

    gold = load_gold(args.gold)
    runs = [(str(d), observation_predictions(d), gold) for d in args.obs]
    runs += [(str(d), v2_predictions(d), gold) for d in args.v2]
    if args.registry:
        registry = Registry.model_validate_json(
            args.registry.read_text(encoding="utf-8")
        )
        resolved_gold = resolve_gold(gold, registry)
        runs += [
            (
                f"{d} + registry",
                resolve_predictions(observation_predictions(d), d.name, registry),
                resolved_gold,
            )
            for d in args.obs
        ]

    table = Table(title=f"Speaker attribution vs {args.gold.name}")
    for column in (
        "extraction",
        "pages",
        "accuracy",
        "wrong speaker",
        "unresolved name",
        "missing",
        "uncertain ok",
    ):
        table.add_column(column)
    for name, predicted, run_gold in runs:
        s = score_speakers(predicted, run_gold)
        c = s.certain
        table.add_row(
            name,
            str(s.pages),
            f"{c.accuracy:.1%} ({c.correct}/{c.total})",
            str(c.total - c.correct - c.missing - c.unresolved),
            str(c.unresolved),
            str(c.missing),
            f"{s.uncertain.correct}/{s.uncertain.total}",
        )
        if args.errors:
            for e in s.errors:
                console.print(
                    f"[dim]{name}[/dim] {e['image']} "
                    f'"{e["text"]}": gold={e["gold"]} got={e["got"]}'
                )
    console.print(table)


if __name__ == "__main__":
    main()
