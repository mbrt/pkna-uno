"""Fact links proposed per issue, and their combination with manual corrections.

An agent reads each issue's facts beside the facts before them and proposes
which state the same proposition as an earlier fact, or its negation. The
proposals of all issues and the manual links are combined into propositions
for the state views.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pkna.extract.events import SERIES_FACT_PREFIX, Fact, IssueLog
from pkna.extract.registry import Registry
from pkna.extract.world_state import FactLink, FactLinks, global_fact_id, issue_key


class ProposedLink(BaseModel):
    fact: str = Field(description="Id of a fact of this issue, e.g. 'f12'.")
    same: list[str] = Field(
        default=[],
        description="Global ids (e.g. 'pkna-2/f07') of earlier facts stating the "
        "same proposition.",
    )
    opposite: list[str] = Field(
        default=[], description="Global ids of earlier facts stating its negation."
    )
    note: str = Field(description="Why, in one sentence, citing what the facts say.")


class IssueLinks(BaseModel):
    links: list[ProposedLink]


class IssueLinkLog(IssueLinks):
    issue: str
    meta: dict[str, Any] = {}


class LinkIndex(BaseModel):
    """What an issue's proposals may refer to, for validation."""

    issue: str
    facts: list[str] = Field(description="Local ids of the issue's facts.")
    earlier: list[str] = Field(description="Global ids of the facts before each.")


def fact_order(logs: Sequence[IssueLog]) -> list[str]:
    """Global ids of all facts, in replay order."""
    return [
        global_fact_id(log.issue, f.id)
        for log in sorted(logs, key=lambda log: issue_key(log.issue))
        for f in log.facts
    ]


def link_index(
    logs: Sequence[IssueLog], issue: str, series: Sequence[Fact] = ()
) -> LinkIndex:
    """An issue's facts may link to series facts, earlier issues' facts, and
    each other."""
    order = fact_order(logs)
    own = [f"{issue}/{f.id}" for log in logs if log.issue == issue for f in log.facts]
    first = order.index(own[0]) if own else len(order)
    return LinkIndex(
        issue=issue,
        facts=[fid.split("/", 1)[1] for fid in own],
        earlier=[f.id for f in series] + order[: first + len(own)],
    )


def link_problems(links: IssueLinks, index: LinkIndex) -> list[str]:
    """Links to unknown facts, and facts linked twice or both ways."""
    problems: list[str] = []
    known = set(index.earlier)
    seen: set[str] = set()
    for i, link in enumerate(links.links, 1):
        where = f"link {i} ({link.fact})"
        if link.fact not in index.facts:
            problems.append(f"{where}: {link.fact!r} is not a fact of {index.issue}")
        if link.fact in seen:
            problems.append(f"{where}: {link.fact!r} is linked more than once")
        seen.add(link.fact)
        own = f"{index.issue}/{link.fact}"
        if not link.same and not link.opposite:
            problems.append(f"{where}: no 'same' or 'opposite' facts")
        problems += [
            f"{where}: {t!r} is not a fact id (use global ids such as 'pkna-2/f07')"
            for t in [*link.same, *link.opposite]
            if t not in known
        ]
        if own in [*link.same, *link.opposite]:
            problems.append(f"{where}: links to itself")
        problems += [
            f"{where}: {t!r} is both 'same' and 'opposite'"
            for t in sorted(set(link.same) & set(link.opposite))
        ]
    return problems


def combine_links(
    proposed: Sequence[IssueLinkLog], manual: FactLinks, order: Sequence[str]
) -> tuple[FactLinks, list[str]]:
    """Join proposed and manual links into propositions.

    Links chain across issues: a fact linked to one linked to a third states
    the same proposition as the third, or its negation. Manual links apply
    first and win over proposals that contradict them. Links to facts not in
    `order` (stale after an issue's log was redone) are dropped and reported.
    """
    position = {fid: i for i, fid in enumerate(order)}
    parent: dict[str, str] = {}
    flip: dict[str, bool] = {}
    problems: list[str] = []

    def known(fid: str) -> bool:
        return fid in position or fid.startswith(SERIES_FACT_PREFIX)

    def find(fid: str) -> tuple[str, bool]:
        negated = False
        while parent.get(fid, fid) != fid:
            negated ^= flip[fid]
            fid = parent[fid]
        return fid, negated

    def link(a: str, b: str, opposite: bool, source: str) -> None:
        for fid in (a, b):
            if not known(fid):
                problems.append(f"{source}: unknown fact {fid!r}")
                return
        ra, na = find(a)
        rb, nb = find(b)
        if ra == rb:
            if na ^ nb != opposite:
                relation = "opposite" if opposite else "same"
                problems.append(
                    f"{source}: {a} and {b} as {relation} contradicts other links"
                )
            return
        parent[ra], flip[ra] = rb, na ^ nb ^ opposite

    for i, group in enumerate(manual.links, 1):
        head = group.same[0]
        for fid in group.same[1:]:
            link(head, fid, False, f"manual link {i}")
        for fid in group.opposite:
            link(head, fid, True, f"manual link {i}")
    detached = set(manual.unlink)
    for log in proposed:
        for p in log.links:
            fid = f"{log.issue}/{p.fact}"
            if fid in detached:
                continue
            for targets, opposite in ((p.same, False), (p.opposite, True)):
                for target in targets:
                    if target not in detached:
                        link(fid, target, opposite, fid)

    def sort_key(fid: str) -> tuple[int, str]:
        return position.get(fid, -1), fid

    members: dict[str, list[tuple[str, bool]]] = {}
    for fid in sorted(set(parent) | set(parent.values()), key=sort_key):
        root, negated = find(fid)
        members.setdefault(root, []).append((fid, negated))
    links: list[FactLink] = []
    for group in members.values():
        # The earliest fact states the proposition, so ids stay stable as
        # later issues are linked.
        first_negated = group[0][1]
        links.append(
            FactLink(
                same=[fid for fid, n in group if n == first_negated],
                opposite=[fid for fid, n in group if n != first_negated],
            )
        )
    links.sort(key=lambda link: sort_key(link.same[0]))
    return FactLinks(links=links), problems


def load_proposals(links_root: Path) -> list[IssueLinkLog]:
    return [
        IssueLinkLog.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(links_root.glob("*.json"))
    ]


def load_fact_links(
    links_root: Path, manual_path: Path, logs: Sequence[IssueLog]
) -> tuple[FactLinks, list[str]]:
    """The combined links of the proposals and the manual corrections."""
    manual = FactLinks()
    if manual_path.exists():
        manual = FactLinks.model_validate_json(manual_path.read_text(encoding="utf-8"))
    return combine_links(load_proposals(links_root), manual, fact_order(logs))


def _facts_by_id(
    logs: Sequence[IssueLog], series: Sequence[Fact]
) -> dict[str, tuple[str | None, Fact]]:
    """Each fact with its issue, None for series facts."""
    by_id: dict[str, tuple[str | None, Fact]] = {f.id: (None, f) for f in series}
    for log in logs:
        for f in log.facts:
            by_id[global_fact_id(log.issue, f.id)] = (log.issue, f)
    return by_id


def render_facts(
    logs: Sequence[IssueLog],
    registry: Registry,
    fids: Sequence[str],
    series: Sequence[Fact] = (),
) -> str:
    """One line per fact: id, truth in its issue, characters, and statement.

    Characters are named by their registry name, the same across issues.
    """
    by_id = _facts_by_id(logs, series)
    names = {c.id: c.name for c in registry.characters}
    lines: list[str] = []
    for fid in fids:
        issue, fact = by_id[fid]
        about = ", ".join(
            names[registry.member(issue, n)] if issue else n for n in fact.about
        )
        lines.append(f"- {fid} [{fact.truth}] ({about}): {fact.statement}")
    return "\n".join(lines)


def fact_hash_payload(
    logs: Sequence[IssueLog], fids: Sequence[str], series: Sequence[Fact] = ()
) -> str:
    """The facts' ids, statements, and truth, for detecting changed inputs.

    Character names are left out: a registry correction renames characters
    without changing what the facts say.
    """
    by_id = {fid: fact for fid, (_, fact) in _facts_by_id(logs, series).items()}
    return json.dumps(
        [[fid, by_id[fid].statement, by_id[fid].truth] for fid in fids],
        ensure_ascii=False,
    )
