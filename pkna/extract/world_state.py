"""State views: the world as of any point in the story, replayed from the event logs.

Plain code over the event and knowledge log, so a view can be recomputed for
any cutoff: what each character believes, and where each was last seen.
Characters are registry ids; facts have global ids ('pkna-0/f03', or the
'serie:' ids of series facts).

Each issue names its own facts, so a later issue revising an earlier one
('Due è stato cancellato' in pkna-2, shown false in pkna-8) states it as a new
fact. Fact links join the facts stating one proposition or its negation, so a
later stance replaces an earlier one and the latest issue settling the
proposition decides its truth.
"""

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from pkna.extract.events import (
    SERIES_FACT_PREFIX,
    Fact,
    IssueLog,
    KnowledgeSource,
    Stance,
    Truth,
)
from pkna.extract.registry import Registry
from pkna.extract.scenes import natural_sort_key

REF_KEY_PATTERN = re.compile(r"^(?P<page>.+)#p(?P<panel>\d+)(?:\.t(?P<text>\d+))?$")


class Cutoff(BaseModel):
    """A point in the story: before a panel or line of an issue, or after the issue."""

    issue: str
    ref: str | None = Field(
        default=None,
        description="Panel or lettering reference; None for the end of the issue.",
    )


class FactLink(BaseModel):
    """One proposition, as stated by facts of one or more issues."""

    same: list[str] = Field(
        min_length=1,
        description="Global ids of facts stating the proposition; the first "
        "identifies it.",
    )
    opposite: list[str] = Field(
        default=[], description="Global ids of facts stating its negation."
    )


class FactLinks(BaseModel):
    links: list[FactLink] = []
    unlink: list[str] = Field(
        default=[],
        description="Facts to detach from proposed links (manual corrections only).",
    )


class Belief(BaseModel):
    fact: str = Field(description="The fact the character's latest stance is on.")
    stance: Stance
    source: KnowledgeSource
    informant: str | None
    issue: str
    at: str


class Sighting(BaseModel):
    issue: str
    scene: str
    place: list[str]
    remote: bool


class WorldState(BaseModel):
    beliefs: dict[str, dict[str, Belief]] = Field(
        description="Latest belief of each character about each proposition, by "
        "the id of its first linked fact, or the fact's own id when unlinked."
    )
    last_seen: dict[str, Sighting] = Field(
        description="The last scene each character was present in."
    )
    facts: dict[str, Fact] = Field(description="Facts introduced so far, by global id.")
    truth: dict[str, Truth] = Field(
        description="Truth of each fact introduced so far, as settled by the "
        "latest issue that settles its proposition."
    )


def ref_key(ref: str) -> tuple[str, int, int]:
    """Reading order of a reference within its issue.

    Page image names sort in reading order. A panel reference sorts before the
    panel's lettering.
    """
    match = REF_KEY_PATTERN.match(ref)
    if not match:
        raise ValueError(f"Not a panel or lettering reference: {ref!r}")
    return match["page"], int(match["panel"]), int(match["text"] or 0)


def global_fact_id(issue: str, fact: str) -> str:
    return fact if fact.startswith(SERIES_FACT_PREFIX) else f"{issue}/{fact}"


def issue_key(issue: str) -> tuple:
    return natural_sort_key(Path(issue))


def propositions(links: FactLinks) -> dict[str, tuple[str, bool]]:
    """For each linked fact, its proposition's id and whether it negates it."""
    return {
        fid: (link.same[0], negated)
        for link in links.links
        for negated, fids in ((False, link.same), (True, link.opposite))
        for fid in fids
    }


def world_state(
    logs: Sequence[IssueLog],
    registry: Registry,
    cutoff: Cutoff,
    links: FactLinks | None = None,
) -> WorldState:
    """Replay the logs in issue order up to the cutoff, excluding it.

    Facts of the cutoff issue count with the truth that issue gives them.
    """
    proposition = propositions(links or FactLinks())
    beliefs: dict[str, dict[str, Belief]] = {}
    last_seen: dict[str, Sighting] = {}
    facts: dict[str, Fact] = {}
    settled: dict[str, bool] = {}
    end = issue_key(cutoff.issue)
    for log in sorted(logs, key=lambda log: issue_key(log.issue)):
        if issue_key(log.issue) > end:
            break
        limit = (
            ref_key(cutoff.ref) if log.issue == cutoff.issue and cutoff.ref else None
        )
        for fact in log.facts:
            fid = global_fact_id(log.issue, fact.id)
            facts[fid] = fact
            if fact.truth != "unknown" and fid in proposition:
                pid, negated = proposition[fid]
                settled[pid] = (fact.truth == "true") != negated
        for scene in log.scenes:
            if limit and ref_key(scene.start) >= limit:
                break
            for p in scene.present:
                last_seen[registry.member(log.issue, p.name)] = Sighting(
                    issue=log.issue, scene=scene.id, place=scene.place, remote=p.remote
                )
            for k in sorted(scene.knowledge, key=lambda k: ref_key(k.at)):
                if limit and ref_key(k.at) >= limit:
                    break
                fid = global_fact_id(log.issue, k.fact)
                pid = proposition.get(fid, (fid, False))[0]
                beliefs.setdefault(registry.member(log.issue, k.character), {})[pid] = (
                    Belief(
                        fact=fid,
                        stance=k.stance,
                        source=k.source,
                        informant=(
                            registry.member(log.issue, k.informant)
                            if k.informant
                            else None
                        ),
                        issue=log.issue,
                        at=k.at,
                    )
                )
    truth: dict[str, Truth] = {}
    for fid, fact in facts.items():
        pid, negated = proposition.get(fid, (fid, False))
        truth[fid] = (
            ("true" if settled[pid] != negated else "false")
            if pid in settled
            else fact.truth
        )
    return WorldState(beliefs=beliefs, last_seen=last_seen, facts=facts, truth=truth)
