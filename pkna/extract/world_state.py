"""State views: the world as of any point in the story, replayed from the event logs.

Plain code over the event and knowledge log, so a view can be recomputed for
any cutoff: what each character believes, and where each was last seen.
Characters are registry ids; facts have global ids ('pkna-0/f03', or the
'serie:' ids of series facts).
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


class Belief(BaseModel):
    fact: str
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
        description="Latest belief of each character about each fact."
    )
    last_seen: dict[str, Sighting] = Field(
        description="The last scene each character was present in."
    )
    facts: dict[str, Fact] = Field(description="Facts introduced so far, by global id.")


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


def world_state(
    logs: Sequence[IssueLog], registry: Registry, cutoff: Cutoff
) -> WorldState:
    """Replay the logs in issue order up to the cutoff, excluding it."""
    beliefs: dict[str, dict[str, Belief]] = {}
    last_seen: dict[str, Sighting] = {}
    facts: dict[str, Fact] = {}
    end = issue_key(cutoff.issue)
    for log in sorted(logs, key=lambda log: issue_key(log.issue)):
        if issue_key(log.issue) > end:
            break
        limit = (
            ref_key(cutoff.ref) if log.issue == cutoff.issue and cutoff.ref else None
        )
        for fact in log.facts:
            facts[global_fact_id(log.issue, fact.id)] = fact
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
                beliefs.setdefault(registry.member(log.issue, k.character), {})[fid] = (
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
    return WorldState(beliefs=beliefs, last_seen=last_seen, facts=facts)
