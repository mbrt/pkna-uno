"""Tests for replaying event logs into state views."""

import pytest

from pkna.extract.events import (
    Fact,
    IssueLog,
    KnowledgeChange,
    LoggedScene,
    Presence,
)
from pkna.extract.registry import IssueCast, Persona, ResolvedCharacter, merge_casts
from pkna.extract.world_state import Cutoff, Sighting, ref_key, world_state


def character(name: str, *personas: str) -> ResolvedCharacter:
    return ResolvedCharacter(
        name=name,
        kind="named",
        personas=[Persona(name=p, labels=[p]) for p in personas or (name,)],
        description="d",
    )


REGISTRY = merge_casts(
    {
        "pkna-0": IssueCast(
            characters=[
                character("Paolino Paperino", "Paperino", "Paperinik"),
                character("Uno"),
                character("Paperilla Starry"),
            ]
        ),
        "pkna-2": IssueCast(characters=[character("Paperinik"), character("Uno")]),
        "pkna-10": IssueCast(characters=[character("Paperinik"), character("Uno")]),
    },
    preferred_names={"Paperino"},
)


def scene(
    sid: str, start: str, place: str, present: list[str], *knowledge: KnowledgeChange
) -> LoggedScene:
    return LoggedScene(
        id=sid,
        start=start,
        end=start,
        place=[place],
        summary="s",
        present=[Presence(name=n, personas=[n], remote=False) for n in present],
        knowledge=list(knowledge),
    )


def learns(at: str, who: str, fact: str, stance: str = "believes", informant=None):
    return KnowledgeChange.model_validate(
        {
            "at": at,
            "character": who,
            "fact": fact,
            "stance": stance,
            "source": "told" if informant else "already_known",
            "informant": informant,
        }
    )


LOGS = [
    IssueLog(
        issue="pkna-10",
        facts=[],
        scenes=[
            scene(
                "s01",
                "pkna10-001#p1",
                "Ducklair Tower",
                ["Uno"],
                learns(
                    "pkna10-001#p1.t1", "Uno", "serie:identita-paperinik", "suspects"
                ),
            )
        ],
    ),
    IssueLog(
        issue="pkna-0",
        facts=[Fact(id="f01", statement="Uno è un'IA.", truth="true")],
        scenes=[
            scene(
                "s01",
                "pkna-0-001#p1",
                "Ducklair Tower",
                ["Uno", "Paolino Paperino"],
                learns("pkna-0-001#p3.t1", "Paolino Paperino", "f01", informant="Uno"),
                learns("pkna-0-001#p1.t2", "Uno", "serie:identita-paperinik"),
            ),
            scene("s02", "pkna-0-002#p1", "Terrazza", ["Paperilla Starry", "Uno"]),
        ],
    ),
    IssueLog(
        issue="pkna-2",
        facts=[],
        scenes=[
            scene(
                "s01",
                "pkna2-001#p1",
                "Paperopoli",
                ["Paperinik"],
                learns("pkna2-001#p1", "Paperinik", "serie:identita-paperinik"),
            )
        ],
    ),
]


def test_cutoff_excludes_the_line_and_later_scenes():
    state = world_state(LOGS, REGISTRY, Cutoff(issue="pkna-0", ref="pkna-0-001#p3.t1"))

    assert {c: sorted(b) for c, b in state.beliefs.items()} == {
        "uno": ["serie:identita-paperinik"]
    }
    assert state.last_seen == {
        "uno": Sighting(
            issue="pkna-0", scene="s01", place=["Ducklair Tower"], remote=False
        ),
        "paperino": Sighting(
            issue="pkna-0", scene="s01", place=["Ducklair Tower"], remote=False
        ),
    }
    assert list(state.facts) == ["pkna-0/f01"]


def test_beliefs_persist_across_issues_in_publication_order():
    state = world_state(LOGS, REGISTRY, Cutoff(issue="pkna-10"))

    uno = state.beliefs["uno"]["serie:identita-paperinik"]
    paperino = state.beliefs["paperino"]
    assert (uno.stance, uno.issue) == ("suspects", "pkna-10")
    assert sorted(paperino) == ["pkna-0/f01", "serie:identita-paperinik"]
    assert paperino["pkna-0/f01"].informant == "uno"
    assert state.last_seen["paperilla-starry"].place == ["Terrazza"]

    before = world_state(LOGS, REGISTRY, Cutoff(issue="pkna-2"))
    assert before.beliefs["uno"]["serie:identita-paperinik"].stance == "believes"


def test_ref_key_orders_panels_before_their_lettering():
    assert ref_key("pkna-0-001#p3") < ref_key("pkna-0-001#p3.t1")
    assert ref_key("pkna-0-001#p3.t2") < ref_key("pkna-0-001#p10")
    with pytest.raises(ValueError, match="Not a panel"):
        ref_key("pkna-0-001")
