"""Event and knowledge log: what happens in each scene, and who learns what.

Derived from the observation records and the issue's character resolution,
without page images. Records cite panels ('pkna-0-012#p3') and lettering
('pkna-0-012#p3.t2') of the observation layer, so state views can replay the
story up to any point and check who could know what at that moment.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from pkna.extract.observations import (
    CharacterPresence,
    ObservationRecord,
    ObservedPanel,
    TextElement,
)
from pkna.extract.registry import IssueCast

Truth = Literal["true", "false", "unknown"]
Timeline = Literal["present", "flashback", "dream", "imagined"]
Stance = Literal["believes", "suspects", "disbelieves"]
KnowledgeSource = Literal["witnessed", "told", "overheard", "inferred", "already_known"]

SERIES_FACT_PREFIX = "serie:"

REF_PATTERN = re.compile(r"^(?P<panel>.+#p\d+)(?:\.t\d+)?$")

DEPICTION_NOTES = {
    "hologram": "ologramma",
    "on_screen": "su schermo",
    "off_panel": "fuori campo",
    "image": "immagine",
    "memory": "ricordo",
}
TEXT_KIND_NAMES = {"narration": "didascalia", "sign": "scritta"}


class Fact(BaseModel):
    id: str = Field(
        description="Short id unique in the issue, e.g. 'f01'. Series facts use "
        "their 'serie:' id and are not redefined."
    )
    statement: str = Field(
        description="A self-contained proposition in Italian, naming characters "
        "in full (e.g. 'Paperilla Starry è tenuta in ostaggio dagli Evroniani')."
    )
    truth: Truth = Field(
        description="Whether the statement is true in the story, judged from this "
        "issue only: 'false' for lies and mistaken beliefs shown as such, "
        "'unknown' when the issue does not settle it."
    )
    about: list[str] = Field(
        default=[], description="Characters the fact concerns, by name."
    )


class IssueFacts(BaseModel):
    facts: list[Fact]


class Event(BaseModel):
    at: str = Field(description="The panel where the event happens.")
    description: str = Field(
        description="One literal sentence in Italian, present tense."
    )
    participants: list[str] = Field(description="Characters who act or are acted on.")
    witnesses: list[str] = Field(
        default=[], description="Other characters who see or hear it happen."
    )
    evidence: list[str] = Field(
        default=[], description="Further panels or lettering showing the event."
    )


class KnowledgeChange(BaseModel):
    at: str = Field(
        description="The lettering or panel where the character comes to know, "
        "suspect, or reject the fact, or shows they already knew it."
    )
    character: str
    fact: str = Field(description="Id of an issue fact or a series fact.")
    stance: Stance = Field(
        description="believes: holds the fact as true (a false fact makes this a "
        "mistaken belief). suspects: considers it likely. disbelieves: rejects it."
    )
    source: KnowledgeSource = Field(
        description="witnessed: sees or hears it happen. told: someone says it to "
        "them. overheard: hears words not addressed to them (eavesdropping, "
        "broadcasts). inferred: deduces it. already_known: words or behavior show "
        "they knew it before."
    )
    informant: str | None = Field(
        default=None,
        description="Who told them, or whose words they overheard. Required for "
        "'told' and 'overheard'.",
    )
    evidence: list[str] = Field(default=[], description="Further supporting refs.")


class Scene(BaseModel):
    """A continuous stretch of panels in one place and time, as written by an agent."""

    start: str = Field(description="The first panel of the scene.")
    place: list[str] = Field(
        min_length=1,
        description="From general to specific, e.g. ['Paperopoli', 'Ducklair "
        "Tower', 'Sala di controllo'].",
    )
    time: str | None = Field(
        default=None,
        description="Time of day or relation to other scenes, when shown "
        "(e.g. 'notte', 'il mattino dopo').",
    )
    timeline: Timeline = Field(
        default="present",
        description="present, or a flashback, dream, or imagined scene.",
    )
    summary: str = Field(description="Two or three literal sentences in Italian.")
    events: list[Event] = []
    knowledge: list[KnowledgeChange] = []


class Presence(BaseModel):
    name: str
    personas: list[str] = Field(description="Personas depicted in the scene.")
    remote: bool = Field(description="Only seen or heard through screens or devices.")


class LoggedScene(Scene):
    id: str
    end: str = Field(description="The last panel of the scene.")
    present: list[Presence] = Field(
        description="Characters drawn or speaking in the scene, from the observations."
    )


class IssueLog(BaseModel):
    issue: str
    facts: list[Fact]
    scenes: list[LoggedScene]
    meta: dict[str, Any] = {}


class IssueIndex(BaseModel):
    """Valid references and character names of an issue, for validation."""

    panels: list[str] = Field(description="Panel references in reading order.")
    texts: list[str]
    characters: list[str]


def panel_ref(record: ObservationRecord, panel_index: int) -> str:
    return f"{Path(record.source.image).stem}#p{panel_index + 1}"


def issue_panels(
    records: Sequence[ObservationRecord],
) -> list[tuple[str, ObservedPanel]]:
    return [
        (panel_ref(r, pi), panel) for r in records for pi, panel in enumerate(r.panels)
    ]


def build_index(records: Sequence[ObservationRecord], cast: IssueCast) -> IssueIndex:
    panels = issue_panels(records)
    return IssueIndex(
        panels=[ref for ref, _ in panels],
        texts=[
            f"{ref}.t{ti}"
            for ref, panel in panels
            for ti in range(1, len(panel.texts) + 1)
        ],
        characters=[c.name for c in cast.characters],
    )


def _label_owners(cast: IssueCast) -> dict[str, tuple[str, str]]:
    """Map each label to its character name and persona name."""
    return {
        label: (c.name, p.name)
        for c in cast.characters
        for p in c.personas
        for label in p.labels
    }


def _display_names(cast: IssueCast) -> dict[str, str]:
    """Label to character name, with the persona when a character has several."""
    multi = {c.name for c in cast.characters if len(c.personas) > 1}
    return {
        label: f"{name} [{persona}]" if name in multi else name
        for label, (name, persona) in _label_owners(cast).items()
    }


def _presence_text(c: CharacterPresence, display: dict[str, str]) -> str:
    notes = [n for n in (DEPICTION_NOTES.get(c.depiction), c.expression) if n]
    return display[c.name] + (f" ({'; '.join(notes)})" if notes else "")


def _text_line(t: TextElement, display: dict[str, str]) -> str:
    if not t.speaker:
        return f'({TEXT_KIND_NAMES.get(t.kind, t.kind)}) "{t.text}"'
    to = f" → {', '.join(display[a] for a in t.addressees)}" if t.addressees else ""
    notes = [f"via {t.via}"] if t.via else []
    if t.kind == "thought":
        notes.append("pensiero")
    extra = f" ({'; '.join(notes)})" if notes else ""
    return f'{display[t.speaker]}{to}{extra}: "{t.text}"'


def render_log_transcript(records: Sequence[ObservationRecord], cast: IssueCast) -> str:
    """The issue's records with references and resolved character names."""
    display = _display_names(cast)
    out: list[str] = []
    for r in records:
        stem = Path(r.source.image).stem
        page = f" (p. {r.printed_page_number})" if r.printed_page_number else ""
        out += [f"## {stem}{page}", ""]
        for pi, panel in enumerate(r.panels):
            header = f"[{panel_ref(r, pi)}]"
            if panel.new_scene:
                cue = f" ({panel.transition_cue})" if panel.transition_cue else ""
                header += f" NUOVA SCENA{cue}."
            if panel.location:
                header += f" Luogo: {panel.location}."
            out.append(header)
            if panel.characters:
                present = "; ".join(
                    _presence_text(c, display) for c in panel.characters
                )
                out.append(f"  Presenti: {present}")
            out.append(f"  {panel.description}")
            out += [
                f"  t{ti} {_text_line(t, display)}"
                for ti, t in enumerate(panel.texts, 1)
                if t.kind != "sfx"
            ]
        out.append("")
    return "\n".join(out)


def _panel_of(ref: str) -> str | None:
    match = REF_PATTERN.match(ref)
    return match["panel"] if match else None


def log_problems(
    scenes: Sequence[Scene],
    facts: Sequence[Fact],
    index: IssueIndex,
    series_facts: set[str],
) -> list[str]:
    """References, names, and fact ids that do not match the issue."""
    position = {ref: i for i, ref in enumerate(index.panels)}
    refs = set(index.panels) | set(index.texts)
    names = set(index.characters)

    if not scenes:
        return ["no scenes"]
    starts = [position.get(s.start) for s in scenes]
    structure = [
        f"scene {n}: start {scene.start!r} is not a panel reference"
        for n, (scene, start) in enumerate(zip(scenes, starts), 1)
        if start is None
    ]
    if starts[0] is not None and starts[0] != 0:
        structure.append(f"scene 1: must start at the first panel, {index.panels[0]}")
    valid_starts = [s for s in starts if s is not None]
    if valid_starts != sorted(set(valid_starts)):
        structure.append("scenes: starts are not in reading order")
    if structure:
        return structure

    fact_ids = [f.id for f in facts]
    problems = [
        f"fact {fid}: defined more than once"
        for fid in sorted({f for f in fact_ids if fact_ids.count(f) > 1})
    ]
    for f in facts:
        problems += [
            f"fact {f.id}: unknown character {n!r}" for n in f.about if n not in names
        ]
    known_facts = set(fact_ids) | series_facts
    used_facts: set[str] = set()
    bounds = [*valid_starts, len(index.panels)]
    for n, scene in enumerate(scenes, 1):
        lo, hi = bounds[n - 1], bounds[n]

        def check_ref(ref: str, where: str) -> None:
            panel = _panel_of(ref)
            if ref not in refs or panel is None or not lo <= position[panel] < hi:
                problems.append(
                    f"{where}: {ref!r} is not a panel or lettering of this scene "
                    f"({index.panels[lo]} to {index.panels[hi - 1]})"
                )

        def check_names(names_used: Sequence[str | None], where: str) -> None:
            problems.extend(
                f"{where}: unknown character {name!r}"
                for name in names_used
                if name is not None and name not in names
            )

        for i, e in enumerate(scene.events, 1):
            where = f"scene {n} event {i}"
            for ref in [e.at, *e.evidence]:
                check_ref(ref, where)
            check_names([*e.participants, *e.witnesses], where)
        for i, k in enumerate(scene.knowledge, 1):
            where = f"scene {n} knowledge {i}"
            for ref in [k.at, *k.evidence]:
                check_ref(ref, where)
            check_names([k.character, k.informant], where)
            if k.fact not in known_facts:
                problems.append(f"{where}: unknown fact {k.fact!r}")
            used_facts.add(k.fact)
            if k.source in ("told", "overheard") and not k.informant:
                problems.append(f"{where}: source {k.source!r} needs an informant")
    problems += [
        f"fact {fid}: no knowledge change refers to it"
        for fid in fact_ids
        if fid not in used_facts
    ]
    return problems


def scene_presence(
    panels: Sequence[ObservedPanel], cast: IssueCast, include_memories: bool
) -> list[Presence]:
    """Characters drawn or speaking in the panels, merged by character.

    Pictures of a character don't count; memories count only in flashback,
    dream, and imagined scenes, where everyone is drawn as a memory.
    """
    owners = _label_owners(cast)
    present: dict[str, Presence] = {}

    def add(label: str, remote: bool) -> None:
        name, persona = owners[label]
        p = present.setdefault(name, Presence(name=name, personas=[], remote=True))
        if persona not in p.personas:
            p.personas.append(persona)
        p.remote = p.remote and remote

    for panel in panels:
        for c in panel.characters:
            if c.depiction == "image" or (
                c.depiction == "memory" and not include_memories
            ):
                continue
            add(c.name, remote=c.depiction == "on_screen")
        for t in panel.texts:
            if t.speaker:
                add(t.speaker, remote=t.via is not None)
    return list(present.values())


def assemble_log(
    issue: str,
    scenes: Sequence[Scene],
    facts: Sequence[Fact],
    records: Sequence[ObservationRecord],
    cast: IssueCast,
    meta: dict[str, Any],
) -> IssueLog:
    """Add ids, end panels, and presence to validated scenes."""
    panels = issue_panels(records)
    position = {ref: i for i, (ref, _) in enumerate(panels)}
    bounds = [position[s.start] for s in scenes] + [len(panels)]
    logged = [
        LoggedScene(
            **scene.model_dump(),
            id=f"s{n:02d}",
            end=panels[bounds[n] - 1][0],
            present=scene_presence(
                [panel for _, panel in panels[bounds[n - 1] : bounds[n]]],
                cast,
                include_memories=scene.timeline != "present",
            ),
        )
        for n, scene in enumerate(scenes, 1)
    ]
    return IssueLog(issue=issue, facts=list(facts), scenes=logged, meta=meta)
