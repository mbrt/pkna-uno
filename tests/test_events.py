"""Tests for the event and knowledge log schemas and helpers."""

from pkna.extract.events import (
    Event,
    Fact,
    KnowledgeChange,
    Presence,
    Scene,
    assemble_log,
    build_index,
    log_problems,
    render_log_transcript,
)
from pkna.extract.observations import (
    CharacterPresence,
    ObservationRecord,
    ObservedPanel,
    PageSource,
    TextElement,
)
from pkna.extract.registry import IssueCast, Persona, ResolvedCharacter


def record(image: str, panels: list[ObservedPanel], **kwargs) -> ObservationRecord:
    return ObservationRecord(
        source=PageSource(issue="pkna-0", image=image, index=1), panels=panels, **kwargs
    )


RECORDS = [
    record(
        "pkna-0-001.jpg",
        [
            ObservedPanel(
                description="Una sfera verde parla a un papero mascherato.",
                characters=[
                    CharacterPresence(name="Uno", depiction="hologram"),
                    CharacterPresence(
                        name="Paperinik", depiction="in_person", expression="sorpreso"
                    ),
                ],
                texts=[
                    TextElement(
                        kind="speech",
                        text="Io so chi sei!",
                        speaker="Uno",
                        addressees=["Paperinik"],
                    ),
                    TextElement(kind="sfx", text="Zap"),
                ],
            ),
            ObservedPanel(
                description="La sfera mostra un'immagine di una papera.",
                characters=[
                    CharacterPresence(name="Uno", depiction="hologram"),
                    CharacterPresence(
                        name="Papera dai capelli arancioni", depiction="image"
                    ),
                ],
            ),
        ],
        printed_page_number=2,
    ),
    record(
        "pkna-0-002.jpg",
        [
            ObservedPanel(
                new_scene=True,
                transition_cue="Intanto...",
                location="terrazza",
                description="Una papera sulla terrazza guarda la TV.",
                characters=[
                    CharacterPresence(
                        name="Papera dai capelli arancioni", depiction="in_person"
                    )
                ],
                texts=[
                    TextElement(
                        kind="thought",
                        text="Che notte!",
                        speaker="Papera dai capelli arancioni",
                    ),
                    TextElement(kind="narration", text="Intanto..."),
                    TextElement(
                        kind="speech", text="Notizie!", speaker="Paperino", via="TV"
                    ),
                ],
            )
        ],
    ),
]


def character(name: str, *personas: tuple[str, list[str]]) -> ResolvedCharacter:
    return ResolvedCharacter(
        name=name,
        kind="named",
        personas=[Persona(name=p, labels=labels) for p, labels in personas],
        description="d",
    )


CAST = IssueCast(
    characters=[
        character(
            "Paolino Paperino", ("Paperino", ["Paperino"]), ("Paperinik", ["Paperinik"])
        ),
        character("Uno", ("Uno", ["Uno"])),
        character(
            "Paperilla Starry", ("Paperilla Starry", ["Papera dai capelli arancioni"])
        ),
    ]
)

INDEX = build_index(RECORDS, CAST)
SERIES = {"serie:identita-paperinik"}

FACTS = [
    Fact(id="f01", statement="Uno conosce Paperinik.", truth="true", about=["Uno"])
]


def valid_scenes() -> list[Scene]:
    return [
        Scene(
            start="pkna-0-001#p1",
            place=["Paperopoli", "Ducklair Tower"],
            summary="Uno parla con Paperinik.",
            events=[
                Event(
                    at="pkna-0-001#p1",
                    description="Uno appare a Paperinik.",
                    participants=["Uno", "Paolino Paperino"],
                )
            ],
            knowledge=[
                KnowledgeChange(
                    at="pkna-0-001#p1.t1",
                    character="Uno",
                    fact="serie:identita-paperinik",
                    stance="believes",
                    source="already_known",
                ),
                KnowledgeChange(
                    at="pkna-0-001#p1.t1",
                    character="Paolino Paperino",
                    fact="f01",
                    stance="believes",
                    source="told",
                    informant="Uno",
                    evidence=["pkna-0-001#p2"],
                ),
            ],
        ),
        Scene(
            start="pkna-0-002#p1", place=["Terrazza"], summary="Paperilla guarda la TV."
        ),
    ]


def test_transcript_shows_references_resolved_names_and_personas():
    assert render_log_transcript(RECORDS, CAST).splitlines() == [
        "## pkna-0-001 (p. 2)",
        "",
        "[pkna-0-001#p1]",
        "  Presenti: Uno (ologramma); Paolino Paperino [Paperinik] (sorpreso)",
        "  Una sfera verde parla a un papero mascherato.",
        '  t1 Uno → Paolino Paperino [Paperinik]: "Io so chi sei!"',
        "[pkna-0-001#p2]",
        "  Presenti: Uno (ologramma); Paperilla Starry (immagine)",
        "  La sfera mostra un'immagine di una papera.",
        "",
        "## pkna-0-002",
        "",
        "[pkna-0-002#p1] NUOVA SCENA (Intanto...). Luogo: terrazza.",
        "  Presenti: Paperilla Starry",
        "  Una papera sulla terrazza guarda la TV.",
        '  t1 Paperilla Starry (pensiero): "Che notte!"',
        '  t2 (didascalia) "Intanto..."',
        '  t3 Paolino Paperino [Paperino] (via TV): "Notizie!"',
    ]


def test_valid_log_has_no_problems():
    assert log_problems(valid_scenes(), FACTS, INDEX, SERIES) == []


def test_scenes_must_cover_the_issue_in_order():
    first, second = valid_scenes()

    assert log_problems([second, first], FACTS, INDEX, SERIES) == [
        "scene 1: must start at the first panel, pkna-0-001#p1",
        "scenes: starts are not in reading order",
    ]
    assert log_problems(
        [first.model_copy(update={"start": "pkna-0-009#p1"})], FACTS, INDEX, SERIES
    ) == ["scene 1: start 'pkna-0-009#p1' is not a panel reference"]


def test_references_names_and_facts_are_checked_per_scene():
    first, second = valid_scenes()
    first.events[0].participants.append("Paperinik")
    first.knowledge[1].informant = None
    first.knowledge[0].fact = "f99"
    second.events.append(
        Event(at="pkna-0-001#p2", description="x", participants=["Paperilla Starry"])
    )

    assert log_problems([first, second], FACTS, INDEX, SERIES) == [
        "scene 1 event 1: unknown character 'Paperinik'",
        "scene 1 knowledge 1: unknown fact 'f99'",
        "scene 1 knowledge 2: source 'told' needs an informant",
        "scene 2 event 1: 'pkna-0-001#p2' is not a panel or lettering of this "
        "scene (pkna-0-002#p1 to pkna-0-002#p1)",
    ]


def test_unused_and_duplicate_facts_are_problems():
    facts = [*FACTS, *FACTS, Fact(id="f02", statement="x", truth="unknown")]

    assert log_problems(valid_scenes(), facts, INDEX, SERIES) == [
        "fact f01: defined more than once",
        "fact f02: no knowledge change refers to it",
    ]


def test_assembled_log_adds_ids_ends_and_presence():
    log = assemble_log("pkna-0", valid_scenes(), FACTS, RECORDS, CAST, {"k": "v"})

    assert [(s.id, s.start, s.end) for s in log.scenes] == [
        ("s01", "pkna-0-001#p1", "pkna-0-001#p2"),
        ("s02", "pkna-0-002#p1", "pkna-0-002#p1"),
    ]
    assert log.scenes[0].present == [
        Presence(name="Uno", personas=["Uno"], remote=False),
        Presence(name="Paolino Paperino", personas=["Paperinik"], remote=False),
    ]
    assert log.scenes[1].present == [
        Presence(name="Paperilla Starry", personas=["Paperilla Starry"], remote=False),
        Presence(name="Paolino Paperino", personas=["Paperino"], remote=True),
    ]
    assert log.meta == {"k": "v"}
