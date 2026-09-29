"""Tests for the character registry schemas and helpers."""

from pkna.extract.observations import (
    CastMember,
    CharacterPresence,
    ObservationRecord,
    ObservedPanel,
    PageSource,
    TextElement,
)
from pkna.extract.registry import (
    IssueCast,
    LabelRef,
    Overrides,
    ResolvedCharacter,
    cast_problems,
    collect_mentions,
    merge_casts,
    render_transcript,
)


def record(image: str, panels: list[ObservedPanel], **kwargs) -> ObservationRecord:
    return ObservationRecord(
        source=PageSource(issue="pkna-0", image=image, index=1),
        panels=panels,
        **kwargs,
    )


RECORDS = [
    record(
        "pkna-0-001.jpg",
        [
            ObservedPanel(
                description="Una sfera verde parla a un papero mascherato.",
                characters=[
                    CharacterPresence(name="Uno", depiction="hologram"),
                    CharacterPresence(name="Paperinik", depiction="in_person"),
                ],
                texts=[
                    TextElement(
                        kind="speech",
                        text="Piacere!",
                        speaker="Uno",
                        addressees=["Paperinik"],
                    ),
                    TextElement(kind="sfx", text="Zap"),
                ],
            )
        ],
        printed_page_number=2,
    ),
    record(
        "pkna-0-002.jpg",
        [
            ObservedPanel(
                new_scene=True,
                location="terrazza, di notte",
                description="Una papera dai capelli arancioni.",
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
                ],
            )
        ],
        unlisted_characters=[
            CastMember(name="Papera dai capelli arancioni", appearance="abito verde")
        ],
    ),
]


def test_collect_mentions_counts_usage_and_gathers_appearance():
    cast = [CastMember(name="Uno", appearance="sfera verde")]

    mentions = {m.label: m for m in collect_mentions(RECORDS, cast)}

    assert mentions["Uno"].model_dump() == {
        "label": "Uno",
        "lines": 1,
        "panels": 1,
        "addressed": 0,
        "page_count": 1,
        "pages": ["pkna-0-001"],
        "appearance": ["sfera verde"],
        "sample_lines": ["Piacere!"],
    }
    assert mentions["Paperinik"].addressed == 1
    assert mentions["Papera dai capelli arancioni"].appearance == ["abito verde"]


def test_render_transcript_shows_speakers_addressees_and_scene_changes():
    text = render_transcript(RECORDS)

    assert text.splitlines() == [
        "## pkna-0-001 (p. 2)",
        "[1] Personaggi: Uno (hologram), Paperinik.",
        "    Una sfera verde parla a un papero mascherato.",
        '    - Uno → Paperinik: "Piacere!"',
        "",
        "## pkna-0-002",
        "[1] NUOVA SCENA Luogo: terrazza, di notte. Personaggi: Papera dai capelli arancioni.",
        "    Una papera dai capelli arancioni.",
        '    - Papera dai capelli arancioni (pensiero): "Che notte!"',
        '    - (narration) "Intanto..."',
    ]


def character(
    name: str, *personas: tuple[str, list[str]], kind: str = "named"
) -> ResolvedCharacter:
    return ResolvedCharacter.model_validate(
        {
            "name": name,
            "kind": kind,
            "personas": [{"name": p, "labels": labels} for p, labels in personas],
            "description": "d",
        }
    )


CASTS = {
    "pkna-0": IssueCast(
        characters=[
            character(
                "Paolino Paperino",
                ("Paperino", ["Paperino"]),
                ("Paperinik", ["Paperinik", "Papero mascherato"]),
            ),
            character("Zondag", ("Zondag", ["Zondag"])),
            character(
                "Soldato evroniano",
                ("Soldato evroniano", ["Soldato evroniano"]),
                kind="unnamed",
            ),
        ]
    ),
    "pkna-1": IssueCast(
        characters=[
            character("Paperinik", ("Paperinik", ["Paperinik", "PK"])),
            character("Generale Zondag", ("Generale Zondag", ["Generale Zondag"])),
            character(
                "Soldato evroniano",
                ("Soldato evroniano", ["Soldato Evroniano"]),
                kind="unnamed",
            ),
        ]
    ),
}


class TestMergeCasts:
    def test_links_named_characters_by_persona_names_and_keeps_extras_per_issue(self):
        registry = merge_casts(CASTS, preferred_names={"Paperino"})

        paperino = registry.find("Paperinik")
        assert paperino is not None
        assert (paperino.id, paperino.name, paperino.issues) == (
            "paperino",
            "Paperino",
            ["pkna-0", "pkna-1"],
        )
        assert paperino.names == ["Paolino Paperino", "Paperinik"]
        assert {p.name: p.labels for p in paperino.personas} == {
            "Paperino": ["Paperino"],
            "Paperinik": ["PK", "Paperinik", "Papero mascherato"],
        }
        assert registry.resolve("pkna-1", "PK") == LabelRef(
            id="paperino", persona="Paperinik"
        )
        assert registry.resolve("pkna-0", "Soldato evroniano") == LabelRef(
            id="pkna-0:soldato-evroniano", persona="Soldato evroniano"
        )
        assert registry.resolve("pkna-1", "Soldato Evroniano") == LabelRef(
            id="pkna-1:soldato-evroniano", persona="Soldato evroniano"
        )
        assert registry.find("Soldato evroniano") is None
        assert registry.find("Generale Zondag") != registry.find("Zondag")
        assert registry.member("pkna-0", "Paolino Paperino") == "paperino"
        assert registry.member("pkna-1", "Paperinik") == "paperino"
        assert registry.member("pkna-1", "Soldato evroniano") == (
            "pkna-1:soldato-evroniano"
        )

    def test_most_complete_name_wins_over_most_common(self):
        casts = {
            "pkna-0": IssueCast(characters=[character("Lyla Lay", ("Lyla", ["Lyla"]))]),
            "pkna-1": IssueCast(characters=[character("Lyla", ("Lyla", ["Lyla"]))]),
            "pkna-2": IssueCast(characters=[character("Lyla", ("Lyla", ["Lyla"]))]),
        }

        registry = merge_casts(casts)

        assert [(c.id, c.name) for c in registry.characters] == [
            ("lyla-lay", "Lyla Lay")
        ]

    def test_characters_distinct_in_one_issue_are_never_merged(self):
        casts = {
            "pkna-11": IssueCast(
                characters=[
                    character(
                        "Urk", ("Urk", ["Urk"]), ("Dexter Brundle", ["Dexter Brundle"])
                    ),
                    character("Dexter Brundle", ("Dexter Brundle", ["Vero Dexter"])),
                ]
            ),
            "pkna-12": IssueCast(characters=[character("Urk", ("Urk", ["Urk"]))]),
        }

        registry = merge_casts(casts)

        assert registry.member("pkna-11", "Urk") == registry.member("pkna-12", "Urk")
        assert registry.member("pkna-11", "Dexter Brundle") == "dexter-brundle"
        assert registry.kept_apart == [("dexter-brundle", "urk")]

    def test_overrides_merge_names_the_pages_spell_differently(self):
        registry = merge_casts(CASTS, Overrides(merge=[["Generale Zondag", "Zondag"]]))

        zondag = registry.find("Generale Zondag")
        assert zondag is not None
        assert zondag == registry.find("Zondag")
        assert zondag.issues == ["pkna-0", "pkna-1"]
        assert [(p.name, p.labels) for p in zondag.personas] == [
            ("Zondag", ["Generale Zondag", "Zondag"])
        ]
        assert registry.resolve("pkna-1", "Generale Zondag") == LabelRef(
            id="generale-zondag", persona="Zondag"
        )


def test_cast_problems_accepts_a_complete_assignment_with_personas():
    cast = IssueCast(
        characters=[
            character("Paperino", ("Paperino", ["Paperino"]), ("Paperinik", ["PK"])),
            character("Uno", ("Uno", ["Uno"])),
        ]
    )

    assert cast_problems(cast, {"Paperino", "PK", "Uno"}) == []


def test_cast_problems_reports_duplicate_unknown_and_missing_labels():
    cast = IssueCast(
        characters=[
            character("Uno", ("Uno", ["Uno", "Sfera"])),
            character("Sfera verde", ("Sfera verde", ["Sfera"])),
            character("Vuoto"),
        ]
    )

    assert cast_problems(cast, {"Uno", "Paperinik"}) == [
        "character 'Vuoto' has no personas",
        "label 'Sfera' is assigned to several characters: ['Uno', 'Sfera verde']",
        "label 'Sfera' is not in the records",
        "label 'Paperinik' is not assigned to any character",
    ]
