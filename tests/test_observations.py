"""Unit tests for the observation layer schema and helpers."""

import json
from pathlib import Path

from PIL import Image

from pkna.extract.observations import (
    CastMember,
    ObservationRecord,
    ObservedPanel,
    PageSource,
    TextElement,
    apply_cast_notes,
    build_cast_sheet,
    list_issue_pages,
    load_v2_pages,
    page_strips,
    previous_page_dialogue,
)


def v2_page(
    image: str,
    dialogues: list[list[tuple[str, str]]],
    introduced: list[tuple[str, str]] | None = None,
) -> dict:
    return {
        "summary": "Riassunto con la trama completa.",
        "panels": [
            {
                "is_new_scene": False,
                "description": "Descrizione scritta conoscendo la trama.",
                "dialogues": [
                    {"character": c, "line": line, "tone": "neutral"}
                    for c, line in panel
                ],
            }
            for panel in dialogues
        ],
        "characters_introduced": [
            {"name": n, "appearance": a} for n, a in (introduced or [])
        ],
        "meta": {"input_page_path": f"/abs/input/pkna/pkna-0/{image}"},
    }


class TestListIssuePages:
    def test_sorted_with_one_based_index_and_case_insensitive_suffix(
        self, tmp_path: Path
    ):
        issue = tmp_path / "pkna-40"
        issue.mkdir()
        for name in ["02.jpg", "01.jpeg", "03.JPG", "notes.txt"]:
            (issue / name).touch()

        pages = list_issue_pages(issue)

        assert [(s.image, s.index, p.name) for s, p in pages] == [
            ("01.jpeg", 1, "01.jpeg"),
            ("02.jpg", 2, "02.jpg"),
            ("03.JPG", 3, "03.JPG"),
        ]
        assert all(s.issue == "pkna-40" for s, _ in pages)


def test_page_strips_overlap_cover_the_page_and_share_a_width():
    page = Image.new("RGB", (100, 300))

    strips = page_strips(page, count=3, overlap=0.1, width=200)

    # Strip bounds are [0, 110], [90, 210], [190, 300] before scaling by 2.
    assert [s.size for s in strips] == [(200, 220), (200, 240), (200, 220)]


class TestObservationRecordIds:
    def test_ids_use_image_stem_and_one_based_positions(self):
        record = ObservationRecord(
            source=PageSource(issue="pkna-0", image="pkna-0-070.jpg", index=71),
            panels=[
                ObservedPanel(
                    description="a",
                    texts=[TextElement(kind="speech", text="x", speaker="Uno")],
                ),
                ObservedPanel(
                    description="b",
                    texts=[
                        TextElement(kind="sfx", text="Tap"),
                        TextElement(kind="speech", text="y", speaker="Paperinik"),
                    ],
                ),
            ],
        )

        ids = [(text_id, t.text) for text_id, _, t in record.iter_texts()]

        assert record.source.page_id == "pkna-0/pkna-0-070"
        assert ids == [
            ("pkna-0/pkna-0-070#p1.t1", "x"),
            ("pkna-0/pkna-0-070#p2.t1", "Tap"),
            ("pkna-0/pkna-0-070#p2.t2", "y"),
        ]

    def test_json_round_trip(self):
        record = ObservationRecord(
            source=PageSource(issue="pkna-0", image="p.jpg", index=1),
            printed_page_number=71,
            panels=[ObservedPanel(description="Uno appare.", location="151° piano")],
            unlisted_characters=[CastMember(name="Voce dalla TV")],
            meta={"config_id": "abc"},
        )

        restored = ObservationRecord.model_validate_json(record.model_dump_json())

        assert restored == record


class TestLoadV2Pages:
    def test_keys_by_image_file_name_not_page_file_index(self, tmp_path: Path):
        page = v2_page("pkna-0-070.jpg", [[("Uno", "Meglio di no!")]])
        (tmp_path / "page_071.json").write_text(json.dumps(page), encoding="utf-8")

        pages = load_v2_pages(tmp_path)

        assert list(pages) == ["pkna-0-070.jpg"]


class TestBuildCastSheet:
    def test_introduced_characters_first_then_other_speakers(self):
        pages = [
            v2_page("1.jpg", [[("Paperinik", "Chi va là?")]], [("Uno", "Sfera verde")]),
            v2_page(
                "2.jpg",
                [[("Uno", "Salve."), ("Paperinik", "Ciao.")]],
                [("Uno", "Altra descrizione"), ("Lyla", "Androide")],
            ),
        ]

        cast = build_cast_sheet(pages)

        assert cast == [
            CastMember(name="Uno", appearance="Sfera verde"),
            CastMember(name="Lyla", appearance="Androide"),
            CastMember(name="Paperinik", appearance=None),
        ]


class TestApplyCastNotes:
    def test_appends_to_existing_members_and_adds_missing_ones(self):
        cast = [
            CastMember(name="Uno", appearance="Sfera verde."),
            CastMember(name="Paperinik"),
            CastMember(name="Lyla", appearance="Androide."),
        ]
        notes = {
            "Uno": "Balloon a punte.",
            "Paperinik": "Mantello blu.",
            "Due": "Voce metallica.",
        }

        result = apply_cast_notes(cast, notes)

        assert result == [
            CastMember(name="Uno", appearance="Sfera verde. Balloon a punte."),
            CastMember(name="Paperinik", appearance="Mantello blu."),
            CastMember(name="Lyla", appearance="Androide."),
            CastMember(name="Due", appearance="Voce metallica."),
        ]


class TestPreviousPageDialogue:
    def test_keeps_only_speaker_and_line_of_last_panels(self):
        page = v2_page(
            "1.jpg",
            [[("A", "uno")], [("B", "due")], [("C", "tre"), ("D", "quattro")]],
        )

        lines = previous_page_dialogue(page, max_panels=2)

        assert lines == ["B: due", "C: tre", "D: quattro"]

    def test_no_previous_page(self):
        assert previous_page_dialogue(None) == []
