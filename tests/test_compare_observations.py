"""Tests for compare_observations.py alignment and summary metrics."""

from extract.compare_observations import (
    Line,
    align_lines,
    compare_page,
    normalize_text,
    silent_presences,
    summarize,
)
from pkna.extract.observations import (
    CharacterPresence,
    ObservationRecord,
    ObservedPanel,
    PageSource,
    TextElement,
)


def line(speaker: str, text: str) -> Line:
    return Line(speaker=speaker, text=text, ref=f"{speaker}:{text}")


def record(panels: list[ObservedPanel], image: str = "p.jpg") -> ObservationRecord:
    return ObservationRecord(
        source=PageSource(issue="pkna-0", image=image, index=1), panels=panels
    )


def v2_page(panels: list[tuple[str, list[tuple[str, str]]]]) -> dict:
    return {
        "panels": [
            {
                "is_new_scene": False,
                "description": desc,
                "dialogues": [{"character": c, "line": t} for c, t in dialogues],
            }
            for desc, dialogues in panels
        ]
    }


def test_normalize_text_ignores_case_accents_and_punctuation():
    assert normalize_text("Perché?! Sul  TETTO...") == normalize_text(
        "perche sul tetto"
    )


class TestAlignLines:
    def test_pairs_similar_lines_one_to_one(self):
        new = [line("Uno", "Cosa? Sul tetto?!"), line("Uno", "Dammi retta.")]
        old = [line("Uno", "Dammi retta!"), line("Uno", "Cosa? Sul tetto?")]

        pairs, only_new, only_old = align_lines(new, old)

        assert [(n.text, o.text) for n, o in pairs] == [
            ("Cosa? Sul tetto?!", "Cosa? Sul tetto?"),
            ("Dammi retta.", "Dammi retta!"),
        ]
        assert only_new == []
        assert only_old == []

    def test_dissimilar_and_surplus_lines_stay_unmatched(self):
        new = [line("Uno", "Meglio di no!"), line("Uno", "Meglio di no!")]
        old = [line("Uno", "Meglio di no!"), line("TV", "Sempre su...")]

        pairs, only_new, only_old = align_lines(new, old)

        assert len(pairs) == 1
        assert [n.text for n in only_new] == ["Meglio di no!"]
        assert [o.text for o in only_old] == ["Sempre su..."]


def test_compare_page_reports_speaker_mismatch_and_unmatched_lines():
    rec = record(
        [
            ObservedPanel(
                description="Paperinik davanti alla TV.",
                texts=[
                    TextElement(
                        kind="speech",
                        text="Sempre su...",
                        speaker="Voce dalla TV",
                        via="TV",
                    ),
                    TextElement(kind="sfx", text="Click"),
                    TextElement(kind="thought", text="Che noia", speaker="Paperinik"),
                ],
            )
        ]
    )
    v2 = v2_page([("desc", [("Uno", "Sempre su..."), ("Paperinik", "Ehi!")])])

    result = compare_page(rec, v2)

    assert result.matched == 1
    assert result.speaker_mismatches == [
        {
            "ref": "pkna-0/p#p1.t1",
            "text": "Sempre su...",
            "new": "Voce dalla TV",
            "v2": "Uno",
        }
    ]
    assert [n["text"] for n in result.only_new] == ["Che noia"]
    assert [o["text"] for o in result.only_v2] == ["Ehi!"]


def test_silent_presences_count_drawn_characters_who_do_not_speak():
    rec = record(
        [
            ObservedPanel(
                description="d",
                characters=[
                    CharacterPresence(name="Uno", depiction="hologram"),
                    CharacterPresence(name="Paperinik", depiction="in_person"),
                    CharacterPresence(name="Lyla", depiction="off_panel"),
                ],
                texts=[TextElement(kind="speech", text="Ciao", speaker="Uno")],
            )
        ]
    )

    assert dict(silent_presences(rec)) == {"Paperinik": 1}


def test_summarize_counts_leakage_heuristics_for_both_extractions():
    rec = record(
        [
            ObservedPanel(description="Paperinik si allontana verso l'uscita."),
            ObservedPanel(description="La sfera di Uno si ingrandisce."),
        ]
    )
    v2 = v2_page(
        [
            ("Uno risponde con sarcasmo.", []),
            ("Paperinik, ignaro della vera minaccia, sorride.", []),
        ]
    )

    summary = summarize([rec], {"p.jpg": v2})

    assert summary["descriptions_paraphrasing_speech"] == {"new": 0, "v2": 1}
    assert summary["descriptions_interpretive"] == {"new": 0, "v2": 1}
    assert summary["panels"] == {"new": 2, "v2": 2}
