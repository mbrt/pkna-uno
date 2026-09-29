"""Tests for score_speakers.py."""

from extract.compare_observations import Line
from extract.score_speakers import (
    GoldLine,
    GoldSet,
    resolve_gold,
    resolve_predictions,
    score_speakers,
)
from pkna.extract.registry import IssueCast, ResolvedCharacter, merge_casts


def line(speaker: str, text: str) -> Line:
    return Line(speaker=speaker, text=text, ref="r")


GOLD = GoldSet(
    aliases={"Paperino": "Paperinik"},
    lines=[
        GoldLine(image="a.jpg", text="Piacere! Io sono Uno!", speaker="Uno"),
        GoldLine(image="a.jpg", text="Uno, eh?", speaker="Paperinik"),
        GoldLine(image="a.jpg", text="Sempre su...", speaker="*other*"),
        GoldLine(
            image="a.jpg", text="Vai pure a dormire.", speaker="Uno", uncertain=True
        ),
        GoldLine(image="b.jpg", text="Ciao", speaker="Uno"),
    ],
)


def test_aliases_and_other_labels_count_as_correct():
    predicted = {
        "a.jpg": [
            line("uno", "Piacere! Io sono Uno"),
            line("Paperino", "Uno, eh?"),
            line("Voce dalla TV", "Sempre su..."),
            line("Uno", "Vai pure a dormire."),
        ]
    }

    score = score_speakers(predicted, GOLD)

    assert score.pages == 1
    assert (score.certain.total, score.certain.correct) == (3, 3)
    assert (score.uncertain.total, score.uncertain.correct) == (1, 1)
    assert score.errors == []


def test_wrong_named_speaker_other_and_missing_lines_are_errors():
    predicted = {
        "a.jpg": [
            line("Paperinik", "Piacere! Io sono Uno!"),
            line("Uno", "Sempre su..."),
        ]
    }

    score = score_speakers(predicted, GOLD)

    assert (score.certain.total, score.certain.correct, score.certain.missing) == (
        3,
        0,
        1,
    )
    assert score.certain.unresolved == 0
    assert score.uncertain.missing == 1
    assert [(e["text"], e["got"]) for e in score.errors] == [
        ("Piacere! Io sono Uno!", "Paperinik"),
        ("Sempre su...", "Uno"),
        ("Uno, eh?", None),
        ("Vai pure a dormire.", None),
    ]


def test_descriptive_label_for_named_speaker_is_unresolved():
    predicted = {"b.jpg": [line("Sfera verde parlante", "Ciao")]}

    score = score_speakers(predicted, GOLD)

    assert (score.certain.correct, score.certain.unresolved) == (0, 1)


def test_registry_resolves_descriptive_labels_to_the_gold_character():
    registry = merge_casts(
        {
            "pkna-0": IssueCast(
                characters=[
                    ResolvedCharacter.model_validate(
                        {
                            "name": "Uno",
                            "kind": "named",
                            "personas": [
                                {
                                    "name": "Uno",
                                    "labels": ["Uno", "Sfera verde parlante"],
                                }
                            ],
                            "description": "d",
                        }
                    )
                ]
            )
        }
    )
    predicted = {"b.jpg": [line("Sfera verde parlante", "Ciao")]}

    score = score_speakers(
        resolve_predictions(predicted, "pkna-0", registry), resolve_gold(GOLD, registry)
    )

    assert (score.certain.correct, score.certain.unresolved) == (1, 0)
