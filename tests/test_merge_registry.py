"""Tests for merge_registry.py review helpers."""

from extract.merge_registry import format_review, similar_names
from pkna.extract.registry import Registry, RegistryCharacter, RegistryPersona


def named(cid: str, name: str, names: list[str] | None = None) -> RegistryCharacter:
    return RegistryCharacter(
        id=cid,
        name=name,
        kind="named",
        personas=[RegistryPersona(name=name, labels=[name])],
        issues=["pkna-0"],
        names=names or [name],
        description="d",
    )


def test_similar_names_flags_contained_and_near_identical_names():
    characters = [
        named("zondag", "Zondag"),
        named("generale-zondag", "Generale Zondag"),
        named("xadhoom", "Xadhoom"),
        named("xadhom", "Xadhom"),
        named("paperone", "Paperone"),
        named("paperon-de-paperoni", "Paperon de' Paperoni"),
        named("paperino", "Paperino"),
        named("qui", "Qui"),
        named("quo", "Quo"),
    ]

    assert similar_names(characters) == [
        ("zondag", "generale-zondag"),
        ("xadhoom", "xadhom"),
        ("paperone", "paperon-de-paperoni"),
    ]


def test_review_lists_merges_under_different_names():
    registry = Registry(
        characters=[named("paperino", "Paperino", ["Paolino Paperino", "Paperino"])],
        labels={"pkna-0": {}},
    )

    report = format_review(registry)

    assert "- `paperino`: Paolino Paperino, Paperino (1 issues)" in report
    assert "| `paperino` | Paperino | Paperino (1 labels) | 1 |" in report
