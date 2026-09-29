"""Tests for resolve_characters_agent.py workspaces, validation, and resume."""

import json
from pathlib import Path

import pytest
from PIL import Image

from extract.resolve_characters_agent import (
    complete_issues,
    input_hash,
    is_done,
    issue_inputs,
    output_path,
    resolve_issue,
)
from pkna.extract.observations import (
    CharacterPresence,
    ObservationRecord,
    ObservedPanel,
    PageSource,
    TextElement,
)
from pkna.extract.registry import IssueCast, Persona, ResolvedCharacter


def page_record(index: int, speaker: str) -> ObservationRecord:
    return ObservationRecord(
        source=PageSource(issue="pkna-0", image=f"pkna-0-{index:03d}.jpg", index=index),
        panels=[
            ObservedPanel(
                description="d",
                characters=[CharacterPresence(name=speaker, depiction="in_person")],
                texts=[TextElement(kind="speech", text="Ciao", speaker=speaker)],
            )
        ],
    )


RECORDS = [page_record(1, "Paperinik"), page_record(2, "Papero mascherato")]


@pytest.fixture
def pages(tmp_path: Path) -> list[Path]:
    issue_dir = tmp_path / "pages" / "pkna-0"
    issue_dir.mkdir(parents=True)
    paths = [issue_dir / f"pkna-0-{i:03d}.jpg" for i in (1, 2)]
    for path in paths:
        Image.new("RGB", (10, 10)).save(path)
    return paths


@pytest.fixture
def series(tmp_path: Path) -> Path:
    path = tmp_path / "series.json"
    path.write_text('{"identities": []}')
    return path


def valid_cast() -> IssueCast:
    return IssueCast(
        characters=[
            ResolvedCharacter(
                name="Paperino",
                kind="named",
                personas=[
                    Persona(name="Paperinik", labels=["Paperinik", "Papero mascherato"])
                ],
                description="papero in costume",
                evidence="Stesso costume a pkna-0-001 e pkna-0-002.",
            )
        ]
    )


def writing(cast: IssueCast, seen: list[list[str]] | None = None):
    def runner(workspace: Path, model: str) -> dict:
        if seen is not None:
            seen.append(
                sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*.*"))
            )
        (workspace / "out" / "cast.json").write_text(cast.model_dump_json())
        return {"session_id": "s1"}

    return runner


def test_valid_resolution_is_written_with_provenance_and_resumable(
    tmp_path: Path, pages: list[Path], series: Path
):
    out_root = tmp_path / "out"
    seen: list[list[str]] = []

    ok = resolve_issue(
        "pkna-0",
        RECORDS,
        pages,
        writing(valid_cast(), seen),
        "m",
        out_root,
        "cfg",
        v2_root=tmp_path / "v2",
        series_path=series,
    )

    result = json.loads(output_path(out_root, "pkna-0").read_text())
    inputs_id = input_hash(*issue_inputs(RECORDS, tmp_path / "v2" / "pkna-0"))
    assert ok
    assert seen == [
        [
            "INSTRUCTIONS.md",
            "mentions.json",
            "pages/pkna-0-001.jpg",
            "pages/pkna-0-002.jpg",
            "schema.json",
            "series.json",
            "transcript.md",
        ]
    ]
    assert IssueCast.model_validate(result["cast"]) == valid_cast()
    assert result["meta"]["agent"] == {"session_id": "s1"}
    assert list((out_root / "_work").iterdir()) == []
    assert is_done(output_path(out_root, "pkna-0"), "cfg", inputs_id)
    assert not is_done(output_path(out_root, "pkna-0"), "cfg", "other-input")
    assert not is_done(output_path(out_root, "pkna-0"), "other-cfg", inputs_id)


def test_resolution_missing_a_label_is_a_failure_and_keeps_workspace(
    tmp_path: Path, pages: list[Path], series: Path
):
    out_root = tmp_path / "out"
    cast = valid_cast()
    cast.characters[0].personas[0].labels = ["Paperinik"]

    ok = resolve_issue(
        "pkna-0",
        RECORDS,
        pages,
        writing(cast),
        "m",
        out_root,
        "cfg",
        v2_root=tmp_path / "v2",
        series_path=series,
    )

    [failure] = [
        json.loads(line)
        for line in (out_root / "failures.jsonl").read_text().splitlines()
    ]
    assert not ok
    assert not output_path(out_root, "pkna-0").exists()
    assert failure["issue"] == "pkna-0"
    assert "'Papero mascherato' is not assigned" in failure["error"]
    assert (out_root / "_work" / "pkna-0" / "out" / "cast.json").exists()


def test_complete_issues_skips_issues_with_missing_pages(
    tmp_path: Path, pages: list[Path]
):
    obs_root = tmp_path / "obs"
    (obs_root / "pkna-0").mkdir(parents=True)
    (obs_root / "pkna-0" / "pkna-0-001.json").write_text(RECORDS[0].model_dump_json())

    assert complete_issues(["pkna-0"], tmp_path / "pages", obs_root) == {}

    (obs_root / "pkna-0" / "pkna-0-002.json").write_text(RECORDS[1].model_dump_json())
    ready = complete_issues(["pkna-0"], tmp_path / "pages", obs_root)

    assert [r.source.image for r in ready["pkna-0"][0]] == [
        "pkna-0-001.jpg",
        "pkna-0-002.jpg",
    ]
    assert ready["pkna-0"][1] == pages
