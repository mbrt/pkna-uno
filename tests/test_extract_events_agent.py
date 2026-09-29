"""Tests for extract_events_agent.py workspaces, validation, and resume."""

import json
from pathlib import Path

import pytest

from extract.event_agent_tools import validate_log
from extract.extract_events_agent import (
    input_hash,
    is_done,
    issue_inputs,
    log_issue,
    output_path,
    ready_issues,
)
from pkna.extract.events import Fact, IssueFacts, IssueLog, Scene
from tests.test_events import CAST, FACTS, RECORDS, valid_scenes


@pytest.fixture
def series(tmp_path: Path) -> Path:
    path = tmp_path / "series-facts.json"
    path.write_text(
        IssueFacts(
            facts=[
                Fact(
                    id="serie:identita-paperinik",
                    statement="Paolino Paperino è Paperinik.",
                    truth="true",
                )
            ]
        ).model_dump_json()
    )
    return path


def writing(
    scenes: list[Scene], facts: list[Fact], seen: list[list[str]] | None = None
):
    def runner(workspace: Path, model: str) -> dict:
        if seen is not None:
            seen.append(
                sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*.*"))
            )
        out = workspace / "out"
        (out / "facts.json").write_text(IssueFacts(facts=facts).model_dump_json())
        for n, scene in enumerate(scenes, 1):
            (out / "scenes" / f"{n:03d}.json").write_text(scene.model_dump_json())
        return {"session_id": "s1"}

    return runner


def test_valid_log_is_written_with_provenance_and_resumable(
    tmp_path: Path, series: Path
):
    out_root = tmp_path / "out"
    seen: list[list[str]] = []

    ok = log_issue(
        "pkna-0",
        RECORDS,
        CAST,
        writing(valid_scenes(), FACTS, seen),
        "m",
        out_root,
        "cfg",
        series_path=series,
    )

    result = IssueLog.model_validate_json(output_path(out_root, "pkna-0").read_text())
    inputs_id = input_hash(*issue_inputs(RECORDS, CAST))
    assert ok
    assert seen == [
        [
            "INSTRUCTIONS.md",
            "characters.json",
            "facts-schema.json",
            "index.json",
            "scene-schema.json",
            "series-facts.json",
            "transcript.md",
        ]
    ]
    assert [(s.id, s.end, s.summary) for s in result.scenes] == [
        ("s01", "pkna-0-001#p2", "Uno parla con Paperinik."),
        ("s02", "pkna-0-002#p1", "Paperilla guarda la TV."),
    ]
    assert result.facts == FACTS
    assert result.meta["agent"] == {"session_id": "s1"}
    assert list((out_root / "_work").iterdir()) == []
    assert is_done(output_path(out_root, "pkna-0"), "cfg", inputs_id)
    assert not is_done(output_path(out_root, "pkna-0"), "cfg", "other-input")
    assert not is_done(output_path(out_root, "pkna-0"), "other-cfg", inputs_id)


def test_invalid_log_is_a_failure_and_keeps_workspace(tmp_path: Path, series: Path):
    out_root = tmp_path / "out"
    scenes = valid_scenes()
    scenes[0].knowledge[0].character = "Pikappa"

    ok = log_issue(
        "pkna-0",
        RECORDS,
        CAST,
        writing(scenes, FACTS),
        "m",
        out_root,
        "cfg",
        series_path=series,
    )

    [failure] = [
        json.loads(line)
        for line in (out_root / "failures.jsonl").read_text().splitlines()
    ]
    assert not ok
    assert not output_path(out_root, "pkna-0").exists()
    assert failure["error"] == "scene 1 knowledge 1: unknown character 'Pikappa'"
    assert (out_root / "_work" / "pkna-0" / "out" / "scenes" / "001.json").exists()


def test_validate_log_reports_unparsable_files(tmp_path: Path, series: Path):
    out = tmp_path / "out"
    (out / "scenes").mkdir(parents=True)
    (out / "scenes" / "001.json").write_text('{"start": "pkna-0-001#p1"}')

    problems = validate_log(out, tmp_path / "index.json", series)

    assert [p.split(":")[0] for p in problems] == ["facts.json", "scenes/001.json"]


def test_ready_issues_need_complete_observations_and_a_resolution(tmp_path: Path):
    pages_root = tmp_path / "pages"
    obs_root = tmp_path / "obs"
    casts_root = tmp_path / "casts"
    for root in (pages_root / "pkna-0", obs_root / "pkna-0", casts_root):
        root.mkdir(parents=True)
    for r in RECORDS:
        (pages_root / "pkna-0" / r.source.image).write_bytes(b"")
        stem = Path(r.source.image).stem
        (obs_root / "pkna-0" / f"{stem}.json").write_text(r.model_dump_json())

    assert ready_issues(["pkna-0"], pages_root, obs_root, casts_root) == {}

    (casts_root / "pkna-0.json").write_text(
        json.dumps({"issue": "pkna-0", "cast": CAST.model_dump()})
    )
    ready = ready_issues(["pkna-0"], pages_root, obs_root, casts_root)

    assert list(ready) == ["pkna-0"]
    assert ready["pkna-0"][1] == CAST
