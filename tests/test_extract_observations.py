"""Tests for extract_observations.py task building and orchestration."""

import json
from pathlib import Path

import pytest

from extract.extract_observations import (
    CAST_NOTES,
    PageObservation,
    PageTask,
    build_tasks,
    compute_config_id,
    output_path,
    process_tasks,
)
from pkna.extract.observations import (
    ObservationRecord,
    ObservedPanel,
    PageSource,
    TextElement,
)


def write_v2_page(v2_issue: Path, index: int, image: str, line: str) -> None:
    page = {
        "panels": [
            {"description": "d", "dialogues": [{"character": "Uno", "line": line}]}
        ],
        "characters_introduced": [],
        "meta": {"input_page_path": f"/x/{image}"},
    }
    (v2_issue / f"page_{index:03d}.json").write_text(json.dumps(page), encoding="utf-8")


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    pages_root = tmp_path / "pages"
    v2_root = tmp_path / "v2"
    (pages_root / "pkna-0").mkdir(parents=True)
    (v2_root / "pkna-0").mkdir(parents=True)
    for i in range(3):
        image = f"pkna-0-{i:03d}.jpg"
        (pages_root / "pkna-0" / image).touch()
        write_v2_page(v2_root / "pkna-0", i + 1, image, f"battuta {i}")
    return pages_root, v2_root


def make_task(image: str) -> PageTask:
    return PageTask(
        source=PageSource(issue="pkna-0", image=image, index=1),
        image_path=Path(image),
        cast=[],
        previous_dialogue=[],
    )


def fake_observe(task: PageTask) -> PageObservation:
    return PageObservation(
        printed_page_number=task.source.index,
        panels=[
            ObservedPanel(
                description=f"pagina {task.source.image}",
                texts=[TextElement(kind="speech", text="Ciao", speaker="Uno")],
            )
        ],
        unlisted_characters=[],
    )


class TestBuildTasks:
    def test_previous_dialogue_comes_from_preceding_image_when_filtering(
        self, roots: tuple[Path, Path]
    ):
        pages_root, v2_root = roots

        tasks = build_tasks("pkna-0", pages_root, v2_root, only_pages={"pkna-0-002"})

        assert [t.source.image for t in tasks] == ["pkna-0-002.jpg"]
        assert tasks[0].previous_dialogue == ["Uno: battuta 1"]

    def test_first_page_has_no_previous_dialogue(self, roots: tuple[Path, Path]):
        pages_root, v2_root = roots

        tasks = build_tasks("pkna-0", pages_root, v2_root)

        assert [t.previous_dialogue for t in tasks] == [
            [],
            ["Uno: battuta 0"],
            ["Uno: battuta 1"],
        ]
        uno = next(c for c in tasks[0].cast if c.name == "Uno")
        assert uno.appearance == CAST_NOTES["Uno"]


class TestProcessTasks:
    def test_writes_records_with_provenance(self, tmp_path: Path):
        task = make_task("pkna-0-001.jpg")

        result = process_tasks([task], fake_observe, tmp_path, "m", "cfg1", workers=2)

        record = ObservationRecord.model_validate_json(
            output_path(tmp_path, task.source).read_text(encoding="utf-8")
        )
        assert result == (1, 0)
        assert record.source == task.source
        assert record.panels[0].texts[0].text == "Ciao"
        assert record.meta["model_name"] == "m"
        assert record.meta["config_id"] == "cfg1"

    def test_skips_pages_with_matching_config_and_redoes_others(self, tmp_path: Path):
        tasks = [make_task("a.jpg"), make_task("b.jpg")]
        process_tasks(tasks[:1], fake_observe, tmp_path, "m", "cfg1")
        calls: list[str] = []

        def observe(task: PageTask) -> PageObservation:
            calls.append(task.source.image)
            return fake_observe(task)

        assert process_tasks(tasks, observe, tmp_path, "m", "cfg1") == (1, 0)
        assert calls == ["b.jpg"]
        assert process_tasks(tasks, observe, tmp_path, "m", "cfg2") == (2, 0)

    def test_failures_are_logged_and_do_not_stop_other_pages(self, tmp_path: Path):
        tasks = [make_task("bad.jpg"), make_task("good.jpg")]

        def observe(task: PageTask) -> PageObservation:
            if task.source.image == "bad.jpg":
                raise ValueError("parse error")
            return fake_observe(task)

        result = process_tasks(tasks, observe, tmp_path, "m", "cfg1")

        failures = [
            json.loads(line)
            for line in (tmp_path / "pkna-0" / "failures.jsonl")
            .read_text()
            .splitlines()
        ]
        assert result == (1, 1)
        assert [(f["image"], f["error"]) for f in failures] == [
            ("bad.jpg", "ValueError('parse error')")
        ]
        assert not output_path(tmp_path, tasks[0].source).exists()
        assert output_path(tmp_path, tasks[1].source).exists()


def test_config_id_depends_on_model():
    assert compute_config_id("a") == compute_config_id("a")
    assert compute_config_id("a") != compute_config_id("b")
