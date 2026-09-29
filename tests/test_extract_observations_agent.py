"""Tests for extract_observations_agent.py batching, workspaces, and collection."""

import json
from pathlib import Path

import pytest
from PIL import Image

from extract.extract_observations import output_path
from extract.extract_observations_agent import (
    TOOLS,
    Batch,
    compute_config_id,
    issue_batches,
    make_batches,
    process_batches,
    render_instructions,
)
from pkna.extract.observations import (
    CastMember,
    ObservationRecord,
    ObservedPage,
    ObservedPanel,
    PageSource,
    TextElement,
    list_issue_pages,
)


@pytest.fixture
def pages_root(tmp_path: Path) -> Path:
    root = tmp_path / "pages"
    (root / "pkna-0").mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (60, 90), "white").save(
            root / "pkna-0" / f"pkna-0-{i:03d}.jpg"
        )
    return root


def issue_pages(pages_root: Path) -> list[tuple[PageSource, Path]]:
    return list_issue_pages(pages_root / "pkna-0")


def page_json(speaker: str | None = "Uno") -> str:
    page = ObservedPage(
        panels=[
            ObservedPanel(
                description="Una sfera verde.",
                texts=[TextElement(kind="speech", text="Ciao", speaker=speaker)],
            )
        ]
    )
    return page.model_dump_json()


def write_all_pages(workspace: Path, model: str) -> dict:
    for page in (workspace / "pages").iterdir():
        (workspace / "out" / f"{page.stem}.json").write_text(page_json())
    return {"session_id": "s1"}


def read_failures(out_root: Path) -> list[str]:
    path = out_root / "pkna-0" / "failures.jsonl"
    return [json.loads(line)["image"] for line in path.read_text().splitlines()]


class TestMakeBatches:
    def test_splits_at_gaps_and_size_with_preceding_page_as_context(
        self, pages_root: Path
    ):
        pages = issue_pages(pages_root)
        pending = {
            "pkna-0-000.jpg",
            "pkna-0-001.jpg",
            "pkna-0-002.jpg",
            "pkna-0-004.jpg",
        }

        batches = make_batches(pages, pending, [], size=2)

        assert [[s.image for s, _ in b.pages] for b in batches] == [
            ["pkna-0-000.jpg", "pkna-0-001.jpg"],
            ["pkna-0-002.jpg"],
            ["pkna-0-004.jpg"],
        ]
        assert [b.context.name if b.context else None for b in batches] == [
            None,
            "pkna-0-001.jpg",
            "pkna-0-003.jpg",
        ]

    def test_issue_batches_skip_pages_done_with_same_config(
        self, pages_root: Path, tmp_path: Path
    ):
        out_root = tmp_path / "out"

        def batches(config_id: str) -> list[Batch]:
            return issue_batches(
                "pkna-0",
                out_root,
                config_id,
                8,
                pages_root=pages_root,
                v2_root=tmp_path / "v2",
            )

        process_batches(batches("cfg"), write_all_pages, "m", out_root, "cfg")

        assert batches("cfg") == []
        assert [len(b.pages) for b in batches("other")] == [5]


def test_instructions_list_pages_context_tools_and_rules(pages_root: Path):
    pages = issue_pages(pages_root)
    batch = Batch(pages=pages[2:4], context=pages[1][1], cast=[])

    text = render_instructions(batch)

    assert "- pkna-0-002\n- pkna-0-003\n" in text
    assert "`context/pkna-0-001.jpg` is the page before the first one" in text
    assert f"{TOOLS} zoom pages/<name>.jpg" in text
    assert "The balloon tail determines the speaker" in text


class TestProcessBatches:
    def test_records_have_batch_provenance_and_workspace_is_removed(
        self, pages_root: Path, tmp_path: Path
    ):
        pages = issue_pages(pages_root)
        cast = [CastMember(name="Uno", appearance="sfera verde")]
        batch = Batch(pages=pages[1:3], context=pages[0][1], cast=cast)
        seen: list[list[str]] = []

        def runner(workspace: Path, model: str) -> dict:
            seen.append(
                sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*.*"))
            )
            return write_all_pages(workspace, model)

        result = process_batches([batch], runner, "m", tmp_path, "cfg")

        record = ObservationRecord.model_validate_json(
            output_path(tmp_path, pages[2][0]).read_text()
        )
        assert result == (2, 0)
        assert seen == [
            [
                "INSTRUCTIONS.md",
                "cast.json",
                "context/pkna-0-000.jpg",
                "pages/pkna-0-001.jpg",
                "pages/pkna-0-002.jpg",
                "schema.json",
                "strips/pkna-0-001-1.jpg",
                "strips/pkna-0-001-2.jpg",
                "strips/pkna-0-001-3.jpg",
                "strips/pkna-0-002-1.jpg",
                "strips/pkna-0-002-2.jpg",
                "strips/pkna-0-002-3.jpg",
            ]
        ]
        assert record.panels[0].texts[0].speaker == "Uno"
        assert record.meta["agent"] == {"session_id": "s1"}
        assert record.meta["batch"] == ["pkna-0-001.jpg", "pkna-0-002.jpg"]
        assert record.meta["context"] == {
            "cast_size": 1,
            "context_page": "pkna-0-000.jpg",
        }
        assert list((tmp_path / "_work" / "pkna-0").iterdir()) == []

    def test_invalid_and_missing_pages_are_failures_and_workspace_is_kept(
        self, pages_root: Path, tmp_path: Path
    ):
        pages = issue_pages(pages_root)
        batch = Batch(pages=pages[:3], context=None, cast=[])

        def runner(workspace: Path, model: str) -> dict:
            (workspace / "out" / "pkna-0-000.json").write_text(page_json())
            (workspace / "out" / "pkna-0-001.json").write_text(page_json(speaker=None))
            return {}

        result = process_batches([batch], runner, "m", tmp_path, "cfg")

        assert result == (1, 2)
        assert read_failures(tmp_path) == ["pkna-0-001.jpg", "pkna-0-002.jpg"]
        assert (tmp_path / "_work" / "pkna-0" / "pkna-0-000" / "out").is_dir()

    def test_same_page_names_in_different_issues_get_separate_workspaces(
        self, pages_root: Path, tmp_path: Path
    ):
        (pages_root / "pkna-1").mkdir()
        Image.new("RGB", (60, 90), "black").save(
            pages_root / "pkna-1" / "pkna-0-000.jpg"
        )
        batches = [
            Batch(pages=list_issue_pages(pages_root / issue)[:1], context=None, cast=[])
            for issue in ("pkna-0", "pkna-1")
        ]
        workspaces: list[Path] = []

        def runner(workspace: Path, model: str) -> dict:
            workspaces.append(workspace)
            return {}

        process_batches(batches, runner, "m", tmp_path, "cfg", parallel=1)

        assert workspaces == [
            tmp_path / "_work" / "pkna-0" / "pkna-0-000",
            tmp_path / "_work" / "pkna-1" / "pkna-0-000",
        ]

    def test_pages_written_before_an_agent_error_are_kept(
        self, pages_root: Path, tmp_path: Path
    ):
        pages = issue_pages(pages_root)
        batch = Batch(pages=pages[:2], context=None, cast=[])

        def runner(workspace: Path, model: str) -> dict:
            (workspace / "out" / "pkna-0-000.json").write_text(page_json())
            raise RuntimeError("claude exited with 1")

        result = process_batches([batch], runner, "m", tmp_path, "cfg")

        record = ObservationRecord.model_validate_json(
            output_path(tmp_path, pages[0][0]).read_text()
        )
        assert result == (1, 1)
        assert record.meta["agent"] == {"error": "RuntimeError('claude exited with 1')"}
        assert read_failures(tmp_path) == ["pkna-0-001.jpg"]

    def test_os_error_from_the_agent_run_is_a_failure(
        self, pages_root: Path, tmp_path: Path
    ):
        batch = Batch(pages=issue_pages(pages_root)[:1], context=None, cast=[])

        def runner(workspace: Path, model: str) -> dict:
            raise FileNotFoundError("agent-output.json")

        result = process_batches([batch], runner, "m", tmp_path, "cfg")

        assert result == (0, 1)
        assert read_failures(tmp_path) == ["pkna-0-000.jpg"]


def test_config_id_depends_on_model():
    assert compute_config_id("a") == compute_config_id("a")
    assert compute_config_id("a") != compute_config_id("b")
