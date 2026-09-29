"""Tests for the zoom and validate tools used by extraction agents."""

from pathlib import Path

import pytest
from PIL import Image

from extract.observation_agent_tools import (
    MAX_ZOOM,
    ZOOM_SIZE,
    validate,
    validate_cast,
    zoom,
)
from pkna.extract.observations import ObservedPage, ObservedPanel, TextElement
from pkna.extract.registry import IssueCast, Persona, ResolvedCharacter


@pytest.fixture
def page(tmp_path: Path) -> Path:
    path = tmp_path / "pkna-0-001.jpg"
    Image.new("RGB", (1000, 1500), "white").save(path)
    return path


class TestZoom:
    def test_enlarges_crop_to_zoom_size(self, page: Path, tmp_path: Path):
        out = zoom(page, (0.0, 0.0, 0.5, 0.4), tmp_path / "zoom")

        assert out == tmp_path / "zoom" / "pkna-0-001_0.00_0.00_0.50_0.40.png"
        with Image.open(out) as crop:
            assert crop.size == (round(500 * ZOOM_SIZE / 600), ZOOM_SIZE)

    def test_enlargement_is_capped(self, page: Path, tmp_path: Path):
        out = zoom(page, (0.0, 0.0, 0.1, 0.1), tmp_path)

        with Image.open(out) as crop:
            assert crop.size == (round(100 * MAX_ZOOM), round(150 * MAX_ZOOM))

    def test_rejects_box_outside_page(self, page: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="Box must satisfy"):
            zoom(page, (0.5, 0.0, 1.2, 0.5), tmp_path)


class TestValidate:
    def write(self, tmp_path: Path, speaker: str | None) -> Path:
        path = tmp_path / "page.json"
        page = ObservedPage(
            panels=[
                ObservedPanel(
                    description="d",
                    texts=[
                        TextElement(kind="sfx", text="Zap"),
                        TextElement(kind="speech", text="Ciao", speaker=speaker),
                    ],
                )
            ]
        )
        path.write_text(page.model_dump_json())
        return path

    def test_valid_page_has_no_problems(self, tmp_path: Path):
        assert validate(self.write(tmp_path, "Uno")) == []

    def test_reports_speech_without_speaker(self, tmp_path: Path):
        assert validate(self.write(tmp_path, None)) == [
            "panel 1 text 2: speech without a speaker"
        ]

    def test_reports_schema_errors(self, tmp_path: Path):
        path = tmp_path / "page.json"
        path.write_text('{"panels": [{"texts": []}]}')

        [problem] = validate(path)

        assert "description" in problem and "Field required" in problem


def test_validate_cast_checks_labels_against_mentions(tmp_path: Path):
    mentions = tmp_path / "mentions.json"
    mentions.write_text('[{"label": "Uno"}, {"label": "Paperinik"}]')
    cast = tmp_path / "cast.json"
    cast.write_text(
        IssueCast(
            characters=[
                ResolvedCharacter(
                    name="Uno",
                    kind="named",
                    personas=[Persona(name="Uno", labels=["Uno"])],
                    description="sfera verde",
                )
            ]
        ).model_dump_json()
    )

    assert validate_cast(cast, mentions) == [
        "label 'Paperinik' is not assigned to any character"
    ]
