"""Observation layer: what each comic page shows, without interpretation.

This is the only pipeline layer that reads page images. The entity registry,
event and knowledge log, and character state are derived from these records,
so they must not contain plot knowledge from outside the page.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

SCHEMA_VERSION = "obs-v1"
IMAGE_SUFFIXES = (".jpg", ".jpeg")

TextKind = Literal["speech", "thought", "narration", "sign", "sfx"]
Depiction = Literal[
    "in_person", "hologram", "on_screen", "off_panel", "image", "memory"
]
Attribution = Literal["tail", "context", "uncertain"]


class TextElement(BaseModel):
    """A piece of lettering on the page, in reading order within its panel.

    Balloon outline and tail come before text and speaker so that they are
    recorded as observations, not rationalized from an attribution.
    """

    kind: TextKind = Field(
        description=(
            "speech: spoken balloon. thought: thought balloon. narration: caption box. "
            "sign: text drawn inside the scene (signs, screens, labels, 'Fine'). "
            "sfx: sound effect or onomatopoeia."
        )
    )
    balloon_style: str | None = Field(
        default=None,
        description=(
            "For balloons, the outline as drawn: 'liscio' (plain rounded), 'a punte' "
            "(spiky, electronic voice), 'esplosione frastagliata' (shouting), "
            "'tratteggiato' (whisper), 'nuvola' (thought). Null for captions, signs, "
            "and sfx."
        ),
    )
    tail_target: str | None = Field(
        default=None,
        description=(
            "For balloons, what the tail points at: a character in the panel, a "
            "device, a building, or the panel edge it leaves through (e.g. 'fuori dal "
            "bordo destro'). Null when there is no tail."
        ),
    )
    text: str = Field(description="The lettered text.")
    speaker: str | None = Field(
        default=None,
        description=(
            "Who speaks or thinks the text: the cast sheet name of the persona as "
            "depicted, or a short descriptive label for an unknown speaker. "
            "Null for narration, sign, and sfx."
        ),
    )
    addressees: list[str] = Field(
        default=[],
        description=(
            "Characters the text is directed to, when evident from vocatives, gaze, "
            "or the conversational turn. Empty when general or unclear."
        ),
    )
    via: str | None = Field(
        default=None,
        description=(
            "Device or medium carrying the voice, e.g. 'TV', 'radio', 'altoparlante', "
            "'schermo'. Null when speaking in person (for Uno, this includes his hologram)."
        ),
    )
    attribution: Attribution | None = Field(
        default=None,
        description=(
            "How the speaker was determined: 'tail' when the balloon tail points at "
            "them, 'context' when inferred from balloon style, content, or "
            "neighboring panels, 'uncertain' otherwise. Null when there is no speaker."
        ),
    )


class CharacterPresence(BaseModel):
    """A character shown in a panel, or clearly present just outside it."""

    name: str = Field(
        description=(
            "Cast sheet name of the persona as depicted, or a short descriptive "
            "label for an unknown character."
        )
    )
    depiction: Depiction = Field(
        description=(
            "in_person: physically drawn in the scene. hologram: projected hologram. "
            "on_screen: shown on a screen or monitor. off_panel: not drawn but present "
            "(e.g. speaking from outside the frame). image: photo, poster, or statue. "
            "memory: flashback, dream, or imagination."
        )
    )
    expression: str | None = Field(
        default=None,
        description="Visible facial expression, gesture, or posture. Only what is drawn.",
    )


class ObservedPanel(BaseModel):
    """A single panel, recorded as drawn."""

    new_scene: bool = Field(
        default=False,
        description="True when this panel visibly moves to a different place or time than the previous panel.",
    )
    transition_cue: str | None = Field(
        default=None,
        description="What shows the change of place or time, e.g. a caption such as 'Intanto...' or a different setting.",
    )
    location: str | None = Field(
        default=None,
        description="Where the panel takes place, if recognizable from the drawing or captions.",
    )
    description: str = Field(
        description=(
            "What is drawn: characters, actions, poses, objects, setting, framing. "
            "Do not repeat or summarize the lettering, and do not explain motives, "
            "hidden causes, or what characters know."
        )
    )
    characters: list[CharacterPresence] = Field(
        default=[],
        description="Everyone drawn in the panel, speaking or not, plus off-panel speakers.",
    )
    texts: list[TextElement] = Field(
        default=[],
        description="All lettering in the panel, in reading order.",
    )


class CastMember(BaseModel):
    """A character expected in an issue, used to keep names consistent."""

    name: str
    appearance: str | None = None


class ObservedPage(BaseModel):
    """What one page shows, as written by an extractor before provenance is added."""

    printed_page_number: int | None = Field(
        default=None, description="The page number printed on the page, if visible."
    )
    panels: list[ObservedPanel] = Field(
        description="The panels of the page in reading order."
    )
    unlisted_characters: list[CastMember] = Field(
        default=[],
        description=(
            "Characters on this page missing from the cast sheet: the label used "
            "and a brief appearance description."
        ),
    )


def page_problems(page: ObservedPage) -> list[str]:
    """Consistency problems that the schema alone does not catch."""
    return [
        f"panel {pi} text {ti}: {t.kind} without a speaker"
        for pi, panel in enumerate(page.panels, 1)
        for ti, t in enumerate(panel.texts, 1)
        if t.kind in ("speech", "thought") and not t.speaker
    ]


class PageSource(BaseModel):
    """Where a record comes from. The image file name is the stable page identifier."""

    issue: str
    image: str
    index: int = Field(description="1-based position among the issue's sorted images.")

    @property
    def page_id(self) -> str:
        return f"{self.issue}/{Path(self.image).stem}"


class ObservationRecord(BaseModel):
    """Observations for one page, with provenance."""

    source: PageSource
    printed_page_number: int | None = None
    panels: list[ObservedPanel]
    unlisted_characters: list[CastMember] = []
    meta: dict[str, Any] = {}

    def panel_id(self, panel_index: int) -> str:
        return f"{self.source.page_id}#p{panel_index + 1}"

    def iter_texts(self) -> Iterator[tuple[str, ObservedPanel, TextElement]]:
        """Yield (text_id, panel, text) for every text element, in page order."""
        for pi, panel in enumerate(self.panels):
            for ti, text in enumerate(panel.texts):
                yield f"{self.panel_id(pi)}.t{ti + 1}", panel, text


def list_issue_pages(issue_dir: Path) -> list[tuple[PageSource, Path]]:
    """Page images of an issue in reading order, with their sources."""
    images = sorted(
        p for p in issue_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    return [
        (PageSource(issue=issue_dir.name, image=p.name, index=i + 1), p)
        for i, p in enumerate(images)
    ]


def page_strips(
    page: Image.Image, count: int, overlap: float, width: int
) -> list[Image.Image]:
    """Split a page into overlapping horizontal strips resized to a common width.

    At full-page scale, balloon outlines and tails are too small for the model
    to resolve reliably, and they decide speaker attribution.
    """
    page_width, page_height = page.size
    step = page_height / count
    pad = int(step * overlap)
    scale = width / page_width
    strips: list[Image.Image] = []
    for i in range(count):
        top = max(0, int(i * step) - pad)
        bottom = min(page_height, int((i + 1) * step) + pad)
        strip = page.crop((0, top, page_width, bottom))
        strips.append(strip.resize((width, round((bottom - top) * scale))))
    return strips


def load_v2_pages(v2_issue_dir: Path) -> dict[str, dict]:
    """Map image file names to extract-emotional v2 page records for one issue."""
    pages: dict[str, dict] = {}
    for path in sorted(v2_issue_dir.glob("page_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages[Path(data["meta"]["input_page_path"]).name] = data
    return pages


def build_cast_sheet(v2_pages: Sequence[dict]) -> list[CastMember]:
    """Introduced characters with their appearance, then remaining speakers by name."""
    cast: dict[str, CastMember] = {}
    for page in v2_pages:
        for ca in page.get("characters_introduced", []):
            cast.setdefault(
                ca["name"], CastMember(name=ca["name"], appearance=ca["appearance"])
            )
    for page in v2_pages:
        for panel in page["panels"]:
            for dl in panel["dialogues"]:
                cast.setdefault(dl["character"], CastMember(name=dl["character"]))
    return list(cast.values())


def apply_cast_notes(cast: list[CastMember], notes: dict[str, str]) -> list[CastMember]:
    """Append curated identification cues to cast members, adding missing ones."""
    result = [
        CastMember(
            name=c.name,
            appearance=" ".join(filter(None, [c.appearance, notes[c.name]])),
        )
        if c.name in notes
        else c
        for c in cast
    ]
    known = {c.name for c in cast}
    result += [
        CastMember(name=name, appearance=note)
        for name, note in notes.items()
        if name not in known
    ]
    return result


def previous_page_dialogue(v2_page: dict | None, max_panels: int = 2) -> list[str]:
    """Last dialogue lines of the previous page, as speaker continuity context.

    Only speakers and lines are kept: v2 descriptions and summaries were written
    with the whole issue's plot in view.
    """
    if v2_page is None:
        return []
    return [
        f"{dl['character']}: {dl['line']}"
        for panel in v2_page["panels"][-max_panels:]
        for dl in panel["dialogues"]
    ]
