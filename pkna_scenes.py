"""Scene data structures and extraction from extract-emotional output.

Provides dataclasses for representing comic scenes with emotional annotations,
functions to extract scenes from per-page JSON files, and formatting for LLM prompts.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnnotatedDialogue:
    """A dialogue line with emotional annotations from extract_emotional."""

    character: str
    line: str
    tone: str = "neutral"
    speech_act: str = "informing"


@dataclass
class Panel:
    """A single comic panel with its description, dialogues, and visual cues."""

    description: str
    dialogues: list[AnnotatedDialogue]
    visual_cues: list[str] = field(default_factory=list)


@dataclass
class Scene:
    """A scene from the comics containing Uno, with enriched emotional data."""

    issue: str
    page_numbers: list[int]
    panels: list[Panel]
    other_characters: set[str]

    @property
    def scene_id(self) -> str:
        return f"{self.issue}_{self.page_numbers[0]}"

    @property
    def summary(self) -> str:
        return " ".join(p.description for p in self.panels)

    def to_context_string(self) -> str:
        pages_str = ", ".join(f"page {p}" for p in self.page_numbers)
        chars_str = (
            ", ".join(sorted(self.other_characters))
            if self.other_characters
            else "none"
        )
        return (
            f"Issue: {self.issue}, {pages_str}. Other characters present: {chars_str}"
        )

    def to_dict(self) -> dict:
        return {
            "issue": self.issue,
            "page_numbers": self.page_numbers,
            "panels": [
                {
                    "description": p.description,
                    "dialogues": [
                        {
                            "character": d.character,
                            "line": d.line,
                            "tone": d.tone,
                            "speech_act": d.speech_act,
                        }
                        for d in p.dialogues
                    ],
                    "visual_cues": p.visual_cues,
                }
                for p in self.panels
            ],
            "other_characters": list(self.other_characters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        panels = [
            Panel(
                description=p["description"],
                dialogues=[AnnotatedDialogue(**d) for d in p["dialogues"]],
                visual_cues=p.get("visual_cues", []),
            )
            for p in data["panels"]
        ]
        return cls(
            issue=data["issue"],
            page_numbers=data["page_numbers"],
            panels=panels,
            other_characters=set(data["other_characters"]),
        )


# ============================================================================
# Scene Extraction from emotional output
# ============================================================================


@dataclass
class _PanelAccumulator:
    """Temporary accumulator for building scenes from panels."""

    panels: list[dict] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def extract_scenes_from_issue(issue_dir: Path) -> list[Scene]:
    page_files = sorted(issue_dir.glob("page_*.json"))
    scenes: list[Scene] = []
    current = _PanelAccumulator()

    for page_file in page_files:
        page_num = int(page_file.stem.split("_")[1])

        with open(page_file, encoding="utf-8") as f:
            page_data = json.load(f)

        panels = page_data.get("panels", [])
        if not panels:
            continue

        for panel in panels:
            if panel.get("is_new_scene", False) and current.panels:
                scene = _create_scene_from_panels(
                    issue_dir.name, current.pages, current.panels
                )
                if scene:
                    scenes.append(scene)
                current = _PanelAccumulator()

            current.panels.append(panel)
            if page_num not in current.pages:
                current.pages.append(page_num)

    if current.panels:
        scene = _create_scene_from_panels(issue_dir.name, current.pages, current.panels)
        if scene:
            scenes.append(scene)

    return scenes


def _create_scene_from_panels(
    issue: str, page_numbers: list[int], raw_panels: list[dict]
) -> Scene | None:
    """Create a Scene from panels, only if Uno is present."""
    panels: list[Panel] = []
    other_characters: set[str] = set()
    has_uno = False

    for raw in raw_panels:
        desc = raw.get("description", "")
        cues = [c for c in raw.get("visual_cues", []) if c]
        dialogues: list[AnnotatedDialogue] = []

        for dialogue in raw.get("dialogues", []):
            character = dialogue.get("character", "").strip()
            line = dialogue.get("line", "").strip()
            if not character or not line:
                continue

            ad = AnnotatedDialogue(
                character=character,
                line=line,
                tone=dialogue.get("tone", "neutral"),
                speech_act=dialogue.get("speech_act", "informing"),
            )
            dialogues.append(ad)

            if character.lower() == "uno":
                has_uno = True
            else:
                other_characters.add(character)

        panels.append(Panel(description=desc, dialogues=dialogues, visual_cues=cues))

    if not has_uno:
        return None

    return Scene(
        issue=issue,
        page_numbers=page_numbers,
        panels=panels,
        other_characters=other_characters,
    )


def natural_sort_key(path: Path) -> tuple:
    parts = path.name.split("-")
    key: list[int | str] = []
    for part in parts:
        try:
            key.append(int(part))
        except ValueError:
            key.append(part)
    return tuple(key)


# ============================================================================
# Formatting
# ============================================================================


def format_scene_view(scene: Scene) -> str:
    lines = [
        f"Scene: {scene.scene_id}",
        f"Issue: {scene.issue}, pages {'-'.join(map(str, scene.page_numbers))}",
        f"Characters present: {', '.join(sorted(scene.other_characters)) if scene.other_characters else 'Uno only'}",
    ]

    for i, panel in enumerate(scene.panels, 1):
        lines.append("")
        lines.append(f"--- Panel {i} ---")
        lines.append(f"[{panel.description}]")
        if panel.visual_cues:
            lines.append(f"Visual: {'; '.join(panel.visual_cues)}")
        for d in panel.dialogues:
            lines.append(f'{d.character}: "{d.line}" [{d.tone}, {d.speech_act}]')

    return "\n".join(lines)
