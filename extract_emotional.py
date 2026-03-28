#!/usr/bin/env python3
"""Extract structured information from comic book pages with enriched emotional annotations.

Compared to dspy-extract-full.py, this script adds per-dialogue tone and speech act
classification, and per-panel visual cues. The autoregressive page-by-page architecture
is the same: each page is processed with context from the previous page's output.
"""

import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)

# Settings
MODEL_NAME = "vertex_ai/claude-sonnet-4-6"
VERSION = "v2"
MAX_WORKERS = 1

# Paths
BASE_DIR = Path(__file__).parent
PAGES_ROOT = BASE_DIR / "input/pkna"
WIKI_ROOT = BASE_DIR / "output/wiki/fandom/crawl/storie/storie-di-pkna"
OUT_ROOT = BASE_DIR / f"output/extract-emotional/{VERSION}"

# Allowed values for constrained fields
TONE_VALUES = (
    "neutral",
    "sarcastic",
    "concerned",
    "playful",
    "authoritative",
    "melancholic",
    "angry",
    "affectionate",
    "humorous",
    "urgent",
)

SPEECH_ACT_VALUES = (
    "informing",
    "joking",
    "commanding",
    "deflecting",
    "comforting",
    "warning",
    "questioning",
    "narrating",
)

Tone = Literal[
    "neutral",
    "sarcastic",
    "concerned",
    "playful",
    "authoritative",
    "melancholic",
    "angry",
    "affectionate",
    "humorous",
    "urgent",
]

SpeechAct = Literal[
    "informing",
    "joking",
    "commanding",
    "deflecting",
    "comforting",
    "warning",
    "questioning",
    "narrating",
]

# Global progress bar
PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)


def configure_lm() -> None:
    load_dotenv()
    lm = dspy.LM(
        model=MODEL_NAME,
        vertex_credentials=os.getenv("VERTEX_AI_CREDS"),
        vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        temperature=1.0,
        max_tokens=60000,
    )
    dspy.configure(lm=lm, track_usage=True)


# ============================================================================
# Pydantic Models
# ============================================================================


class DialogueLine(BaseModel):
    """A single line of dialogue spoken by a character, with emotional annotation."""

    character: str = Field(description="The name of the character speaking the line.")
    line: str = Field(description="The dialogue line spoken by the character.")
    tone: Tone = Field(
        default="neutral",
        description=(
            "The emotional tone of this dialogue line, inferred from the character's "
            "expression, the speech bubble shape, and the surrounding context."
        ),
    )
    speech_act: SpeechAct = Field(
        default="informing",
        description="The pragmatic function of this dialogue line.",
    )


class Panel(BaseModel):
    """A single panel from a comic book page, with visual cues."""

    is_new_scene: bool = Field(
        default=False,
        description="Indicates whether this panel is a break from the previous scene, in terms of location or time.",
    )
    description: str = Field(
        description="A detailed description of the events happening in the panel."
    )
    caption_text: str | None = Field(
        default=None,
        description="The text of any caption present in the panel, if applicable.",
    )
    visual_cues: list[str] = Field(
        default=[],
        description=(
            "Notable visual details that convey character emotion or state but are not "
            "part of the narrative description. Examples: 'Uno's hologram flickers "
            "nervously', 'Paperinik clenches his fists', 'screen displays warning symbols'."
        ),
    )
    dialogues: list[DialogueLine] = Field(
        default=[],
        description="A list of dialogue lines spoken by characters in the panel. Dialogues are in order of appearance.",
    )


class CharacterAppearance(BaseModel):
    """A character's name paired with a brief visual description from their first appearance."""

    name: str = Field(description="The character's name.")
    appearance: str = Field(
        description=(
            "Brief visual description of the character as they first appear: "
            "species, build, distinctive clothing, colors, notable features. "
            "Keep to 1-2 sentences."
        ),
    )


# ============================================================================
# DSPy Signatures
# ============================================================================


class PlotExtractor(dspy.Signature):
    """Take a comic book issue summary and extract structured information about its plot.

    Follow these instructions:
    - Identify the main characters involved in the plot.
    - Each character should be returned by itself and represented by their name. Do not group characters together.
    - Summarize the key events that drive the story forward.
    - Use the language of the comic book (Italian) for all summaries and descriptions.
    """

    issue_summary: str = dspy.InputField(
        description="A brief summary of the comic book issue."
    )

    main_characters: list[str] = dspy.OutputField(
        description="A list of main characters involved in the plot."
    )
    key_events: list[str] = dspy.OutputField(
        description="A list of key events that drive the story forward."
    )


class PageExtractor(dspy.Signature):
    """Take a comic book page image and extract structured information about its content.

    FOLLOW THESE IMPORTANT INSTRUCTIONS

    Story continuity:
    - Use the overall plot summary and key events to inform the page content.
    - Use the previous page summary and panels to maintain continuity.
    - Use the same character names as previously introduced, if the character is the same.
    - Use the appearance descriptions of previously introduced characters to correctly
      re-identify them, even if they appear in a different context (e.g. on a screen,
      in flashback, without costume).
    - New characters can be introduced if they haven't appeared before. For each character
      appearing for the first time on this page, provide a brief visual description
      (species, build, clothing, colors, notable features) in the characters_introduced
      output.

    Output ordering:
    - Maintain the order of panels as they should be read on the page.
    - Within each panel, use dialogue context and conversational coherence to determine
      the correct reading order. The flow of conversation (questions before answers,
      calls before responses) should guide ordering. Do NOT assume a rigid
      top-to-bottom, left-to-right order.

    Panel boundaries and bubble assignment:
    - Carefully identify panel boundaries. Do not merge dialogues from adjacent panels
      into a single panel entry.
    - A speech bubble belongs to the panel where its tail (pointer) originates, even if
      the bubble body visually overlaps into an adjacent panel.

    Last event tracking:
    - Use the last event to keep track of story progression.
    - Update the last event if a new key event occurs on this page.
    - In the last event, refer to the key events by their exact wording as provided. DO NOT invent new key events.

    Character attribution:
    - The tail (pointer) of each speech bubble determines the speaker. Follow the tail
      to identify who is speaking, regardless of who else is visible or prominent in the
      panel.
    - When a character appears on a screen or monitor and another character is physically
      present, follow the bubble tail to determine the speaker. Do not assume the
      on-screen character is speaking just because their face is prominent.
    - Use the dialogue content itself as a signal for attribution. A character would not
      call out their own name or alias (e.g. "Pikappa!" cannot be spoken by Paperinik,
      it must be someone addressing him). Check that attributions make conversational
      sense.
    - The speaker of a line might not always be visible in the panel. Use context from
      previous and following panels to infer the speaker.
    - Sounds or onomatopoeias should not be considered dialogue lines. Mention them in
      the panel description if relevant, but do not include them in the dialogues list.

    Emotional tone:
    - For each dialogue line, infer the emotional tone from the speech bubble shape
      (jagged = shouting/angry, wavy = whispering/fearful, heart-shaped = affectionate),
      the character's facial expression, and the surrounding narrative context.
    - Use "neutral" only when no clear emotion is conveyed.

    Speech act:
    - For each dialogue line, classify its pragmatic function: is the character informing,
      joking, commanding, deflecting, comforting, warning, questioning, or narrating?

    Visual cues:
    - For each panel, list notable visual details that convey character emotion or state
      beyond what the narrative description captures. Focus on body language, facial
      expressions, holographic displays, screen states, environmental reactions.
    - Keep each cue to a short phrase. Omit if no notable cues are present.

    Output text:
    - Use the language of the comic book (Italian) for all summaries, descriptions, and visual cues.
    - Normalize the text by using normal caps instead of all caps, remove line-break hyphens, and accented letters instead of apostrophes when appropriate.
    """

    page: dspy.Image = dspy.InputField(
        description="The comic book page image to be analyzed."
    )
    previous_page_summary: str | None = dspy.InputField(
        default=None,
        description="A brief summary of the previous comic book page, if available.",
    )
    previous_page_panels: list[Panel] | None = dspy.InputField(
        default=None,
        description="The panels from the previous comic book page, if available.",
    )
    characters_already_introduced: list[CharacterAppearance] = dspy.InputField(
        default=[],
        description="Characters already introduced in previous pages, with their visual appearance descriptions.",
    )
    plot_summary: str = dspy.InputField(
        description="A brief summary of the overall plot of the comic book issue."
    )
    key_events: list[str] = dspy.InputField(
        description="A list of the key events in the overall story of the comic book issue."
    )
    last_event: str | None = dspy.InputField(
        default=None,
        description="The last key event that occurred in the comic book issue, if available.",
    )

    summary: str = dspy.OutputField(
        description="A brief summary of the comic book page."
    )
    panels: list[Panel] = dspy.OutputField(
        description="A list of panels extracted from the comic book page."
    )
    new_last_event: str | None = dspy.OutputField(
        default=None,
        description="The last key event that occurred in the comic book issue after this page, if available.",
    )
    characters_introduced: list[CharacterAppearance] = dspy.OutputField(
        default=[],
        description="Characters appearing for the first time on this page, with a brief visual description.",
    )


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class IssueSummary:
    summary: str
    key_events: list[str]
    main_characters: list[str]

    def to_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.summary,
                    "key_events": self.key_events,
                    "main_characters": self.main_characters,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def from_json(path: Path) -> "IssueSummary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IssueSummary(
            summary=data["summary"],
            key_events=data["key_events"],
            main_characters=data["main_characters"],
        )


@dataclass
class ExtractedPageMeta:
    model_name: str
    input_page_path: str
    lm_usage: dict | None = None


@dataclass
class ExtractedPage:
    summary: str
    panels: list[Panel]
    last_event: str | None
    characters_introduced: list[CharacterAppearance]
    meta: ExtractedPageMeta

    def to_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.summary,
                    "panels": [
                        {
                            "is_new_scene": panel.is_new_scene,
                            "description": panel.description,
                            "caption_text": panel.caption_text,
                            "visual_cues": panel.visual_cues,
                            "dialogues": [
                                {
                                    "character": d.character,
                                    "line": d.line,
                                    "tone": d.tone,
                                    "speech_act": d.speech_act,
                                }
                                for d in panel.dialogues
                            ],
                        }
                        for panel in self.panels
                    ],
                    "last_event": self.last_event,
                    "characters_introduced": [
                        {"name": ca.name, "appearance": ca.appearance}
                        for ca in self.characters_introduced
                    ],
                    "meta": {
                        "model_name": self.meta.model_name,
                        "input_page_path": self.meta.input_page_path,
                        "lm_usage": self.meta.lm_usage,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def from_prediction(pred: dspy.Prediction, input_page: Path) -> "ExtractedPage":
        return ExtractedPage(
            summary=pred.summary,
            panels=pred.panels,
            last_event=pred.new_last_event,
            characters_introduced=pred.characters_introduced,
            meta=ExtractedPageMeta(
                model_name=MODEL_NAME,
                input_page_path=input_page.as_posix(),
                lm_usage=pred.get_lm_usage(),
            ),
        )

    @staticmethod
    def from_json(path: Path) -> "ExtractedPage":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data["meta"]
        return ExtractedPage(
            summary=data["summary"],
            panels=[Panel(**panel) for panel in data["panels"]],
            last_event=data.get("last_event"),
            characters_introduced=[
                CharacterAppearance(**ca)
                for ca in data.get("characters_introduced", [])
            ],
            meta=ExtractedPageMeta(
                model_name=meta["model_name"],
                input_page_path=meta["input_page_path"],
                lm_usage=meta.get("lm_usage"),
            ),
        )


# ============================================================================
# DSPy Modules
# ============================================================================


class PlotSummarizer(dspy.Module):
    def __init__(self):
        self.plot_extractor = dspy.ChainOfThought(PlotExtractor)

    def forward(self, issue_summary: str) -> dspy.Prediction:
        pred = self.plot_extractor(issue_summary=issue_summary)
        pred.issue_summary = issue_summary
        return pred

    def to_dataclass(self, pred: dspy.Prediction) -> IssueSummary:
        return IssueSummary(
            summary=pred.issue_summary,
            key_events=pred.key_events,
            main_characters=pred.main_characters,
        )


class Extractor(dspy.Module):
    def __init__(self, summary: IssueSummary):
        self.issue_summary = summary

        self.prev_characters: dict[str, str] = {}
        self.prev_page_summary: str | None = None
        self.prev_page_panels: list[Panel] | None = None
        self.prev_event: str | None = None

        self.page_extractor = dspy.ChainOfThought(PageExtractor)

    def _characters_as_appearances(self) -> list[CharacterAppearance]:
        return [
            CharacterAppearance(name=name, appearance=desc)
            for name, desc in self.prev_characters.items()
        ]

    def forward(self, page: dspy.Image) -> dspy.Prediction:
        page_pred = self.page_extractor(
            page=page,
            previous_page_summary=self.prev_page_summary,
            previous_page_panels=self.prev_page_panels,
            characters_already_introduced=self._characters_as_appearances(),
            plot_summary=self.issue_summary.summary,
            key_events=self.issue_summary.key_events,
            last_event=self.prev_event,
        )
        self.prev_page_summary = page_pred.summary
        self.prev_page_panels = page_pred.panels
        for ca in page_pred.characters_introduced:
            self.prev_characters.setdefault(ca.name, ca.appearance)
        self.prev_event = page_pred.new_last_event
        return page_pred

    def update_from_extracted(self, extracted: ExtractedPage) -> None:
        self.prev_page_summary = extracted.summary
        self.prev_page_panels = extracted.panels
        self.prev_event = extracted.last_event
        for ca in extracted.characters_introduced:
            self.prev_characters.setdefault(ca.name, ca.appearance)


# ============================================================================
# Work Items and Processing
# ============================================================================


@dataclass
class WorkItem:
    id: str
    pages_paths: list[Path]
    summary_path: Path


def make_work_item(issue_id: str, wiki_name: str) -> WorkItem:
    issue_path = PAGES_ROOT / issue_id
    wiki_path = WIKI_ROOT / wiki_name
    pages_paths = sorted(
        list(issue_path.glob("*.jpg")) + list(issue_path.glob("*.jpeg"))
    )
    return WorkItem(id=issue_id, pages_paths=pages_paths, summary_path=wiki_path)


def get_items_to_process() -> list[WorkItem]:
    return [
        make_work_item(n, w)
        for n, w in [
            ("pkna-0", "evroniani-albo.md"),
            ("pkna-0-2", "quando-soffia-il-vento-del-tempo.md"),
            ("pkna-0-3", "xadhoom.md"),
            ("pkna-1", "ombre-su-venere.md"),
            ("pkna-2", "due-albo.md"),
            ("pkna-3", "il-giorno-del-sole-freddo.md"),
            ("pkna-4", "terremoto.md"),
            ("pkna-5", "ritratto-dell-eroe-da-giovane.md"),
            ("pkna-6", "spore.md"),
            ("pkna-7", "invasione.md"),
            ("pkna-8", "silicio.md"),
            ("pkna-9", "le-sorgenti-della-luna.md"),
            ("pkna-10", "trauma-albo.md"),
            ("pkna-11", "urk-albo.md"),
            ("pkna-12", "seconda-stesura.md"),
            ("pkna-13", "la-notte-pi-buia.md"),
            ("pkna-14", "carpe-diem.md"),
            ("pkna-15", "motore-azione.md"),
            ("pkna-16", "manutenzione-straordinaria.md"),
            ("pkna-17", "stella-cadente.md"),
            ("pkna-18", "antico-futuro.md"),
            ("pkna-19", "zero-assoluto.md"),
            ("pkna-20", "mekkano.md"),
            ("pkna-21", "tyrannic.md"),
            ("pkna-22", "frammenti-d-autunno.md"),
            ("pkna-23", "vuoto-di-memoria.md"),
            ("pkna-24", "crepuscolo.md"),
            ("pkna-25", "fuoco-incrociato.md"),
            ("pkna-26", "il-tempo-fugge.md"),
            ("pkna-27", "i-mastini-dell-universo.md"),
            ("pkna-28", "metamorfosi.md"),
            ("pkna-29", "virus.md"),
            ("pkna-30", "fase-due.md"),
            ("pkna-31", "beato-angelico.md"),
            ("pkna-32", "underground.md"),
            ("pkna-33", "il-giorno-che-verr.md"),
            ("pkna-34", "niente-di-personale.md"),
            ("pkna-35", "clandestino-a-bordo.md"),
            ("pkna-36", "lontano-lontano.md"),
            ("pkna-37", "sotto-un-nuovo-sole.md"),
            ("pkna-38", "nella-nebbia.md"),
            ("pkna-39", "cronaufragio.md"),
            ("pkna-40", "un-solo-respiro.md"),
            ("pkna-41", "agdy-days.md"),
            ("pkna-42", "la-sindrome-di-ulisse.md"),
            ("pkna-43", "tempo-al-tempo.md"),
            ("pkna-44", "sul-lato-oscuro.md"),
            ("pkna-45", "operazione-efesto.md"),
            ("pkna-46", "nell-ombra.md"),
            ("pkna-47", "prima-dell-alba.md"),
            ("pkna-48", "le-parti-e-il-tutto.md"),
            ("pkna-49", "se.md"),
        ]
    ]


def process_item(it: WorkItem) -> None:
    log.info(f"Starting processing for {it.id}")
    out_dir = OUT_ROOT / it.id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = [out_dir / f"page_{i + 1:03d}.json" for i in range(len(it.pages_paths))]
    extracted: list[Path] = []
    need_work: list[tuple[Path, Path]] = []

    for in_path, out_path in zip(it.pages_paths, out_paths):
        if out_path.exists():
            extracted.append(out_path)
        else:
            need_work.append((in_path, out_path))

    if not need_work:
        log.info(
            f"All pages ({len(extracted)}) already processed for {it.id}, skipping"
        )
        return
    if len(extracted) > 0:
        log.info(
            f"Resuming processing for {it.id}, {len(extracted)} pages done, {len(need_work)} to go"
        )

    summary_out = out_dir / "issue_summary.json"
    if summary_out.exists():
        summary = IssueSummary.from_json(summary_out)
    else:
        with open(it.summary_path, "r", encoding="utf-8") as f:
            issue_text = f.read()
        ps = PlotSummarizer()
        pred = ps(issue_summary=issue_text)
        summary = ps.to_dataclass(pred)
        summary.to_json(summary_out)

    extractor = Extractor(summary=summary)
    for extracted_path in extracted:
        extracted_page = ExtractedPage.from_json(extracted_path)
        extractor.update_from_extracted(extracted_page)

    with PROGRESS as progress:
        for in_path, out_path in progress.track(
            need_work,
            total=len(need_work),
            description=f"Processing pages for {it.id}...",
        ):
            page_image = dspy.Image(in_path.as_posix())
            pred = extractor(page=page_image)
            extracted_page = ExtractedPage.from_prediction(pred, input_page=in_path)
            extracted_page.to_json(out_path)

    log.info(f"Finished processing for {it.id}")


def main():
    configure_lm()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    items = get_items_to_process()

    max_workers = min(MAX_WORKERS, len(items))
    log.info(f"Starting pool with {max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, it): it for it in items}

        with PROGRESS as progress:
            for future in progress.track(
                concurrent.futures.as_completed(future_to_item),
                total=len(future_to_item),
                description="Processing issues...",
            ):
                item = future_to_item[future]
                try:
                    future.result()
                except Exception as exc:
                    log.exception(
                        f"Item {item.id} generated an exception: {exc}", exc_info=exc
                    )

    log.info("All tasks completed")


if __name__ == "__main__":
    main()
