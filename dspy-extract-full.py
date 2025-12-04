#!/usr/bin/env python3

import concurrent.futures
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import dspy
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler


# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(__file__).parent
PAGES_ROOT = SCRIPT_DIR.parent / "input/pkna"
WIKI_ROOT = SCRIPT_DIR.parent / "output/wiki/fandom/crawl/storie/storie-di-pkna"
OUT_ROOT = SCRIPT_DIR.parent / "output/dspy-extract-full/v0"


class DialogueLine(BaseModel):
    """A single line of dialogue spoken by a character."""

    character: str = Field(description="The name of the character speaking the line.")
    line: str = Field(description="The dialogue line spoken by the character.")


class Panel(BaseModel):
    """A single panel from a comic book page."""

    description: str = Field(
        description="A detailed description of the events happening in the panel."
    )
    caption_text: str | None = Field(
        default=None,
        description="The text of any caption present in the panel, if applicable.",
    )
    dialogues: list[DialogueLine] = Field(
        description="A list of dialogue lines spoken by characters in the panel. Dialogues are in order of appearance."
    )


class PlotExtractor(dspy.Signature):
    """Take a comic book issue summary and extract structured information about its plot.

    Follow these instructions:
    - Identify the main characters involved in the plot.
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
    - New character can be introduced if they haven't appeared before.

    Last event tracking:
    - Use the last event to keep track of story progression.
    - Update the last event if a new key event occurs on this page.
    - In the last event, refer to the key events by their exact wording as provided. DO NOT invent new key events.

    Character attribution:
    - Make extra effort to correctly attribute dialogue lines to the right characters.
    - The speaker of a line might not always be visible in the panel. Use context from previous and following panels to infer the speaker.

    Output text:
    - Use the language of the comic book (Italian) for all summaries and descriptions.
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
    characters_already_introduced: list[str] = dspy.InputField(
        default=[],
        description="A list of character names that have already been introduced in previous pages.",
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


class PlotSummarizer(dspy.Module):
    def __init__(self):
        self.plot_extractor = dspy.ChainOfThought(PlotExtractor)

    def forward(self, issue_summary: str) -> dspy.Prediction:
        pred = self.plot_extractor(issue_summary=issue_summary)
        pred.issue_summary = issue_summary  # Store original summary
        return pred

    def to_dataclass(self, pred: dspy.Prediction) -> IssueSummary:
        return IssueSummary(
            summary=pred.issue_summary,
            key_events=pred.key_events,
            main_characters=pred.main_characters,
        )


@dataclass
class ExtractedPage:
    summary: str
    panels: list[Panel]
    last_event: str | None
    processing_time_seconds: float
    model_name: str
    input_page_path: str

    def to_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.summary,
                    "panels": [
                        {
                            "description": panel.description,
                            "caption_text": panel.caption_text,
                            "dialogues": [
                                {
                                    "character": dialogue.character,
                                    "line": dialogue.line,
                                }
                                for dialogue in panel.dialogues
                            ],
                        }
                        for panel in self.panels
                    ],
                    "last_event": self.last_event,
                    "processing_time_seconds": self.processing_time_seconds,
                    "model_name": self.model_name,
                    "input_page_path": self.input_page_path,
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
            processing_time_seconds=pred.processing_time_seconds,
            model_name=pred.model_name,
            input_page_path=input_page.as_posix(),
        )

    @staticmethod
    def from_json(path: Path) -> "ExtractedPage":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExtractedPage(
            summary=data["summary"],
            panels=[Panel(**panel) for panel in data["panels"]],
            last_event=data.get("last_event"),
            processing_time_seconds=data["processing_time_seconds"],
            model_name=data["model_name"],
            input_page_path=data["input_page_path"],
        )


class Extractor(dspy.Module):
    def __init__(self, summary: IssueSummary):
        self.issue_summary = summary

        self.prev_characters = set[str]()
        self.prev_page_summary: str | None = None
        self.prev_page_panels: list[Panel] | None = None
        self.prev_event: str | None = None

        self.plot_extractor = dspy.ChainOfThought(PlotExtractor)
        self.page_extractor = dspy.ChainOfThought(PageExtractor)

    def forward(self, page: dspy.Image) -> dspy.Prediction:
        page_pred = self.page_extractor(
            page=page,
            previous_page_summary=self.prev_page_summary,
            previous_page_panels=self.prev_page_panels,
            characters_already_introduced=list(self.prev_characters),
            plot_summary=self.issue_summary.summary,
            key_events=self.issue_summary.key_events,
            last_event=self.prev_event,
        )
        # Update state for next page
        self.prev_page_summary = page_pred.summary
        self.prev_page_panels = page_pred.panels
        self.prev_characters.update(
            [
                dialogue.character
                for panel in page_pred.panels
                for dialogue in panel.dialogues
            ]
        )
        self.prev_event = page_pred.new_last_event
        return page_pred

    def update_from_extracted(self, extracted: ExtractedPage) -> None:
        self.prev_page_summary = extracted.summary
        self.prev_page_panels = extracted.panels
        self.prev_event = extracted.last_event
        self.prev_characters.update(
            [
                dialogue.character
                for panel in extracted.panels
                for dialogue in panel.dialogues
            ]
        )


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
        ]
    ]


def process_item(it: WorkItem) -> None:
    log.info(f"Starting processing for {it.id}")
    out_dir = OUT_ROOT / it.id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = [
        out_dir / f"page_{i + 1:03d}.json"
        for i in range(len(it.pages_paths))
    ]
    extracted = []
    need_work = []

    for in_path, out_path in zip(it.pages_paths, out_paths):
        if out_path.exists():
            extracted.append(out_path)
        else:
            need_work.append((in_path, out_path))

    if not need_work:
        log.info(f"All pages ({len(extracted)}) already processed for {it.id}, skipping")
        return
    if len(extracted) > 0:
        log.info(f"Resuming processing for {it.id}, {len(extracted)} pages done, {len(need_work)} to go")

    # Otherwise, we need to reconstruct the work that was done so far
    # Starting from the issue summary
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

    # Now create the extractor and replay previous pages
    extractor = Extractor(summary=summary)
    for extracted_path in extracted:
        extracted_page = ExtractedPage.from_json(extracted_path)
        extractor.update_from_extracted(extracted_page)

    # Now process the remaining pages
    for in_path, out_path in need_work:
        log.info(f"Processing page {in_path.name} for {it.id}")
        page_image = dspy.Image.from_file(in_path)
        pred = extractor(page=page_image)
        # Save extracted page
        extracted_page = ExtractedPage.from_prediction(pred, input_page=in_path)
        extracted_page.to_json(out_path)

    log.info(f"Finished processing for {it.id}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    items = get_items_to_process()

    # Number of concurrent threads
    max_workers = min(4, len(items))
    log.info(f"Starting pool with {max_workers} workers")

    # Using ThreadPoolExecutor to manage threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Dictionary to map futures to their original items
        future_to_item = {executor.submit(process_item, it): it for it in items}

        # Iterate over futures as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result()
            except Exception as exc:
                log.error(f"Item {item} generated an exception: {exc}")

    log.info("All tasks completed")


if __name__ == "__main__":
    main()
