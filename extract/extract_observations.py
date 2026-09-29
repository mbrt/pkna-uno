#!/usr/bin/env python3
"""Extract the observation layer from comic page images.

Every page is extracted independently, so pages run in parallel. Continuity
comes from a per-issue cast sheet and the previous page's dialogue from
extract-emotional v2, never from the issue's plot summary: records describe
only what the page shows.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dotenv import load_dotenv
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from pkna.extract.observations import (
    SCHEMA_VERSION,
    CastMember,
    apply_cast_notes,
    ObservationRecord,
    ObservedPanel,
    PageSource,
    build_cast_sheet,
    list_issue_pages,
    load_v2_pages,
    page_strips,
    previous_page_dialogue,
)
from pkna.logging import setup_logging

console, log = setup_logging()

DEFAULT_MODEL = "bedrock/eu.anthropic.claude-sonnet-4-6"
VERSION = "v1"
MAX_WORKERS = 8
TEMPERATURE = 0.0
STRIP_COUNT = 3
STRIP_OVERLAP = 0.1
STRIP_WIDTH = 1600

BASE_DIR = Path(__file__).parent.parent
PAGES_ROOT = BASE_DIR / "input/pkna"
V2_ROOT = BASE_DIR / "output/extract-emotional/v2"
OUT_ROOT = BASE_DIR / f"output/observations/{VERSION}"

# Curated identification cues, verified against pages from several issues.
CAST_NOTES = {
    "Uno": (
        "Appare come una grande sfera olografica verde e traslucida, punteggiata "
        "di bolle, con un volto; su schermi e comunicatori si vede il suo volto "
        "verde. La sua voce è scritta in balloon dal contorno a punte (voce "
        "elettronica), anche quando proviene da un dispositivo come il "
        "comunicatore da polso o dalla torre."
    ),
}

PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)


class PageObserver(dspy.Signature):
    """Record what a comic book page shows, panel by panel, without interpretation.

    These records are the factual base for later analysis of who knows what and
    when, so they must contain only what a reader can see on this page.

    What to record:
    - Only what is drawn or lettered on this page. Do not explain motives, hidden
      threats, secrets, or what characters know or intend, and do not anticipate
      later events. If something is not drawn, do not describe it.
    - Descriptions cover characters, actions, poses, objects, setting, and framing.
      Do not repeat or summarize the lettering: it is recorded separately in texts.

    Panels and reading order:
    - Keep panels in reading order and do not merge adjacent panels.
    - Within a panel, order texts by conversational coherence (questions before
      answers, calls before responses), not by rigid top-to-bottom position.
    - A balloon belongs to the panel where its tail originates.

    Texts:
    - Record every balloon, caption, and legible in-world text, including sound
      effects (kind "sfx").
    - Thought balloons (cloud shape or a trail of small bubbles) have kind "thought".
    - Everything inside a balloon is speech or thought, including interjections
      such as "Eeeh?!". Sound effects are lettered outside balloons and have no
      speaker.
    - Normalize text: normal caps instead of all caps, no line-break hyphens,
      accented letters instead of apostrophes where appropriate.

    Speakers:
    - For each balloon, look at its outline and where its tail points, and record
      them before the text and speaker. Decide the speaker from these visual
      observations; use what the line says only to break ties. A plausible-sounding
      line is not evidence of who said it.
    - The balloon tail determines the speaker. Follow it even when another
      character is more prominent in the panel.
    - A balloon style that the cast sheet associates with a character is strong
      evidence for that speaker, especially when the tail leads off-panel or to a
      building or device.
    - When the tail points to a TV, radio, screen, or loudspeaker, set "via" and
      attribute the line to the actual source if identifiable, otherwise use a
      descriptive label such as "Voce dalla TV".
    - Use dialogue content as a check: a character does not call out their own
      name or alias.
    - Use the cast sheet names. Name the persona as depicted: a character in
      civilian clothes and the same character in costume may have different names
      in the cast sheet.
    - Unknown characters get a short descriptive label (in Italian); list them in
      unlisted_characters with a brief appearance description.

    Characters:
    - List everyone drawn in each panel, whether or not they speak, and speakers
      who are present but outside the frame (depiction "off_panel").

    Locations:
    - Name a place only when the drawing or captions identify it. Otherwise
      describe the setting generically (e.g. "città vista dall'alto, di notte").
      Do not guess planets, cities, or buildings.

    Language: write descriptions, locations, expressions, and labels in Italian,
    the language of the comic.
    """

    page: dspy.Image = dspy.InputField(desc="The comic book page image.")
    page_strips: list[dspy.Image] = dspy.InputField(
        desc=(
            "The same page split into overlapping horizontal strips, top to bottom, "
            "enlarged to show balloon outlines, tails, and small lettering."
        )
    )
    cast_sheet: list[CastMember] = dspy.InputField(
        desc="Characters known to appear in this issue, with appearance descriptions where available."
    )
    previous_page_dialogue: list[str] = dspy.InputField(
        desc="Last dialogue lines of the previous page, for speaker continuity. May contain errors."
    )

    printed_page_number: int | None = dspy.OutputField(
        desc="The page number printed on the page, if visible."
    )
    panels: list[ObservedPanel] = dspy.OutputField(
        desc="The panels of the page in reading order."
    )
    unlisted_characters: list[CastMember] = dspy.OutputField(
        desc="Characters on this page missing from the cast sheet: the label used and a brief appearance description."
    )


@dataclass
class PageTask:
    source: PageSource
    image_path: Path
    cast: list[CastMember]
    previous_dialogue: list[str]


@dataclass
class PageObservation:
    printed_page_number: int | None
    panels: list[ObservedPanel]
    unlisted_characters: list[CastMember]
    lm_usage: dict | None = None


ObserveFn = Callable[[PageTask], PageObservation]


def compute_config_id(model_name: str) -> str:
    """Hash of everything that changes the output for the same page."""
    payload = {
        "model": model_name,
        "schema_version": SCHEMA_VERSION,
        "instructions": PageObserver.instructions,
        "fields": {
            name: field.json_schema_extra for name, field in PageObserver.fields.items()
        },
        "panel_schema": ObservedPanel.model_json_schema(),
        "cast_notes": CAST_NOTES,
        "temperature": TEMPERATURE,
        "strips": [STRIP_COUNT, STRIP_OVERLAP, STRIP_WIDTH],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def output_path(out_root: Path, source: PageSource) -> Path:
    return out_root / source.issue / f"{Path(source.image).stem}.json"


def is_done(path: Path, config_id: str) -> bool:
    if not path.exists():
        return False
    meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
    return meta.get("config_id") == config_id


def build_tasks(
    issue: str,
    pages_root: Path,
    v2_root: Path,
    only_pages: set[str] | None = None,
) -> list[PageTask]:
    """Build one task per page image, optionally restricted to image stems."""
    v2_pages = load_v2_pages(v2_root / issue)
    cast = apply_cast_notes(build_cast_sheet(list(v2_pages.values())), CAST_NOTES)

    tasks: list[PageTask] = []
    previous_image: str | None = None
    for source, image_path in list_issue_pages(pages_root / issue):
        previous = v2_pages.get(previous_image) if previous_image else None
        previous_image = source.image
        if only_pages is not None and image_path.stem not in only_pages:
            continue
        tasks.append(
            PageTask(
                source=source,
                image_path=image_path,
                cast=cast,
                previous_dialogue=previous_page_dialogue(previous),
            )
        )
    return tasks


def write_record(
    out_root: Path,
    task: PageTask,
    obs: PageObservation,
    model_name: str,
    config_id: str,
) -> None:
    record = ObservationRecord(
        source=task.source,
        printed_page_number=obs.printed_page_number,
        panels=obs.panels,
        unlisted_characters=obs.unlisted_characters,
        meta={
            "model_name": model_name,
            "config_id": config_id,
            "schema_version": SCHEMA_VERSION,
            "lm_usage": obs.lm_usage,
            "context": {
                "cast_size": len(task.cast),
                "previous_dialogue": task.previous_dialogue,
            },
        },
    )
    path = output_path(out_root, task.source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def record_failure(
    out_root: Path, source: PageSource, config_id: str, error: Exception
) -> None:
    path = out_root / source.issue / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "image": source.image,
        "config_id": config_id,
        "error": repr(error),
        "time": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_tasks(
    tasks: list[PageTask],
    observe: ObserveFn,
    out_root: Path,
    model_name: str,
    config_id: str,
    workers: int = MAX_WORKERS,
) -> tuple[int, int]:
    """Run pending tasks in parallel. Returns (succeeded, failed) counts."""
    pending = [
        t for t in tasks if not is_done(output_path(out_root, t.source), config_id)
    ]
    skipped = len(tasks) - len(pending)
    if skipped:
        log.info(f"Skipping {skipped} pages already extracted with config {config_id}")
    if not pending:
        return 0, 0

    succeeded = failed = 0
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
        PROGRESS as progress,
    ):
        futures = {executor.submit(observe, t): t for t in pending}
        for future in progress.track(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            description="Extracting pages...",
        ):
            task = futures[future]
            try:
                obs = future.result()
            except Exception as e:
                log.exception(f"Failed on {task.source.page_id}: {e}")
                record_failure(out_root, task.source, config_id, e)
                failed += 1
                continue
            write_record(out_root, task, obs, model_name, config_id)
            succeeded += 1
    return succeeded, failed


def jpeg_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def observe_with_dspy(task: PageTask) -> PageObservation:
    with Image.open(task.image_path) as page:
        strips = [
            dspy.Image(jpeg_data_uri(s))
            for s in page_strips(page, STRIP_COUNT, STRIP_OVERLAP, STRIP_WIDTH)
        ]
    # A fresh module per call keeps worker threads from sharing module state.
    module = dspy.ChainOfThought(PageObserver)
    pred = module(
        page=dspy.Image(task.image_path.as_posix()),
        page_strips=strips,
        cast_sheet=task.cast,
        previous_page_dialogue=task.previous_dialogue,
    )
    return PageObservation(
        printed_page_number=pred.printed_page_number,
        panels=pred.panels,
        unlisted_characters=pred.unlisted_characters,
        lm_usage=pred.get_lm_usage(),
    )


def configure_lm(model_name: str) -> None:
    load_dotenv()
    lm = dspy.LM(model=model_name, temperature=TEMPERATURE, max_tokens=60000)
    dspy.configure(lm=lm, track_usage=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the observation layer from comic page images"
    )
    parser.add_argument("--issues", nargs="+", default=["pkna-0"])
    parser.add_argument(
        "--pages",
        nargs="+",
        default=None,
        help="Restrict to these image stems, e.g. pkna-0-069 pkna-0-070",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    configure_lm(args.model)
    config_id = compute_config_id(args.model)
    only_pages = set(args.pages) if args.pages else None
    tasks = [
        t
        for issue in args.issues
        for t in build_tasks(issue, PAGES_ROOT, V2_ROOT, only_pages)
    ]
    console.print(
        f"[bold cyan]Observation extraction[/bold cyan]: {len(tasks)} pages, "
        f"model {args.model}, config {config_id}"
    )

    succeeded, failed = process_tasks(
        tasks, observe_with_dspy, args.out_dir, args.model, config_id, args.workers
    )
    console.print(f"Succeeded: {succeeded}, failed: {failed}. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
