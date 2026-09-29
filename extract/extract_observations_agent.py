#!/usr/bin/env python3
"""Extract the observation layer with Cursor agents, several pages per request.

Each batch of consecutive pages runs as one headless `cursor-agent` session in
an isolated workspace holding only the page images, enlarged strips, the cast
sheet, the output schema, and instructions. The agent zooms into balloons it
cannot resolve and validates every file it writes. Plot summaries and earlier
extractions are not reachable from the workspace, so records describe only
what the pages show.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from extract.extract_observations import (
    CAST_NOTES,
    STRIP_COUNT,
    STRIP_OVERLAP,
    STRIP_WIDTH,
    PageObserver,
    is_done,
    output_path,
    record_failure,
)
from pkna.extract.observations import (
    SCHEMA_VERSION,
    CastMember,
    ObservationRecord,
    ObservedPage,
    PageSource,
    apply_cast_notes,
    build_cast_sheet,
    list_issue_pages,
    load_v2_pages,
    page_problems,
    page_strips,
)
from pkna.llm.cursor_agent import run_cursor_agent
from pkna.logging import setup_logging

console, log = setup_logging()

DEFAULT_MODEL = "claude-opus-5-5-high"
VERSION = "v2"
BATCH_SIZE = 8
MAX_PARALLEL = 4
ROUNDS = 3
AGENT_TIMEOUT_S = 90 * 60

BASE_DIR = Path(__file__).parent.parent
PAGES_ROOT = BASE_DIR / "input/pkna"
V2_ROOT = BASE_DIR / "output/extract-emotional/v2"
OUT_ROOT = BASE_DIR / f"output/observations/{VERSION}"
TOOLS = shlex.join(
    [sys.executable, str(BASE_DIR / "extract/observation_agent_tools.py")]
)

PROMPT = "Follow the instructions in INSTRUCTIONS.md in this directory."

INSTRUCTIONS = """\
# Observation extraction

Record what each page of an Italian comic book (PKNA, Paperinik New Adventures)
shows, one JSON file per page.

Pages to extract, in reading order:
{pages}
{context}
## Workflow

Process the pages one at a time, in order. For each page:

1. Read `pages/<name>.jpg`, then its enlarged strips `strips/<name>-1.jpg` to
   `strips/<name>-{strip_count}.jpg` (top to bottom).
2. Zoom into every balloon whose outline or tail you cannot see clearly, and
   always before attributing a line whose tail does not clearly point at a
   character:
   `{tools} zoom pages/<name>.jpg X0 Y0 X1 Y1`
   The box is in fractions of the page width and height (0 to 1). Read the
   image file it prints.
3. Write `out/<name>.json`, a JSON object that follows `schema.json`.
4. Run `{tools} validate out/<name>.json` and fix every problem it reports.

Use only the files in this directory: do not look up plot summaries or other
information about the comic. `cast.json` is the cast sheet: characters known to
appear in this issue, with appearance descriptions where available.

When every page file is written and valid, reply with the list of files written.

## Recording rules

{rules}
"""

PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)


@dataclass
class Batch:
    pages: list[tuple[PageSource, Path]]
    context: Path | None
    cast: list[CastMember]


# Runs the agent in a prepared workspace with a model; returns run metadata.
AgentRunner = Callable[[Path, str], dict]


def compute_config_id(model_name: str) -> str:
    """Hash of everything that changes the output for the same page."""
    payload = {
        "model": model_name,
        "extractor": "cursor-agent",
        "schema_version": SCHEMA_VERSION,
        "instructions": INSTRUCTIONS,
        "rules": PageObserver.instructions,
        "page_schema": ObservedPage.model_json_schema(),
        "cast_notes": CAST_NOTES,
        "strips": [STRIP_COUNT, STRIP_OVERLAP, STRIP_WIDTH],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def make_batches(
    issue_pages: list[tuple[PageSource, Path]],
    pending: set[str],
    cast: list[CastMember],
    size: int,
) -> list[Batch]:
    """Group pending pages into runs of consecutive pages of at most `size`.

    Each batch gets the page before its first page as context, so the agent can
    follow a conversation across the batch boundary.
    """
    groups: list[list[int]] = []
    for i, (source, _) in enumerate(issue_pages):
        if source.image not in pending:
            continue
        if groups and groups[-1][-1] == i - 1 and len(groups[-1]) < size:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [
        Batch(
            pages=[issue_pages[i] for i in group],
            context=issue_pages[group[0] - 1][1] if group[0] > 0 else None,
            cast=cast,
        )
        for group in groups
    ]


def render_instructions(batch: Batch) -> str:
    pages = "\n".join(f"- {path.stem}" for _, path in batch.pages)
    context = ""
    if batch.context:
        context = (
            f"\n`context/{batch.context.stem}.jpg` is the page before the first one. "
            "Read it to follow the conversation into the first page, but do not "
            "write a file for it.\n"
        )
    return INSTRUCTIONS.format(
        pages=pages,
        context=context,
        strip_count=STRIP_COUNT,
        tools=TOOLS,
        rules=PageObserver.instructions,
    )


def prepare_workspace(batch: Batch, workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    for sub in ("pages", "strips", "context", "out"):
        (workspace / sub).mkdir(parents=True)
    for _, path in batch.pages:
        shutil.copyfile(path, workspace / "pages" / f"{path.stem}.jpg")
        with Image.open(path) as page:
            strips = page_strips(page, STRIP_COUNT, STRIP_OVERLAP, STRIP_WIDTH)
        for i, strip in enumerate(strips, 1):
            strip.convert("RGB").save(
                workspace / "strips" / f"{path.stem}-{i}.jpg", quality=90
            )
    if batch.context:
        shutil.copyfile(
            batch.context, workspace / "context" / f"{batch.context.stem}.jpg"
        )
    cast = [c.model_dump(exclude_none=True) for c in batch.cast]
    (workspace / "cast.json").write_text(
        json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (workspace / "schema.json").write_text(
        json.dumps(ObservedPage.model_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (workspace / "INSTRUCTIONS.md").write_text(
        render_instructions(batch), encoding="utf-8"
    )


def run_agent(workspace: Path, model: str) -> dict:
    return run_cursor_agent(workspace, model, PROMPT, AGENT_TIMEOUT_S)


def collect_batch(
    batch: Batch,
    workspace: Path,
    out_root: Path,
    model_name: str,
    config_id: str,
    agent: dict,
) -> tuple[int, int]:
    """Turn valid page files into records; log the rest as failures."""
    succeeded = failed = 0
    for source, path in batch.pages:
        page_file = workspace / "out" / f"{path.stem}.json"
        try:
            page = ObservedPage.model_validate_json(
                page_file.read_text(encoding="utf-8")
            )
            problems = page_problems(page)
            if problems:
                raise ValueError("; ".join(problems))
        except (OSError, ValueError) as e:
            record_failure(out_root, source, config_id, e)
            failed += 1
            continue
        record = ObservationRecord(
            source=source,
            printed_page_number=page.printed_page_number,
            panels=page.panels,
            unlisted_characters=page.unlisted_characters,
            meta={
                "model_name": model_name,
                "config_id": config_id,
                "schema_version": SCHEMA_VERSION,
                "extractor": "cursor-agent",
                "agent": agent,
                "batch": [s.image for s, _ in batch.pages],
                "context": {
                    "cast_size": len(batch.cast),
                    "context_page": batch.context.name if batch.context else None,
                },
            },
        )
        dest = output_path(out_root, source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        succeeded += 1
    return succeeded, failed


def process_batch(
    batch: Batch,
    run_agent: AgentRunner,
    model_name: str,
    out_root: Path,
    config_id: str,
) -> tuple[int, int]:
    """Run one batch. Valid pages are kept even when the agent run fails midway."""
    # Page file names repeat across issues, so the first page alone is not unique.
    first_source, first_path = batch.pages[0]
    workspace = out_root / "_work" / first_source.issue / first_path.stem
    prepare_workspace(batch, workspace)
    try:
        agent = run_agent(workspace, model_name)
    except (
        subprocess.SubprocessError,
        RuntimeError,
        json.JSONDecodeError,
        OSError,
    ) as e:
        log.error(f"Agent run failed for {workspace}: {e}")
        agent = {"error": repr(e)}
    succeeded, failed = collect_batch(
        batch, workspace, out_root, model_name, config_id, agent
    )
    if failed:
        log.warning(f"{failed} pages failed; workspace kept at {workspace}")
    else:
        shutil.rmtree(workspace)
    return succeeded, failed


def issue_batches(
    issue: str,
    out_root: Path,
    config_id: str,
    batch_size: int,
    only_pages: set[str] | None = None,
    pages_root: Path = PAGES_ROOT,
    v2_root: Path = V2_ROOT,
) -> list[Batch]:
    """Batches of the issue's pages not yet extracted with this config."""
    issue_pages = list_issue_pages(pages_root / issue)
    v2_pages = load_v2_pages(v2_root / issue)
    cast = apply_cast_notes(build_cast_sheet(list(v2_pages.values())), CAST_NOTES)
    pending = {
        source.image
        for source, path in issue_pages
        if (only_pages is None or path.stem in only_pages)
        and not is_done(output_path(out_root, source), config_id)
    }
    return make_batches(issue_pages, pending, cast, batch_size)


def process_batches(
    batches: list[Batch],
    run_agent: AgentRunner,
    model_name: str,
    out_root: Path,
    config_id: str,
    parallel: int = MAX_PARALLEL,
) -> tuple[int, int]:
    """Run batches in parallel. Returns (succeeded, failed) page counts."""
    succeeded = failed = 0
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor,
        PROGRESS as progress,
    ):
        futures = [
            executor.submit(
                process_batch, b, run_agent, model_name, out_root, config_id
            )
            for b in batches
        ]
        for future in progress.track(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            description="Extracting batches...",
        ):
            ok, bad = future.result()
            succeeded += ok
            failed += bad
    return succeeded, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the observation layer with Cursor agents"
    )
    parser.add_argument(
        "--issues", nargs="+", default=["pkna-0"], help="Issue directories, or 'all'"
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        default=None,
        help="Restrict to these image stems, e.g. pkna-0-069 pkna-0-070",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="Passes over the pages still missing, so failed pages are retried",
    )
    args = parser.parse_args()

    issues = args.issues
    if issues == ["all"]:
        issues = sorted(p.name for p in PAGES_ROOT.iterdir() if p.is_dir())
    config_id = compute_config_id(args.model)
    only_pages = set(args.pages) if args.pages else None
    for round_number in range(1, args.rounds + 1):
        batches = [
            b
            for issue in issues
            for b in issue_batches(
                issue, args.out_dir, config_id, args.batch_size, only_pages
            )
        ]
        if not batches:
            break
        pages = sum(len(b.pages) for b in batches)
        console.print(
            f"[bold cyan]Agent observation extraction[/bold cyan], round "
            f"{round_number}: {pages} pages in {len(batches)} batches, model "
            f"{args.model}, config {config_id}"
        )
        succeeded, failed = process_batches(
            batches,
            run_agent,
            args.model,
            args.out_dir,
            config_id,
            args.parallel,
        )
        console.print(f"Succeeded: {succeeded}, failed: {failed}")
    console.print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
