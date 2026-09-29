#!/usr/bin/env python3
"""Resolve the character labels of each issue with a Cursor agent.

Observation records name characters as each page shows them, so one character
can appear under several labels (a descriptive label before being named, a
civilian and a costumed persona). One agent session per issue groups the
labels into characters, from a transcript of the issue's records and the page
images. Issues run in parallel; a global merge links characters across issues.
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
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from extract.compare_observations import load_records
from extract.extract_observations import CAST_NOTES
from pkna.extract.observations import (
    ObservationRecord,
    apply_cast_notes,
    build_cast_sheet,
    list_issue_pages,
    load_v2_pages,
)
from pkna.extract.registry import (
    IssueCast,
    cast_problems,
    collect_mentions,
    render_transcript,
)
from pkna.llm.cursor_agent import run_cursor_agent
from pkna.logging import setup_logging

console, log = setup_logging()

DEFAULT_MODEL = "claude-opus-5-5-high"
VERSION = "v1"
MAX_PARALLEL = 8
ROUNDS = 2
AGENT_TIMEOUT_S = 90 * 60

BASE_DIR = Path(__file__).parent.parent
PAGES_ROOT = BASE_DIR / "input/pkna"
OBS_ROOT = BASE_DIR / "output/observations/v2"
V2_ROOT = BASE_DIR / "output/extract-emotional/v2"
SERIES_PATH = BASE_DIR / "data/registry/series.json"
OUT_ROOT = BASE_DIR / f"output/registry/{VERSION}/issues"
TOOLS = shlex.join(
    [sys.executable, str(BASE_DIR / "extract/observation_agent_tools.py")]
)

PROMPT = "Follow the instructions in INSTRUCTIONS.md in this directory."

INSTRUCTIONS = """\
# Character resolution

The observation records of one issue of an Italian comic book (PKNA, Paperinik
New Adventures) name characters as each page shows them: by name when the page
or the cast sheet gives one, otherwise by a descriptive label such as
"Evroniano in frac viola". One character can appear under several labels, for
example a descriptive label on pages before the one where they are named.

Group the labels into characters and write `out/cast.json`, a JSON object that
follows `schema.json`.

## Files

- `transcript.md`: the issue's records, page by page. Each panel lists its
  location, the characters present, a description of what is drawn, and the
  lettering with speakers and addressees.
- `mentions.json`: every label, with how often it speaks, appears, and is
  addressed, the pages where it occurs, appearance descriptions, and sample lines.
- `series.json`: identities established by the premise of the series.
- `pages/`: the page images, named as in the transcript.

## Workflow

1. Read `mentions.json` and the whole transcript.
2. For labels that may be the same character, compare appearance descriptions,
   the pages where they occur, what they say, and how others address them. When
   the text is not conclusive, look at the page images; to enlarge a region run
   `{tools} zoom "pages/<name>.jpg" X0 Y0 X1 Y1` (fractions of the page width
   and height) and read the image file it prints.
3. Write `out/cast.json`, run `{tools} validate-cast out/cast.json`, and fix
   every problem it reports.

Use only the files in this directory: do not look up plot summaries or other
information about the comic.

## Rules

- Every label in `mentions.json` belongs to exactly one persona of one
  character. Copy labels exactly as written.
- Link a descriptive label to a named character only on evidence from the
  pages: the character is named on the same or a nearby page, the appearance
  matches (clothes, colors, features), or the dialogue identifies them. Record
  the evidence with page names. When the evidence is weak, keep them separate.
- A character with a secret or alternate identity has one persona per
  identity; put each label under the persona it depicts. `series.json` lists the
  identities known from the premise of the series.
- Name each character with the most complete name the pages give (e.g.
  "Paperilla Starry" rather than "Paperilla").
- A label that may cover several different individuals, such as "Soldato
  evroniano" used for different soldiers, is "unnamed" or "group" even if one
  of those individuals is named elsewhere.
- A voice from a device (e.g. "Voce dalla TV") belongs to the character who is
  speaking when the pages identify them; otherwise it is "unnamed".
- Write names of unnamed characters, descriptions, and evidence in Italian.

When `out/cast.json` is valid, reply with the number of characters written.
"""

PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)

# Runs the agent in a prepared workspace with a model; returns run metadata.
AgentRunner = Callable[[Path, str], dict]


def compute_config_id(model_name: str) -> str:
    """Hash of everything besides the input records that changes the output."""
    payload = {
        "model": model_name,
        "instructions": INSTRUCTIONS,
        "cast_schema": IssueCast.model_json_schema(),
        "series": SERIES_PATH.read_text(encoding="utf-8"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def issue_inputs(
    records: list[ObservationRecord], v2_issue_dir: Path
) -> tuple[str, list[dict]]:
    """The transcript and label mentions an agent works from."""
    cast = apply_cast_notes(
        build_cast_sheet(list(load_v2_pages(v2_issue_dir).values())), CAST_NOTES
    )
    mentions = [m.model_dump() for m in collect_mentions(records, cast)]
    return render_transcript(records), mentions


def input_hash(transcript: str, mentions: list[dict]) -> str:
    payload = transcript + json.dumps(mentions, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def output_path(out_root: Path, issue: str) -> Path:
    return out_root / f"{issue}.json"


def is_done(path: Path, config_id: str, inputs_id: str) -> bool:
    if not path.exists():
        return False
    meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
    return meta.get("config_id") == config_id and meta.get("input_hash") == inputs_id


def prepare_workspace(
    workspace: Path,
    transcript: str,
    mentions: list[dict],
    page_paths: list[Path],
    series_path: Path,
) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "pages").mkdir(parents=True)
    (workspace / "out").mkdir()
    for path in page_paths:
        shutil.copyfile(path, workspace / "pages" / f"{path.stem}.jpg")
    (workspace / "transcript.md").write_text(transcript, encoding="utf-8")
    (workspace / "mentions.json").write_text(
        json.dumps(mentions, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    shutil.copyfile(series_path, workspace / "series.json")
    (workspace / "schema.json").write_text(
        json.dumps(IssueCast.model_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (workspace / "INSTRUCTIONS.md").write_text(
        INSTRUCTIONS.format(tools=TOOLS), encoding="utf-8"
    )


def record_failure(out_root: Path, issue: str, config_id: str, error: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    entry = {
        "issue": issue,
        "config_id": config_id,
        "error": error,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_root / "failures.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def resolve_issue(
    issue: str,
    records: list[ObservationRecord],
    page_paths: list[Path],
    run_agent: AgentRunner,
    model_name: str,
    out_root: Path,
    config_id: str,
    v2_root: Path = V2_ROOT,
    series_path: Path = SERIES_PATH,
) -> bool:
    """Resolve one issue. Returns whether a valid resolution was written."""
    transcript, mentions = issue_inputs(records, v2_root / issue)
    inputs_id = input_hash(transcript, mentions)
    workspace = out_root / "_work" / issue
    prepare_workspace(workspace, transcript, mentions, page_paths, series_path)
    try:
        agent = run_agent(workspace, model_name)
    except (subprocess.SubprocessError, RuntimeError, json.JSONDecodeError) as e:
        log.error(f"Agent run failed for {issue}: {e}")
        agent = {"error": repr(e)}
    try:
        cast = IssueCast.model_validate_json(
            (workspace / "out" / "cast.json").read_text(encoding="utf-8")
        )
        problems = cast_problems(cast, {m["label"] for m in mentions})
        if problems:
            raise ValueError("; ".join(problems))
    except (OSError, ValueError) as e:
        record_failure(out_root, issue, config_id, repr(e))
        log.warning(f"No valid resolution for {issue}; workspace kept at {workspace}")
        return False
    result = {
        "issue": issue,
        "cast": cast.model_dump(),
        "meta": {
            "model_name": model_name,
            "config_id": config_id,
            "input_hash": inputs_id,
            "agent": agent,
        },
    }
    output_path(out_root, issue).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.rmtree(workspace)
    return True


def run_agent(workspace: Path, model: str) -> dict:
    return run_cursor_agent(workspace, model, PROMPT, AGENT_TIMEOUT_S)


def complete_issues(
    issues: list[str], pages_root: Path, obs_root: Path
) -> dict[str, tuple[list[ObservationRecord], list[Path]]]:
    """Records and page images of the issues whose pages are all extracted."""
    ready: dict[str, tuple[list[ObservationRecord], list[Path]]] = {}
    for issue in issues:
        pages = [path for _, path in list_issue_pages(pages_root / issue)]
        issue_dir = obs_root / issue
        records = load_records(issue_dir) if issue_dir.is_dir() else []
        if len(records) < len(pages):
            log.info(
                f"Skipping {issue}: {len(records)} of {len(pages)} pages extracted"
            )
            continue
        ready[issue] = (records, pages)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve character labels per issue with Cursor agents"
    )
    parser.add_argument(
        "--issues", nargs="+", default=["pkna-0"], help="Issue directories, or 'all'"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    issues = args.issues
    if issues == ["all"]:
        issues = sorted(p.name for p in PAGES_ROOT.iterdir() if p.is_dir())
    config_id = compute_config_id(args.model)
    ready = complete_issues(issues, PAGES_ROOT, OBS_ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for round_number in range(1, args.rounds + 1):
        pending = [
            issue
            for issue, (records, _) in ready.items()
            if not is_done(
                output_path(args.out_dir, issue),
                config_id,
                input_hash(*issue_inputs(records, V2_ROOT / issue)),
            )
        ]
        if not pending:
            break
        console.print(
            f"[bold cyan]Character resolution[/bold cyan], round {round_number}: "
            f"{len(pending)} issues, model {args.model}, config {config_id}"
        )
        succeeded = 0
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex,
            PROGRESS as progress,
        ):
            futures = [
                ex.submit(
                    resolve_issue,
                    issue,
                    *ready[issue],
                    run_agent,
                    args.model,
                    args.out_dir,
                    config_id,
                )
                for issue in pending
            ]
            for future in progress.track(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                description="Resolving issues...",
            ):
                succeeded += future.result()
        console.print(f"Succeeded: {succeeded}, failed: {len(pending) - succeeded}")
    console.print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
