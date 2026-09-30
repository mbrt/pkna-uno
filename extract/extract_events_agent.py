#!/usr/bin/env python3
"""Build the event and knowledge log of each issue with a Claude Code agent.

One agent session per issue reads the issue's observation records, with
character labels resolved, and writes its scenes: place, events, and changes in
what characters know. The log is derived from text only; page images were read
once by the observation layer. Issues run in parallel.
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

from extract.event_agent_tools import load_log
from extract.resolve_characters_agent import complete_issues
from pkna.extract.events import (
    IssueFacts,
    IssueIndex,
    Scene,
    assemble_log,
    build_index,
    log_problems,
    render_log_transcript,
)
from pkna.extract.observations import ObservationRecord
from pkna.extract.registry import IssueCast
from pkna.llm.claude_agent import run_claude_agent
from pkna.logging import setup_logging

console, log = setup_logging()

DEFAULT_MODEL = "claude-opus-5-5"
VERSION = "v1"
MAX_PARALLEL = 8
ROUNDS = 2
AGENT_TIMEOUT_S = 120 * 60

BASE_DIR = Path(__file__).parent.parent
PAGES_ROOT = BASE_DIR / "input/pkna"
OBS_ROOT = BASE_DIR / "output/observations/v2"
CASTS_ROOT = BASE_DIR / "output/registry/v1/issues"
SERIES_FACTS_PATH = BASE_DIR / "data/events/series-facts.json"
OUT_ROOT = BASE_DIR / f"output/events/{VERSION}"
TOOLS = shlex.join([sys.executable, str(BASE_DIR / "extract/event_agent_tools.py")])

PROMPT = "Follow the instructions in INSTRUCTIONS.md in this directory."

INSTRUCTIONS = """\
# Event and knowledge log

`transcript.md` records one issue of an Italian comic book (PKNA, Paperinik
New Adventures) page by page and panel by panel: where each panel takes place,
who is drawn in it, what is drawn, and the lettering with speakers and
addressees. Turn it into a log of scenes: what happens in each scene, and who
comes to know, suspect, or reject which facts.

## Files

- `transcript.md`: each panel is headed by its reference, e.g.
  `[pkna-0-012#p3]`, and each line of lettering starts with its number in the
  panel, e.g. `t2`, so that line's reference is `pkna-0-012#p3.t2`. Characters
  are named as in `characters.json`, with the persona in brackets for
  characters who have more than one (e.g. a civilian and a costumed identity).
  `NUOVA SCENA` marks where the page-by-page extraction saw a change of place
  or time.
- `characters.json`: the characters of the issue. Refer to characters by these
  names, exactly.
- `series-facts.json`: facts from the premise of the series, with `serie:` ids.
  Refer to them by id instead of redefining them.
- `scene-schema.json` and `facts-schema.json`: the formats of the files to write.

## Workflow

1. Read the whole transcript.
2. Divide the issue into scenes. Each scene starts at a panel and lasts until
   the next scene starts; together the scenes cover every panel. Start a new
   scene at every change of place or time, including cuts back and forth
   between places. The `NUOVA SCENA` markers are hints: the page-by-page
   extraction can miss a change at a page turn, or mark one where the scene
   continues.
3. Write the log in batches of about ten scenes, in reading order: write each
   scene as `out/scenes/NNN.json` (`001.json`, `002.json`, ...) following
   `scene-schema.json`, and add the facts the batch uses to `out/facts.json`
   following `facts-schema.json`. Write each batch to disk before working out
   the next one; do not compose the whole log in a single step.
4. Run `{tools} validate-log out` and fix every problem it reports.

If `out/` already holds scenes and facts from an interrupted session, check that
they are consistent, and continue from the panel where they end instead of
starting over.

Use only the files in this directory: do not read files outside it (the
validator's messages say what to fix), keep any scratch files in this
directory rather than in `/tmp`, and do not look up plot summaries or other
information about the comic.

## Rules

- Work from the transcript: what is drawn, said, and written. Do not add
  motives, feelings, or explanations it does not show.
- Places go from general to specific. Use the same names for the same place
  throughout the issue, taken from captions and dialogue when they give one.
- Events are the actions and happenings that change the situation: arrivals
  and departures, fights, captures, rescues and escapes, uses of devices and
  powers, transformations, discoveries, announced decisions, deaths.
  Participants act or are acted on; witnesses are other characters who see or
  hear it happen.
- Record a fact when a character learns, reveals, hides, suspects, or is wrong
  about it, and it can matter later: identities and secret identities, what
  someone or something is, plans and intentions, threats, where characters and
  objects are, what happened off-page or in the past, abilities and weaknesses,
  allegiances, lies. Skip small talk and what everyone present obviously knows.
- For each fact, record a knowledge change for every character concerned, where
  it happens: whoever is told it (the speaker is the informant), overhears it,
  sees it happen, or deduces it; and whoever shows by words or behavior that
  they already knew it, the first time this shows. Only characters present in
  the scene, or reached through a device, learn from it.
- A lie is a false fact that a listener believes. When the liar's knowledge of
  the truth matters, record the true fact too, already known to the liar.
- Knowledge of secret identities matters: when a character learns or shows
  they know that two personas are the same person, record it.
- Cite the lettering (`...#p3.t2`) where a knowledge change happens when there
  is one, otherwise the panel. Every reference of a scene's events and
  knowledge changes must be inside that scene.
- Write statements, summaries, descriptions, and place names in Italian.

When the log is valid, reply with the number of scenes and facts written.
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


def compute_config_id(model_name: str, series_path: Path = SERIES_FACTS_PATH) -> str:
    """Hash of everything besides the issue inputs that changes the output."""
    payload = {
        "model": model_name,
        "instructions": INSTRUCTIONS,
        "scene_schema": Scene.model_json_schema(),
        "facts_schema": IssueFacts.model_json_schema(),
        "series_facts": series_path.read_text(encoding="utf-8"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def issue_inputs(
    records: list[ObservationRecord], cast: IssueCast
) -> tuple[str, list[dict]]:
    """The transcript and character list an agent works from."""
    characters = [
        {
            "name": c.name,
            "kind": c.kind,
            "personas": [p.name for p in c.personas],
            "description": c.description,
        }
        for c in cast.characters
    ]
    return render_log_transcript(records, cast), characters


def input_hash(transcript: str, characters: list[dict]) -> str:
    payload = transcript + json.dumps(characters, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def output_path(out_root: Path, issue: str) -> Path:
    return out_root / f"{issue}.json"


def is_done(
    path: Path, config_id: str, inputs_id: str, redo_stale: bool = False
) -> bool:
    """Whether a log exists for these inputs.

    A log written with other instructions or model still counts, unless
    redo_stale: changes that only make runs cheaper should not redo good logs.
    """
    if not path.exists():
        return False
    meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
    if meta.get("input_hash") != inputs_id:
        return False
    return not redo_stale or meta.get("config_id") == config_id


def load_cast(casts_root: Path, issue: str) -> IssueCast | None:
    path = casts_root / f"{issue}.json"
    if not path.exists():
        return None
    return IssueCast.model_validate(
        json.loads(path.read_text(encoding="utf-8"))["cast"]
    )


def same_inputs(workspace: Path, transcript: str, characters_json: str) -> bool:
    try:
        old_transcript = (workspace / "transcript.md").read_text(encoding="utf-8")
        old_characters = (workspace / "characters.json").read_text(encoding="utf-8")
    except OSError:
        return False
    return old_transcript == transcript and old_characters == characters_json


def prepare_workspace(
    workspace: Path,
    transcript: str,
    characters: list[dict],
    index: IssueIndex,
    series_path: Path,
) -> None:
    """Write the agent's inputs.

    Output left by an interrupted session on the same inputs is kept, so that a
    retry continues from it instead of paying for the same work again.
    """
    characters_json = json.dumps(characters, ensure_ascii=False, indent=1)
    if workspace.exists() and not same_inputs(workspace, transcript, characters_json):
        shutil.rmtree(workspace)
    (workspace / "out" / "scenes").mkdir(parents=True, exist_ok=True)
    (workspace / "transcript.md").write_text(transcript, encoding="utf-8")
    (workspace / "characters.json").write_text(characters_json, encoding="utf-8")
    (workspace / "index.json").write_text(index.model_dump_json(), encoding="utf-8")
    shutil.copyfile(series_path, workspace / "series-facts.json")
    for name, model in (
        ("scene-schema.json", Scene),
        ("facts-schema.json", IssueFacts),
    ):
        (workspace / name).write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2),
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


def log_issue(
    issue: str,
    records: list[ObservationRecord],
    cast: IssueCast,
    run_agent: AgentRunner,
    model_name: str,
    out_root: Path,
    config_id: str,
    series_path: Path = SERIES_FACTS_PATH,
) -> bool:
    """Build one issue's log. Returns whether a valid log was written."""
    transcript, characters = issue_inputs(records, cast)
    inputs_id = input_hash(transcript, characters)
    index = build_index(records, cast)
    workspace = out_root / "_work" / issue
    prepare_workspace(workspace, transcript, characters, index, series_path)
    try:
        agent = run_agent(workspace, model_name)
    except (subprocess.SubprocessError, RuntimeError, json.JSONDecodeError) as e:
        log.error(f"Agent run failed for {issue}: {e}")
        agent = {"error": repr(e)}
    scenes, facts, problems = load_log(workspace / "out")
    if not problems:
        series = IssueFacts.model_validate_json(series_path.read_text(encoding="utf-8"))
        problems = log_problems(scenes, facts, index, {f.id for f in series.facts})
    if problems:
        record_failure(out_root, issue, config_id, "; ".join(problems))
        log.warning(f"No valid log for {issue}; workspace kept at {workspace}")
        return False
    meta = {
        "model_name": model_name,
        "config_id": config_id,
        "input_hash": inputs_id,
        "agent": agent,
    }
    issue_log = assemble_log(issue, scenes, facts, records, cast, meta)
    output_path(out_root, issue).write_text(
        issue_log.model_dump_json(indent=1), encoding="utf-8"
    )
    shutil.rmtree(workspace)
    return True


def run_agent(workspace: Path, model: str) -> dict:
    return run_claude_agent(workspace, model, PROMPT, AGENT_TIMEOUT_S)


def ready_issues(
    issues: list[str], pages_root: Path, obs_root: Path, casts_root: Path
) -> dict[str, tuple[list[ObservationRecord], IssueCast]]:
    """Records and cast of the issues with complete observations and a resolution."""
    ready: dict[str, tuple[list[ObservationRecord], IssueCast]] = {}
    for issue, (records, _) in complete_issues(issues, pages_root, obs_root).items():
        cast = load_cast(casts_root, issue)
        if cast is None:
            log.info(f"Skipping {issue}: no character resolution")
            continue
        ready[issue] = (records, cast)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-issue event and knowledge logs with Claude Code agents"
    )
    parser.add_argument(
        "--issues", nargs="+", default=["pkna-0"], help="Issue directories, or 'all'"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument(
        "--redo-stale",
        action="store_true",
        help="Also redo logs written with other instructions or model",
    )
    args = parser.parse_args()

    issues = args.issues
    if issues == ["all"]:
        issues = sorted(p.name for p in PAGES_ROOT.iterdir() if p.is_dir())
    config_id = compute_config_id(args.model)
    ready = ready_issues(issues, PAGES_ROOT, OBS_ROOT, CASTS_ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for round_number in range(1, args.rounds + 1):
        pending = [
            issue
            for issue, (records, cast) in ready.items()
            if not is_done(
                output_path(args.out_dir, issue),
                config_id,
                input_hash(*issue_inputs(records, cast)),
                args.redo_stale,
            )
        ]
        if not pending:
            break
        console.print(
            f"[bold cyan]Event log[/bold cyan], round {round_number}: "
            f"{len(pending)} issues, model {args.model}, config {config_id}"
        )
        succeeded = 0
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex,
            PROGRESS as progress,
        ):
            futures = [
                ex.submit(
                    log_issue,
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
                description="Logging issues...",
            ):
                succeeded += future.result()
        console.print(f"Succeeded: {succeeded}, failed: {len(pending) - succeeded}")
    console.print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
