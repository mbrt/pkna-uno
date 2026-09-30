#!/usr/bin/env python3
"""Link each issue's facts to earlier facts stating the same proposition.

Every issue's event log names its own facts, so a later issue revising what an
earlier one showed states it as a new fact. One Claude Code session per issue
reads the issue's facts beside all facts before them and proposes which state
the same proposition as an earlier fact, or its negation. Issues run in
parallel; the state views combine the proposals with the manual corrections in
data/events/fact-links.json. A review report lists the combined propositions.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from extract.fact_link_tools import load_links
from extract.show_world_state import load_logs
from pkna.extract.events import SERIES_FACT_PREFIX, Fact, IssueFacts, IssueLog
from pkna.extract.fact_links import (
    IssueLinkLog,
    IssueLinks,
    LinkIndex,
    fact_hash_payload,
    link_index,
    link_problems,
    load_fact_links,
    render_facts,
)
from pkna.extract.registry import Registry
from pkna.extract.world_state import FactLinks, global_fact_id, issue_key
from pkna.llm.claude_agent import run_claude_agent
from pkna.logging import setup_logging

console, log = setup_logging()

DEFAULT_MODEL = "claude-opus-5-5"
MAX_PARALLEL = 8
ROUNDS = 2
AGENT_TIMEOUT_S = 60 * 60

BASE_DIR = Path(__file__).parent.parent
EVENTS_ROOT = BASE_DIR / "output/events/v1"
OUT_ROOT = EVENTS_ROOT / "links"
REGISTRY_PATH = BASE_DIR / "output/registry/v1/registry.json"
MANUAL_LINKS_PATH = BASE_DIR / "data/events/fact-links.json"
SERIES_FACTS_PATH = BASE_DIR / "data/events/series-facts.json"
TOOLS = shlex.join([sys.executable, str(BASE_DIR / "extract/fact_link_tools.py")])

PROMPT = "Follow the instructions in INSTRUCTIONS.md in this directory."

INSTRUCTIONS = """\
The facts in this directory come from the event logs of an Italian comic book
series (PKNA, Paperinik New Adventures), one log per issue. Each log records the
facts its characters come to know, suspect, or reject, and names them with its
own ids. So when a later issue revisits something an earlier one showed (a
character thought destroyed turns out to have survived, a suspicion is
confirmed), it states it as a new fact. Link the facts of issue {issue} to the
earlier facts that state the same proposition or its negation, so that a
character's later stance replaces the earlier one.

## Files

- `issue-facts.md`: the facts of {issue}, one per line: global id, truth as
  judged from that issue alone, characters concerned, statement.
- `earlier-facts.md`: the facts from the premise of the series (`serie:` ids),
  then all facts before those of {issue} in publication order, grouped by
  issue, in the same format. The facts of {issue} itself also count as earlier
  for the facts after them. Characters are named the same in every issue.
- `links-schema.json`: the format of the file to write.

## Workflow

1. Read `issue-facts.md`.
2. For each fact, search the earlier facts for the same proposition or its
   negation: search by the characters, objects, and places it concerns, and by
   key words and their variants. `earlier-facts.md` is long; use grep rather
   than reading it whole.
3. Write `out/links.json` following `links-schema.json`, with an entry for each
   fact of {issue} that has links, and only for those. Refer to facts of
   {issue} by their local id (`f12`) in `fact`, and to linked facts by their
   global id (`pkna-2/f07`).
4. Run `{tools} validate-links out/links.json` and fix every problem it reports.

Use only the files in this directory: do not read files outside it, and do not
look up plot summaries or other information about the comic.

## Rules

- Same: the two facts are true or false together, as statements about the same
  story situation, even if worded differently, with more or fewer details that
  identify the same event, or judged differently by different issues (one says
  true, the other false). E.g. 'Due è stato cancellato.' and 'Due è stato
  cancellato durante il precedente scontro con Uno e Paperinik.'
- Opposite: one is true exactly when the other is false. E.g. 'Due è stato
  distrutto insieme alla sua scialuppa.' and 'Due è sopravvissuto alla
  distruzione della sua scialuppa.'
- Do not link facts that are only related: one implying the other, overlapping
  in part, concerning the same topic, or describing a similar situation at
  another time (a different fight, a different disappearance).
- A fact may link to several earlier facts. When earlier facts are already the
  same proposition, linking to the most recent one suffices.
- Most facts have no links. Do not force any.
- Write each note in English, citing what the linked statements say.

When the file is valid, reply with the number of facts linked.
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
    """Hash of everything besides the issue inputs that changes the output."""
    payload = {
        "model": model_name,
        "instructions": INSTRUCTIONS,
        "links_schema": IssueLinks.model_json_schema(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def input_hash(
    logs: Sequence[IssueLog], index: LinkIndex, series: Sequence[Fact] = ()
) -> str:
    own = [f"{index.issue}/{f}" for f in index.facts]
    payload = fact_hash_payload(logs, [*index.earlier, *own], series)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def output_path(out_root: Path, issue: str) -> Path:
    return out_root / f"{issue}.json"


def is_done(
    path: Path, config_id: str, inputs_id: str, redo_stale: bool = False
) -> bool:
    """Whether links exist for these inputs.

    Links written with other instructions or model still count, unless
    redo_stale.
    """
    if not path.exists():
        return False
    meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
    if meta.get("input_hash") != inputs_id:
        return False
    return not redo_stale or meta.get("config_id") == config_id


def render_earlier(
    logs: Sequence[IssueLog],
    registry: Registry,
    index: LinkIndex,
    series: Sequence[Fact] = (),
) -> str:
    by_issue: dict[str, list[str]] = {}
    for fid in index.earlier:
        group = "serie" if fid.startswith(SERIES_FACT_PREFIX) else fid.split("/")[0]
        by_issue.setdefault(group, []).append(fid)
    return "\n\n".join(
        f"## {issue}\n\n{render_facts(logs, registry, fids, series)}"
        for issue, fids in by_issue.items()
    )


def prepare_workspace(
    workspace: Path,
    logs: Sequence[IssueLog],
    registry: Registry,
    index: LinkIndex,
    series: Sequence[Fact] = (),
) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "out").mkdir(parents=True)
    own = [f"{index.issue}/{f}" for f in index.facts]
    (workspace / "issue-facts.md").write_text(
        render_facts(logs, registry, own) + "\n", encoding="utf-8"
    )
    (workspace / "earlier-facts.md").write_text(
        render_earlier(logs, registry, index, series) + "\n", encoding="utf-8"
    )
    (workspace / "index.json").write_text(index.model_dump_json(), encoding="utf-8")
    (workspace / "links-schema.json").write_text(
        json.dumps(IssueLinks.model_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (workspace / "INSTRUCTIONS.md").write_text(
        INSTRUCTIONS.format(issue=index.issue, tools=TOOLS), encoding="utf-8"
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


def link_issue(
    issue: str,
    logs: Sequence[IssueLog],
    registry: Registry,
    run_agent: AgentRunner,
    model_name: str,
    out_root: Path,
    config_id: str,
    series: Sequence[Fact] = (),
) -> bool:
    """Propose one issue's links. Returns whether valid links were written."""
    index = link_index(logs, issue, series)
    workspace = out_root / "_work" / issue
    prepare_workspace(workspace, logs, registry, index, series)
    try:
        agent = run_agent(workspace, model_name)
    except (subprocess.SubprocessError, RuntimeError, json.JSONDecodeError) as e:
        log.error(f"Agent run failed for {issue}: {e}")
        agent = {"error": repr(e)}
    links, problems = load_links(workspace / "out" / "links.json")
    if links is not None:
        problems = link_problems(links, index)
    if links is None or problems:
        record_failure(out_root, issue, config_id, "; ".join(problems))
        log.warning(f"No valid links for {issue}; workspace kept at {workspace}")
        return False
    meta = {
        "model_name": model_name,
        "config_id": config_id,
        "input_hash": input_hash(logs, index, series),
        "agent": agent,
    }
    output_path(out_root, issue).write_text(
        IssueLinkLog(issue=issue, links=links.links, meta=meta).model_dump_json(
            indent=1
        ),
        encoding="utf-8",
    )
    shutil.rmtree(workspace)
    return True


def format_review(
    links: FactLinks,
    problems: Sequence[str],
    logs: Sequence[IssueLog],
    series: Sequence[Fact] = (),
) -> str:
    facts = {f.id: f for f in series}
    facts |= {global_fact_id(log.issue, f.id): f for log in logs for f in log.facts}

    def line(fid: str) -> str:
        fact = facts.get(fid)
        return (
            f"  - `{fid}` [{fact.truth}] {fact.statement}" if fact else f"  - `{fid}`"
        )

    lines = [
        "# Fact links review",
        "",
        f"{len(links.links)} propositions stated by "
        f"{sum(len(g.same) + len(g.opposite) for g in links.links)} facts.",
        "",
        "Fix wrong links with `links` and `unlink` in `data/events/fact-links.json`.",
        "",
        "## Ignored links",
        "",
        *[f"- {p}" for p in problems],
        "",
        "## Propositions",
    ]
    for group in links.links:
        lines += ["", f"- `{group.same[0]}`", *map(line, group.same)]
        if group.opposite:
            lines += ["  - opposite:", *("  " + line(fid) for fid in group.opposite)]
    return "\n".join(lines) + "\n"


def run_agent(workspace: Path, model: str) -> dict:
    return run_claude_agent(workspace, model, PROMPT, AGENT_TIMEOUT_S)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose links between facts of different issues"
    )
    parser.add_argument(
        "--issues", nargs="+", default=["pkna-0"], help="Issues, or 'all'"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--events-dir", type=Path, default=EVENTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument(
        "--redo-stale",
        action="store_true",
        help="Also redo links written with other instructions or model",
    )
    args = parser.parse_args()

    logs = load_logs(args.events_dir)
    registry = Registry.model_validate_json(REGISTRY_PATH.read_text(encoding="utf-8"))
    series = IssueFacts.model_validate_json(
        SERIES_FACTS_PATH.read_text(encoding="utf-8")
    ).facts
    issues = sorted((log.issue for log in logs), key=issue_key)
    if args.issues != ["all"]:
        issues = [i for i in issues if i in args.issues]
    config_id = compute_config_id(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for round_number in range(1, args.rounds + 1):
        pending = [
            issue
            for issue in issues
            if not is_done(
                output_path(args.out_dir, issue),
                config_id,
                input_hash(logs, link_index(logs, issue, series), series),
                args.redo_stale,
            )
        ]
        if not pending:
            break
        console.print(
            f"[bold cyan]Fact links[/bold cyan], round {round_number}: "
            f"{len(pending)} issues, model {args.model}, config {config_id}"
        )
        succeeded = 0
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex,
            PROGRESS as progress,
        ):
            futures = [
                ex.submit(
                    link_issue,
                    issue,
                    logs,
                    registry,
                    run_agent,
                    args.model,
                    args.out_dir,
                    config_id,
                    series,
                )
                for issue in pending
            ]
            for future in progress.track(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                description="Linking facts...",
            ):
                succeeded += future.result()
        console.print(f"Succeeded: {succeeded}, failed: {len(pending) - succeeded}")

    links, problems = load_fact_links(args.out_dir, MANUAL_LINKS_PATH, logs)
    (args.out_dir / "review.md").write_text(
        format_review(links, problems, logs, series), encoding="utf-8"
    )
    console.print(
        f"{len(links.links)} propositions, {len(problems)} ignored links. "
        f"Output: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
