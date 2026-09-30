"""Tests for link_facts_agent.py workspaces, validation, resume, and review."""

import json
from pathlib import Path

from extract.fact_link_tools import validate_links
from extract.link_facts_agent import (
    format_review,
    input_hash,
    is_done,
    link_issue,
    output_path,
)
from pkna.extract.events import Fact, IssueLog
from pkna.extract.fact_links import IssueLinkLog, IssueLinks, ProposedLink, link_index
from pkna.extract.registry import IssueCast, Persona, ResolvedCharacter, merge_casts
from pkna.extract.world_state import FactLink, FactLinks

DUE = ResolvedCharacter(
    name="Due",
    kind="named",
    personas=[Persona(name="Due", labels=["Due"])],
    description="d",
)
REGISTRY = merge_casts(
    {issue: IssueCast(characters=[DUE]) for issue in ("pkna-2", "pkna-8")}
)
LOGS = [
    IssueLog(
        issue="pkna-2",
        facts=[
            Fact(
                id="due-cancellato",
                statement="Due è stato cancellato.",
                truth="true",
                about=["Due"],
            )
        ],
        scenes=[],
    ),
    IssueLog(
        issue="pkna-8",
        facts=[
            Fact(
                id="f61",
                statement="Due è stato cancellato durante il precedente scontro.",
                truth="false",
                about=["Due"],
            )
        ],
        scenes=[],
    ),
]
LINK = ProposedLink(fact="f61", same=["pkna-2/due-cancellato"], note="n")
SERIES = [
    Fact(
        id="serie:identita-paperinik",
        statement="Paolino Paperino è Paperinik.",
        truth="true",
    )
]


def writing(links: list[ProposedLink], seen: dict[str, str] | None = None):
    def runner(workspace: Path, model: str) -> dict:
        if seen is not None:
            seen.update(
                (p.name, p.read_text()) for p in workspace.iterdir() if p.is_file()
            )
        (workspace / "out" / "links.json").write_text(
            IssueLinks(links=links).model_dump_json()
        )
        return {"session_id": "s1"}

    return runner


def test_valid_links_are_written_with_provenance_and_resumable(tmp_path: Path):
    seen: dict[str, str] = {}

    ok = link_issue(
        "pkna-8", LOGS, REGISTRY, writing([LINK], seen), "m", tmp_path, "c1", SERIES
    )

    assert ok
    assert seen["issue-facts.md"] == (
        "- pkna-8/f61 [false] (Due): "
        "Due è stato cancellato durante il precedente scontro.\n"
    )
    assert seen["earlier-facts.md"] == (
        "## serie\n\n- serie:identita-paperinik [true] (): "
        "Paolino Paperino è Paperinik.\n\n"
        "## pkna-2\n\n- pkna-2/due-cancellato [true] (Due): Due è stato cancellato."
        "\n\n## pkna-8\n\n- pkna-8/f61 [false] (Due): "
        "Due è stato cancellato durante il precedente scontro.\n"
    )
    assert "issue pkna-8" in seen["INSTRUCTIONS.md"]
    written = IssueLinkLog.model_validate_json(
        output_path(tmp_path, "pkna-8").read_text()
    )
    assert written.links == [LINK]
    assert written.meta["agent"] == {"session_id": "s1"}
    assert not (tmp_path / "_work" / "pkna-8").exists()

    inputs_id = input_hash(LOGS, link_index(LOGS, "pkna-8", SERIES), SERIES)
    assert is_done(output_path(tmp_path, "pkna-8"), "c2", inputs_id)
    assert not is_done(output_path(tmp_path, "pkna-8"), "c2", inputs_id, True)
    assert not is_done(output_path(tmp_path, "pkna-8"), "c1", "other")


def test_input_hash_ignores_character_names():
    renamed = [
        log.model_copy(
            update={"facts": [f.model_copy(update={"about": []}) for f in log.facts]}
        )
        for log in LOGS
    ]
    index = link_index(LOGS, "pkna-8")

    assert input_hash(renamed, index) == input_hash(LOGS, index)


def test_invalid_links_are_recorded_and_the_workspace_kept(tmp_path: Path):
    bad = ProposedLink(fact="f61", same=["due-cancellato"], note="n")

    ok = link_issue("pkna-8", LOGS, REGISTRY, writing([bad]), "m", tmp_path, "c1")

    assert not ok
    assert not output_path(tmp_path, "pkna-8").exists()
    failure = json.loads((tmp_path / "failures.jsonl").read_text())
    assert "'due-cancellato' is not a fact id" in failure["error"]
    workspace = tmp_path / "_work" / "pkna-8"
    assert validate_links(workspace / "out" / "links.json", workspace / "index.json")


def test_review_lists_propositions_and_ignored_links():
    links = FactLinks(
        links=[FactLink(same=["pkna-2/due-cancellato"], opposite=["pkna-8/f61"])]
    )

    review = format_review(links, ["pkna-8/f61: unknown fact 'x'"], LOGS)

    assert "- pkna-8/f61: unknown fact 'x'" in review
    assert (
        "- `pkna-2/due-cancellato`\n"
        "  - `pkna-2/due-cancellato` [true] Due è stato cancellato.\n"
        "  - opposite:\n"
        "    - `pkna-8/f61` [false] "
        "Due è stato cancellato durante il precedente scontro.\n"
    ) in review
