"""Tests for proposed fact links, their validation, and their combination."""

import pytest

from pkna.extract.events import Fact, IssueLog
from pkna.extract.fact_links import (
    IssueLinkLog,
    IssueLinks,
    LinkIndex,
    ProposedLink,
    combine_links,
    fact_order,
    link_index,
    link_problems,
    render_facts,
)
from pkna.extract.registry import IssueCast, Persona, ResolvedCharacter, merge_casts
from pkna.extract.world_state import FactLink, FactLinks


def facts(issue: str, *statements: str, about: list[str] | None = None) -> IssueLog:
    return IssueLog(
        issue=issue,
        facts=[
            Fact(id=f"f{i}", statement=s, truth="true", about=about or [])
            for i, s in enumerate(statements, 1)
        ],
        scenes=[],
    )


LOGS = [
    facts("pkna-10", "Due è in rete."),
    facts("pkna-2", "Due è stato cancellato.", "Due è sopravvissuto."),
    facts("pkna-0", "Uno è un'IA."),
]
ORDER = ["pkna-0/f1", "pkna-2/f1", "pkna-2/f2", "pkna-10/f1"]


def proposal(issue: str, fact: str, same=(), opposite=()) -> IssueLinkLog:
    return IssueLinkLog(
        issue=issue,
        links=[
            ProposedLink(fact=fact, same=list(same), opposite=list(opposite), note="n")
        ],
    )


def test_fact_order_follows_publication_order():
    assert fact_order(LOGS) == ORDER


def test_link_index_offers_series_earlier_issues_and_the_issue_itself():
    series = [Fact(id="serie:x", statement="s", truth="true")]

    assert link_index(LOGS, "pkna-2", series) == LinkIndex(
        issue="pkna-2",
        facts=["f1", "f2"],
        earlier=["serie:x", "pkna-0/f1", "pkna-2/f1", "pkna-2/f2"],
    )


@pytest.mark.parametrize(
    "link, problem",
    [
        (ProposedLink(fact="f9", same=["pkna-0/f1"], note="n"), "not a fact of pkna-2"),
        (ProposedLink(fact="f1", note="n"), "no 'same' or 'opposite'"),
        (ProposedLink(fact="f1", same=["f2"], note="n"), "'f2' is not a fact id"),
        (
            ProposedLink(fact="f1", same=["pkna-10/f1"], note="n"),
            "'pkna-10/f1' is not a fact id",
        ),
        (ProposedLink(fact="f1", same=["pkna-2/f1"], note="n"), "links to itself"),
        (
            ProposedLink(
                fact="f2", same=["pkna-2/f1"], opposite=["pkna-2/f1"], note="n"
            ),
            "both 'same' and 'opposite'",
        ),
    ],
)
def test_link_problems(link: ProposedLink, problem: str):
    problems = link_problems(IssueLinks(links=[link]), link_index(LOGS, "pkna-2"))

    assert len(problems) == 1
    assert problem in problems[0]


def test_link_problems_accepts_valid_links_and_rejects_duplicates():
    index = link_index(LOGS, "pkna-2")
    valid = ProposedLink(fact="f2", opposite=["pkna-2/f1"], note="n")

    assert link_problems(IssueLinks(links=[valid]), index) == []
    assert link_problems(IssueLinks(links=[valid, valid]), index) == [
        "link 2 (f2): 'f2' is linked more than once"
    ]


def test_combine_links_chains_negations_across_issues():
    proposed = [
        proposal("pkna-2", "f2", opposite=["pkna-2/f1"]),
        proposal("pkna-10", "f1", opposite=["pkna-2/f2"]),
    ]

    links, problems = combine_links(proposed, FactLinks(), ORDER)

    assert problems == []
    assert links == FactLinks(
        links=[
            FactLink(same=["pkna-2/f1", "pkna-10/f1"], opposite=["pkna-2/f2"]),
        ]
    )


def test_manual_links_win_and_unlink_detaches_proposals():
    proposed = [
        proposal("pkna-2", "f2", same=["pkna-2/f1"]),
        proposal("pkna-10", "f1", same=["pkna-0/f1"]),
    ]
    manual = FactLinks(
        links=[FactLink(same=["pkna-2/f1"], opposite=["pkna-2/f2"])],
        unlink=["pkna-0/f1"],
    )

    links, problems = combine_links(proposed, manual, ORDER)

    assert links == FactLinks(
        links=[FactLink(same=["pkna-2/f1"], opposite=["pkna-2/f2"])]
    )
    assert problems == [
        "pkna-2/f2: pkna-2/f2 and pkna-2/f1 as same contradicts other links"
    ]


def test_combine_links_drops_links_to_unknown_facts():
    proposed = [proposal("pkna-10", "f1", same=["pkna-3/f1"])]

    links, problems = combine_links(proposed, FactLinks(), ORDER)

    assert links == FactLinks()
    assert problems == ["pkna-10/f1: unknown fact 'pkna-3/f1'"]


def test_render_facts_names_characters_by_registry_name():
    registry = merge_casts(
        {
            "pkna-2": IssueCast(
                characters=[
                    ResolvedCharacter(
                        name="Due",
                        kind="named",
                        personas=[Persona(name="Due", labels=["Due"])],
                        description="d",
                    )
                ]
            )
        }
    )
    logs = [facts("pkna-2", "Due è stato cancellato.", about=["Due"])]

    assert render_facts(logs, registry, ["pkna-2/f1"]) == (
        "- pkna-2/f1 [true] (Due): Due è stato cancellato."
    )
