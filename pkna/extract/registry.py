"""Character registry: which labels in the observation records are the same character.

Observation records name characters as the page shows them: a name when the
page or cast sheet gives one, otherwise a descriptive label such as
"Evroniano in frac viola". Resolution groups the labels of one issue into
characters with personas (Paperino in civilian clothes, Paperinik in costume),
and a global merge links the characters of different issues.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from pkna.extract.observations import CastMember, ObservationRecord

CharacterKind = Literal["named", "unnamed", "group"]

MAX_SAMPLE_LINES = 3
MAX_LISTED_PAGES = 8


class Persona(BaseModel):
    name: str = Field(
        description=(
            "Name of the persona: the character's name, or the name of one identity "
            "when the character has several (e.g. 'Paperino' and 'Paperinik')."
        )
    )
    labels: list[str] = Field(
        description="Labels from the observation records that refer to this persona."
    )


class ResolvedCharacter(BaseModel):
    name: str = Field(
        description=(
            "The character's name as the pages give it (in a caption, a vocative, "
            "a self-introduction, or the cast sheet). For characters never named, "
            "a short descriptive label."
        )
    )
    kind: CharacterKind = Field(
        description=(
            "named: an individual whose proper name the pages give. unnamed: an "
            "individual without a proper name (e.g. 'Direttore', 'Ostaggio'), or a "
            "descriptive label that may cover several individuals (e.g. 'Soldato "
            "evroniano'). group: a collective (soldiers, passers-by, audience)."
        )
    )
    personas: list[Persona] = Field(
        description="One persona for most characters; several for secret identities."
    )
    description: str = Field(description="Brief appearance, as drawn in this issue.")
    evidence: str | None = Field(
        default=None,
        description=(
            "Why labels were linked to this character, citing pages, whenever the "
            "link is not simply the same name. Null otherwise."
        ),
    )

    @property
    def labels(self) -> list[str]:
        return [label for p in self.personas for label in p.labels]


class IssueCast(BaseModel):
    """The characters of one issue, covering every label in its records."""

    characters: list[ResolvedCharacter]


class Mention(BaseModel):
    """How one label is used across an issue's observation records."""

    label: str
    lines: int = 0
    panels: int = 0
    addressed: int = 0
    page_count: int = 0
    pages: list[str] = []
    appearance: list[str] = []
    sample_lines: list[str] = []


def collect_mentions(
    records: Sequence[ObservationRecord], cast: Sequence[CastMember]
) -> list[Mention]:
    """Label usage for an issue, most frequent first."""
    mentions: dict[str, Mention] = {}
    pages: dict[str, list[str]] = {}

    def get(label: str, page: str) -> Mention:
        m = mentions.setdefault(label, Mention(label=label))
        seen = pages.setdefault(label, [])
        if page not in seen:
            seen.append(page)
        return m

    for r in records:
        page = r.source.page_id.split("/", 1)[1]
        for panel in r.panels:
            for c in panel.characters:
                get(c.name, page).panels += 1
            for t in panel.texts:
                if t.speaker:
                    m = get(t.speaker, page)
                    m.lines += 1
                    if len(m.sample_lines) < MAX_SAMPLE_LINES:
                        m.sample_lines.append(t.text)
                for a in t.addressees:
                    get(a, page).addressed += 1
        for c in r.unlisted_characters:
            if c.name in mentions and c.appearance:
                appearance = mentions[c.name].appearance
                if c.appearance not in appearance:
                    appearance.append(c.appearance)
    for c in cast:
        if c.name in mentions and c.appearance:
            mentions[c.name].appearance.insert(0, c.appearance)
    for label, m in mentions.items():
        m.page_count = len(pages[label])
        m.pages = pages[label][:MAX_LISTED_PAGES]
    return sorted(
        mentions.values(), key=lambda m: (-(m.lines + m.panels + m.addressed), m.label)
    )


def render_transcript(records: Sequence[ObservationRecord]) -> str:
    """A compact, readable rendering of an issue's records, page by page."""
    out: list[str] = []
    for r in records:
        page = r.source.page_id.split("/", 1)[1]
        printed = f" (p. {r.printed_page_number})" if r.printed_page_number else ""
        out.append(f"## {page}{printed}")
        for i, panel in enumerate(r.panels, 1):
            header = f"[{i}]"
            if panel.new_scene:
                header += " NUOVA SCENA"
            if panel.location:
                header += f" Luogo: {panel.location}."
            chars = ", ".join(
                c.name if c.depiction == "in_person" else f"{c.name} ({c.depiction})"
                for c in panel.characters
            )
            if chars:
                header += f" Personaggi: {chars}."
            out.append(header)
            out.append(f"    {panel.description}")
            for t in panel.texts:
                if t.kind == "sfx":
                    continue
                if t.speaker:
                    to = f" → {', '.join(t.addressees)}" if t.addressees else ""
                    via = f" [via {t.via}]" if t.via else ""
                    marker = " (pensiero)" if t.kind == "thought" else ""
                    out.append(f'    - {t.speaker}{to}{via}{marker}: "{t.text}"')
                else:
                    out.append(f'    - ({t.kind}) "{t.text}"')
        out.append("")
    return "\n".join(out)


class RegistryPersona(BaseModel):
    name: str
    labels: list[str]


class RegistryCharacter(BaseModel):
    """A character across the series; unnamed ones are scoped to one issue."""

    id: str
    name: str
    kind: CharacterKind
    personas: list[RegistryPersona]
    issues: list[str]
    names: list[str] = Field(description="Names the issue resolutions used.")
    description: str


class LabelRef(BaseModel):
    id: str
    persona: str


class Registry(BaseModel):
    characters: list[RegistryCharacter]
    labels: dict[str, dict[str, LabelRef]] = Field(
        description="For each issue, the character and persona of every label."
    )
    members: dict[str, dict[str, str]] = Field(
        default={},
        description="For each issue, the id of every character of its resolution, "
        "by the name the resolution gives it.",
    )
    kept_apart: list[tuple[str, str]] = Field(
        default=[],
        description="Characters sharing a name that were not merged because one "
        "issue's resolution keeps them distinct.",
    )

    def resolve(self, issue: str, label: str) -> LabelRef | None:
        return self.labels.get(issue, {}).get(label)

    def member(self, issue: str, name: str) -> str:
        """The id of a character named as in the issue's resolution."""
        return self.members[issue][name]

    def find(self, name: str) -> RegistryCharacter | None:
        """The named character with this name, a name variant, or persona name."""
        key = name_key(name)
        for c in self.characters:
            if c.kind == "named" and key in character_keys(c.names, c.personas):
                return c
        return None


class Overrides(BaseModel):
    """Manual corrections applied by every merge."""

    merge: list[list[str]] = Field(
        default=[],
        description="Groups of names that are the same named character; the "
        "first name of a group is the character's canonical name.",
    )


def name_key(name: str) -> str:
    """Casefolded name without accents or punctuation, for matching."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", plain).split())


def character_keys(
    names: Sequence[str], personas: Sequence[Persona | RegistryPersona]
) -> set[str]:
    return {name_key(n) for n in names} | {name_key(p.name) for p in personas}


def slugify(name: str) -> str:
    return name_key(name).replace(" ", "-")


def _choose_name(
    names: Sequence[str], persona_names: Sequence[str], preferred: set[str]
) -> str:
    """A preferred name if the cluster uses one, else its most complete name.

    Preferring the name with the most words keeps ids stable as issues are
    added: a later issue calling 'Lyla Lay' just 'Lyla' doesn't rename her.
    """
    for name in [*names, *persona_names]:
        if name in preferred:
            return name
    counts = Counter(names)
    return max(counts, key=lambda n: (len(name_key(n).split()), counts[n], len(n)))


def _unique_id(base: str, used: set[str]) -> str:
    cid, n = base, 2
    while cid in used:
        cid, n = f"{base}-{n}", n + 1
    used.add(cid)
    return cid


def merge_casts(
    casts: dict[str, IssueCast],
    overrides: Overrides | None = None,
    preferred_names: set[str] | None = None,
) -> Registry:
    """Link named characters across issues by shared names or persona names.

    Descriptive labels never link issues: the same label can denote different
    characters in different issues. Unnamed characters and groups stay scoped
    to their issue, with ids prefixed by the issue.
    """
    overrides = overrides or Overrides()
    preferred = (preferred_names or set()) | {g[0] for g in overrides.merge}
    named = [
        (issue, c)
        for issue, cast in sorted(casts.items())
        for c in cast.characters
        if c.kind == "named"
    ]
    parent = list(range(len(named)))
    cluster_issues = [{issue} for issue, _ in named]
    refused: list[tuple[int, int]] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        # One issue's resolution is authoritative about which of its
        # characters are distinct, e.g. an impostor using a real person's name.
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if cluster_issues[ri] & cluster_issues[rj]:
            refused.append((i, j))
            return
        parent[ri] = rj
        cluster_issues[rj] |= cluster_issues[ri]

    alias = {name_key(n): name_key(g[0]) for g in overrides.merge for n in g}
    owner: dict[str, int] = {}
    for i, (_, c) in enumerate(named):
        for key in character_keys([c.name], c.personas):
            key = alias.get(key, key)
            if key in owner:
                union(i, owner[key])
            else:
                owner[key] = i

    clusters: dict[int, list[tuple[str, ResolvedCharacter]]] = {}
    for i, member in enumerate(named):
        clusters.setdefault(find(i), []).append(member)

    characters: list[RegistryCharacter] = []
    labels: dict[str, dict[str, LabelRef]] = {issue: {} for issue in casts}
    members: dict[str, dict[str, str]] = {issue: {} for issue in casts}
    used_ids: set[str] = set()
    cluster_ids: dict[int, str] = {}
    for root, cluster in clusters.items():
        names = [c.name for _, c in cluster]
        persona_names = [p.name for _, c in cluster for p in c.personas]
        name = _choose_name(names, persona_names, preferred)
        cid = _unique_id(slugify(name), used_ids)
        cluster_ids[root] = cid
        personas: dict[str, RegistryPersona] = {}
        for issue, c in cluster:
            members[issue][c.name] = cid
            for p in c.personas:
                key = name_key(p.name)
                rp = personas.setdefault(
                    alias.get(key, key), RegistryPersona(name=p.name, labels=[])
                )
                rp.labels = sorted(set(rp.labels) | set(p.labels))
                for label in p.labels:
                    labels[issue][label] = LabelRef(id=cid, persona=rp.name)
        characters.append(
            RegistryCharacter(
                id=cid,
                name=name,
                kind="named",
                personas=list(personas.values()),
                issues=sorted({issue for issue, _ in cluster}),
                names=sorted(set(names)),
                description=cluster[0][1].description,
            )
        )
    for issue, cast in sorted(casts.items()):
        for c in cast.characters:
            if c.kind == "named":
                continue
            cid = _unique_id(f"{issue}:{slugify(c.name)}", used_ids)
            members[issue][c.name] = cid
            for p in c.personas:
                for label in p.labels:
                    labels[issue][label] = LabelRef(id=cid, persona=p.name)
            characters.append(
                RegistryCharacter(
                    id=cid,
                    name=c.name,
                    kind=c.kind,
                    personas=[
                        RegistryPersona(name=p.name, labels=p.labels)
                        for p in c.personas
                    ],
                    issues=[issue],
                    names=[c.name],
                    description=c.description,
                )
            )
    refused_ids = [(cluster_ids[find(i)], cluster_ids[find(j)]) for i, j in refused]
    return Registry(
        characters=characters,
        labels=labels,
        members=members,
        kept_apart=sorted({(min(a, b), max(a, b)) for a, b in refused_ids}),
    )


def cast_problems(cast: IssueCast, labels: set[str]) -> list[str]:
    """Labels not assigned exactly once, and assignments of unknown labels."""
    assigned: dict[str, list[str]] = {}
    for c in cast.characters:
        if not c.personas:
            assigned.setdefault("", []).append(c.name)
        for label in c.labels:
            assigned.setdefault(label, []).append(c.name)
    problems = [f"character {name!r} has no personas" for name in assigned.pop("", [])]
    problems += [
        f"label {label!r} is assigned to several characters: {owners}"
        for label, owners in assigned.items()
        if len(owners) > 1
    ]
    problems += [
        f"label {label!r} is not in the records"
        for label in sorted(set(assigned) - labels)
    ]
    problems += [
        f"label {label!r} is not assigned to any character"
        for label in sorted(labels - set(assigned))
    ]
    return problems
