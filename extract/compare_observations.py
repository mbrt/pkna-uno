#!/usr/bin/env python3
"""Compare observation-layer records with extract-emotional v2 for one issue.

Reports dialogue agreement, speaker disagreements, fields that v2 lacks, and
description leakage heuristics. Writes a Markdown summary and a JSON file with
per-page disagreements for manual review.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from pkna.extract.observations import ObservationRecord, load_v2_pages
from pkna.logging import setup_logging

console, log = setup_logging()

BASE_DIR = Path(__file__).parent.parent
OBS_ROOT = BASE_DIR / "output/observations/v1"
V2_ROOT = BASE_DIR / "output/extract-emotional/v2"

MATCH_THRESHOLD = 0.8
DIALOGUE_KINDS = ("speech", "thought")

# Descriptions that report what is said duplicate the lettering and can reveal
# a response before its balloon is reached.
PARAPHRASE_RE = re.compile(
    r"\b(risponde|dice|chiede|esclama|replica|ribatte|commenta|urla|grida|"
    r"dichiara|annuncia|spiega|ordina|avverte|rassicura|scherza|ironizza|protesta)\b",
    re.IGNORECASE,
)
# Words that attribute hidden knowledge, motives, or outcomes rather than
# describing the drawing.
INTERPRETIVE_RE = re.compile(
    r"\b(ignar[oiae]|inconsapevole|non sa|nasconde|nascondendo|segretamente|"
    r"in realtà|rivela|vera natura|vera minaccia|intende|vuole|sospetta)\b",
    re.IGNORECASE,
)


@dataclass
class Line:
    speaker: str
    text: str
    ref: str


@dataclass
class PageComparison:
    image: str
    matched: int = 0
    speaker_mismatches: list[dict] = field(default_factory=list)
    only_new: list[dict] = field(default_factory=list)
    only_v2: list[dict] = field(default_factory=list)
    new_scene_breaks: int = 0
    v2_scene_breaks: int = 0


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", no_marks).split())


def align_lines(
    new: list[Line], old: list[Line], threshold: float = MATCH_THRESHOLD
) -> tuple[list[tuple[Line, Line]], list[Line], list[Line]]:
    """Greedily pair lines by text similarity. Returns (pairs, only_new, only_old)."""
    new_norm = [normalize_text(n.text) for n in new]
    old_norm = [normalize_text(o.text) for o in old]
    candidates = sorted(
        (
            (SequenceMatcher(None, a, b).ratio(), i, j)
            for i, a in enumerate(new_norm)
            for j, b in enumerate(old_norm)
        ),
        reverse=True,
    )
    used_new: set[int] = set()
    used_old: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, i, j in candidates:
        if score < threshold:
            break
        if i in used_new or j in used_old:
            continue
        used_new.add(i)
        used_old.add(j)
        pairs.append((i, j))
    pairs.sort()
    return (
        [(new[i], old[j]) for i, j in pairs],
        [n for i, n in enumerate(new) if i not in used_new],
        [o for j, o in enumerate(old) if j not in used_old],
    )


def new_lines(record: ObservationRecord) -> list[Line]:
    return [
        Line(speaker=t.speaker or "", text=t.text, ref=text_id)
        for text_id, _, t in record.iter_texts()
        if t.kind in DIALOGUE_KINDS
    ]


def v2_lines(page: dict, page_ref: str) -> list[Line]:
    return [
        Line(speaker=dl["character"], text=dl["line"], ref=f"{page_ref}#p{pi}.d{di}")
        for pi, panel in enumerate(page["panels"], 1)
        for di, dl in enumerate(panel["dialogues"], 1)
    ]


def compare_page(record: ObservationRecord, v2_page: dict) -> PageComparison:
    result = PageComparison(image=record.source.image)
    pairs, only_new, only_old = align_lines(
        new_lines(record), v2_lines(v2_page, record.source.page_id)
    )
    result.matched = len(pairs)
    for n, o in pairs:
        if n.speaker.casefold() != o.speaker.casefold():
            result.speaker_mismatches.append(
                {"ref": n.ref, "text": n.text, "new": n.speaker, "v2": o.speaker}
            )
    result.only_new = [asdict(n) for n in only_new]
    result.only_v2 = [asdict(o) for o in only_old]
    result.new_scene_breaks = sum(p.new_scene for p in record.panels)
    result.v2_scene_breaks = sum(
        p.get("is_new_scene", False) for p in v2_page["panels"]
    )
    return result


def count_matches(pattern: re.Pattern[str], descriptions: list[str]) -> int:
    return sum(1 for d in descriptions if pattern.search(d))


def silent_presences(record: ObservationRecord) -> Counter[str]:
    """Characters drawn in a panel without speaking or thinking in it."""
    counts: Counter[str] = Counter()
    for panel in record.panels:
        speakers = {t.speaker for t in panel.texts if t.kind in DIALOGUE_KINDS}
        for c in panel.characters:
            if c.depiction != "off_panel" and c.name not in speakers:
                counts[c.name] += 1
    return counts


def summarize(records: list[ObservationRecord], v2_pages: dict[str, dict]) -> dict:
    pages = [compare_page(r, v2_pages[r.source.image]) for r in records]
    texts = [t for r in records for _, _, t in r.iter_texts()]
    new_desc = [p.description for r in records for p in r.panels]
    v2_desc = [
        p["description"] for r in records for p in v2_pages[r.source.image]["panels"]
    ]
    silent: Counter[str] = Counter()
    for r in records:
        silent.update(silent_presences(r))

    dialogue = [t for t in texts if t.kind in DIALOGUE_KINDS]
    matched = sum(p.matched for p in pages)
    return {
        "pages": len(records),
        "panels": {"new": len(new_desc), "v2": len(v2_desc)},
        "dialogue_lines": {
            "new": len(dialogue),
            "v2": sum(len(v2_lines(v2_pages[r.source.image], "")) for r in records),
            "matched": matched,
            "only_new": sum(len(p.only_new) for p in pages),
            "only_v2": sum(len(p.only_v2) for p in pages),
            "speaker_mismatches": sum(len(p.speaker_mismatches) for p in pages),
        },
        "text_kinds": dict(Counter(t.kind for t in texts)),
        "attribution": dict(Counter(t.attribution for t in dialogue)),
        "via_device": sum(1 for t in dialogue if t.via),
        "with_addressees": sum(1 for t in dialogue if t.addressees),
        "silent_presences": dict(silent.most_common(15)),
        "unlisted_characters": sorted(
            {c.name for r in records for c in r.unlisted_characters}
        ),
        "printed_page_numbers": sum(
            1 for r in records if r.printed_page_number is not None
        ),
        "panels_with_location": sum(1 for r in records for p in r.panels if p.location),
        "scene_breaks": {
            "new": sum(p.new_scene_breaks for p in pages),
            "v2": sum(p.v2_scene_breaks for p in pages),
        },
        "descriptions_paraphrasing_speech": {
            "new": count_matches(PARAPHRASE_RE, new_desc),
            "v2": count_matches(PARAPHRASE_RE, v2_desc),
        },
        "descriptions_interpretive": {
            "new": count_matches(INTERPRETIVE_RE, new_desc),
            "v2": count_matches(INTERPRETIVE_RE, v2_desc),
        },
        "page_details": [asdict(p) for p in pages],
    }


def format_report(issue: str, summary: dict) -> str:
    d = summary["dialogue_lines"]
    lines = [
        f"# Observation layer vs extract-emotional v2: {issue}",
        "",
        f"Pages compared: {summary['pages']}. "
        f"Panels: {summary['panels']['new']} new, {summary['panels']['v2']} v2.",
        "",
        "## Dialogue (speech and thought)",
        "",
        f"- Lines: {d['new']} new, {d['v2']} v2; {d['matched']} matched by text",
        f"- Only in new: {d['only_new']}; only in v2: {d['only_v2']}",
        f"- Speaker disagreements among matched lines: {d['speaker_mismatches']}",
        "",
        "## Fields v2 does not have",
        "",
        f"- Text kinds: {summary['text_kinds']}",
        f"- Attribution: {summary['attribution']}",
        f"- Lines carried by a device: {summary['via_device']}",
        f"- Lines with addressees: {summary['with_addressees']}",
        f"- Panels with a location: {summary['panels_with_location']}",
        f"- Pages with a printed page number: {summary['printed_page_numbers']}",
        f"- Silent presences (top): {summary['silent_presences']}",
        f"- Characters missing from the cast sheet: {summary['unlisted_characters']}",
        "",
        "## Scene breaks and descriptions",
        "",
        f"- Scene breaks: {summary['scene_breaks']}",
        f"- Descriptions paraphrasing speech: {summary['descriptions_paraphrasing_speech']}",
        f"- Descriptions with interpretive wording: {summary['descriptions_interpretive']}",
        "",
        "## Speaker disagreements",
        "",
    ]
    for page in summary["page_details"]:
        for m in page["speaker_mismatches"]:
            lines.append(f'- `{m["ref"]}` "{m["text"]}": new={m["new"]}, v2={m["v2"]}')
    lines += ["", "## Unmatched lines", ""]
    for page in summary["page_details"]:
        for n in page["only_new"]:
            lines.append(f'- new `{n["ref"]}` {n["speaker"]}: "{n["text"]}"')
        for o in page["only_v2"]:
            lines.append(f'- v2 `{o["ref"]}` {o["speaker"]}: "{o["text"]}"')
    return "\n".join(lines) + "\n"


def load_records(issue_dir: Path) -> list[ObservationRecord]:
    records = [
        ObservationRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in issue_dir.glob("*.json")
        if p.name != "comparison.json"
    ]
    return sorted(records, key=lambda r: r.source.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare observation-layer records with extract-emotional v2"
    )
    parser.add_argument("--issue", default="pkna-0")
    args = parser.parse_args()

    issue_dir = OBS_ROOT / args.issue
    records = load_records(issue_dir)
    v2_pages = load_v2_pages(V2_ROOT / args.issue)
    missing = [r.source.image for r in records if r.source.image not in v2_pages]
    if missing:
        log.warning(f"No v2 record for {missing}; excluded from comparison")
    records = [r for r in records if r.source.image in v2_pages]

    summary = summarize(records, v2_pages)
    (issue_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = format_report(args.issue, summary)
    (issue_dir / "comparison.md").write_text(report, encoding="utf-8")
    console.print(report)


if __name__ == "__main__":
    main()
