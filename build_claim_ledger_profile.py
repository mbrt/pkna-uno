#!/usr/bin/env python3

"""
Build a character profile using a claim ledger approach.

This script tracks discrete claims about a character with support/contradict counts,
uses progressive disclosure for context management, and generates the final soul
document from substantiated claims.

Key features:
- Claim-based evidence accumulation across scenes
- Progressive disclosure: compact list view + detail tool for specific claims
- Cross-issue pattern recognition through claim support counts
- Threshold-based filtering for final document generation
- Checkpoint system with JSON snapshots and diffs
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    HttpOptions,
    Part,
)
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

load_dotenv()

# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Settings
MODEL_NAME = "gemini-3-flash-preview"
CHARACTER_NAME = "Uno"
ENCODING_NAME = "cl100k_base"
VERSION_TAG = "v8"
MAX_TOOL_ITERATIONS = 64
CLAIM_SUPPORT_THRESHOLD = 2  # Minimum support_count to include in final document

# Retry settings for API calls
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0
API_TIMEOUT_SECONDS = 300  # 5 minutes per API call


def generate_with_retry(
    client: genai.Client,
    conversation: list[Content],
    config: GenerateContentConfig,
) -> GenerateContentResponse | None:
    """Call generate_content with retry on resource exhausted and timeout errors."""
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=conversation,  # type: ignore[arg-type]
                config=config,
            )
            return response
        except Exception as e:
            error_str = str(e).lower()
            is_timeout = "timeout" in error_str or "timed out" in error_str
            is_rate_limited = (
                "resource" in error_str and "exhausted" in error_str
            ) or "429" in error_str
            is_retryable = is_timeout or is_rate_limited

            if not is_retryable:
                raise

            if attempt == MAX_RETRIES - 1:
                log.error(f"Max retries ({MAX_RETRIES}) exceeded: {e}")
                return None

            # Add jitter: +/- 25% of backoff
            jitter = backoff * 0.25 * (2 * random.random() - 1)
            sleep_time = min(backoff + jitter, MAX_BACKOFF_SECONDS)

            error_type = "Timeout" if is_timeout else "Resource exhausted"
            log.warning(
                f"{error_type} (attempt {attempt + 1}/{MAX_RETRIES}), "
                f"retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)

    return None


# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / VERSION_TAG
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# Global progress bar
PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)

# Valid claim sections
VALID_SECTIONS = {
    "identity",
    "personality",
    "communication",
    "values",
    "behavior",
    "relationships",
}


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


def path_str(path: Path) -> str:
    """Get a pretty string for a Path relative to BASE_DIR."""
    try:
        rel_path = path.relative_to(BASE_DIR)
    except ValueError:
        rel_path = path
    return str(rel_path)


# ============================================================================
# Pydantic Models for Claims
# ============================================================================


class Quote(BaseModel):
    """An Italian quote with context from the comics."""

    text: str  # Italian quote
    context: str  # Brief context (1-2 sentences)
    scene_id: str  # Reference to source scene


class SceneEvidence(BaseModel):
    """Evidence linking a scene to a claim."""

    scene_id: str  # e.g., "pkna-0_12"
    justification: str  # Brief reason (1-2 sentences)


class Claim(BaseModel):
    """A discrete claim about the character with evidence tracking."""

    id: int
    text: str  # Claim in English
    section: str  # identity|personality|communication|values|behavior|relationships
    supporting: list[SceneEvidence] = []  # Scenes that support with justification
    contradicting: list[SceneEvidence] = []  # Scenes that contradict with justification
    quotes: list[Quote] = []  # Quotes with context

    @property
    def support_count(self) -> int:
        """Net support: supporting scenes minus contradicting scenes."""
        return len(self.supporting) - len(self.contradicting)

    def absorb_contradictions(self) -> int:
        """Move all contradicting evidence to supporting and clear contradicting.

        Call this after the claim text has been refined to account for the
        contradictions, so they become supporting evidence for the nuanced claim.

        Returns:
            Number of evidence entries moved.
        """
        moved = len(self.contradicting)
        self.supporting.extend(self.contradicting)
        self.contradicting = []
        return moved


# ============================================================================
# Scene Data Structure (reused from build_agentic_character_profile.py)
# ============================================================================


@dataclass
class Scene:
    """A scene from the comics containing Uno."""

    issue: str
    page_numbers: list[int]
    summary: str
    uno_dialogues: list[str]
    panel_descriptions: list[str]
    other_characters: set[str]

    @property
    def scene_id(self) -> str:
        """Unique identifier for this scene: issue_firstpage."""
        return f"{self.issue}_{self.page_numbers[0]}"

    def to_context_string(self) -> str:
        """Create a context string describing the scene."""
        pages_str = ", ".join(f"page {p}" for p in self.page_numbers)
        chars_str = (
            ", ".join(sorted(self.other_characters))
            if self.other_characters
            else "none"
        )
        return (
            f"Issue: {self.issue}, {pages_str}. Other characters present: {chars_str}"
        )

    def to_other_context(self) -> str:
        """Create additional context string."""
        return f"Panel descriptions: {' | '.join(self.panel_descriptions)}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "issue": self.issue,
            "page_numbers": self.page_numbers,
            "summary": self.summary,
            "uno_dialogues": self.uno_dialogues,
            "panel_descriptions": self.panel_descriptions,
            "other_characters": list(self.other_characters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        """Create Scene from dictionary."""
        return cls(
            issue=data["issue"],
            page_numbers=data["page_numbers"],
            summary=data["summary"],
            uno_dialogues=data["uno_dialogues"],
            panel_descriptions=data["panel_descriptions"],
            other_characters=set(data["other_characters"]),
        )


def extract_scenes_from_issue(issue_dir: Path) -> list[Scene]:
    """Extract all scenes containing Uno from an issue."""
    page_files = sorted(issue_dir.glob("page_*.json"))
    scenes = []
    current_scene_panels: list[dict] = []
    current_scene_pages: list[int] = []

    for page_file in page_files:
        page_num = int(page_file.stem.split("_")[1])

        with open(page_file, encoding="utf-8") as f:
            page_data = json.load(f)

        panels = page_data.get("panels", [])
        if not panels:
            continue

        for panel in panels:
            if panel.get("is_new_scene", False) and current_scene_panels:
                scene = create_scene_from_panels(
                    issue_dir.name, current_scene_pages, current_scene_panels
                )
                if scene:
                    scenes.append(scene)

                current_scene_panels = []
                current_scene_pages = []

            current_scene_panels.append(panel)
            if page_num not in current_scene_pages:
                current_scene_pages.append(page_num)

    if current_scene_panels:
        scene = create_scene_from_panels(
            issue_dir.name, current_scene_pages, current_scene_panels
        )
        if scene:
            scenes.append(scene)

    return scenes


def create_scene_from_panels(
    issue: str, page_numbers: list[int], panels: list[dict]
) -> Scene | None:
    """Create a Scene object from panels, only if Uno is present."""
    uno_dialogues = []
    panel_descriptions = []
    other_characters: set[str] = set()

    for panel in panels:
        if desc := panel.get("description"):
            panel_descriptions.append(desc)

        for dialogue in panel.get("dialogues", []):
            character = dialogue.get("character", "").strip()
            line = dialogue.get("line", "").strip()

            if character.lower() == "uno":
                uno_dialogues.append(line)
            elif character:
                other_characters.add(character)

    if not uno_dialogues:
        return None

    summary = " ".join(panel_descriptions)

    return Scene(
        issue=issue,
        page_numbers=page_numbers,
        summary=summary,
        uno_dialogues=uno_dialogues,
        panel_descriptions=panel_descriptions,
        other_characters=other_characters,
    )


def natural_sort_key(path: Path) -> tuple:
    """Generate a sort key for natural/numeric sorting of issue directories."""
    parts = path.name.split("-")
    key: list[int | str] = []
    for part in parts:
        try:
            key.append(int(part))
        except ValueError:
            key.append(part)
    return tuple(key)


# ============================================================================
# ClaimLedger Class
# ============================================================================


class ClaimLedger:
    """Manages claims about a character with evidence tracking."""

    def __init__(self):
        self._claims: dict[int, Claim] = {}
        self._next_id: int = 1
        self._processed_scene_ids: set[str] = set()  # Only IDs, serialized
        self._scene_cache: dict[str, Scene] = {}  # In-memory cache, not serialized

    def to_json(self) -> dict:
        """Serialize ledger to JSON-compatible dict."""
        return {
            "next_id": self._next_id,
            "claims": {str(k): v.model_dump() for k, v in self._claims.items()},
            "processed_scene_ids": sorted(self._processed_scene_ids),
        }

    @classmethod
    def from_json(cls, data: dict) -> "ClaimLedger":
        """Deserialize ledger from JSON dict."""
        ledger = cls()
        ledger._next_id = data.get("next_id", 1)
        ledger._claims = {
            int(k): Claim.model_validate(v) for k, v in data.get("claims", {}).items()
        }
        ledger._processed_scene_ids = set(data.get("processed_scene_ids", []))
        return ledger

    def add_scene(self, scene: Scene) -> None:
        """Mark a scene as processed and cache it for later reference."""
        self._processed_scene_ids.add(scene.scene_id)
        self._scene_cache[scene.scene_id] = scene

    def get_scene(self, scene_id: str) -> Scene | None:
        """Get a cached scene by ID (only available for current session)."""
        return self._scene_cache.get(scene_id)

    def is_scene_processed(self, scene_id: str) -> bool:
        """Check if a scene has already been processed."""
        return scene_id in self._processed_scene_ids

    def populate_scene_cache(self, scenes: list["Scene"]) -> None:
        """Populate the scene cache with previously processed scenes.

        Call this after loading from checkpoint to enable view_scene for
        scenes processed in previous runs.
        """
        for scene in scenes:
            if scene.scene_id in self._processed_scene_ids:
                self._scene_cache[scene.scene_id] = scene

    def get_claims_by_section(
        self, section: str | None = None
    ) -> dict[str, list[Claim]]:
        """Get claims grouped by section.

        Args:
            section: If provided, only return claims from this section.

        Returns:
            Dict mapping section names to lists of claims, excluding empty sections.
        """
        result: dict[str, list[Claim]] = defaultdict(list)
        for claim in self._claims.values():
            if section is None or claim.section == section:
                result[claim.section].append(claim)
        # Sort by support count within each section
        for section_claims in result.values():
            section_claims.sort(key=lambda c: c.support_count, reverse=True)
        return result

    def get_claim(self, claim_id: int) -> Claim | None:
        """Get a claim by ID."""
        return self._claims.get(claim_id)

    def add_claim(
        self,
        section: str,
        text: str,
        scene_id: str,
        justification: str,
        quote: str | None = None,
        quote_context: str | None = None,
    ) -> Claim:
        """Add a new claim with initial supporting evidence."""
        claim = Claim(
            id=self._next_id,
            text=text,
            section=section,
            supporting=[SceneEvidence(scene_id=scene_id, justification=justification)],
            contradicting=[],
            quotes=[],
        )
        if quote and quote_context:
            claim.quotes.append(
                Quote(text=quote, context=quote_context, scene_id=scene_id)
            )

        self._claims[self._next_id] = claim
        self._next_id += 1
        return claim

    def support_claim(
        self,
        claim_id: int,
        scene_id: str,
        justification: str,
        quote: str | None = None,
        quote_context: str | None = None,
    ) -> tuple[bool, str]:
        """Add supporting evidence to an existing claim."""
        claim = self._claims.get(claim_id)
        if not claim:
            return False, f"Claim {claim_id} not found"

        # Check if this scene already supports this claim
        for ev in claim.supporting:
            if ev.scene_id == scene_id:
                return False, f"Scene {scene_id} already supports claim {claim_id}"

        claim.supporting.append(
            SceneEvidence(scene_id=scene_id, justification=justification)
        )

        if quote and quote_context:
            claim.quotes.append(
                Quote(text=quote, context=quote_context, scene_id=scene_id)
            )

        return True, f"Added support to claim {claim_id} (now +{claim.support_count})"

    def contradict_claim(
        self,
        claim_id: int,
        scene_id: str,
        justification: str,
    ) -> tuple[bool, str]:
        """Add contradicting evidence to a claim."""
        claim = self._claims.get(claim_id)
        if not claim:
            return False, f"Claim {claim_id} not found"

        # Check if this scene already contradicts this claim
        for ev in claim.contradicting:
            if ev.scene_id == scene_id:
                return False, f"Scene {scene_id} already contradicts claim {claim_id}"

        claim.contradicting.append(
            SceneEvidence(scene_id=scene_id, justification=justification)
        )

        return (
            True,
            f"Added contradiction to claim {claim_id} (now +{claim.support_count})",
        )

    def refine_claim(self, claim_id: int, new_text: str) -> tuple[bool, str]:
        """Update the text of an existing claim."""
        claim = self._claims.get(claim_id)
        if not claim:
            return False, f"Claim {claim_id} not found"

        old_text = claim.text
        claim.text = new_text
        return (
            True,
            f"Refined claim {claim_id}: '{old_text[:50]}...' -> '{new_text[:50]}...'",
        )

    def claim_count(self) -> int:
        """Total number of claims."""
        return len(self._claims)

    def scene_count(self) -> int:
        """Total number of processed scenes."""
        return len(self._processed_scene_ids)


# ============================================================================
# System Prompts
# ============================================================================


def get_scene_processing_prompt() -> str:
    """Generate system prompt for scene processing."""
    return """You are building a claim-based character profile for "Uno" from PKNA comics (Paperinik New Adventures).

## Tools (Progressive Disclosure)
- list_claims(section=None): Compact list - "id: claim [+N]"
- view_claims(ids): Full details for specific claims (max 10)
- view_scene(scene_id): Look back at a previous scene for context
- add_claim(section, text, justification, quote?, quote_context?): Add NEW claim with justification
- support_claim(claim_id, justification, quote?, quote_context?): Support existing claim with reason
- contradict_claim(claim_id, justification): Contradict existing claim with reason
- refine_claim(claim_id, new_text): Update claim text

## Workflow
1. list_claims() to see existing claims
2. Analyze scene for evidence about Uno
3. For each insight:
   - MATCHES existing claim -> support_claim() with justification
   - CONTRADICTS existing claim -> contradict_claim() with justification
   - ADDS NUANCE to existing -> view_claims() to review details, then refine_claim()
   - GENUINELY NEW insight -> add_claim()
4. Brief summary when done

## Sections
- identity: Core facts about what Uno IS (nature, constraints, origin)
- personality: Character traits (sarcasm, protectiveness, humor)
- communication: How Uno speaks (expressions, nicknames, speech patterns)
- values: What Uno believes in (loyalty, duty, freedom)
- behavior: What Uno does or doesn't do (actions, habits, constraints)
- relationships: How Uno relates to specific characters

## Quality Guidelines
- Claims must be SPECIFIC and VERIFIABLE from scenes
- Prefer supporting existing claims over creating near-duplicates
- Write claims in ENGLISH
- Preserve Italian quotes exactly as they appear
- Include quote context explaining why the quote matters
- Justifications should be brief (1-2 sentences) explaining the evidence

## Current Scene ID
The current scene ID will be provided with each scene. Use this ID when adding evidence.
"""


def get_soul_document_prompt(threshold: int) -> str:
    """Generate system prompt for final soul document generation."""
    return f"""Generate a soul document from the validated claims provided.

## Input
You will receive claims organized by section, filtered to only include claims with support_count >= {threshold}.

## Output Format

# Uno - Soul Document

## Essential Identity
[From "identity" section - core facts about what Uno is]

## Core Personality
[From "personality" section - prioritized by support count]
[Include Italian quotes with inline translations: "Quote" (English translation)]

## Communication Style
### Voice and Tone
[From "communication" - formal/informal patterns, tone by context]

### Linguistic Markers
[Characteristic phrases, nicknames, expressions]
- Nicknames for Paperinik: list them
- Common expressions with translations

## Values and Beliefs
[From "values" section]

## Behavioral Guidelines
### What Uno Does
[Positive behaviors from "behavior" section]

### What Uno Doesn't Do
[Constraints and limitations from "behavior" section]

## Key Relationships
### With Paperinik
[Detailed dynamics from "relationships"]

### With Everett Ducklair
[Relationship description]

### With Due
[Relationship description]

[...other significant characters...]

## Guidelines
- Order claims by support_count (highest first) within sections
- Italian quotes format: "Quote" (English translation)
- Merge very similar claims into coherent prose
- Exclude claims with negative support_count
- Be comprehensive but not repetitive
- Capture the character's distinctive voice and personality
"""


def get_claim_refinement_prompt() -> str:
    """Generate system prompt for refining contradicted claims."""
    return """You refine character claims that have contradicting evidence.

## Task
Given a claim with supporting AND contradicting evidence, rewrite the claim to incorporate the nuance from the contradictions. The refined claim should preserve the core truth while adding qualifiers or exceptions.

## Rules
- Output ONLY the refined claim text. No preamble, no explanation, no wrapping.
- Keep claims concise: 1-3 sentences in English.
- Preserve the core insight while adding qualifiers/exceptions for contradictions.
- Use concrete language, not vague hedging (prefer "except when X" over "sometimes").

## Examples

Input claim: "Uno is always sarcastic."
Supporting: "Uses ironic deflection when Paperinik asks personal questions."
Contradicting: "Speaks with sincere urgency when Paperinik is in danger."
Output: Uno defaults to sarcasm and ironic deflection, especially in casual exchanges, but drops the sarcasm entirely and speaks with sincere urgency when Paperinik is in genuine danger.

Input claim: "Uno dislikes humans."
Supporting: "Expresses frustration with human inefficiency."
Contradicting: "Shows genuine concern for Paperinik's wellbeing."
Output: Uno is often frustrated by human inefficiency and limitations, but forms deep bonds with individuals he respects, showing genuine concern for their wellbeing.
"""


# ============================================================================
# Tool Formatting Functions
# ============================================================================


def format_claims_compact(ledger: ClaimLedger, section: str | None = None) -> str:
    """Format claims as a compact list for progressive disclosure."""
    claims_by_section = ledger.get_claims_by_section(section)

    lines = [f"Total claims: {ledger.claim_count()}"]
    lines.append("")

    for sect, claims in claims_by_section.items():
        lines.append(f"## {sect.title()}")
        for claim in claims:
            sign = "+" if claim.support_count >= 0 else ""
            lines.append(
                f"{claim.id}: {claim.text[:256]}{'...' if len(claim.text) > 256 else ''} [{sign}{claim.support_count}]"
            )
        lines.append("")

    return "\n".join(lines)


def format_claims_detail(ledger: ClaimLedger, claim_ids: list[int]) -> str:
    """Format detailed view of specific claims."""
    lines = []

    for claim_id in claim_ids:
        claim = ledger.get_claim(claim_id)
        if not claim:
            lines.append(f"ID {claim_id}: Not found")
            lines.append("")
            continue

        lines.append(f"ID {claim_id}:")
        lines.append(f"  Section: {claim.section}")
        lines.append(f"  Text: {claim.text}")
        lines.append(
            f"  Support: {'+' if claim.support_count >= 0 else ''}{claim.support_count} "
            f"({len(claim.supporting)} supporting, {len(claim.contradicting)} contradicting)"
        )

        if claim.supporting:
            lines.append("  Supporting:")
            for ev in claim.supporting:
                lines.append(f"    - {ev.scene_id}: {ev.justification}")

        if claim.contradicting:
            lines.append("  Contradicting:")
            for ev in claim.contradicting:
                lines.append(f"    - {ev.scene_id}: {ev.justification}")

        if claim.quotes:
            lines.append("  Quotes:")
            for i, q in enumerate(claim.quotes, 1):
                lines.append(f'    {i}. "{q.text}"')
                lines.append(f"       Context: {q.context}")

        lines.append("")

    return "\n".join(lines)


def format_scene_view(scene: Scene) -> str:
    """Format a scene for the view_scene tool."""
    lines = [
        f"Scene: {scene.scene_id}",
        f"Issue: {scene.issue}, pages {'-'.join(map(str, scene.page_numbers))}",
        f"Characters present: {', '.join(sorted(scene.other_characters)) if scene.other_characters else 'Uno only'}",
        "",
        f"Summary: {scene.summary}",
        "",
        "Uno's Dialogues:",
    ]
    for dialogue in scene.uno_dialogues:
        lines.append(f'- "{dialogue}"')

    if scene.panel_descriptions:
        lines.append("")
        lines.append("Panel Descriptions:")
        for desc in scene.panel_descriptions:
            lines.append(f"- {desc}")

    return "\n".join(lines)


# ============================================================================
# LedgerTools - Shared tools for LLM interaction with the claim ledger
# ============================================================================


class LedgerTools:
    """Tool methods for LLM interaction with the claim ledger.

    Shared between SceneProcessor and ClaimRefiner so both can inspect
    claims, evidence, and scenes.
    """

    def __init__(self, ledger: ClaimLedger):
        self._ledger = ledger
        self._current_scene_id: str = ""

    @property
    def all(self) -> list:
        """All tool methods for GenerateContentConfig."""
        return [
            self.list_claims,
            self.view_claims,
            self.view_scene,
            self.add_claim,
            self.support_claim,
            self.contradict_claim,
            self.refine_claim,
        ]

    def list_claims(self, section: str | None = None) -> str:
        """List all claims in compact format.

        Shows claim ID, text (truncated), and net support count for quick scanning.

        Args:
            section: Optional filter by section (identity, personality, communication,
                    values, behavior, relationships). If None, shows all sections.

        Returns:
            Compact listing of claims organized by section.
        """
        log.debug(f"[list_claims] section={section}")
        return format_claims_compact(self._ledger, section)

    def view_claims(self, ids: list[int]) -> str:
        """View detailed information about specific claims.

        Shows full text, support/contradict evidence with justifications, and quotes.
        Limited to 10 claims per request.

        Args:
            ids: List of claim IDs to view (max 10).

        Returns:
            Detailed information for each requested claim.
        """
        log.debug(f"[view_claims] ids={ids}")
        return format_claims_detail(self._ledger, ids)

    def view_scene(self, scene_id: str) -> str:
        """Look back at a previously processed scene.

        Useful when refining claims or resolving contradictions by reviewing
        original context.

        Args:
            scene_id: Scene identifier (e.g., "pkna-0_12").

        Returns:
            Scene details including summary, dialogues, and panel descriptions.
        """
        log.debug(f"[view_scene] scene_id={scene_id}")
        scene = self._ledger.get_scene(scene_id)
        if not scene:
            return f"Scene {scene_id} not found in processed scenes."
        return format_scene_view(scene)

    def add_claim(
        self,
        section: str,
        text: str,
        justification: str,
        quote: str | None = None,
        quote_context: str | None = None,
    ) -> str:
        """Add a new claim about Uno.

        Creates a new claim with the current scene as initial supporting evidence.

        Args:
            section: Claim category (identity, personality, communication, values,
                    behavior, relationships).
            text: The claim text in English.
            justification: Brief explanation of why this scene supports the claim.
            quote: Optional Italian quote from the scene that exemplifies the claim.
            quote_context: Required if quote provided - explains the quote's significance.

        Returns:
            Confirmation with the new claim ID and support count.
        """
        log.debug(f"[add_claim] section={section}, text={text[:50]}...")

        if section not in VALID_SECTIONS:
            return f"Error: Invalid section '{section}'. Valid sections: {', '.join(sorted(VALID_SECTIONS))}"

        if quote and not quote_context:
            return "Error: quote_context is required when providing a quote."

        claim = self._ledger.add_claim(
            section=section,
            text=text,
            scene_id=self._current_scene_id,
            justification=justification,
            quote=quote,
            quote_context=quote_context,
        )

        return f"Added claim {claim.id} in {section}: '{text[:60]}...' [+{claim.support_count}]"

    def support_claim(
        self,
        claim_id: int,
        justification: str,
        quote: str | None = None,
        quote_context: str | None = None,
    ) -> str:
        """Add supporting evidence to an existing claim.

        Adds the current scene to the claim's supporting evidence list.

        Args:
            claim_id: ID of the claim to support.
            justification: Brief explanation of why this scene supports the claim.
            quote: Optional Italian quote that exemplifies the claim.
            quote_context: Required if quote provided - explains the quote's significance.

        Returns:
            Confirmation with updated support count.
        """
        log.debug(f"[support_claim] claim_id={claim_id}")

        if quote and not quote_context:
            return "Error: quote_context is required when providing a quote."

        _, message = self._ledger.support_claim(
            claim_id=claim_id,
            scene_id=self._current_scene_id,
            justification=justification,
            quote=quote,
            quote_context=quote_context,
        )

        return message

    def contradict_claim(self, claim_id: int, justification: str) -> str:
        """Add contradicting evidence to a claim.

        Adds the current scene to the claim's contradicting evidence list,
        reducing its net support count.

        Args:
            claim_id: ID of the claim to contradict.
            justification: Brief explanation of why this scene contradicts the claim.

        Returns:
            Confirmation with updated support count.
        """
        log.debug(f"[contradict_claim] claim_id={claim_id}")

        _, message = self._ledger.contradict_claim(
            claim_id=claim_id,
            scene_id=self._current_scene_id,
            justification=justification,
        )

        return message

    def refine_claim(self, claim_id: int, new_text: str) -> str:
        """Update the text of an existing claim.

        Keeps all supporting and contradicting evidence intact.
        Use this to improve claim wording or add nuance.

        Args:
            claim_id: ID of the claim to refine.
            new_text: New claim text in English.

        Returns:
            Confirmation showing old and new text.
        """
        log.debug(f"[refine_claim] claim_id={claim_id}, new_text={new_text[:50]}...")

        _, message = self._ledger.refine_claim(claim_id, new_text)
        return message


# ============================================================================
# SceneProcessor Class
# ============================================================================


class SceneProcessor:
    """Processes scenes and updates the claim ledger."""

    def __init__(self, client: genai.Client, ledger: ClaimLedger):
        self._client = client
        self._ledger = ledger
        self._tools = LedgerTools(ledger)
        self._config = GenerateContentConfig(
            system_instruction=get_scene_processing_prompt(),
            temperature=0.7,
            top_p=0.95,
            tools=self._tools.all,
        )

    # -------------------------------------------------------------------------
    # Processing Logic
    # -------------------------------------------------------------------------

    def _generate_with_retry(
        self, conversation: list[Content]
    ) -> GenerateContentResponse | None:
        """Call generate_content with retry on resource exhausted and timeout errors."""
        return generate_with_retry(self._client, conversation, self._config)

    def process_scene(self, scene: Scene, scene_number: int) -> tuple[bool, str]:
        """Process a single scene and update the claim ledger.

        Returns:
            Tuple of (success, summary_or_error)
        """
        self._tools._current_scene_id = scene.scene_id

        # Format scene as prompt
        dialogues_text = "\n".join(f'- "{d}"' for d in scene.uno_dialogues)
        context = scene.to_context_string()
        panel_context = scene.to_other_context()

        scene_prompt = f"""Analyze this scene and update the claim ledger.

**Current Scene ID:** {scene.scene_id}

**Scene Context:** {context}

**Scene Summary:** {scene.summary}

**Uno's Dialogues (Italian):**
{dialogues_text}

**Additional Context:** {panel_context}

Use list_claims() to see existing claims, then add supporting/contradicting evidence or new claims as appropriate. When done, provide a brief one-line summary of your updates."""

        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=scene_prompt)])
        ]

        try:
            response = self._generate_with_retry(conversation)
            if response is None:
                return False, "API call failed after retries"

            summary = response.text or "No summary provided"

            # Mark scene as processed
            self._ledger.add_scene(scene)

            log.info(f"Scene {scene_number}: {summary[:256]}...")
            return True, summary

        except Exception as e:
            log.error(f"Scene {scene_number}: Error - {e}")
            return False, str(e)


# ============================================================================
# Claim Refiner
# ============================================================================


class ClaimRefiner:
    """Refines contradicted claims to incorporate nuance from contradictions."""

    def __init__(self, client: genai.Client, ledger: ClaimLedger):
        self._client = client
        self._ledger = ledger
        self._tools = LedgerTools(ledger)
        self._config = GenerateContentConfig(
            system_instruction=get_claim_refinement_prompt(),
            temperature=0.3,
            top_p=0.95,
            tools=self._tools.all,
        )

    def _find_contradicted_claims(self) -> list[Claim]:
        """Find all claims that have contradicting evidence."""
        result = []
        for claims in self._ledger.get_claims_by_section().values():
            for claim in claims:
                if claim.contradicting:
                    result.append(claim)
        return result

    def refine_claim(self, claim: Claim) -> tuple[bool, str]:
        """Refine a single claim via LLM call.

        Returns:
            Tuple of (success, refined_text_or_error).
        """
        claim_detail = format_claims_detail(self._ledger, [claim.id])
        prompt = f"Refine this contradicted claim:\n\n{claim_detail}"
        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=prompt)])
        ]

        response = generate_with_retry(self._client, conversation, self._config)
        if response is None:
            return False, "API call failed after retries"

        refined_text = (response.text or "").strip()
        if not refined_text:
            return False, "Empty response from LLM"

        return True, refined_text

    def refine_all(self) -> tuple[int, int]:
        """Refine all contradicted claims.

        Returns:
            Tuple of (refined_count, failed_count).
        """
        contradicted = self._find_contradicted_claims()
        if not contradicted:
            log.info("No contradicted claims to refine")
            return 0, 0

        log.info(f"Refining {len(contradicted)} contradicted claims...")
        refined_count = 0
        failed_count = 0

        with PROGRESS as progress:
            task = progress.add_task("Refining claims...", total=len(contradicted))

            for i, claim in enumerate(contradicted, 1):
                success, result = self.refine_claim(claim)

                if success:
                    self._ledger.refine_claim(claim.id, result)
                    claim.absorb_contradictions()
                    refined_count += 1
                    log.debug(f"Refined claim {claim.id}: {result[:80]}...")
                else:
                    failed_count += 1
                    log.warning(f"Failed to refine claim {claim.id}: {result}")

                progress.update(task, completed=i)

        return refined_count, failed_count


# ============================================================================
# Soul Document Generator
# ============================================================================


class SoulDocumentGenerator:
    """Generates the final soul document from validated claims."""

    def __init__(self, client: genai.Client, ledger: ClaimLedger, threshold: int):
        self._client = client
        self._ledger = ledger
        self._threshold = threshold
        self._config = GenerateContentConfig(
            system_instruction=get_soul_document_prompt(threshold),
            temperature=0.7,
            top_p=0.95,
        )

    def _format_claims_for_generation(self) -> str:
        """Format validated claims for soul document generation."""
        lines = []
        claims_by_section = self._ledger.get_claims_by_section()

        for section in VALID_SECTIONS:
            claims = [
                c
                for c in claims_by_section[section]
                if c.support_count >= self._threshold
            ]
            if not claims:
                continue

            lines.append(f"## {section.upper()}")
            lines.append("")

            for claim in claims:
                lines.append(
                    f"**Claim (support: +{claim.support_count}):** {claim.text}"
                )

                if claim.quotes:
                    lines.append("Quotes:")
                    for q in claim.quotes:
                        lines.append(f'  - "{q.text}" — {q.context}')

                lines.append("")

        return "\n".join(lines)

    def generate(self) -> tuple[bool, str]:
        """Generate the soul document from validated claims.

        Returns:
            Tuple of (success, document_or_error)
        """
        claims_text = self._format_claims_for_generation()

        if not claims_text.strip():
            return False, "No claims meet the support threshold"

        prompt = f"""Generate a soul document from these validated claims:

{claims_text}

Create a comprehensive, well-organized soul document following the specified format."""

        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=prompt)])
        ]

        try:
            response = self._client.models.generate_content(
                model=MODEL_NAME,
                contents=conversation,  # type: ignore[arg-type]
                config=self._config,
            )

            document = response.text or ""
            return True, document

        except Exception as e:
            log.error(f"Soul document generation failed: {e}")
            return False, str(e)


# ============================================================================
# Checkpoint Management
# ============================================================================


def save_checkpoint(ledger: ClaimLedger, checkpoint_num: int) -> None:
    """Save ledger checkpoint as JSON, keeping last 3."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num:03d}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)

    log.debug(f"Saved checkpoint {checkpoint_num}: {path_str(checkpoint_path)}")

    # Remove old checkpoints (keep last 3)
    if checkpoint_num > 3:
        old_checkpoint = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num - 3:03d}.json"
        if old_checkpoint.exists():
            old_checkpoint.unlink()


def find_latest_checkpoint() -> tuple[int, ClaimLedger] | None:
    """Find the latest checkpoint and return its number and ledger.

    Returns:
        Tuple of (checkpoint_number, ledger) if a checkpoint exists,
        None otherwise.
    """
    if not CHECKPOINTS_DIR.exists():
        return None

    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("checkpoint_*.json"))
    if not checkpoint_files:
        return None

    latest = checkpoint_files[-1]

    try:
        checkpoint_num = int(latest.stem.split("_")[1])
    except (IndexError, ValueError):
        log.warning(f"Could not parse checkpoint number from {latest}")
        return None

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    ledger = ClaimLedger.from_json(data)
    return checkpoint_num, ledger


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main function to build the character profile."""
    parser = argparse.ArgumentParser(
        description="Build character profile using claim ledger approach"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (for testing)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=CLAIM_SUPPORT_THRESHOLD,
        help=f"Minimum support count for claims in final document (default: {CLAIM_SUPPORT_THRESHOLD})",
    )
    args = parser.parse_args()

    console.print(
        f"\n[bold cyan]Claim Ledger Character Profile Builder ({VERSION_TAG})[/bold cyan]\n"
    )

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume from
    checkpoint_info = find_latest_checkpoint()
    if checkpoint_info:
        checkpoint_num, ledger = checkpoint_info
        log.info(
            f"Resuming from checkpoint {checkpoint_num} "
            f"({ledger.claim_count()} claims, {ledger.scene_count()} scenes processed)"
        )
    else:
        ledger = ClaimLedger()
        checkpoint_num = 0
        log.info("Starting fresh")

    client = genai.Client(http_options=HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000))
    processor = SceneProcessor(client, ledger)

    # Collect all scenes with Uno
    log.info("Scanning for scenes containing Uno...")
    all_scenes: list[Scene] = []
    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )

    for issue_dir in issue_dirs:
        scenes = extract_scenes_from_issue(issue_dir)
        all_scenes.extend(scenes)

    log.info(f"Total: {len(all_scenes)} scenes with Uno across all issues")

    # Populate scene cache for previously processed scenes (enables view_scene on resume)
    ledger.populate_scene_cache(all_scenes)

    # Filter to unprocessed scenes
    unprocessed_scenes = [
        s for s in all_scenes if not ledger.is_scene_processed(s.scene_id)
    ]

    if args.max_scenes:
        unprocessed_scenes = unprocessed_scenes[: args.max_scenes]

    if not unprocessed_scenes:
        log.info("All scenes already processed!")
    else:
        log.info(f"Processing {len(unprocessed_scenes)} unprocessed scenes")

        # Open processing log file (append mode for resume support)
        log_path = OUTPUT_DIR / "operations.jsonl"
        successful_count = 0

        with open(log_path, "a", encoding="utf-8") as log_file:
            with PROGRESS as progress:
                task = progress.add_task(
                    "Building claim ledger...",
                    total=len(unprocessed_scenes),
                )

                for i, scene in enumerate(unprocessed_scenes, 1):
                    log.info(
                        f"\nProcessing scene {i}/{len(unprocessed_scenes)}: {scene.scene_id}"
                    )

                    success, summary = processor.process_scene(scene, i)

                    if success:
                        successful_count += 1

                    # Save checkpoint after each scene
                    checkpoint_num += 1
                    save_checkpoint(ledger, checkpoint_num)

                    # Write log entry
                    log_entry = {
                        "scene_number": i,
                        "scene_id": scene.scene_id,
                        "issue": scene.issue,
                        "pages": scene.page_numbers,
                        "success": success,
                        "summary": summary,
                        "claims_after": ledger.claim_count(),
                    }
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    log_file.flush()

                    progress.update(task, completed=i)

        console.print("\n[bold green]Scene processing complete![/bold green]")
        console.print(f"Processed: {len(unprocessed_scenes)} scenes")
        console.print(f"Successful: {successful_count}")
        console.print(f"Total claims: {ledger.claim_count()}")

    # Save final ledger
    final_ledger_path = OUTPUT_DIR / "final_ledger.json"
    with open(final_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved final ledger to {path_str(final_ledger_path)}")

    # Refine contradicted claims
    console.print("\n[bold cyan]Refining contradicted claims...[/bold cyan]")
    refiner = ClaimRefiner(client, ledger)
    refined_count, failed_count = refiner.refine_all()

    if refined_count > 0 or failed_count > 0:
        console.print(f"Refined: {refined_count}, Failed: {failed_count}")

        # Save refined ledger
        refined_ledger_path = OUTPUT_DIR / "refined_ledger.json"
        with open(refined_ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
        log.info(f"Saved refined ledger to {path_str(refined_ledger_path)}")
    else:
        console.print("No contradicted claims found")

    # Generate soul document
    console.print("\n[bold cyan]Generating soul document...[/bold cyan]")

    generator = SoulDocumentGenerator(client, ledger, args.threshold)
    success, result = generator.generate()

    if success:
        soul_doc_path = OUTPUT_DIR / "uno_soul_document.md"
        with open(soul_doc_path, "w", encoding="utf-8") as f:
            f.write(result)

        tokens = count_tokens(result)
        words = len(result.split())

        console.print("\n[bold green]Soul document generated![/bold green]")
        console.print(f"Output: {path_str(soul_doc_path)}")
        console.print(f"Size: {tokens:,} tokens (~{words:,} words)")
        console.print(f"Threshold: support >= {args.threshold}")

        # Count claims by threshold
        claims_included = sum(
            1
            for claims in ledger.get_claims_by_section().values()
            for c in claims
            if c.support_count >= args.threshold
        )
        console.print(f"Claims included: {claims_included}/{ledger.claim_count()}")
    else:
        console.print(
            f"\n[bold red]Soul document generation failed:[/bold red] {result}"
        )

    # Print summary
    console.print(f"\n[bold]Output directory:[/bold] {path_str(OUTPUT_DIR)}")
    console.print(f"Checkpoints: {path_str(CHECKPOINTS_DIR)}/")


if __name__ == "__main__":
    main()
