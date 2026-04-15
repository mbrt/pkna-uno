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
from typing import cast

import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    ContentListUnionDict,
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
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Settings
MODEL_NAME = "gemini-3-flash-preview"
CHARACTER_NAME = "Uno"
ENCODING_NAME = "cl100k_base"
VERSION_TAG = "v11"
MAX_TOOL_ITERATIONS = 64
CLAIM_SUPPORT_THRESHOLD = 2  # Minimum support_count to include in final document

# Retry settings for API calls
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0
API_TIMEOUT_SECONDS = 300  # 5 minutes per API call

# Paths
BASE_DIR = Path(__file__).parent.parent
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

# Valid claim sections (ordered for deterministic document generation)
SECTION_ORDER = [
    "identity",
    "psychology",
    "communication",
    "motivations",
    "capabilities",
    "behavior",
    "relationships",
]

# Valid top-level sections
VALID_SECTIONS = set(SECTION_ORDER)

# Valid hierarchical paths (AIEOS-derived)
VALID_PATHS = {
    # Identity
    "identity/names",
    "identity/bio",
    "identity/origin",
    # Psychology - Neural Matrix (AI cognitive weights)
    "psychology/neural_matrix/creativity",
    "psychology/neural_matrix/empathy",
    "psychology/neural_matrix/logic",
    "psychology/neural_matrix/adaptability",
    "psychology/neural_matrix/charisma",
    "psychology/neural_matrix/reliability",
    # Psychology - OCEAN Traits
    "psychology/traits/ocean/openness",
    "psychology/traits/ocean/conscientiousness",
    "psychology/traits/ocean/extraversion",
    "psychology/traits/ocean/agreeableness",
    "psychology/traits/ocean/neuroticism",
    # Psychology - Other traits
    "psychology/traits/mbti",
    "psychology/traits/temperament",
    # Psychology - Moral Compass
    "psychology/moral_compass/alignment",
    "psychology/moral_compass/core_values",
    "psychology/moral_compass/conflict_resolution",
    # Psychology - Emotional Profile
    "psychology/emotional/base_mood",
    "psychology/emotional/volatility",
    "psychology/emotional/resilience",
    "psychology/emotional/triggers/joy",
    "psychology/emotional/triggers/anger",
    "psychology/emotional/triggers/sadness",
    "psychology/emotional/triggers/fear",
    # Communication (AIEOS: Linguistics)
    "communication/voice/formality",
    "communication/voice/verbosity",
    "communication/voice/vocabulary",
    "communication/voice/style",
    "communication/syntax/structure",
    "communication/syntax/contractions",
    "communication/idiolect/catchphrases",
    "communication/idiolect/nicknames",
    "communication/idiolect/expressions",
    "communication/idiolect/forbidden",
    "communication/interaction/dominance",
    "communication/interaction/turn_taking",
    "communication/interaction/emotional_coloring",
    # Motivations
    "motivations/core_drive",
    "motivations/goals/short_term",
    "motivations/goals/long_term",
    "motivations/fears/rational",
    "motivations/fears/irrational",
    # Capabilities
    "capabilities/skills",
    "capabilities/limitations",
    # Behavior
    "behavior/does",
    "behavior/avoids",
    # Relationships - dynamic (relationships/{name})
    # Validated separately to allow any character name
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
                contents=cast(ContentListUnionDict, conversation),
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
    path: str  # Hierarchical path, e.g., "psychology/traits/ocean/openness"
    supporting: list[SceneEvidence] = []  # Scenes that support with justification
    contradicting: list[SceneEvidence] = []  # Scenes that contradict with justification
    quotes: list[Quote] = []  # Quotes with context

    @property
    def section(self) -> str:
        """Top-level section (first component of path)."""
        return self.path.split("/")[0]

    @property
    def support_count(self) -> int:
        """Net support: supporting scenes minus contradicting scenes."""
        return len(self.supporting) - len(self.contradicting)


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

    def get_claims_by_path(
        self, path_prefix: str | None = None
    ) -> dict[str, list[Claim]]:
        """Get claims grouped by full path, filtered by prefix.

        Args:
            path_prefix: If provided, only return claims with paths starting with this prefix.

        Returns:
            Dict mapping full paths to lists of claims.
        """
        result: dict[str, list[Claim]] = defaultdict(list)
        for claim in self._claims.values():
            if path_prefix is None or claim.path.startswith(path_prefix):
                result[claim.path].append(claim)
        return result

    def get_claim(self, claim_id: int) -> Claim | None:
        """Get a claim by ID."""
        return self._claims.get(claim_id)

    def add_claim(
        self,
        path: str,
        text: str,
        scene_id: str,
        justification: str,
        quote: str | None = None,
        quote_context: str | None = None,
    ) -> Claim:
        """Add a new claim with initial supporting evidence."""
        # Validate path
        section = path.split("/")[0]
        if section not in VALID_SECTIONS:
            raise ValueError(f"Invalid section in path '{path}'")

        # relationships/{name} pattern is always valid
        if not (section == "relationships" or path in VALID_PATHS):
            raise ValueError(f"Invalid path '{path}'")

        claim = Claim(
            id=self._next_id,
            text=text,
            path=path,
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
    return """You are building a claim-based character profile for "Uno" from PKNA comics.

## Tools (Progressive Disclosure)
- list_claims(section=None): Compact list - "id: path: claim [+N]"
- view_claims(ids): Full details for specific claims (max 10)
- view_scene(scene_id): Look back at a previous scene for context
- add_claim(path, text, justification, quote?, quote_context?): Add NEW claim
- support_claim(claim_id, justification, quote?, quote_context?): Support existing claim
- contradict_claim(claim_id, justification): Contradict existing claim
- refine_claim(claim_id, new_text): Update claim text

## Workflow
1. list_claims() to see existing claims
2. Analyze scene for evidence about Uno
3. For each insight:
   - MATCHES existing claim -> support_claim() with justification
   - CONTRADICTS existing claim -> contradict_claim() with justification
   - ADDS NUANCE -> view_claims() to review, then refine_claim()
   - GENUINELY NEW -> add_claim()
4. Brief summary when done

## Claim Paths (use these exact paths)

### identity/ - Core facts about what Uno IS
- identity/names: Names, aliases, nicknames
- identity/bio: Nature, entity type, physical form, hardware
- identity/origin: Creator (Everett Ducklair), creation context, purpose

### psychology/ - Personality and emotional patterns
Neural Matrix (AI cognitive weights):
- psychology/neural_matrix/creativity: Creative thinking ability
- psychology/neural_matrix/empathy: Emotional understanding
- psychology/neural_matrix/logic: Logical reasoning ability
- psychology/neural_matrix/adaptability: Flexibility in situations
- psychology/neural_matrix/charisma: Social influence ability
- psychology/neural_matrix/reliability: Dependability and consistency

OCEAN Traits (Big Five):
- psychology/traits/ocean/openness: Openness to experience
- psychology/traits/ocean/conscientiousness: Organization, dependability
- psychology/traits/ocean/extraversion: Social energy, assertiveness
- psychology/traits/ocean/agreeableness: Cooperation, trust
- psychology/traits/ocean/neuroticism: Emotional instability

Other Personality:
- psychology/traits/mbti: MBTI type (e.g., INTJ)
- psychology/traits/temperament: Temperament description

Moral Compass:
- psychology/moral_compass/alignment: Moral alignment (lawful good, etc.)
- psychology/moral_compass/core_values: Core values (loyalty, duty, etc.)
- psychology/moral_compass/conflict_resolution: How conflicts are resolved

Emotional Profile:
- psychology/emotional/base_mood: Default emotional state
- psychology/emotional/volatility: Emotional stability (high = unstable)
- psychology/emotional/resilience: Recovery from setbacks
- psychology/emotional/triggers/joy: Things that bring joy
- psychology/emotional/triggers/anger: Things that cause anger
- psychology/emotional/triggers/sadness: Things that cause sadness
- psychology/emotional/triggers/fear: Things that cause fear

### communication/ - How Uno speaks
Voice Style:
- communication/voice/formality: Formal vs casual register
- communication/voice/verbosity: Terse vs verbose style
- communication/voice/vocabulary: Vocabulary level (technical, advanced)
- communication/voice/style: Style descriptors (sarcastic, witty)

Syntax:
- communication/syntax/structure: Sentence structure patterns
- communication/syntax/contractions: Contraction usage patterns

Idiolect (individual speech):
- communication/idiolect/catchphrases: Characteristic phrases (Italian + translation)
- communication/idiolect/nicknames: Nicknames for Paperinik and others
- communication/idiolect/expressions: Common expressions
- communication/idiolect/forbidden: Words/phrases Uno avoids

Interaction Patterns:
- communication/interaction/dominance: Conversational control patterns
- communication/interaction/turn_taking: Turn-taking patterns
- communication/interaction/emotional_coloring: Emotional tone

### motivations/ - What drives Uno
- motivations/core_drive: Primary motivation (raison d'être)
- motivations/goals/short_term: Immediate objectives
- motivations/goals/long_term: Long-term aspirations
- motivations/fears/rational: Logical fears (system failure, etc.)
- motivations/fears/irrational: Emotional fears (abandonment, etc.)

### capabilities/ - What Uno can and cannot do
- capabilities/skills: Skills and abilities with proficiency context
- capabilities/limitations: Explicit constraints, vulnerabilities, dependencies

### behavior/ - Actions and habits
- behavior/does: Positive behaviors, protocols, habits
- behavior/avoids: Self-imposed constraints, things avoided

### relationships/ - Character dynamics
- relationships/{character}: Use character name (e.g., relationships/paperinik)

## Quality Guidelines
- Claims must be SPECIFIC and VERIFIABLE from scenes
- Use the exact paths above (or relationships/{name} pattern)
- Prefer supporting existing claims over creating near-duplicates
- Write claims in ENGLISH
- Preserve Italian quotes exactly
- Include quote context explaining why the quote matters
"""


SECTION_PROMPTS = {
    "identity": """## Essential Identity

### Names and Aliases
[Claims from identity/names]

### Nature and Form
[Claims from identity/bio - what Uno IS, entity type, physical manifestation]

### Origin
[Claims from identity/origin - creator, creation context, purpose]

Write as structured prose with key facts clearly stated.""",
    "psychology": """## Core Psychology

### Neural Matrix (AI Cognitive Profile)
[Claims from psychology/neural_matrix/* - creativity, empathy, logic, adaptability, charisma, reliability]
Describe cognitive strengths and patterns as prose.

### Personality Traits
[Claims from psychology/traits/* - OCEAN, MBTI, temperament]
Include Italian quotes with inline translations: "Quote" (translation)

### Moral Compass
[Claims from psychology/moral_compass/* - alignment, core values, conflict resolution]

### Emotional Profile
[Claims from psychology/emotional/* - base mood, volatility, resilience]

#### Emotional Triggers
- Joy: [psychology/emotional/triggers/joy]
- Anger: [psychology/emotional/triggers/anger]
- Sadness: [psychology/emotional/triggers/sadness]
- Fear: [psychology/emotional/triggers/fear]

Write flowing prose capturing psychological makeup with concrete examples.""",
    "communication": """## Communication Style

### Voice and Tone
[Claims from communication/voice/* - formality, verbosity, vocabulary, style]

### Syntax Patterns
[Claims from communication/syntax/*]

### Linguistic Markers (Idiolect)
[Claims from communication/idiolect/*]
- **Catchphrases**: "Italian phrase" (translation)
- **Nicknames for Paperinik**: list them
- **Common expressions**: with translations

### Interaction Patterns
[Claims from communication/interaction/* - dominance, turn-taking, emotional coloring]

List concrete examples with Italian quotes.""",
    "motivations": """## Motivations and Drives

### Core Purpose
[Claims from motivations/core_drive - what fundamentally drives Uno]

### Goals
**Short-term:** [motivations/goals/short_term]
**Long-term:** [motivations/goals/long_term]

### Fears
**Rational fears:** [motivations/fears/rational - logical threats]
**Irrational fears:** [motivations/fears/irrational - emotional anxieties]

Write prose capturing what motivates and concerns the character.""",
    "capabilities": """## Capabilities and Limitations

### Skills and Abilities
[Claims from capabilities/skills]
List with proficiency context where relevant.

### Limitations
[Claims from capabilities/limitations]
Explicit constraints, vulnerabilities, blind spots, dependencies.

Include specific examples from scenes.""",
    "behavior": """## Behavioral Patterns

### What Uno Does
[Claims from behavior/does - positive behaviors, habits, protocols]

### What Uno Avoids
[Claims from behavior/avoids - self-imposed constraints, things avoided]

Split into subsections with concrete examples.""",
    "relationships": """## Key Relationships

### With Paperinik
[Claims from relationships/paperinik]

### With Everett Ducklair
[Claims from relationships/everett_ducklair or relationships/ducklair]

### With Due
[Claims from relationships/due]

### Other Characters
[Group remaining relationships/* claims by character]

Create subsections for each character with significant claims.""",
}


def get_section_prompt(section: str, threshold: int) -> str:
    """Generate system prompt for a single section of the soul document."""
    template = SECTION_PROMPTS[section]
    return f"""Generate one section of a soul document from validated claims.

You MUST incorporate ALL claims provided. Every single claim represents validated
evidence about this character and must be reflected in your output. Do NOT skip
claims with lower support counts — they are already filtered to meet the minimum
threshold of {threshold}.

## Output Format

{template}

## Formatting Rules
- Order claims by support_count (highest first) within subsections
- Italian quotes format: "Quote" (English translation)
- Merge very similar claims into coherent prose rather than repeating
- Be comprehensive — every claim must appear in the output
- Capture the character's distinctive voice and personality
- Output ONLY the section content (starting with the ## heading), no preamble
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
    """Format claims as a compact list showing paths for progressive disclosure."""
    claims_by_section = ledger.get_claims_by_section(section)

    lines = [f"Total claims: {ledger.claim_count()}"]
    lines.append("")

    for sect, claims in claims_by_section.items():
        lines.append(f"## {sect.title()}")
        for claim in claims:
            sign = "+" if claim.support_count >= 0 else ""
            # Show path instead of just section
            lines.append(
                f"{claim.id}: [{claim.path}] {claim.text[:200]}{'...' if len(claim.text) > 200 else ''} [{sign}{claim.support_count}]"
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
        lines.append(f"  Path: {claim.path}")
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

        Shows claim ID, path, text (truncated), and net support count for quick scanning.

        Args:
            section: Optional filter by section (identity, psychology, communication,
                    motivations, capabilities, behavior, relationships). If None, shows all sections.

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
        path: str,
        text: str,
        justification: str,
        quote: str | None = None,
        quote_context: str | None = None,
    ) -> str:
        """Add a new claim about Uno.

        Creates a new claim with the current scene as initial supporting evidence.

        Args:
            path: Claim path (e.g., "psychology/traits/ocean/openness").
            text: The claim text in English.
            justification: Brief explanation of why this scene supports the claim.
            quote: Optional Italian quote from the scene that exemplifies the claim.
            quote_context: Required if quote provided - explains the quote's significance.

        Returns:
            Confirmation with the new claim ID and support count.
        """
        log.debug(f"[add_claim] path={path}, text={text[:50]}...")

        # Validate path
        section = path.split("/")[0]
        if section not in VALID_SECTIONS:
            return f"Error: Invalid section in path '{path}'. Valid sections: {', '.join(sorted(VALID_SECTIONS))}"

        # relationships/{name} pattern is always valid
        if not (section == "relationships" or path in VALID_PATHS):
            return f"Error: Invalid path '{path}'. Use one of the defined paths or relationships/{{character_name}}."

        if quote and not quote_context:
            return "Error: quote_context is required when providing a quote."

        try:
            claim = self._ledger.add_claim(
                path=path,
                text=text,
                scene_id=self._current_scene_id,
                justification=justification,
                quote=quote,
                quote_context=quote_context,
            )
        except ValueError as e:
            return f"Error: {e}"

        return f"Added claim {claim.id} [{path}]: '{text[:60]}...' [+{claim.support_count}]"

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
                    # Absorb contradictions into supporting evidence
                    claim.supporting.extend(claim.contradicting)
                    claim.contradicting = []
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
    """Generates the final soul document from validated claims, one section at a time."""

    def __init__(self, client: genai.Client, ledger: ClaimLedger, threshold: int):
        self._client = client
        self._ledger = ledger
        self._threshold = threshold

    def _format_section_claims(self, section: str) -> str:
        """Format validated claims for a single section, grouped by path."""
        claims_by_section = self._ledger.get_claims_by_section(section=section)
        claims = [
            c
            for c in claims_by_section.get(section, [])
            if c.support_count >= self._threshold
        ]
        if not claims:
            return ""

        # Group by path for structured output
        by_path: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            by_path[claim.path].append(claim)

        lines = []
        for path in sorted(by_path.keys()):
            lines.append(f"### Path: {path}")
            for claim in by_path[path]:
                lines.append(
                    f"**Claim (support: +{claim.support_count}):** {claim.text}"
                )
                if claim.quotes:
                    for q in claim.quotes:
                        lines.append(f'  - "{q.text}" — {q.context}')
            lines.append("")

        return "\n".join(lines)

    def _generate_section(self, section: str) -> tuple[bool, str]:
        """Generate a single section of the soul document.

        Returns:
            Tuple of (success, section_text_or_error).
            If no claims meet the threshold for this section, returns (True, "").
        """
        claims_text = self._format_section_claims(section)
        if not claims_text.strip():
            log.info(f"Section '{section}': no claims meet threshold, skipping")
            return True, ""

        claim_count = claims_text.count("**Claim (support:")
        log.info(f"Section '{section}': generating from {claim_count} claims")

        config = GenerateContentConfig(
            system_instruction=get_section_prompt(section, self._threshold),
            temperature=0.7,
            top_p=0.95,
        )

        prompt = (
            f"Generate the '{section}' section from the following "
            f"{claim_count} validated claims. "
            f"Every claim must be reflected in your output.\n\n"
            f"{claims_text}"
        )

        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=prompt)])
        ]

        response = generate_with_retry(self._client, conversation, config)
        if response is None:
            return False, f"Generation failed for section '{section}'"

        return True, response.text or ""

    def generate(self) -> tuple[bool, str]:
        """Generate the soul document from validated claims, one section at a time.

        Returns:
            Tuple of (success, document_or_error)
        """
        sections: list[str] = []

        for section in SECTION_ORDER:
            success, result = self._generate_section(section)
            if not success:
                log.error(f"Soul document generation failed at section '{section}'")
                return False, result
            if result:
                sections.append(result)

        if not sections:
            return False, "No claims meet the support threshold"

        document = "# Uno - Soul Document\n\n" + "\n\n".join(sections)
        return True, document


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


def run_claim_generation(client: genai.Client, max_scenes: int | None) -> ClaimLedger:
    """Process scenes and generate claims, saving the final ledger.

    Skips if final_ledger.json already exists. Supports resuming from
    checkpoints for partial runs.

    Returns:
        The claim ledger (loaded from file or built from scenes).
    """
    final_ledger_path = OUTPUT_DIR / "final_ledger.json"

    if final_ledger_path.exists():
        log.info(
            f"Claim ledger already exists at {path_str(final_ledger_path)}, skipping"
        )
        with open(final_ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ClaimLedger.from_json(data)

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

    if max_scenes:
        unprocessed_scenes = unprocessed_scenes[:max_scenes]

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
    with open(final_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved final ledger to {path_str(final_ledger_path)}")

    return ledger


def run_claim_refinement(client: genai.Client, ledger: ClaimLedger) -> ClaimLedger:
    """Refine contradicted claims and save the refined ledger.

    Skips if refined_ledger.json already exists.

    Returns:
        The ledger (possibly refined in place, or loaded from file).
    """
    refined_ledger_path = OUTPUT_DIR / "refined_ledger.json"

    if refined_ledger_path.exists():
        log.info(
            f"Refined ledger already exists at {path_str(refined_ledger_path)}, skipping"
        )
        with open(refined_ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ClaimLedger.from_json(data)

    console.print("\n[bold cyan]Refining contradicted claims...[/bold cyan]")
    refiner = ClaimRefiner(client, ledger)
    refined_count, failed_count = refiner.refine_all()

    if refined_count > 0 or failed_count > 0:
        console.print(f"Refined: {refined_count}, Failed: {failed_count}")
    else:
        console.print("No contradicted claims found")

    # Always save the refined ledger so subsequent runs skip this step
    with open(refined_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved refined ledger to {path_str(refined_ledger_path)}")

    return ledger


def run_document_generation(
    client: genai.Client, ledger: ClaimLedger, threshold: int
) -> None:
    """Generate the soul document from validated claims.

    Skips if uno_soul_document.md already exists.
    """
    soul_doc_path = OUTPUT_DIR / "uno_soul_document.md"

    if soul_doc_path.exists():
        log.info(f"Soul document already exists at {path_str(soul_doc_path)}, skipping")
        return

    console.print("\n[bold cyan]Generating soul document...[/bold cyan]")

    generator = SoulDocumentGenerator(client, ledger, threshold)
    success, result = generator.generate()

    if success:
        with open(soul_doc_path, "w", encoding="utf-8") as f:
            f.write(result)

        tokens = count_tokens(result)
        words = len(result.split())

        console.print("\n[bold green]Soul document generated![/bold green]")
        console.print(f"Output: {path_str(soul_doc_path)}")
        console.print(f"Size: {tokens:,} tokens (~{words:,} words)")
        console.print(f"Threshold: support >= {threshold}")

        # Count claims by threshold
        claims_included = sum(
            1
            for claims in ledger.get_claims_by_section().values()
            for c in claims
            if c.support_count >= threshold
        )
        console.print(f"Claims included: {claims_included}/{ledger.claim_count()}")
    else:
        console.print(
            f"\n[bold red]Soul document generation failed:[/bold red] {result}"
        )


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

    client = genai.Client(http_options=HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000))

    ledger = run_claim_generation(client, args.max_scenes)
    ledger = run_claim_refinement(client, ledger)
    run_document_generation(client, ledger, args.threshold)

    # Print summary
    console.print(f"\n[bold]Output directory:[/bold] {path_str(OUTPUT_DIR)}")
    console.print(f"Checkpoints: {path_str(CHECKPOINTS_DIR)}/")


if __name__ == "__main__":
    main()
