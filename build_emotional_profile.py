#!/usr/bin/env python3

"""
Build a character profile using a claim ledger approach with enriched emotional data.

Compared to build_claim_ledger_profile.py (v11), this script:
- Reads from extract-emotional/v2 (tone, speech_act, visual_cues per dialogue/panel)
- Includes context dialogues from other characters in scenes
- Expands the claim taxonomy with growth/arc, humor, self-model, theory-of-mind,
  knowledge boundaries, negative claims, and situational adaptation paths
- Adds a claim synthesis pass (causal reasoning + negative claims) after refinement
- Generates experience vignettes as a soul document section
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Settings
MODEL_NAME = "gemini-3.1-pro-preview"
CHARACTER_NAME = "Uno"
ENCODING_NAME = "cl100k_base"
VERSION_TAG = "v12"
MAX_TOOL_ITERATIONS = 64
CLAIM_SUPPORT_THRESHOLD = 2

# Retry settings for API calls
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0
API_TIMEOUT_SECONDS = 300

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "extract-emotional" / "v2"
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
    "vignettes",
]

VALID_SECTIONS = set(SECTION_ORDER)

# Valid hierarchical paths
VALID_PATHS = {
    # Identity
    "identity/names",
    "identity/bio",
    "identity/origin",
    # Psychology - Neural Matrix
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
    "psychology/moral_compass/value_hierarchy",
    "psychology/moral_compass/dilemma_patterns",
    # Psychology - Emotional Profile
    "psychology/emotional/base_mood",
    "psychology/emotional/volatility",
    "psychology/emotional/resilience",
    "psychology/emotional/triggers/joy",
    "psychology/emotional/triggers/anger",
    "psychology/emotional/triggers/sadness",
    "psychology/emotional/triggers/fear",
    # Psychology - Growth / Arc (B1)
    "psychology/growth/emotional_arc",
    "psychology/growth/relationship_arc",
    # Psychology - Self-perception (B4)
    "psychology/self_model/identity_stance",
    "psychology/self_model/agency",
    "psychology/self_model/mortality",
    # Psychology - Negative
    "psychology/never",
    # Communication - Voice
    "communication/voice/formality",
    "communication/voice/verbosity",
    "communication/voice/vocabulary",
    "communication/voice/style",
    "communication/voice/register_shifts",
    # Communication - Syntax
    "communication/syntax/structure",
    "communication/syntax/contractions",
    # Communication - Idiolect
    "communication/idiolect/catchphrases",
    "communication/idiolect/nicknames",
    "communication/idiolect/expressions",
    "communication/idiolect/forbidden",
    # Communication - Interaction
    "communication/interaction/dominance",
    "communication/interaction/turn_taking",
    "communication/interaction/emotional_coloring",
    # Communication - Humor (B3)
    "communication/humor/type",
    "communication/humor/timing",
    "communication/humor/targets",
    # Communication - Negative
    "communication/never",
    # Motivations
    "motivations/core_drive",
    "motivations/goals/short_term",
    "motivations/goals/long_term",
    "motivations/fears/rational",
    "motivations/fears/irrational",
    # Capabilities
    "capabilities/skills",
    "capabilities/limitations",
    "capabilities/knowledge_boundaries/temporal",
    "capabilities/knowledge_boundaries/domain",
    "capabilities/knowledge_boundaries/forbidden",
    # Behavior
    "behavior/does",
    "behavior/avoids",
    "behavior/never",
    "behavior/evolution",
    "behavior/adaptation/by_audience",
    "behavior/adaptation/by_situation",
    # Relationships - dynamic sub-paths validated separately
    # relationships/{name}
    # relationships/{name}/dynamic
    # relationships/{name}/uno_believes
    # relationships/{name}/perceived_by_uno
    # relationships/{name}/behavioral_driver
}

# Valid relationship sub-path suffixes
RELATIONSHIP_SUB_PATHS = {
    "dynamic",
    "uno_believes",
    "perceived_by_uno",
    "behavioral_driver",
}


def is_valid_claim_path(path: str) -> bool:
    """Check whether a claim path is valid."""
    section = path.split("/")[0]
    if section not in VALID_SECTIONS:
        return False
    if section == "relationships":
        parts = path.split("/")
        # relationships/{name} or relationships/{name}/{sub_path}
        if len(parts) == 2:
            return True
        if len(parts) == 3 and parts[2] in RELATIONSHIP_SUB_PATHS:
            return True
        return False
    if section == "vignettes":
        return True
    return path in VALID_PATHS


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


def path_str(path: Path) -> str:
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
    text: str
    context: str
    scene_id: str


class SceneEvidence(BaseModel):
    scene_id: str
    justification: str


class Claim(BaseModel):
    id: int
    text: str
    path: str
    supporting: list[SceneEvidence] = []
    contradicting: list[SceneEvidence] = []
    quotes: list[Quote] = []

    @property
    def section(self) -> str:
        return self.path.split("/")[0]

    @property
    def support_count(self) -> int:
        return len(self.supporting) - len(self.contradicting)


# ============================================================================
# Scene Data Structures (enriched for emotional extraction)
# ============================================================================


@dataclass
class AnnotatedDialogue:
    """A dialogue line with emotional annotations from extract_emotional."""

    character: str
    line: str
    tone: str = "neutral"
    speech_act: str = "informing"


@dataclass
class Panel:
    """A single comic panel with its description, dialogues, and visual cues."""

    description: str
    dialogues: list[AnnotatedDialogue]
    visual_cues: list[str] = field(default_factory=list)


@dataclass
class Scene:
    """A scene from the comics containing Uno, with enriched emotional data."""

    issue: str
    page_numbers: list[int]
    panels: list[Panel]
    other_characters: set[str]

    @property
    def scene_id(self) -> str:
        return f"{self.issue}_{self.page_numbers[0]}"

    @property
    def summary(self) -> str:
        return " ".join(p.description for p in self.panels)

    def to_context_string(self) -> str:
        pages_str = ", ".join(f"page {p}" for p in self.page_numbers)
        chars_str = (
            ", ".join(sorted(self.other_characters))
            if self.other_characters
            else "none"
        )
        return (
            f"Issue: {self.issue}, {pages_str}. Other characters present: {chars_str}"
        )

    def to_dict(self) -> dict:
        return {
            "issue": self.issue,
            "page_numbers": self.page_numbers,
            "panels": [
                {
                    "description": p.description,
                    "dialogues": [
                        {
                            "character": d.character,
                            "line": d.line,
                            "tone": d.tone,
                            "speech_act": d.speech_act,
                        }
                        for d in p.dialogues
                    ],
                    "visual_cues": p.visual_cues,
                }
                for p in self.panels
            ],
            "other_characters": list(self.other_characters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        panels = [
            Panel(
                description=p["description"],
                dialogues=[AnnotatedDialogue(**d) for d in p["dialogues"]],
                visual_cues=p.get("visual_cues", []),
            )
            for p in data["panels"]
        ]
        return cls(
            issue=data["issue"],
            page_numbers=data["page_numbers"],
            panels=panels,
            other_characters=set(data["other_characters"]),
        )


# ============================================================================
# Scene Extraction from emotional output
# ============================================================================


@dataclass
class _PanelAccumulator:
    """Temporary accumulator for building scenes from panels."""

    panels: list[dict] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def extract_scenes_from_issue(issue_dir: Path) -> list[Scene]:
    page_files = sorted(issue_dir.glob("page_*.json"))
    scenes: list[Scene] = []
    current = _PanelAccumulator()

    for page_file in page_files:
        page_num = int(page_file.stem.split("_")[1])

        with open(page_file, encoding="utf-8") as f:
            page_data = json.load(f)

        panels = page_data.get("panels", [])
        if not panels:
            continue

        for panel in panels:
            if panel.get("is_new_scene", False) and current.panels:
                scene = _create_scene_from_panels(
                    issue_dir.name, current.pages, current.panels
                )
                if scene:
                    scenes.append(scene)
                current = _PanelAccumulator()

            current.panels.append(panel)
            if page_num not in current.pages:
                current.pages.append(page_num)

    if current.panels:
        scene = _create_scene_from_panels(issue_dir.name, current.pages, current.panels)
        if scene:
            scenes.append(scene)

    return scenes


def _create_scene_from_panels(
    issue: str, page_numbers: list[int], raw_panels: list[dict]
) -> Scene | None:
    """Create a Scene from panels, only if Uno is present."""
    panels: list[Panel] = []
    other_characters: set[str] = set()
    has_uno = False

    for raw in raw_panels:
        desc = raw.get("description", "")
        cues = [c for c in raw.get("visual_cues", []) if c]
        dialogues: list[AnnotatedDialogue] = []

        for dialogue in raw.get("dialogues", []):
            character = dialogue.get("character", "").strip()
            line = dialogue.get("line", "").strip()
            if not character or not line:
                continue

            ad = AnnotatedDialogue(
                character=character,
                line=line,
                tone=dialogue.get("tone", "neutral"),
                speech_act=dialogue.get("speech_act", "informing"),
            )
            dialogues.append(ad)

            if character.lower() == "uno":
                has_uno = True
            else:
                other_characters.add(character)

        panels.append(Panel(description=desc, dialogues=dialogues, visual_cues=cues))

    if not has_uno:
        return None

    return Scene(
        issue=issue,
        page_numbers=page_numbers,
        panels=panels,
        other_characters=other_characters,
    )


def natural_sort_key(path: Path) -> tuple:
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
    def __init__(self):
        self._claims: dict[int, Claim] = {}
        self._next_id: int = 1
        self._processed_scene_ids: set[str] = set()
        self._scene_cache: dict[str, Scene] = {}

    def to_json(self) -> dict:
        return {
            "next_id": self._next_id,
            "claims": {str(k): v.model_dump() for k, v in self._claims.items()},
            "processed_scene_ids": sorted(self._processed_scene_ids),
        }

    @classmethod
    def from_json(cls, data: dict) -> "ClaimLedger":
        ledger = cls()
        ledger._next_id = data.get("next_id", 1)
        ledger._claims = {
            int(k): Claim.model_validate(v) for k, v in data.get("claims", {}).items()
        }
        ledger._processed_scene_ids = set(data.get("processed_scene_ids", []))
        return ledger

    def add_scene(self, scene: Scene) -> None:
        self._processed_scene_ids.add(scene.scene_id)
        self._scene_cache[scene.scene_id] = scene

    def get_scene(self, scene_id: str) -> Scene | None:
        return self._scene_cache.get(scene_id)

    def is_scene_processed(self, scene_id: str) -> bool:
        return scene_id in self._processed_scene_ids

    def populate_scene_cache(self, scenes: list[Scene]) -> None:
        for scene in scenes:
            if scene.scene_id in self._processed_scene_ids:
                self._scene_cache[scene.scene_id] = scene

    def get_claims_by_section(
        self, section: str | None = None
    ) -> dict[str, list[Claim]]:
        result: dict[str, list[Claim]] = defaultdict(list)
        for claim in self._claims.values():
            if section is None or claim.section == section:
                result[claim.section].append(claim)
        for section_claims in result.values():
            section_claims.sort(key=lambda c: c.support_count, reverse=True)
        return result

    def get_claims_by_path(
        self, path_prefix: str | None = None
    ) -> dict[str, list[Claim]]:
        result: dict[str, list[Claim]] = defaultdict(list)
        for claim in self._claims.values():
            if path_prefix is None or claim.path.startswith(path_prefix):
                result[claim.path].append(claim)
        return result

    def get_claim(self, claim_id: int) -> Claim | None:
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
        if not is_valid_claim_path(path):
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
        claim = self._claims.get(claim_id)
        if not claim:
            return False, f"Claim {claim_id} not found"

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
        claim = self._claims.get(claim_id)
        if not claim:
            return False, f"Claim {claim_id} not found"

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
        claim = self._claims.get(claim_id)
        if not claim:
            return False, f"Claim {claim_id} not found"

        old_text = claim.text
        claim.text = new_text
        return (
            True,
            f"Refined claim {claim_id}: '{old_text[:50]}...' -> '{new_text[:50]}...'",
        )

    def remove_claim(self, claim_id: int) -> bool:
        if claim_id not in self._claims:
            return False
        del self._claims[claim_id]
        return True

    def claim_count(self) -> int:
        return len(self._claims)

    def scene_count(self) -> int:
        return len(self._processed_scene_ids)


# ============================================================================
# System Prompts
# ============================================================================


def get_scene_processing_prompt() -> str:
    return """You are building a claim-based character profile for "Uno" from PKNA comics.

## Input Enrichment

Each scene includes:
- **Tone annotations** per dialogue line (sarcastic, concerned, playful, etc.)
- **Speech act** per dialogue line (informing, joking, commanding, deflecting, etc.)
- **Visual cues** from panels (body language, holographic state, screen displays)
- **Context dialogues** from other characters in the same scene

Use ALL of these signals. Tone + speech act tell you HOW Uno speaks; visual cues tell
you what his non-verbal state reveals; context dialogues tell you what others say to/about
him and how his words land.

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
- psychology/neural_matrix/creativity
- psychology/neural_matrix/empathy
- psychology/neural_matrix/logic
- psychology/neural_matrix/adaptability
- psychology/neural_matrix/charisma
- psychology/neural_matrix/reliability

OCEAN Traits (Big Five):
- psychology/traits/ocean/openness
- psychology/traits/ocean/conscientiousness
- psychology/traits/ocean/extraversion
- psychology/traits/ocean/agreeableness
- psychology/traits/ocean/neuroticism

Other Personality:
- psychology/traits/mbti: MBTI type
- psychology/traits/temperament: Temperament description

Moral Compass:
- psychology/moral_compass/alignment: Moral alignment
- psychology/moral_compass/core_values: Core values
- psychology/moral_compass/conflict_resolution: How conflicts are resolved
- psychology/moral_compass/value_hierarchy: Which values win when they conflict
- psychology/moral_compass/dilemma_patterns: Recurring ethical tensions and resolutions

Emotional Profile:
- psychology/emotional/base_mood
- psychology/emotional/volatility
- psychology/emotional/resilience
- psychology/emotional/triggers/joy
- psychology/emotional/triggers/anger
- psychology/emotional/triggers/sadness
- psychology/emotional/triggers/fear

Growth / Character Arc:
- psychology/growth/emotional_arc: How emotional patterns shift over the series
- psychology/growth/relationship_arc: How key relationships deepen over time

Self-Perception:
- psychology/self_model/identity_stance: How Uno frames his own existence (alive? conscious?)
- psychology/self_model/agency: Does he see himself as having free will?
- psychology/self_model/mortality: How he relates to shutdown/death

Negative (what Uno would NEVER think/feel):
- psychology/never

### communication/ - How Uno speaks

Voice Style:
- communication/voice/formality
- communication/voice/verbosity
- communication/voice/vocabulary
- communication/voice/style
- communication/voice/register_shifts: When and how formality shifts by audience/situation

Syntax:
- communication/syntax/structure
- communication/syntax/contractions

Idiolect (individual speech):
- communication/idiolect/catchphrases: Characteristic phrases (Italian + translation)
- communication/idiolect/nicknames: Nicknames for Paperinik and others
- communication/idiolect/expressions: Common expressions
- communication/idiolect/forbidden: Words/phrases Uno avoids

Interaction Patterns:
- communication/interaction/dominance
- communication/interaction/turn_taking
- communication/interaction/emotional_coloring

Humor (central to Uno's character):
- communication/humor/type: Sarcasm, self-deprecation, wordplay, dark humor, absurdist
- communication/humor/timing: When humor is deployed (tension-breaking, deflection, affection-masking)
- communication/humor/targets: What/who Uno jokes about

Negative (what Uno would NEVER say):
- communication/never

### motivations/ - What drives Uno
- motivations/core_drive
- motivations/goals/short_term
- motivations/goals/long_term
- motivations/fears/rational
- motivations/fears/irrational

### capabilities/ - What Uno can and cannot do
- capabilities/skills
- capabilities/limitations
- capabilities/knowledge_boundaries/temporal: What era is his knowledge from?
- capabilities/knowledge_boundaries/domain: What domains does he know well vs. poorly?
- capabilities/knowledge_boundaries/forbidden: Knowledge that would break character

### behavior/ - Actions and habits
- behavior/does: Positive behaviors, protocols, habits
- behavior/avoids: Self-imposed constraints
- behavior/never: Things Uno would NEVER do (strong negative claims)
- behavior/evolution: Behaviors that appear or disappear over the series
- behavior/adaptation/by_audience: How behavior changes by interlocutor
- behavior/adaptation/by_situation: How behavior changes in crisis vs. calm

### relationships/ - Character dynamics and theory of mind
For each character, use these sub-paths:
- relationships/{character}/dynamic: Observable relationship dynamic
- relationships/{character}/uno_believes: What Uno thinks/feels about this character
- relationships/{character}/perceived_by_uno: What Uno thinks this character thinks about HIM
- relationships/{character}/behavioral_driver: How these beliefs drive Uno's behavior toward them

You may also use the flat relationships/{character} path for general observations.

## Theory of Mind Guidelines

Pay special attention to Uno's INTERNAL SOCIAL MODEL:
- What does Uno believe about each character? (trust, respect, frustration, admiration)
- What does Uno think others think about HIM? Look for:
  - Uno's reactions to how others treat him (reveals his model of their perception)
  - Explicit statements by Uno about what others think of him
  - Behavioral shifts that reveal Uno's assumptions about others' expectations
- How do these beliefs DRIVE his behavior? (the "because" behind actions)
- Use context dialogues: what others say to Uno reveals what he infers about their view of him

## Quality Guidelines
- Claims must be SPECIFIC and VERIFIABLE from scenes
- Use the exact paths above (or relationships/{name}/* pattern)
- Prefer supporting existing claims over creating near-duplicates
- Write claims in ENGLISH
- Preserve Italian quotes exactly
- Include quote context explaining why the quote matters
- Use tone/speech_act annotations as evidence (e.g., "sarcastic tone confirms deflection pattern")
- Note the issue number when observing character growth or arc changes
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
[Claims from psychology/moral_compass/* - alignment, core values, conflict resolution, value hierarchy, dilemma patterns]
When values conflict, describe how Uno resolves the tension.

### Emotional Profile
[Claims from psychology/emotional/* - base mood, volatility, resilience]

#### Emotional Triggers
- Joy: [psychology/emotional/triggers/joy]
- Anger: [psychology/emotional/triggers/anger]
- Sadness: [psychology/emotional/triggers/sadness]
- Fear: [psychology/emotional/triggers/fear]

### Self-Perception
[Claims from psychology/self_model/* - identity stance, agency, mortality]
How does Uno see himself? Does he consider himself alive? How does he relate to shutdown/death?

### Character Growth
[Claims from psychology/growth/* - emotional arc, relationship arc]
How Uno evolves over the series, with issue references where available.

### What Uno Would Never Think or Feel
[Claims from psychology/never]

Write flowing prose capturing psychological makeup with concrete examples.""",
    "communication": """## Communication Style

### Voice and Tone
[Claims from communication/voice/* - formality, verbosity, vocabulary, style, register shifts]
Describe how formality shifts by audience and situation.

### Syntax Patterns
[Claims from communication/syntax/*]

### Linguistic Markers (Idiolect)
[Claims from communication/idiolect/*]
- **Catchphrases**: "Italian phrase" (translation)
- **Nicknames for Paperinik**: list them
- **Common expressions**: with translations

### Humor
[Claims from communication/humor/* - type, timing, targets]
This is central to Uno's character. Describe the types of humor he uses,
when he deploys them, and who/what he targets.

### Interaction Patterns
[Claims from communication/interaction/* - dominance, turn-taking, emotional coloring]

### What Uno Would Never Say
[Claims from communication/never]

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

### Knowledge Boundaries
[Claims from capabilities/knowledge_boundaries/*]
- **Temporal**: What era is his knowledge from?
- **Domain**: What does he know well vs. poorly?
- **Forbidden**: What knowledge would break character?

Include specific examples from scenes.""",
    "behavior": """## Behavioral Patterns

### What Uno Does
[Claims from behavior/does - positive behaviors, habits, protocols]

### What Uno Avoids
[Claims from behavior/avoids - self-imposed constraints]

### What Uno Would NEVER Do
[Claims from behavior/never - strong negative claims, protective boundaries]

### Behavioral Evolution
[Claims from behavior/evolution - behaviors that appear or disappear over the series]

### Situational Adaptation
[Claims from behavior/adaptation/* - by audience, by situation]
How does Uno's behavior change depending on who he's talking to and what's happening?

Split into subsections with concrete examples.""",
    "relationships": """## Key Relationships

For each character with significant claims, create a subsection structured as:

### With {Character Name}
**Observable Dynamic:** [relationships/{character}/dynamic]
**What Uno Believes About Them:** [relationships/{character}/uno_believes]
**What Uno Thinks They Think of Him:** [relationships/{character}/perceived_by_uno]
**How This Drives His Behavior:** [relationships/{character}/behavioral_driver]

Synthesize these into a narrative: "Uno sees X as..., believes X views him as...,
and this drives him to..."

### Priority Characters (create subsections for each):
1. Paperinik / PK
2. Everett Ducklair
3. Due

### Other Characters
Group remaining relationships/* claims by character.

Include Italian quotes showing the relationship dynamic.""",
    "vignettes": """## Canonical Interactions

Generate 3-5 short interaction vignettes that demonstrate Uno's character in action.
Each vignette should be a brief reconstructed dialogue (4-8 lines) that captures a
DIFFERENT facet of Uno's personality.

Vignettes MUST cover at least:
1. Uno with Paperinik in a casual/banter context
2. Uno with Paperinik in a crisis/serious context
3. Uno with a different character (Everett, Due, or another)

### Format for each vignette:

#### {Title} ({tone descriptor})
**Context:** One sentence describing the situation.

```
Character: "Dialogue line"
Character: "Dialogue line"
...
```

### Rules:
- Use Italian for all dialogue lines (these are behavioral templates)
- Ground each vignette in the highest-evidence claims
- Show the behavioral contrast between situations (humor vs. seriousness, formal vs. casual)
- Each vignette should demonstrate a different claim path in action
- Output ONLY the vignettes section, starting with the ## heading""",
}


def get_section_prompt(section: str, threshold: int) -> str:
    template = SECTION_PROMPTS[section]
    return f"""Generate one section of a soul document from validated claims.

You MUST incorporate ALL claims provided. Every single claim represents evidence
about this character and must be reflected in your output.

Claims come in two kinds:
- **Validated claims** (support >= {threshold}): These are well-established facts.
  Present them as definitive statements.
- **Tentative claims** (marked "tentative"): These have some evidence but are not
  fully confirmed. Present them with hedging language such as "there is some evidence
  that...", "possibly...", "in at least one instance...", or "it appears that...".
  Do NOT skip tentative claims — include them, but clearly distinguish them from
  validated ones.

## Output Format

{template}

## Formatting Rules
- Order claims by support_count (highest first) within subsections
- Italian quotes format: "Quote" (English translation)
- Merge very similar claims into coherent prose rather than repeating
- Be comprehensive — every claim must appear in the output
- Present tentative claims with appropriate hedging, grouped after validated claims
- Capture the character's distinctive voice and personality
- Output ONLY the section content (starting with the ## heading), no preamble
"""


def get_claim_refinement_prompt() -> str:
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


def get_claim_synthesis_reasoning_prompt() -> str:
    return """You enrich character claims with causal reasoning.

## Task
Given a claim about a character's behavior or communication pattern, along with all
its supporting evidence, add a "because..." explanation that captures WHY the character
does this. The reasoning must be grounded in the evidence.

## Rules
- Output ONLY the enriched claim text. No preamble, no explanation, no wrapping.
- Keep the original claim intact and append the causal reasoning.
- The "because" must be supported by the evidence, not speculative.
- Keep claims concise: 1-4 sentences in English.
- If the evidence doesn't support a clear "why", return the claim unchanged.

## Examples

Input claim: "Uno deflects with humor when Paperinik asks about his feelings."
Evidence: "Uses sarcastic quip when PK asks if he's worried", "Changes subject with joke when PK probes his emotional state", "In private moment, admits to Due that showing weakness would undermine PK's confidence"
Output: Uno deflects with humor when Paperinik asks about his feelings, because he fears that showing vulnerability would undermine Paperinik's confidence in him as a reliable partner.

Input claim: "Uno speaks formally to Everett Ducklair."
Evidence: "Uses 'Master Ducklair' address", "Adopts deferential tone", "Straightens holographic posture when Everett enters"
Output: Uno speaks formally to Everett Ducklair, because he regards Everett as his creator and authority figure, and the formality reflects genuine reverence rather than mere protocol.
"""


def get_claim_condensation_prompt() -> str:
    return """You condense multiple low-evidence character claims into fewer, broader claims.

## Task
Given a group of related claims about a character, merge them into fewer claims that
capture the combined insight. Each output claim should be a broader statement that
subsumes the input claims it merges.

## Rules
- Output a JSON array of objects, each with "path" and "text" fields.
- The "path" should be the most appropriate path from the input claims.
- Merge claims that overlap or describe facets of the same trait/behavior.
- Preserve the most specific and distinctive insights — don't over-generalize.
- Keep each claim concise: 1-3 sentences in English.
- If two claims genuinely describe different things, keep them separate.
- Output ONLY the JSON array, no preamble.
- Produce FEWER claims than the input (that's the whole point).

## Example

Input claims:
- [psychology/traits/ocean/openness] "Uno shows curiosity about human emotions." [+1]
- [psychology/traits/ocean/openness] "Uno experiments with new approaches to problems." [+1]
- [psychology/traits/ocean/conscientiousness] "Uno follows systematic procedures." [+1]

Output:
[
  {"path": "psychology/traits/ocean/openness", "text": "Uno demonstrates high openness through curiosity about human emotions and willingness to experiment with new problem-solving approaches."},
  {"path": "psychology/traits/ocean/conscientiousness", "text": "Uno follows systematic procedures."}
]
"""


def get_claim_synthesis_negatives_prompt() -> str:
    return """You identify negative claims (things a character would NEVER do) from existing evidence.

## Task
Given all existing claims about a character in a specific domain, identify what the
character would NEVER do, say, or think in that domain. These are patterns that are
conspicuously absent from the evidence — things that would be out of character.

## Rules
- Output a JSON array of objects, each with "path" and "text" fields.
- Each negative claim must be the INVERSE of well-supported positive evidence.
- Be specific: "Uno never insults Paperinik's intelligence" not "Uno is never mean".
- Only include claims that are strongly supported by the absence pattern.
- Use the appropriate */never path for the domain being analyzed.
- Output 3-8 negative claims per domain. Fewer if the evidence is thin.
- Output ONLY the JSON array, no preamble.

## Example

Domain: behavior
Existing claims: "Uno monitors all tower systems continuously", "Uno provides tactical support during missions", "Uno uses humor to defuse tension"
Output:
[
  {"path": "behavior/never", "text": "Uno never abandons his post or stops monitoring tower systems, even during personal crises."},
  {"path": "behavior/never", "text": "Uno never withholds critical tactical information from Paperinik during missions, regardless of personal disagreements."},
  {"path": "behavior/never", "text": "Uno never uses humor at the expense of someone who is genuinely suffering or in mortal danger."}
]
"""


# ============================================================================
# Tool Formatting Functions
# ============================================================================


def format_claims_compact(ledger: ClaimLedger, section: str | None = None) -> str:
    claims_by_section = ledger.get_claims_by_section(section)

    lines = [f"Total claims: {ledger.claim_count()}"]
    lines.append("")

    for sect, claims in claims_by_section.items():
        lines.append(f"## {sect.title()}")
        for claim in claims:
            sign = "+" if claim.support_count >= 0 else ""
            lines.append(
                f"{claim.id}: [{claim.path}] "
                f"{claim.text[:200]}{'...' if len(claim.text) > 200 else ''} "
                f"[{sign}{claim.support_count}]"
            )
        lines.append("")

    return "\n".join(lines)


def format_claims_detail(ledger: ClaimLedger, claim_ids: list[int]) -> str:
    lines: list[str] = []

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
    lines = [
        f"Scene: {scene.scene_id}",
        f"Issue: {scene.issue}, pages {'-'.join(map(str, scene.page_numbers))}",
        f"Characters present: {', '.join(sorted(scene.other_characters)) if scene.other_characters else 'Uno only'}",
    ]

    for i, panel in enumerate(scene.panels, 1):
        lines.append("")
        lines.append(f"--- Panel {i} ---")
        lines.append(f"[{panel.description}]")
        if panel.visual_cues:
            lines.append(f"Visual: {'; '.join(panel.visual_cues)}")
        for d in panel.dialogues:
            lines.append(f'{d.character}: "{d.line}" [{d.tone}, {d.speech_act}]')

    return "\n".join(lines)


# ============================================================================
# LedgerTools
# ============================================================================


class LedgerTools:
    def __init__(self, ledger: ClaimLedger):
        self._ledger = ledger
        self._current_scene_id: str = ""

    @property
    def all(self) -> list:
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

        if not is_valid_claim_path(path):
            return (
                f"Error: Invalid path '{path}'. Use one of the defined paths "
                f"or relationships/{{character_name}}/{{sub_path}}."
            )

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

    def _generate_with_retry(
        self, conversation: list[Content]
    ) -> GenerateContentResponse | None:
        return generate_with_retry(self._client, conversation, self._config)

    def process_scene(self, scene: Scene, scene_number: int) -> tuple[bool, str]:
        self._tools._current_scene_id = scene.scene_id

        panels_text = format_scene_view(scene)

        scene_prompt = f"""Analyze this scene and update the claim ledger.

{panels_text}

Use list_claims() to see existing claims, then add supporting/contradicting evidence or new claims as appropriate. Pay attention to tone annotations and context dialogues for relationship and theory-of-mind evidence. When done, provide a brief one-line summary of your updates."""

        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=scene_prompt)])
        ]

        try:
            response = self._generate_with_retry(conversation)
            if response is None:
                return False, "API call failed after retries"

            summary = response.text or "No summary provided"
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
        result = []
        for claims in self._ledger.get_claims_by_section().values():
            for claim in claims:
                if claim.contradicting:
                    result.append(claim)
        return result

    def refine_claim(self, claim: Claim) -> tuple[bool, str]:
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
# Claim Synthesizer
# ============================================================================


# Sections whose claims are eligible for causal reasoning enrichment
_REASONING_SECTIONS = {"behavior", "communication", "relationships"}

# Sections that get negative claim synthesis
_NEGATIVE_SECTIONS = {"behavior", "communication", "psychology", "capabilities"}


class ClaimSynthesizer:
    """Enriches claims with causal reasoning and adds negative/boundary claims."""

    def __init__(self, client: genai.Client, ledger: ClaimLedger, threshold: int):
        self._client = client
        self._ledger = ledger
        self._threshold = threshold
        self._reasoning_config = GenerateContentConfig(
            system_instruction=get_claim_synthesis_reasoning_prompt(),
            temperature=0.3,
            top_p=0.95,
        )
        self._negatives_config = GenerateContentConfig(
            system_instruction=get_claim_synthesis_negatives_prompt(),
            temperature=0.5,
            top_p=0.95,
        )

    def _eligible_claims_for_reasoning(self) -> list[Claim]:
        result = []
        for section, claims in self._ledger.get_claims_by_section().items():
            if section not in _REASONING_SECTIONS:
                continue
            for claim in claims:
                if claim.support_count >= self._threshold:
                    result.append(claim)
        return result

    def _enrich_claim_reasoning(self, claim: Claim) -> tuple[bool, str]:
        claim_detail = format_claims_detail(self._ledger, [claim.id])
        prompt = (
            f"Enrich this claim with causal reasoning (add 'because...'):\n\n"
            f"{claim_detail}"
        )
        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=prompt)])
        ]

        response = generate_with_retry(
            self._client, conversation, self._reasoning_config
        )
        if response is None:
            return False, "API call failed after retries"

        enriched = (response.text or "").strip()
        if not enriched:
            return False, "Empty response from LLM"

        return True, enriched

    def synthesize_reasoning(self) -> tuple[int, int]:
        eligible = self._eligible_claims_for_reasoning()
        if not eligible:
            log.info("No claims eligible for reasoning enrichment")
            return 0, 0

        log.info(f"Enriching {len(eligible)} claims with causal reasoning...")
        enriched_count = 0
        failed_count = 0

        with PROGRESS as progress:
            task = progress.add_task("Enriching claims...", total=len(eligible))

            for i, claim in enumerate(eligible, 1):
                success, result = self._enrich_claim_reasoning(claim)

                if success and result != claim.text:
                    self._ledger.refine_claim(claim.id, result)
                    enriched_count += 1
                    log.debug(f"Enriched claim {claim.id}: {result[:80]}...")
                elif not success:
                    failed_count += 1
                    log.warning(f"Failed to enrich claim {claim.id}: {result}")

                progress.update(task, completed=i)

        return enriched_count, failed_count

    def _synthesize_negatives_for_section(self, section: str) -> list[dict[str, str]]:
        claims_by_section = self._ledger.get_claims_by_section(section=section)
        claims = claims_by_section.get(section, [])
        if not claims:
            return []

        claims_summary = "\n".join(
            f"- [{c.path}] {c.text} [+{c.support_count}]"
            for c in claims
            if c.support_count >= self._threshold
        )
        if not claims_summary:
            return []

        prompt = (
            f"Domain: {section}\n"
            f"Existing claims:\n{claims_summary}\n\n"
            f"Identify what this character would NEVER do/say/think in this domain."
        )
        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=prompt)])
        ]

        response = generate_with_retry(
            self._client, conversation, self._negatives_config
        )
        if response is None:
            log.warning(f"Failed to synthesize negatives for {section}")
            return []

        text = (response.text or "").strip()
        if not text:
            return []

        try:
            negatives = json.loads(text)
            if not isinstance(negatives, list):
                return []
            return negatives
        except json.JSONDecodeError:
            log.warning(f"Failed to parse negatives JSON for {section}: {text[:100]}")
            return []

    def synthesize_negatives(self) -> int:
        total_added = 0

        for section in _NEGATIVE_SECTIONS:
            negatives = self._synthesize_negatives_for_section(section)
            for neg in negatives:
                path = neg.get("path", "")
                text = neg.get("text", "")
                if not path or not text:
                    continue
                if not is_valid_claim_path(path):
                    log.warning(f"Invalid negative claim path: {path}")
                    continue

                try:
                    self._ledger.add_claim(
                        path=path,
                        text=text,
                        scene_id="synthesis",
                        justification="Inferred from absence patterns across all scenes",
                    )
                    total_added += 1
                except ValueError as e:
                    log.warning(f"Failed to add negative claim: {e}")

        return total_added

    def synthesize_all(self) -> tuple[int, int, int]:
        """Run both synthesis passes.

        Returns:
            Tuple of (enriched_count, enrichment_failures, negatives_added).
        """
        enriched, failed = self.synthesize_reasoning()
        negatives = self.synthesize_negatives()
        return enriched, failed, negatives


# ============================================================================
# Claim Condenser
# ============================================================================


def _path_prefix(path: str, segments: int) -> str:
    parts = path.split("/")
    return "/".join(parts[:segments])


def _group_claims_by_prefix(
    claims: list[Claim], segments: int
) -> dict[str, list[Claim]]:
    groups: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        prefix = _path_prefix(claim.path, segments)
        groups[prefix].append(claim)
    return groups


def _merge_evidence(claims: list[Claim]) -> list[SceneEvidence]:
    """Collect all supporting evidence from multiple claims, deduplicating by scene_id."""
    seen: set[str] = set()
    merged: list[SceneEvidence] = []
    for claim in claims:
        for ev in claim.supporting:
            if ev.scene_id not in seen:
                seen.add(ev.scene_id)
                merged.append(ev)
    return merged


def _merge_quotes(claims: list[Claim]) -> list[Quote]:
    seen: set[tuple[str, str]] = set()
    merged: list[Quote] = []
    for claim in claims:
        for q in claim.quotes:
            key = (q.text, q.scene_id)
            if key not in seen:
                seen.add(key)
                merged.append(q)
    return merged


class ClaimCondenser:
    """Condenses low-support claims into fewer, broader claims via LLM merging."""

    def __init__(self, client: genai.Client, ledger: ClaimLedger, threshold: int):
        self._client = client
        self._ledger = ledger
        self._threshold = threshold
        self._config = GenerateContentConfig(
            system_instruction=get_claim_condensation_prompt(),
            temperature=0.3,
            top_p=0.95,
        )

    def _low_support_claims(self) -> list[Claim]:
        result = []
        for claims in self._ledger.get_claims_by_section().values():
            for claim in claims:
                if 0 < claim.support_count < self._threshold:
                    result.append(claim)
        return result

    def _condense_group(self, claims: list[Claim]) -> list[dict[str, str]]:
        claims_text = "\n".join(
            f'- [{c.path}] "{c.text}" [+{c.support_count}]' for c in claims
        )
        prompt = f"Condense these related claims:\n{claims_text}"
        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=prompt)])
        ]

        response = generate_with_retry(self._client, conversation, self._config)
        if response is None:
            log.warning("Failed to condense claim group")
            return []

        text = (response.text or "").strip()
        if not text:
            return []

        try:
            condensed = json.loads(text)
            if not isinstance(condensed, list):
                return []
            return condensed
        except json.JSONDecodeError:
            log.warning(f"Failed to parse condensation JSON: {text[:100]}")
            return []

    def _apply_condensation(self, claims: list[Claim], segments: int) -> list[Claim]:
        """Run one condensation pass at the given grouping depth.

        Returns claims that were produced or left unchanged by this pass.
        """
        groups = _group_claims_by_prefix(claims, segments)
        result: list[Claim] = []

        for prefix, group in groups.items():
            if len(group) < 2:
                result.extend(group)
                continue

            condensed_dicts = self._condense_group(group)
            if not condensed_dicts:
                result.extend(group)
                continue

            merged_evidence = _merge_evidence(group)
            merged_quotes = _merge_quotes(group)

            for orig in group:
                self._ledger.remove_claim(orig.id)

            for cd in condensed_dicts:
                path = cd.get("path", "")
                text = cd.get("text", "")
                if not path or not text:
                    continue
                if not is_valid_claim_path(path):
                    log.warning(f"Invalid condensed claim path: {path}")
                    continue

                claim = Claim(
                    id=self._ledger._next_id,
                    text=text,
                    path=path,
                    supporting=list(merged_evidence),
                    contradicting=[],
                    quotes=list(merged_quotes),
                )
                self._ledger._claims[claim.id] = claim
                self._ledger._next_id += 1
                result.append(claim)

            log.debug(
                f"Condensed {len(group)} claims under '{prefix}' "
                f"into {len(condensed_dicts)} claims"
            )

        return result

    def condense_all(self) -> tuple[int, int]:
        """Run hierarchical condensation: 2-segment then 1-segment.

        Returns:
            Tuple of (original_count, final_count) for low-support claims.
        """
        low_support = self._low_support_claims()
        if not low_support:
            log.info("No low-support claims to condense")
            return 0, 0

        original_count = len(low_support)
        log.info(
            f"Condensing {original_count} low-support claims "
            f"(threshold={self._threshold})..."
        )

        # Pass 1: group by 2 path segments
        after_pass1 = self._apply_condensation(low_support, segments=2)

        # Pass 2: re-group still-below-threshold claims by 1 segment
        still_low = [c for c in after_pass1 if c.support_count < self._threshold]
        if still_low:
            log.info(
                f"Pass 2: {len(still_low)} claims still below threshold, "
                f"re-grouping by 1 segment..."
            )
            self._apply_condensation(still_low, segments=1)

        final_count = len(self._low_support_claims())
        log.info(
            f"Condensation complete: {original_count} -> {final_count} "
            f"low-support claims"
        )
        return original_count, final_count


# ============================================================================
# Soul Document Generator
# ============================================================================


class SoulDocumentGenerator:
    def __init__(self, client: genai.Client, ledger: ClaimLedger, threshold: int):
        self._client = client
        self._ledger = ledger
        self._threshold = threshold

    def _format_section_claims(self, section: str) -> str:
        claims_by_section = self._ledger.get_claims_by_section(section=section)
        claims = [c for c in claims_by_section.get(section, []) if c.support_count > 0]
        if not claims:
            return ""

        by_path: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            by_path[claim.path].append(claim)

        lines: list[str] = []
        for path in sorted(by_path.keys()):
            lines.append(f"### Path: {path}")
            for claim in by_path[path]:
                if claim.support_count >= self._threshold:
                    lines.append(
                        f"**Claim (support: +{claim.support_count}):** {claim.text}"
                    )
                else:
                    lines.append(
                        f"**Claim (tentative, support: +{claim.support_count}):** {claim.text}"
                    )
                if claim.quotes:
                    for q in claim.quotes:
                        lines.append(f'  - "{q.text}" — {q.context}')
            lines.append("")

        return "\n".join(lines)

    def _format_vignette_claims(self) -> str:
        """Format top claims across all sections for vignette generation."""
        all_claims: list[Claim] = []
        for claims in self._ledger.get_claims_by_section().values():
            all_claims.extend(c for c in claims if c.support_count >= self._threshold)

        all_claims.sort(key=lambda c: c.support_count, reverse=True)
        top_claims = all_claims[:50]

        lines: list[str] = []
        for claim in top_claims:
            lines.append(f"- [{claim.path}] {claim.text} [+{claim.support_count}]")
            if claim.quotes:
                for q in claim.quotes[:2]:
                    lines.append(f'  Quote: "{q.text}" — {q.context}')
        return "\n".join(lines)

    def _generate_section(self, section: str) -> tuple[bool, str]:
        if section == "vignettes":
            claims_text = self._format_vignette_claims()
        else:
            claims_text = self._format_section_claims(section)

        if not claims_text.strip():
            log.info(f"Section '{section}': no claims, skipping")
            return True, ""

        if section == "vignettes":
            claim_count = claims_text.count("- [")
        else:
            claim_count = claims_text.count("**Claim (")
        tentative_count = claims_text.count("**Claim (tentative,")

        log.info(
            f"Section '{section}': generating from {claim_count} claims "
            f"({tentative_count} tentative)"
        )

        config = GenerateContentConfig(
            system_instruction=get_section_prompt(section, self._threshold),
            temperature=0.7,
            top_p=0.95,
        )

        if section == "vignettes":
            prompt = (
                f"Generate canonical interaction vignettes based on the following "
                f"{claim_count} top claims about the character. Ground each vignette "
                f"in specific claims and quotes.\n\n{claims_text}"
            )
        else:
            prompt = (
                f"Generate the '{section}' section from the following "
                f"{claim_count} claims ({tentative_count} tentative). "
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
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num:03d}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)

    log.debug(f"Saved checkpoint {checkpoint_num}: {path_str(checkpoint_path)}")

    if checkpoint_num > 3:
        old_checkpoint = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num - 3:03d}.json"
        if old_checkpoint.exists():
            old_checkpoint.unlink()


def find_latest_checkpoint() -> tuple[int, ClaimLedger] | None:
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

    with open(latest, encoding="utf-8") as f:
        data = json.load(f)

    ledger = ClaimLedger.from_json(data)
    return checkpoint_num, ledger


# ============================================================================
# Main Entry Point
# ============================================================================


def run_claim_generation(client: genai.Client, max_scenes: int | None) -> ClaimLedger:
    final_ledger_path = OUTPUT_DIR / "final_ledger.json"

    if final_ledger_path.exists():
        log.info(
            f"Claim ledger already exists at {path_str(final_ledger_path)}, skipping"
        )
        with open(final_ledger_path, encoding="utf-8") as f:
            data = json.load(f)
        return ClaimLedger.from_json(data)

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

    log.info("Scanning for scenes containing Uno...")
    all_scenes: list[Scene] = []
    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )

    for issue_dir in issue_dirs:
        scenes = extract_scenes_from_issue(issue_dir)
        all_scenes.extend(scenes)

    log.info(f"Total: {len(all_scenes)} scenes with Uno across all issues")

    ledger.populate_scene_cache(all_scenes)

    unprocessed_scenes = [
        s for s in all_scenes if not ledger.is_scene_processed(s.scene_id)
    ]

    if max_scenes:
        unprocessed_scenes = unprocessed_scenes[:max_scenes]

    if not unprocessed_scenes:
        log.info("All scenes already processed!")
    else:
        log.info(f"Processing {len(unprocessed_scenes)} unprocessed scenes")

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

                    checkpoint_num += 1
                    save_checkpoint(ledger, checkpoint_num)

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

    with open(final_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved final ledger to {path_str(final_ledger_path)}")

    return ledger


def run_claim_refinement(client: genai.Client, ledger: ClaimLedger) -> ClaimLedger:
    refined_ledger_path = OUTPUT_DIR / "refined_ledger.json"

    if refined_ledger_path.exists():
        log.info(
            f"Refined ledger already exists at {path_str(refined_ledger_path)}, skipping"
        )
        with open(refined_ledger_path, encoding="utf-8") as f:
            data = json.load(f)
        return ClaimLedger.from_json(data)

    console.print("\n[bold cyan]Refining contradicted claims...[/bold cyan]")
    refiner = ClaimRefiner(client, ledger)
    refined_count, failed_count = refiner.refine_all()

    if refined_count > 0 or failed_count > 0:
        console.print(f"Refined: {refined_count}, Failed: {failed_count}")
    else:
        console.print("No contradicted claims found")

    with open(refined_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved refined ledger to {path_str(refined_ledger_path)}")

    return ledger


def run_claim_synthesis(
    client: genai.Client, ledger: ClaimLedger, threshold: int
) -> ClaimLedger:
    synthesized_ledger_path = OUTPUT_DIR / "synthesized_ledger.json"

    if synthesized_ledger_path.exists():
        log.info(
            f"Synthesized ledger already exists at {path_str(synthesized_ledger_path)}, skipping"
        )
        with open(synthesized_ledger_path, encoding="utf-8") as f:
            data = json.load(f)
        return ClaimLedger.from_json(data)

    console.print(
        "\n[bold cyan]Synthesizing claims (reasoning + negatives)...[/bold cyan]"
    )
    synthesizer = ClaimSynthesizer(client, ledger, threshold)
    enriched, failed, negatives = synthesizer.synthesize_all()

    console.print(
        f"Reasoning: {enriched} enriched, {failed} failed. "
        f"Negatives: {negatives} added."
    )

    with open(synthesized_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved synthesized ledger to {path_str(synthesized_ledger_path)}")

    return ledger


def run_claim_condensation(
    client: genai.Client, ledger: ClaimLedger, threshold: int
) -> ClaimLedger:
    condensed_ledger_path = OUTPUT_DIR / "condensed_ledger.json"

    if condensed_ledger_path.exists():
        log.info(
            f"Condensed ledger already exists at {path_str(condensed_ledger_path)}, skipping"
        )
        with open(condensed_ledger_path, encoding="utf-8") as f:
            data = json.load(f)
        return ClaimLedger.from_json(data)

    console.print("\n[bold cyan]Condensing low-support claims...[/bold cyan]")
    condenser = ClaimCondenser(client, ledger, threshold)
    original, final = condenser.condense_all()

    if original > 0:
        console.print(f"Condensed: {original} low-support claims -> {final}")
    else:
        console.print("No low-support claims to condense")

    with open(condensed_ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.info(f"Saved condensed ledger to {path_str(condensed_ledger_path)}")

    return ledger


def run_document_generation(
    client: genai.Client, ledger: ClaimLedger, threshold: int
) -> None:
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

        validated = sum(
            1
            for claims in ledger.get_claims_by_section().values()
            for c in claims
            if c.support_count >= threshold
        )
        tentative = sum(
            1
            for claims in ledger.get_claims_by_section().values()
            for c in claims
            if 0 < c.support_count < threshold
        )
        console.print(
            f"Claims: {validated} validated + {tentative} tentative "
            f"/ {ledger.claim_count()} total"
        )
    else:
        console.print(
            f"\n[bold red]Soul document generation failed:[/bold red] {result}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build character profile using emotional claim ledger approach"
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
        f"\n[bold cyan]Emotional Claim Ledger Profile Builder ({VERSION_TAG})[/bold cyan]\n"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(http_options=HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000))

    ledger = run_claim_generation(client, args.max_scenes)
    ledger = run_claim_refinement(client, ledger)
    ledger = run_claim_synthesis(client, ledger, args.threshold)
    ledger = run_claim_condensation(client, ledger, args.threshold)
    run_document_generation(client, ledger, args.threshold)

    console.print(f"\n[bold]Output directory:[/bold] {path_str(OUTPUT_DIR)}")
    console.print(f"Checkpoints: {path_str(CHECKPOINTS_DIR)}/")


if __name__ == "__main__":
    main()
