#!/usr/bin/env python3

"""
Build a character profile in AIEOS format (https://aieos.org/).

This script extracts a structured character profile from comic dialogues using
DSPy to iteratively explore scenes and build evidence-based character traits
in AIEOS v1.1 JSON format.

Pipeline stages:
1. Evidence Gathering: Extract raw evidence (claims + quotes) per AIEOS section
2. Section Extraction: Process evidence into structured AIEOS sections
3. Assembly & Validation: Merge sections, add metadata, validate schema
"""

import argparse
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dspy
import tiktoken
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

load_dotenv()

# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)

# Settings
MODEL_NAME = "vertex_ai/gemini-3-pro-preview"
CHARACTER_NAME = "Uno"
ENCODING_NAME = "cl100k_base"
VERSION_TAG = "v10"

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / VERSION_TAG
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
SECTIONS_DIR = CHECKPOINTS_DIR / "sections"

# Global progress bar
PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)

# AIEOS section names
AIEOS_SECTIONS = [
    "identity",
    "psychology",
    "linguistics",
    "history",
    "motivations",
    "capabilities",
]


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


def configure_lm() -> None:
    """Configure DSPy language model."""
    lm = dspy.LM(
        model=MODEL_NAME,
        vertex_credentials=os.getenv("VERTEX_AI_CREDS"),
        vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_tokens=65535,
    )
    dspy.configure(lm=lm, track_usage=True)


# ============================================================================
# Scene Data Structure (reused from build_claim_ledger_profile.py)
# ============================================================================


@dataclass
class Scene:
    """A scene from the comics containing Uno.

    Preserves the original panel structure from the extraction JSON,
    including all fields: description, caption_text, dialogues, is_new_scene.
    """

    issue: str
    page_numbers: list[int]
    panels: list[dict]  # Original panel dicts with all fields preserved

    @property
    def scene_id(self) -> str:
        """Unique identifier for this scene: issue_firstpage."""
        return f"{self.issue}_{self.page_numbers[0]}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "issue": self.issue,
            "page_numbers": self.page_numbers,
            "panels": self.panels,
        }

    def get_uno_dialogues(self) -> list[str]:
        """Extract Uno's dialogue lines from panels."""
        return [
            d["line"]
            for panel in self.panels
            for d in panel.get("dialogues", [])
            if d.get("character", "").lower() == "uno"
        ]

    def get_other_characters(self) -> set[str]:
        """Get names of all non-Uno characters in this scene."""
        chars: set[str] = set()
        for panel in self.panels:
            for d in panel.get("dialogues", []):
                char = d.get("character", "")
                if char and char.lower() != "uno":
                    chars.add(char)
        return chars

    def has_uno(self) -> bool:
        """Check if Uno appears in this scene."""
        return len(self.get_uno_dialogues()) > 0


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
    """Create a Scene if Uno appears, preserving original panel structure."""
    # Check if Uno appears in any dialogue
    has_uno = any(
        d.get("character", "").lower() == "uno"
        for panel in panels
        for d in panel.get("dialogues", [])
    )
    if not has_uno:
        return None

    return Scene(
        issue=issue,
        page_numbers=page_numbers,
        panels=panels,  # Preserve original structure
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
# Scene Store
# ============================================================================


class SceneStore:
    """On-demand scene storage for access during profile building."""

    def __init__(self, scenes: list[Scene]):
        self._scenes = {scene.scene_id: scene for scene in scenes}
        self._by_issue: dict[str, list[Scene]] = {}
        for scene in scenes:
            if scene.issue not in self._by_issue:
                self._by_issue[scene.issue] = []
            self._by_issue[scene.issue].append(scene)

    def get_scene(self, scene_id: str) -> Scene | None:
        """Get a scene by its ID."""
        return self._scenes.get(scene_id)

    def get_scenes_by_issue(self, issue: str) -> list[Scene]:
        """Get all scenes from a specific issue."""
        return self._by_issue.get(issue, [])

    def search_dialogues(self, query: str) -> list[tuple[str, str]]:
        """Search all dialogues (not just Uno's) in panels.

        Returns list of (scene_id, matching_dialogue) tuples.
        """
        results = []
        query_lower = query.lower()
        for scene in self._scenes.values():
            for panel in scene.panels:
                for d in panel.get("dialogues", []):
                    line = d.get("line", "")
                    if query_lower in line.lower():
                        results.append((scene.scene_id, line))
        return results[:20]  # Limit results

    def get_index(self) -> list[dict]:
        """Lightweight index computed on-the-fly from panels."""
        return [
            {
                "scene_id": scene.scene_id,
                "issue": scene.issue,
                "pages": scene.page_numbers,
                "panel_count": len(scene.panels),
                "dialogue_count": sum(
                    len(p.get("dialogues", [])) for p in scene.panels
                ),
                "other_characters": list(scene.get_other_characters()),
            }
            for scene in self._scenes.values()
        ]

    def all_scenes(self) -> list[Scene]:
        """Get all scenes in order."""
        return list(self._scenes.values())

    def scene_count(self) -> int:
        """Total number of scenes."""
        return len(self._scenes)


# ============================================================================
# AIEOS Typed Models
# ============================================================================


# --- Identity Section ---


class AIEOSNames(BaseModel):
    """Character naming information."""

    first: str = Field(description="Primary name")
    middle: str | None = Field(default=None, description="Middle name if any")
    last: str | None = Field(default=None, description="Last/family name if any")
    nickname: str | None = Field(default=None, description="Common nickname")


class AIEOSBio(BaseModel):
    """Biographical information."""

    description: str = Field(description="Description of nature and role")
    entity_type: str = Field(
        default="artificial_intelligence",
        description="Type of entity: human, artificial_intelligence, etc.",
    )
    age_biological: int | None = Field(default=None, description="Biological age")
    age_perceived: int | None = Field(default=None, description="Perceived age")
    gender: str | None = Field(default=None, description="Gender identity")


class AIEOSOrigin(BaseModel):
    """Origin information."""

    creator: str | None = Field(default=None, description="Creator of the entity")
    creation_context: str | None = Field(
        default=None, description="Context of creation"
    )
    nationality: str | None = Field(default=None, description="Nationality")
    birthplace_city: str | None = Field(default=None, description="City of origin")
    birthplace_country: str | None = Field(
        default=None, description="Country of origin"
    )


class AIEOSIdentity(BaseModel):
    """AIEOS Identity section."""

    names: AIEOSNames = Field(description="Name information")
    bio: AIEOSBio = Field(description="Biographical information")
    origin: AIEOSOrigin = Field(default_factory=AIEOSOrigin, description="Origin info")


# --- Psychology Section ---


class NeuralMatrix(BaseModel):
    """Cognitive weights for AI personality (0.0-1.0 scale)."""

    creativity: float = Field(ge=0.0, le=1.0, description="Creative thinking ability")
    empathy: float = Field(ge=0.0, le=1.0, description="Emotional understanding")
    logic: float = Field(ge=0.0, le=1.0, description="Logical reasoning ability")
    adaptability: float = Field(ge=0.0, le=1.0, description="Flexibility in situations")
    charisma: float = Field(ge=0.0, le=1.0, description="Social influence ability")
    reliability: float = Field(
        ge=0.0, le=1.0, description="Dependability and consistency"
    )


class OceanTraits(BaseModel):
    """Big Five personality traits (0.0-1.0 scale)."""

    openness: float = Field(ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(
        ge=0.0, le=1.0, description="Organization and dependability"
    )
    extraversion: float = Field(
        ge=0.0, le=1.0, description="Social energy and assertiveness"
    )
    agreeableness: float = Field(ge=0.0, le=1.0, description="Cooperation and trust")
    neuroticism: float = Field(ge=0.0, le=1.0, description="Emotional instability")


class PersonalityTraits(BaseModel):
    """Personality classification traits."""

    ocean: OceanTraits = Field(description="Big Five personality traits")
    mbti: str | None = Field(default=None, description="MBTI type (e.g., INTJ)")
    temperament: str | None = Field(default=None, description="Temperament type")


class MoralCompass(BaseModel):
    """Moral and ethical framework."""

    alignment: str = Field(description="Alignment (e.g., lawful good, chaotic neutral)")
    core_values: list[str] = Field(default_factory=list, description="Core values")
    conflict_resolution_style: str | None = Field(
        default=None, description="How conflicts are resolved"
    )


class EmotionalTriggers(BaseModel):
    """Emotional triggers by emotion type."""

    joy: list[str] = Field(default_factory=list, description="Things that bring joy")
    anger: list[str] = Field(
        default_factory=list, description="Things that cause anger"
    )
    sadness: list[str] = Field(
        default_factory=list, description="Things that cause sadness"
    )
    fear: list[str] = Field(default_factory=list, description="Things that cause fear")


class EmotionalProfile(BaseModel):
    """Emotional characteristics."""

    base_mood: str = Field(description="Default emotional state")
    volatility: float = Field(
        ge=0.0, le=1.0, description="Emotional stability (0=stable)"
    )
    resilience: float = Field(ge=0.0, le=1.0, description="Recovery from setbacks")
    triggers: EmotionalTriggers = Field(
        default_factory=EmotionalTriggers, description="Emotional triggers"
    )


class AIEOSPsychology(BaseModel):
    """AIEOS Psychology section."""

    neural_matrix: NeuralMatrix = Field(description="Cognitive trait weights")
    traits: PersonalityTraits = Field(description="Personality classifications")
    moral_compass: MoralCompass = Field(description="Moral framework")
    emotional_profile: EmotionalProfile = Field(description="Emotional characteristics")


# --- Linguistics Section ---


class VoiceStyle(BaseModel):
    """Voice and speaking style characteristics."""

    formality_level: float = Field(
        ge=0.0, le=1.0, description="Formality (0=casual, 1=formal)"
    )
    verbosity_level: float = Field(
        ge=0.0, le=1.0, description="Verbosity (0=terse, 1=verbose)"
    )
    vocabulary_level: str = Field(
        default="advanced",
        description="Vocabulary: basic|intermediate|advanced|technical",
    )
    style_descriptors: list[str] = Field(
        default_factory=list, description="Style descriptors (e.g., sarcastic, witty)"
    )


class SyntaxStyle(BaseModel):
    """Syntactic patterns in speech."""

    sentence_structure: str = Field(
        default="complex", description="Structure: simple|compound|complex"
    )
    use_contractions: bool = Field(default=True, description="Uses contractions")
    active_passive_ratio: float = Field(
        ge=0.0, le=1.0, description="Active voice ratio (1=all active)"
    )


class Idiolect(BaseModel):
    """Individual speech characteristics."""

    catchphrases: list[str] = Field(
        default_factory=list,
        description="Characteristic phrases (Italian with translations)",
    )
    forbidden_words: list[str] = Field(
        default_factory=list, description="Words/phrases avoided"
    )
    hesitation_markers: bool = Field(
        default=False, description="Uses hesitation markers (um, uh)"
    )


class InteractionStyle(BaseModel):
    """Conversational interaction patterns."""

    turn_taking: str = Field(description="Turn-taking pattern description")
    dominance_score: float = Field(
        ge=0.0, le=1.0, description="Conversational dominance"
    )
    emotional_coloring: str = Field(description="Emotional tone in interactions")


class AIEOSLinguistics(BaseModel):
    """AIEOS Linguistics section."""

    voice: VoiceStyle = Field(description="Voice and style")
    syntax: SyntaxStyle = Field(description="Syntactic patterns")
    idiolect: Idiolect = Field(description="Individual speech patterns")
    interaction: InteractionStyle = Field(description="Interaction patterns")


# --- History Section ---


class KeyLifeEvent(BaseModel):
    """A significant event in the character's history."""

    year: str | None = Field(default=None, description="Year or time period")
    event: str = Field(description="Description of the event")
    impact: str = Field(description="Significance and lasting effects")


class AIEOSHistory(BaseModel):
    """AIEOS History section."""

    origin_story: str = Field(description="Narrative of origin")
    key_life_events: list[KeyLifeEvent] = Field(
        default_factory=list, description="Significant events"
    )
    relationships: dict[str, str] = Field(
        default_factory=dict,
        description="Key relationships: character name -> description",
    )


# --- Motivations Section ---


class Goals(BaseModel):
    """Character goals."""

    short_term: list[str] = Field(default_factory=list, description="Immediate goals")
    long_term: list[str] = Field(
        default_factory=list, description="Long-term aspirations"
    )


class Fears(BaseModel):
    """Character fears."""

    rational: list[str] = Field(default_factory=list, description="Logical fears")
    irrational: list[str] = Field(
        default_factory=list, description="Emotional/irrational fears"
    )


class AIEOSMotivations(BaseModel):
    """AIEOS Motivations section."""

    core_drive: str = Field(description="Primary motivation")
    goals: Goals = Field(default_factory=Goals, description="Goals")
    fears: Fears = Field(default_factory=Fears, description="Fears")


# --- Capabilities Section ---


class Skill(BaseModel):
    """A skill or ability."""

    name: str = Field(description="Skill name")
    description: str = Field(description="What the skill does")
    proficiency: float = Field(ge=0.0, le=1.0, description="Proficiency level")


class AIEOSCapabilities(BaseModel):
    """AIEOS Capabilities section."""

    skills: list[Skill] = Field(
        default_factory=list, description="Skills and abilities"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Explicit limitations"
    )


# --- Full AIEOS Document ---


class AIEOSStandard(BaseModel):
    """AIEOS standard information."""

    protocol: str = Field(default="AIEOS", description="Protocol name")
    version: str = Field(default="1.1.0", description="Protocol version")


class AIEOSMetadata(BaseModel):
    """AIEOS document metadata."""

    instance_id: str = Field(description="Unique instance UUID")
    version: str = Field(default="1.0", description="Document version")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str = Field(description="ISO8601 update timestamp")
    source: str | None = Field(default=None, description="Data source")
    character: str | None = Field(default=None, description="Character name")


class AIEOSDocument(BaseModel):
    """Complete AIEOS v1.1 document."""

    context: dict[str, str] = Field(
        default_factory=lambda: {"aieos": "https://aieos.org/schema/v1.1/"},
        alias="@context",
        description="JSON-LD context",
    )
    type: str = Field(
        default="aieos:AIEntityObject", alias="@type", description="Document type"
    )
    standard: AIEOSStandard = Field(
        default_factory=AIEOSStandard, description="Standard info"
    )
    metadata: AIEOSMetadata = Field(description="Document metadata")
    identity: AIEOSIdentity = Field(description="Identity section")
    psychology: AIEOSPsychology = Field(description="Psychology section")
    linguistics: AIEOSLinguistics = Field(description="Linguistics section")
    history: AIEOSHistory = Field(description="History section")
    motivations: AIEOSMotivations = Field(description="Motivations section")
    capabilities: AIEOSCapabilities = Field(description="Capabilities section")

    model_config = {"populate_by_name": True}


# ============================================================================
# Evidence Models (for RLM output)
# ============================================================================


class SectionEvidence(BaseModel):
    """Evidence supporting a section's content."""

    scene_id: str = Field(description="Scene ID (e.g., 'pkna-1_23')")
    quote: str | None = Field(default=None, description="Italian quote if applicable")
    description: str = Field(description="What this evidence shows")
    trait: str = Field(description="Which trait/field this supports")


# Concrete output types per section (DSPy needs concrete types, not Generic)
class IdentityWithEvidence(BaseModel):
    """Identity section output with evidence citations."""

    section: AIEOSIdentity
    evidence: list[SectionEvidence] = Field(description="Supporting evidence")


class PsychologyWithEvidence(BaseModel):
    """Psychology section output with evidence citations."""

    section: AIEOSPsychology
    evidence: list[SectionEvidence] = Field(description="Supporting evidence")


class LinguisticsWithEvidence(BaseModel):
    """Linguistics section output with evidence citations."""

    section: AIEOSLinguistics
    evidence: list[SectionEvidence] = Field(description="Supporting evidence")


class HistoryWithEvidence(BaseModel):
    """History section output with evidence citations."""

    section: AIEOSHistory
    evidence: list[SectionEvidence] = Field(description="Supporting evidence")


class MotivationsWithEvidence(BaseModel):
    """Motivations section output with evidence citations."""

    section: AIEOSMotivations
    evidence: list[SectionEvidence] = Field(description="Supporting evidence")


class CapabilitiesWithEvidence(BaseModel):
    """Capabilities section output with evidence citations."""

    section: AIEOSCapabilities
    evidence: list[SectionEvidence] = Field(description="Supporting evidence")


# ============================================================================
# RLM Tool Functions (global scene store access)
# ============================================================================

# Global scene store for RLM tool access
SCENE_STORE: SceneStore | None = None


def list_all_scenes() -> list[dict]:
    """Get lightweight index of all scenes with the character.

    Returns:
        List of dicts with: scene_id, issue, pages, dialogue_count, other_characters
    """
    if SCENE_STORE is None:
        return []
    return SCENE_STORE.get_index()


def get_scene(scene_id: str) -> dict | None:
    """Get full details of a specific scene.

    Args:
        scene_id: Scene identifier (e.g., 'pkna-1_23')

    Returns:
        Dict with: issue, pages, summary, uno_dialogues, panel_descriptions, other_characters
        Or None if scene not found.
    """
    if SCENE_STORE is None:
        return None
    scene = SCENE_STORE.get_scene(scene_id)
    if scene is None:
        return None
    return scene.to_dict()


def search_dialogues(query: str, max_results: int = 20) -> list[dict]:
    """Search character's dialogues for a keyword/phrase.

    Args:
        query: Text to search for (case-insensitive)
        max_results: Maximum number of results to return (default 20)

    Returns:
        List of dicts with: scene_id, dialogue (matching line)
    """
    if SCENE_STORE is None:
        return []
    results = SCENE_STORE.search_dialogues(query)[:max_results]
    return [
        {"scene_id": scene_id, "dialogue": dialogue} for scene_id, dialogue in results
    ]


def get_scenes_by_issue(issue: str) -> list[dict]:
    """Get all scenes from a specific issue.

    Args:
        issue: Issue identifier (e.g., 'pkna-1')

    Returns:
        List of scene dicts with full details.
    """
    if SCENE_STORE is None:
        return []
    scenes = SCENE_STORE.get_scenes_by_issue(issue)
    return [s.to_dict() for s in scenes]


def get_scenes_with_character(character: str) -> list[dict]:
    """Get scenes where a specific character appears with the main character.

    Args:
        character: Name of the other character to find

    Returns:
        List of scene dicts where the character appears.
    """
    if SCENE_STORE is None:
        return []
    results = []
    for scene in SCENE_STORE.all_scenes():
        if any(character.lower() in c.lower() for c in scene.get_other_characters()):
            results.append(scene.to_dict())
    return results


# RLM tool list for all section builders
RLM_TOOLS = [
    list_all_scenes,
    get_scene,
    search_dialogues,
    get_scenes_by_issue,
    get_scenes_with_character,
]


# ============================================================================
# Section-Specific RLM Instructions
# ============================================================================

IDENTITY_RLM_INSTRUCTIONS = """
Extract basic identity facts about the character.

EXPLORATION STRATEGY:
1. Use list_all_scenes() to get an overview of available scenes
2. Use search_dialogues() to find mentions of names, origin, creator
3. Use get_scene() to get full context for relevant scenes

WHAT TO FIND:
- Full name and any aliases or nicknames
- Nature (AI, human, other) and entity type
- Creator or origin
- Physical/digital manifestation details

OUTPUT:
- Fill the AIEOSIdentity section with found facts
- Include evidence citations for each claim
"""

PSYCHOLOGY_RLM_INSTRUCTIONS = """
Explore scenes to understand the character's psychological profile.

EXPLORATION STRATEGY:
1. Use list_all_scenes() to get an overview
2. Use search_dialogues() to find emotional expressions, decision statements
3. Use get_scene() to analyze specific interactions in context
4. Look at different issues to see personality consistency

WHAT TO ANALYZE:
- Emotional reactions and triggers (joy, anger, sadness, fear)
- Decision-making patterns (logical vs emotional)
- Big Five personality traits (OCEAN): score each 0.0-1.0 based on evidence
- Neural matrix traits: creativity, empathy, logic, adaptability, charisma, reliability
- Moral alignment and core values
- MBTI type indicators

SCORING RULES:
- All numeric scores must be between 0.0 and 1.0
- Base scores on evidence frequency and strength
- Include quotes that exemplify personality traits

OUTPUT:
- Complete AIEOSPsychology section with all subscores
- Evidence citations linking each trait to specific scenes/quotes
"""

LINGUISTICS_RLM_INSTRUCTIONS = """
Analyze the character's speech patterns across all dialogues.

EXPLORATION STRATEGY:
1. Use list_all_scenes() to see dialogue counts per scene
2. Use search_dialogues() to find recurring expressions and catchphrases
3. Use get_scene() to see dialogue in context
4. Sample dialogues from different issues for consistency

WHAT TO ANALYZE:
- Formality level (formal vs casual)
- Vocabulary complexity (basic, intermediate, advanced, technical)
- Sentence structure patterns
- Catchphrases and recurring expressions (PRESERVE ITALIAN with translations)
- Conversational dominance patterns
- Emotional coloring in speech

IMPORTANT:
- Preserve Italian phrases exactly as found
- Include translations in parentheses: "Frase italiana (English translation)"
- All numeric scores (formality_level, verbosity_level, etc.) must be 0.0-1.0

OUTPUT:
- Complete AIEOSLinguistics section
- Evidence citations with specific Italian quotes
"""

HISTORY_RLM_INSTRUCTIONS = """
Explore scenes chronologically to build the origin story and key events.

EXPLORATION STRATEGY:
1. Use list_all_scenes() to see all available issues
2. Use get_scenes_by_issue() to explore early issues for origin story
3. Use get_scenes_with_character() to map relationships
4. Use search_dialogues() to find references to past events

WHAT TO BUILD:
- Origin story narrative (how the character came to be)
- Key life events with their impact
- Relationships with other characters (use get_scenes_with_character)

OUTPUT:
- Complete AIEOSHistory section
- Evidence citations linking events to specific scenes
"""

MOTIVATIONS_RLM_INSTRUCTIONS = """
Search for scenes revealing the character's goals and fears.

EXPLORATION STRATEGY:
1. Use search_dialogues() with keywords: "voglio", "devo", "obiettivo", "paura", "temo"
2. Use get_scene() to understand context of goal/fear statements
3. Look for scenes of conflict or important decisions

WHAT TO FIND:
- Core drive (primary motivation)
- Short-term goals (immediate objectives)
- Long-term goals (aspirations)
- Rational fears (logical concerns)
- Irrational fears (emotional/deeper fears)

OUTPUT:
- Complete AIEOSMotivations section
- Evidence citations with quotes expressing goals/fears
"""

CAPABILITIES_RLM_INSTRUCTIONS = """
Find scenes demonstrating the character's skills and limitations.

EXPLORATION STRATEGY:
1. Use search_dialogues() to find skill demonstrations and limitations
2. Use get_scene() to see full context of capability usage
3. Look for scenes where character succeeds or fails at tasks

WHAT TO FIND:
- Skills and abilities (with proficiency 0.0-1.0)
- Technical capabilities (for AI characters)
- Explicit limitations or constraints
- Things the character cannot or refuses to do

OUTPUT:
- Complete AIEOSCapabilities section with skills list
- Evidence citations showing each skill/limitation
"""

SECTION_RLM_INSTRUCTIONS = {
    "identity": IDENTITY_RLM_INSTRUCTIONS,
    "psychology": PSYCHOLOGY_RLM_INSTRUCTIONS,
    "linguistics": LINGUISTICS_RLM_INSTRUCTIONS,
    "history": HISTORY_RLM_INSTRUCTIONS,
    "motivations": MOTIVATIONS_RLM_INSTRUCTIONS,
    "capabilities": CAPABILITIES_RLM_INSTRUCTIONS,
}


# ============================================================================
# RLM Signature Factory
# ============================================================================


# Mapping of section names to their WithEvidence output types
SECTION_OUTPUT_TYPES: dict[str, type[BaseModel]] = {
    "identity": IdentityWithEvidence,
    "psychology": PsychologyWithEvidence,
    "linguistics": LinguisticsWithEvidence,
    "history": HistoryWithEvidence,
    "motivations": MotivationsWithEvidence,
    "capabilities": CapabilitiesWithEvidence,
}


def build_section_rlm(section_name: str) -> dspy.RLM:
    """Create an RLM for a specific AIEOS section.

    Args:
        section_name: Name of the section (identity, psychology, etc.)

    Returns:
        Configured RLM instance for the section.
    """
    output_type = SECTION_OUTPUT_TYPES[section_name]
    instructions = SECTION_RLM_INSTRUCTIONS[section_name]

    # Create signature class dynamically
    class SectionSignature(dspy.Signature):
        __doc__ = instructions

        character_name: str = dspy.InputField(
            description="Name of the character to analyze"
        )
        result: output_type = dspy.OutputField(  # type: ignore[valid-type]
            description=f"AIEOS {section_name} section with evidence citations"
        )

    return dspy.RLM(
        signature=SectionSignature,
        tools=RLM_TOOLS,
        max_iterations=30,
        max_llm_calls=100,
        verbose=True,
    )


# ============================================================================
# Profile Builder (RLM-based)
# ============================================================================


class AIEOSProfileBuilder:
    """Builds AIEOS profile from scenes using DSPy RLM."""

    def __init__(self, character_name: str = CHARACTER_NAME):
        self._character = character_name

    def build_section(
        self, section_name: str
    ) -> tuple[BaseModel, list[SectionEvidence]]:
        """Build a section using RLM exploration.

        Args:
            section_name: Name of the section to build.

        Returns:
            Tuple of (section_data, evidence_list).
        """
        log.info(f"Building {section_name} section with RLM...")

        rlm = build_section_rlm(section_name)
        result = rlm(character_name=self._character)

        # Extract section and evidence from result
        with_evidence = result.result
        return with_evidence.section, with_evidence.evidence

    def build_identity_section(self) -> tuple[AIEOSIdentity, list[SectionEvidence]]:
        """Build identity section."""
        section, evidence = self.build_section("identity")
        return section, evidence  # type: ignore[return-value]

    def build_psychology_section(self) -> tuple[AIEOSPsychology, list[SectionEvidence]]:
        """Build psychology section."""
        section, evidence = self.build_section("psychology")
        return section, evidence  # type: ignore[return-value]

    def build_linguistics_section(
        self,
    ) -> tuple[AIEOSLinguistics, list[SectionEvidence]]:
        """Build linguistics section."""
        section, evidence = self.build_section("linguistics")
        return section, evidence  # type: ignore[return-value]

    def build_history_section(self) -> tuple[AIEOSHistory, list[SectionEvidence]]:
        """Build history section."""
        section, evidence = self.build_section("history")
        return section, evidence  # type: ignore[return-value]

    def build_motivations_section(
        self,
    ) -> tuple[AIEOSMotivations, list[SectionEvidence]]:
        """Build motivations section."""
        section, evidence = self.build_section("motivations")
        return section, evidence  # type: ignore[return-value]

    def build_capabilities_section(
        self,
    ) -> tuple[AIEOSCapabilities, list[SectionEvidence]]:
        """Build capabilities section."""
        section, evidence = self.build_section("capabilities")
        return section, evidence  # type: ignore[return-value]


# ============================================================================
# AIEOS Assembly
# ============================================================================


@dataclass
class AIEOSSections:
    """Container for all AIEOS sections."""

    identity: AIEOSIdentity | None = None
    psychology: AIEOSPsychology | None = None
    linguistics: AIEOSLinguistics | None = None
    history: AIEOSHistory | None = None
    motivations: AIEOSMotivations | None = None
    capabilities: AIEOSCapabilities | None = None


def assemble_aieos_document(
    sections: AIEOSSections,
    character_name: str,
) -> AIEOSDocument:
    """Assemble sections into complete AIEOS v1.1 document."""
    now = datetime.now(timezone.utc).isoformat()

    # Create default sections if None
    identity = sections.identity or AIEOSIdentity(
        names=AIEOSNames(first=character_name),
        bio=AIEOSBio(description="Unknown"),
    )
    psychology = sections.psychology or AIEOSPsychology(
        neural_matrix=NeuralMatrix(
            creativity=0.5,
            empathy=0.5,
            logic=0.5,
            adaptability=0.5,
            charisma=0.5,
            reliability=0.5,
        ),
        traits=PersonalityTraits(
            ocean=OceanTraits(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            )
        ),
        moral_compass=MoralCompass(alignment="neutral", core_values=[]),
        emotional_profile=EmotionalProfile(
            base_mood="neutral", volatility=0.5, resilience=0.5
        ),
    )
    linguistics = sections.linguistics or AIEOSLinguistics(
        voice=VoiceStyle(formality_level=0.5, verbosity_level=0.5),
        syntax=SyntaxStyle(active_passive_ratio=0.5),
        idiolect=Idiolect(),
        interaction=InteractionStyle(
            turn_taking="balanced", dominance_score=0.5, emotional_coloring="neutral"
        ),
    )
    history = sections.history or AIEOSHistory(origin_story="Unknown")
    motivations = sections.motivations or AIEOSMotivations(core_drive="Unknown")
    capabilities = sections.capabilities or AIEOSCapabilities()

    return AIEOSDocument(
        metadata=AIEOSMetadata(
            instance_id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            source="PKNA Comics (Paperinik New Adventures)",
            character=character_name,
        ),
        identity=identity,
        psychology=psychology,
        linguistics=linguistics,
        history=history,
        motivations=motivations,
        capabilities=capabilities,
    )


def validate_aieos_document(doc: AIEOSDocument) -> list[str]:
    """Validate AIEOS document and return list of issues."""
    issues = []

    # Check psychology numeric fields are in range
    nm = doc.psychology.neural_matrix
    for field in [
        "creativity",
        "empathy",
        "logic",
        "adaptability",
        "charisma",
        "reliability",
    ]:
        value = getattr(nm, field)
        if not (0 <= value <= 1):
            issues.append(f"psychology.neural_matrix.{field} out of range: {value}")

    ocean = doc.psychology.traits.ocean
    for field in [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]:
        value = getattr(ocean, field)
        if not (0 <= value <= 1):
            issues.append(f"psychology.traits.ocean.{field} out of range: {value}")

    return issues


# ============================================================================
# Checkpoint Management
# ============================================================================


def save_section(
    section_data: BaseModel,
    evidence: list[SectionEvidence],
    section_name: str,
    output_dir: Path,
) -> None:
    """Save a section and its evidence to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save section
    section_path = output_dir / f"{section_name}.json"
    with open(section_path, "w", encoding="utf-8") as f:
        json.dump(section_data.model_dump(), f, ensure_ascii=False, indent=2)
    log.debug(f"Saved section {section_name} to {path_str(section_path)}")

    # Save evidence
    evidence_path = output_dir / f"{section_name}_evidence.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in evidence], f, ensure_ascii=False, indent=2)
    log.debug(f"Saved evidence for {section_name} to {path_str(evidence_path)}")


# Mapping of section names to their model classes
SECTION_MODEL_MAP: dict[str, type[BaseModel]] = {
    "identity": AIEOSIdentity,
    "psychology": AIEOSPsychology,
    "linguistics": AIEOSLinguistics,
    "history": AIEOSHistory,
    "motivations": AIEOSMotivations,
    "capabilities": AIEOSCapabilities,
}


def load_section(section_name: str, sections_dir: Path) -> BaseModel | None:
    """Load a section from JSON file if it exists."""
    path = sections_dir / f"{section_name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model_class = SECTION_MODEL_MAP.get(section_name)
    if model_class:
        return model_class.model_validate(data)
    return None


# ============================================================================
# Pipeline Stages
# ============================================================================


def run_section_extraction(builder: AIEOSProfileBuilder) -> AIEOSSections:
    """Build each AIEOS section using RLM exploration.

    Each section's RLM autonomously explores the scene store via tools,
    extracts relevant evidence, and produces the typed section output.
    """
    sections = AIEOSSections()
    total_evidence_count = 0

    for section_name in AIEOS_SECTIONS:
        # Check for cached section
        cached = load_section(section_name, SECTIONS_DIR)
        if cached:
            log.info(f"Loaded cached section: {section_name}")
            setattr(sections, section_name, cached)
            continue

        log.info(f"Building section: {section_name}")

        try:
            # Build section using RLM (returns section + evidence)
            section_data, evidence = builder.build_section(section_name)

            setattr(sections, section_name, section_data)
            save_section(section_data, evidence, section_name, SECTIONS_DIR)

            total_evidence_count += len(evidence)
            log.info(f"  Completed {section_name} with {len(evidence)} evidence items")

        except Exception as e:
            log.error(f"Error building section {section_name}: {e}")
            import traceback

            traceback.print_exc()

    log.info(
        f"Section extraction complete: {total_evidence_count} total evidence items"
    )
    return sections


def run_assembly(sections: AIEOSSections, character_name: str) -> AIEOSDocument:
    """Stage 3: Assemble and validate final AIEOS document."""
    log.info("Assembling AIEOS document...")

    aieos_doc = assemble_aieos_document(sections, character_name)

    # Validate
    issues = validate_aieos_document(aieos_doc)
    if issues:
        log.warning(f"Validation issues found: {len(issues)}")
        for issue in issues:
            log.warning(f"  - {issue}")
    else:
        log.info("AIEOS document validated successfully")

    return aieos_doc


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main function to build the AIEOS character profile."""
    global SCENE_STORE

    parser = argparse.ArgumentParser(
        description="Build character profile in AIEOS format"
    )
    parser.add_argument(
        "--character",
        type=str,
        default=CHARACTER_NAME,
        help=f"Character name to extract (default: {CHARACTER_NAME})",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (for testing)",
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]AIEOS Profile Builder (RLM)[/bold cyan]\n")
    console.print(f"Character: {args.character}")

    # Check for existing output
    output_path = OUTPUT_DIR / f"{args.character.lower()}_aieos.json"
    if output_path.exists():
        log.info(f"Output already exists: {path_str(output_path)}")
        console.print(
            f"[yellow]Profile already exists at {path_str(output_path)}[/yellow]"
        )
        console.print("[yellow]Delete it to regenerate.[/yellow]")
        return

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Configure DSPy
    configure_lm()

    # Load scenes
    log.info("Loading scenes...")
    all_scenes: list[Scene] = []
    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )

    for issue_dir in issue_dirs:
        scenes = extract_scenes_from_issue(issue_dir)
        all_scenes.extend(scenes)

    # Limit scenes if requested (for testing)
    if args.max_scenes:
        all_scenes = all_scenes[: args.max_scenes]
        log.info(f"Limited to {len(all_scenes)} scenes (--max-scenes)")

    log.info(f"Loaded {len(all_scenes)} scenes with {args.character}")

    # Initialize global scene store for RLM tool access
    SCENE_STORE = SceneStore(all_scenes)

    # Create builder
    builder = AIEOSProfileBuilder(args.character)

    # Section Extraction (RLM explores scenes via tools)
    console.print("\n[bold]Stage 1: Section Extraction (RLM)[/bold]")
    sections = run_section_extraction(builder)

    # Assembly
    console.print("\n[bold]Stage 2: Assembly & Validation[/bold]")
    aieos_doc = run_assembly(sections, args.character)

    # Serialize document using aliases for @context and @type
    doc_dict = aieos_doc.model_dump(by_alias=True)

    # Save final document
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    # Summary
    console.print("\n[bold green]AIEOS Profile Complete![/bold green]\n")
    console.print(f"Output: {path_str(output_path)}")
    console.print(f"Scenes available: {SCENE_STORE.scene_count()}")

    # Token count
    doc_str = json.dumps(doc_dict, ensure_ascii=False)
    tokens = count_tokens(doc_str)
    console.print(f"Document size: {tokens:,} tokens")

    # Write processing log
    log_path = OUTPUT_DIR / "processing_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "character": args.character,
                "scenes_available": SCENE_STORE.scene_count(),
                "output_tokens": tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
