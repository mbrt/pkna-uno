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
MODEL_NAME = "vertex_ai/gemini-3-flash-preview"
CHARACTER_NAME = "Uno"
ENCODING_NAME = "cl100k_base"
VERSION_TAG = "v9"

# Paths
BASE_DIR = Path(__file__).parent.parent
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
        """Search dialogues for a query string.

        Returns list of (scene_id, matching_dialogue) tuples.
        """
        results = []
        query_lower = query.lower()
        for scene in self._scenes.values():
            for dialogue in scene.uno_dialogues:
                if query_lower in dialogue.lower():
                    results.append((scene.scene_id, dialogue))
        return results[:20]  # Limit results

    def get_index(self) -> list[dict]:
        """Get lightweight index of all scenes."""
        return [
            {
                "scene_id": scene.scene_id,
                "issue": scene.issue,
                "pages": scene.page_numbers,
                "dialogue_count": len(scene.uno_dialogues),
                "other_characters": list(scene.other_characters),
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
# Evidence Models
# ============================================================================


class Evidence(BaseModel):
    """A piece of evidence for a character trait."""

    scene_id: str = Field(description="ID of the source scene")
    quote: str | None = Field(default=None, description="Italian quote if applicable")
    description: str = Field(description="What this evidence shows")
    section: str = Field(
        description="AIEOS section: identity|psychology|linguistics|history|motivations|capabilities"
    )
    trait: str = Field(description="Specific trait or field within section")


class EvidenceLedger(BaseModel):
    """Collection of evidence organized by AIEOS section."""

    evidence: list[Evidence] = Field(default_factory=list)
    processed_scene_ids: set[str] = Field(default_factory=set)

    def add_evidence(self, ev: Evidence) -> None:
        """Add an evidence item."""
        self.evidence.append(ev)

    def mark_scene_processed(self, scene_id: str) -> None:
        """Mark a scene as processed."""
        self.processed_scene_ids.add(scene_id)

    def is_scene_processed(self, scene_id: str) -> bool:
        """Check if a scene has been processed."""
        return scene_id in self.processed_scene_ids

    def get_evidence_by_section(self, section: str) -> list[Evidence]:
        """Get all evidence for a specific AIEOS section."""
        return [e for e in self.evidence if e.section == section]

    def get_evidence_by_trait(self, section: str, trait: str) -> list[Evidence]:
        """Get evidence for a specific trait within a section."""
        return [e for e in self.evidence if e.section == section and e.trait == trait]

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "evidence": [e.model_dump() for e in self.evidence],
            "processed_scene_ids": sorted(self.processed_scene_ids),
        }

    @classmethod
    def from_json(cls, data: dict) -> "EvidenceLedger":
        """Deserialize from JSON dict."""
        return cls(
            evidence=[Evidence.model_validate(e) for e in data.get("evidence", [])],
            processed_scene_ids=set(data.get("processed_scene_ids", [])),
        )


# ============================================================================
# DSPy Signatures for Evidence Extraction
# ============================================================================


class SceneEvidenceExtractor(dspy.Signature):
    """Extract evidence about a character from a comic scene for AIEOS profile.

    CRITICAL INSTRUCTIONS:

    1. AIEOS SECTIONS TO EXTRACT:
       - identity: Names, aliases, nature (AI/human), origin facts
       - psychology: Personality traits, emotional patterns, decision-making style
       - linguistics: Speech patterns, catchphrases, formality level, vocabulary
       - history: Origin story, key events, relationships formed
       - motivations: Core drives, goals, fears
       - capabilities: Skills, abilities, limitations

    2. EVIDENCE QUALITY:
       - Each evidence item must be directly supported by the scene
       - Include Italian quotes when they exemplify a trait
       - Be specific about what the evidence demonstrates
       - Focus on observable behavior and explicit statements

    3. TRAIT SPECIFICITY:
       For each section, use these trait categories:
       - psychology.ocean.openness, psychology.ocean.extraversion, etc.
       - psychology.neural_matrix.creativity, psychology.neural_matrix.empathy, etc.
       - psychology.emotional_profile.triggers.joy, etc.
       - linguistics.voice.formality, linguistics.idiolect.catchphrases, etc.
       - motivations.core_drive, motivations.goals.short_term, etc.

    4. OUTPUT:
       - Return a list of Evidence objects
       - Each evidence item links a scene to a specific AIEOS field
       - Empty list is valid if no relevant evidence in scene
    """

    scene_context: str = dspy.InputField(
        description="Context about the scene (issue, pages, characters)"
    )
    scene_summary: str = dspy.InputField(description="Summary of what happens")
    uno_dialogues: list[str] = dspy.InputField(
        description="Uno's dialogue lines in Italian"
    )
    panel_descriptions: str = dspy.InputField(
        description="Descriptions of visual panels"
    )
    scene_id: str = dspy.InputField(description="Unique scene identifier")

    evidence_items: list[Evidence] = dspy.OutputField(
        description="List of evidence items extracted from this scene"
    )


# ============================================================================
# DSPy Signatures for Section Building
# ============================================================================


class PsychologySectionBuilder(dspy.Signature):
    """Build the AIEOS psychology section from evidence.

    STRUCTURE:
    - neural_matrix: creativity, empathy, logic, adaptability, charisma, reliability (0.0-1.0)
    - traits.ocean: openness, conscientiousness, extraversion, agreeableness, neuroticism (0.0-1.0)
    - traits.mbti: 4-letter MBTI type (e.g., INTJ)
    - traits.temperament: temperament description
    - moral_compass: alignment, core_values list, conflict_resolution_style
    - emotional_profile: base_mood, volatility (0.0-1.0), resilience (0.0-1.0), triggers

    SCORING RULES:
    - All numeric scores must be between 0.0 and 1.0
    - Base scores on evidence frequency and strength
    - Provide reasoning for major trait scores
    """

    evidence_items: list[dict] = dspy.InputField(
        description="Evidence items for psychology section"
    )
    character_name: str = dspy.InputField(description="Name of the character")

    psychology: AIEOSPsychology = dspy.OutputField(
        description="AIEOS psychology section"
    )
    scoring_rationale: str = dspy.OutputField(
        description="Brief explanation of how scores were determined"
    )


class LinguisticsSectionBuilder(dspy.Signature):
    """Build the AIEOS linguistics section from evidence.

    STRUCTURE:
    - voice: formality_level (0-1), verbosity_level (0-1), vocabulary_level, style_descriptors
    - syntax: sentence_structure, use_contractions, active_passive_ratio (0-1)
    - idiolect: catchphrases (Italian with translations), forbidden_words, hesitation_markers
    - interaction: turn_taking description, dominance_score (0-1), emotional_coloring

    IMPORTANT:
    - Preserve Italian phrases and expressions
    - Include translations in parentheses for catchphrases: "Frase italiana (English translation)"
    - Base scores on observed dialogue patterns
    - All numeric scores must be between 0.0 and 1.0
    """

    evidence_items: list[dict] = dspy.InputField(
        description="Evidence items for linguistics section"
    )
    sample_dialogues: list[str] = dspy.InputField(
        description="Sample dialogues to analyze patterns"
    )
    character_name: str = dspy.InputField(description="Name of the character")

    linguistics: AIEOSLinguistics = dspy.OutputField(
        description="AIEOS linguistics section"
    )


class IdentitySectionBuilder(dspy.Signature):
    """Build the AIEOS identity section from evidence.

    STRUCTURE:
    - names: first (required), middle, last, nickname
    - bio: description (required), entity_type (e.g., "artificial_intelligence"), age_*, gender
    - origin: creator, creation_context, nationality, birthplace_city, birthplace_country

    For AI characters, adapt human-focused fields appropriately (e.g., creator instead of parents).
    """

    evidence_items: list[dict] = dspy.InputField(
        description="Evidence items for identity section"
    )
    character_name: str = dspy.InputField(description="Name of the character")

    identity: AIEOSIdentity = dspy.OutputField(description="AIEOS identity section")


class HistorySectionBuilder(dspy.Signature):
    """Build the AIEOS history section from evidence.

    STRUCTURE:
    - origin_story: Narrative describing the character's origin
    - key_life_events: List of {year, event, impact} objects
    - relationships: Dict mapping character names to relationship descriptions
    """

    evidence_items: list[dict] = dspy.InputField(
        description="Evidence items for history section"
    )
    character_name: str = dspy.InputField(description="Name of the character")

    history: AIEOSHistory = dspy.OutputField(description="AIEOS history section")


class MotivationsSectionBuilder(dspy.Signature):
    """Build the AIEOS motivations section from evidence.

    STRUCTURE:
    - core_drive: Primary motivation driving the character
    - goals.short_term: List of immediate goals
    - goals.long_term: List of long-term aspirations
    - fears.rational: List of logical/justified fears
    - fears.irrational: List of emotional/unjustified fears
    """

    evidence_items: list[dict] = dspy.InputField(
        description="Evidence items for motivations section"
    )
    character_name: str = dspy.InputField(description="Name of the character")

    motivations: AIEOSMotivations = dspy.OutputField(
        description="AIEOS motivations section"
    )


class CapabilitiesSectionBuilder(dspy.Signature):
    """Build the AIEOS capabilities section from evidence.

    STRUCTURE:
    - skills: List of {name, description, proficiency (0.0-1.0)} objects
    - limitations: List of explicit limitations and constraints

    Proficiency scores must be between 0.0 and 1.0.
    """

    evidence_items: list[dict] = dspy.InputField(
        description="Evidence items for capabilities section"
    )
    character_name: str = dspy.InputField(description="Name of the character")

    capabilities: AIEOSCapabilities = dspy.OutputField(
        description="AIEOS capabilities section"
    )


# ============================================================================
# Profile Builder
# ============================================================================


class AIEOSProfileBuilder:
    """Builds AIEOS profile from scenes using DSPy."""

    def __init__(self, scene_store: SceneStore, character_name: str = CHARACTER_NAME):
        self._store = scene_store
        self._character = character_name
        self._evidence_extractor = dspy.ChainOfThought(SceneEvidenceExtractor)

    def extract_evidence_from_scene(self, scene: Scene) -> list[Evidence]:
        """Extract evidence items from a single scene."""
        try:
            result = self._evidence_extractor(
                scene_context=scene.to_context_string(),
                scene_summary=scene.summary,
                uno_dialogues=scene.uno_dialogues,
                panel_descriptions=scene.to_other_context(),
                scene_id=scene.scene_id,
            )
            return result.evidence_items
        except Exception as e:
            log.warning(f"Error extracting evidence from {scene.scene_id}: {e}")
            return []

    def build_psychology_section(self, evidence: list[Evidence]) -> AIEOSPsychology:
        """Build psychology section from evidence."""
        builder = dspy.ChainOfThought(PsychologySectionBuilder)
        result = builder(
            evidence_items=[e.model_dump() for e in evidence],
            character_name=self._character,
        )
        log.debug(f"Psychology scoring rationale: {result.scoring_rationale}")
        return result.psychology

    def build_linguistics_section(
        self, evidence: list[Evidence], sample_dialogues: list[str]
    ) -> AIEOSLinguistics:
        """Build linguistics section from evidence."""
        builder = dspy.ChainOfThought(LinguisticsSectionBuilder)
        result = builder(
            evidence_items=[e.model_dump() for e in evidence],
            sample_dialogues=sample_dialogues[:50],  # Limit sample size
            character_name=self._character,
        )
        return result.linguistics

    def build_identity_section(self, evidence: list[Evidence]) -> AIEOSIdentity:
        """Build identity section from evidence."""
        builder = dspy.ChainOfThought(IdentitySectionBuilder)
        result = builder(
            evidence_items=[e.model_dump() for e in evidence],
            character_name=self._character,
        )
        return result.identity

    def build_history_section(self, evidence: list[Evidence]) -> AIEOSHistory:
        """Build history section from evidence."""
        builder = dspy.ChainOfThought(HistorySectionBuilder)
        result = builder(
            evidence_items=[e.model_dump() for e in evidence],
            character_name=self._character,
        )
        return result.history

    def build_motivations_section(self, evidence: list[Evidence]) -> AIEOSMotivations:
        """Build motivations section from evidence."""
        builder = dspy.ChainOfThought(MotivationsSectionBuilder)
        result = builder(
            evidence_items=[e.model_dump() for e in evidence],
            character_name=self._character,
        )
        return result.motivations

    def build_capabilities_section(self, evidence: list[Evidence]) -> AIEOSCapabilities:
        """Build capabilities section from evidence."""
        builder = dspy.ChainOfThought(CapabilitiesSectionBuilder)
        result = builder(
            evidence_items=[e.model_dump() for e in evidence],
            character_name=self._character,
        )
        return result.capabilities


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


def save_evidence_ledger(ledger: EvidenceLedger, path: Path) -> None:
    """Save evidence ledger to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger.to_json(), f, ensure_ascii=False, indent=2)
    log.debug(f"Saved evidence ledger to {path_str(path)}")


def load_evidence_ledger(path: Path) -> EvidenceLedger | None:
    """Load evidence ledger from JSON file if it exists."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return EvidenceLedger.from_json(data)


def save_section(section_data: BaseModel, section_name: str, output_dir: Path) -> None:
    """Save a section to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{section_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(section_data.model_dump(), f, ensure_ascii=False, indent=2)
    log.debug(f"Saved section {section_name} to {path_str(path)}")


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


def run_evidence_gathering(
    builder: AIEOSProfileBuilder,
    scene_store: SceneStore,
    max_scenes: int | None,
) -> EvidenceLedger:
    """Stage 1: Gather evidence from all scenes.

    Skips if evidence_ledger.json already exists.
    """
    ledger_path = CHECKPOINTS_DIR / "evidence_ledger.json"

    # Check for existing ledger
    existing_ledger = load_evidence_ledger(ledger_path)
    if existing_ledger:
        log.info(
            f"Loaded existing evidence ledger: {len(existing_ledger.evidence)} items, "
            f"{len(existing_ledger.processed_scene_ids)} scenes processed"
        )
        ledger = existing_ledger
    else:
        ledger = EvidenceLedger()
        log.info("Starting fresh evidence gathering")

    scenes = scene_store.all_scenes()
    if max_scenes:
        scenes = scenes[:max_scenes]

    # Filter to unprocessed scenes
    unprocessed = [s for s in scenes if not ledger.is_scene_processed(s.scene_id)]

    if not unprocessed:
        log.info("All scenes already processed for evidence")
        return ledger

    log.info(f"Gathering evidence from {len(unprocessed)} scenes...")

    with PROGRESS as progress:
        task = progress.add_task("Extracting evidence...", total=len(unprocessed))

        for i, scene in enumerate(unprocessed, 1):
            evidence_items = builder.extract_evidence_from_scene(scene)

            for ev in evidence_items:
                ledger.add_evidence(ev)

            ledger.mark_scene_processed(scene.scene_id)

            # Save checkpoint every 10 scenes
            if i % 10 == 0:
                save_evidence_ledger(ledger, ledger_path)

            progress.update(task, completed=i)

    # Final save
    save_evidence_ledger(ledger, ledger_path)
    log.info(f"Evidence gathering complete: {len(ledger.evidence)} items")

    return ledger


def run_section_extraction(
    builder: AIEOSProfileBuilder,
    ledger: EvidenceLedger,
    scene_store: SceneStore,
) -> AIEOSSections:
    """Stage 2: Extract each AIEOS section from evidence."""
    sections = AIEOSSections()

    # Collect sample dialogues for linguistics
    all_dialogues: list[str] = []
    for scene in scene_store.all_scenes():
        all_dialogues.extend(scene.uno_dialogues)

    for section_name in AIEOS_SECTIONS:
        # Check for cached section
        cached = load_section(section_name, SECTIONS_DIR)
        if cached:
            log.info(f"Loaded cached section: {section_name}")
            setattr(sections, section_name, cached)
            continue

        log.info(f"Building section: {section_name}")
        evidence = ledger.get_evidence_by_section(section_name)
        log.debug(f"  Found {len(evidence)} evidence items")

        try:
            if section_name == "psychology":
                section_data = builder.build_psychology_section(evidence)
            elif section_name == "linguistics":
                section_data = builder.build_linguistics_section(
                    evidence, all_dialogues
                )
            elif section_name == "identity":
                section_data = builder.build_identity_section(evidence)
            elif section_name == "history":
                section_data = builder.build_history_section(evidence)
            elif section_name == "motivations":
                section_data = builder.build_motivations_section(evidence)
            elif section_name == "capabilities":
                section_data = builder.build_capabilities_section(evidence)
            else:
                continue

            setattr(sections, section_name, section_data)
            save_section(section_data, section_name, SECTIONS_DIR)

        except Exception as e:
            log.error(f"Error building section {section_name}: {e}")

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

    console.print("\n[bold cyan]AIEOS Profile Builder[/bold cyan]\n")
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

    log.info(f"Loaded {len(all_scenes)} scenes with {args.character}")

    # Create scene store
    scene_store = SceneStore(all_scenes)

    # Create builder
    builder = AIEOSProfileBuilder(scene_store, args.character)

    # Stage 1: Evidence Gathering
    console.print("\n[bold]Stage 1: Evidence Gathering[/bold]")
    ledger = run_evidence_gathering(builder, scene_store, args.max_scenes)

    # Stage 2: Section Extraction
    console.print("\n[bold]Stage 2: Section Extraction[/bold]")
    sections = run_section_extraction(builder, ledger, scene_store)

    # Stage 3: Assembly
    console.print("\n[bold]Stage 3: Assembly & Validation[/bold]")
    aieos_doc = run_assembly(sections, args.character)

    # Serialize document using aliases for @context and @type
    doc_dict = aieos_doc.model_dump(by_alias=True)

    # Save final document
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    # Summary
    console.print("\n[bold green]AIEOS Profile Complete![/bold green]\n")
    console.print(f"Output: {path_str(output_path)}")
    console.print(f"Evidence items: {len(ledger.evidence)}")
    console.print(f"Scenes processed: {len(ledger.processed_scene_ids)}")

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
                "scenes_processed": len(ledger.processed_scene_ids),
                "evidence_items": len(ledger.evidence),
                "output_tokens": tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
