#!/usr/bin/env python3

"""
Build a condensed character profile directly from scenes without intermediate bloat.

This script processes scenes grouped by issue and uses DSPy to extract generalized
patterns rather than scene-by-scene details. The result is a compact ~7k token
profile, without needing a separate compression step.

Key differences from build_character_profile.py:
- Groups scenes by issue (~50 batches instead of ~hundreds of individual scenes)
- Extracts patterns and generalizations, not granular details
- Maintains size constraints throughout (targets ~7k tokens)
- Single script (no separate compression needed)
- Uses simple search-and-replace edits instead of complex hierarchical editing
"""

import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
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
TARGET_MAX_TOKENS = 7000  # Target profile size
ENCODING_NAME = "cl100k_base"  # GPT-4 tokenizer as approximation
MAX_RETRIES = 3
VERSION_TAG = "v6"

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / VERSION_TAG
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
DIFFS_DIR = OUTPUT_DIR / "diffs"
FAILURES_LOG_PATH = OUTPUT_DIR / "failures_log.jsonl"

# Global progress bar
PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)


def configure_lm() -> None:
    load_dotenv()
    lm = dspy.LM(
        model=MODEL_NAME,
        vertex_credentials=os.getenv("VERTEX_AI_CREDS"),
        vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        temperature=0.8,  # Slightly lower than v2 for more focused output
        top_p=0.95,
        top_k=64,
        max_tokens=65535,
    )
    dspy.configure(lm=lm, track_usage=True)


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


# ============================================================================
# Profile Structure Definition (Single Source of Truth)
# ============================================================================


@dataclass
class SectionDefinition:
    """Definition of a profile section."""

    header: str  # Full header with markdown symbols (e.g., "## Essential Identity")
    placeholder: str  # Initial placeholder content
    purpose: str  # What should go in this section (used in DSPy instructions)
    is_subsection: bool = False  # True for ### subsections


# Define profile structure - this is the ONLY place to edit section structure
PROFILE_STRUCTURE = [
    SectionDefinition(
        header="# Uno - Character Profile",
        placeholder="",
        purpose="Main document title",
    ),
    SectionDefinition(
        header="## Essential Identity",
        placeholder="To be developed based on observed core facts and constraints.",
        purpose="Core facts: what Uno is (AI, not biological), physical constraints (no off switch, power source), origin (created by Everett), capabilities that define identity",
    ),
    SectionDefinition(
        header="## Core Personality",
        placeholder="To be developed with 30-40 most distinctive traits.",
        purpose="30-40 distinctive personality traits. Format: **Trait Name:** Description with examples. Focus on patterns, not individual scenes.",
    ),
    SectionDefinition(
        header="## Communication Style",
        placeholder="To be developed with speech patterns, linguistic markers, and visual interface details.",
        purpose="How Uno communicates: visual interface (hologram colors), speech patterns (calls PK 'socio'), linguistic markers, Italian expressions, tone shifts",
    ),
    SectionDefinition(
        header="## Behavioral Guidelines",
        placeholder="",
        purpose="Section containing What Uno Does and What Uno Doesn't Do subsections",
    ),
    SectionDefinition(
        header="### What Uno Does",
        placeholder="To be developed with characteristic behaviors and capabilities.",
        purpose="Characteristic behaviors and capabilities. Bullet list format. Actions Uno regularly takes.",
        is_subsection=True,
    ),
    SectionDefinition(
        header="### What Uno Doesn't Do",
        placeholder="To be developed with explicit constraints and limitations.",
        purpose="Explicit constraints and limitations. Bullet list format. Things Uno cannot or will not do.",
        is_subsection=True,
    ),
    SectionDefinition(
        header="## Key Relationships",
        placeholder="To be developed for major characters (Paperinik, Everett, Due, Lyla, etc.).",
        purpose="Relationship dynamics with key characters. Can add subsections like '### With Paperinik' for major relationships. Describe interaction patterns.",
    ),
]


def generate_seed_document() -> str:
    """Generate seed document from structure definition.

    This ensures the seed document always matches the defined structure.
    """
    parts = []
    for section in PROFILE_STRUCTURE:
        parts.append(section.header)
        if section.placeholder:
            parts.append("")
            parts.append(section.placeholder)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def get_required_sections() -> list[str]:
    """Get list of required section headers for validation.

    Returns:
        List of all section headers that must be present
    """
    return [section.header for section in PROFILE_STRUCTURE]


def get_structure_description() -> str:
    """Generate structure description for DSPy prompt.

    Returns:
        Formatted structure description with purposes
    """
    lines = ["The document MUST keep this exact structure:"]
    for section in PROFILE_STRUCTURE:
        if section.header.startswith("#"):
            indent = "  " * (section.header.count("#") - 1)
            marker = "* " if not section.is_subsection else "- "
            lines.append(f"{indent}{marker}{section.header}")
            if section.purpose and section.purpose != "Main document title":
                purpose_indent = indent + "  "
                lines.append(f"{purpose_indent}Purpose: {section.purpose}")
    return "\n".join(lines)


# Generate seed document and validation list from structure
SEED_DOCUMENT = generate_seed_document()
REQUIRED_SECTIONS = get_required_sections()


# ============================================================================
# Scene Data Structures
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


# ============================================================================
# DSPy Signatures - Simplified with Search/Replace
# ============================================================================


class IssueInsightExtractor(dspy.Signature):
    """Extract generalized character insights from all scenes in an issue.

    CRITICAL INSTRUCTIONS:

    1. FOCUS ON PATTERNS, NOT DETAILS:
       - Look for recurring behaviors, speech patterns, and personality traits
       - Generalize from specific examples into character patterns
       - Ask: "What does this reveal about Uno's character in general?"
       - Avoid listing individual scene events

    2. LANGUAGE:
       - Write all analysis and pattern descriptions in English
       - Preserve original Italian for the 1-2 BEST dialogue examples only
       - Include English translations in parentheses for Italian quotes

    3. SELECTION CRITERIA FOR EXAMPLES:
       - Choose only the 1-2 MOST distinctive/illustrative quotes from this issue
       - Prioritize quotes that reveal personality, values, or unique traits
       - Skip generic responses or action-oriented dialogue
       - Each example should have clear context (what it reveals about Uno)

    4. RELATIONSHIP INSIGHTS:
       - Note patterns in how Uno interacts with each character
       - What role does Uno play in this relationship?
       - How does Uno's behavior change around this character?

    5. AVOID REDUNDANCY:
       - Assume the profile already has some content
       - Only note truly NEW or DISTINCTIVE observations
       - If a trait seems common/expected for an AI, skip it
    """

    issue_id: str = dspy.InputField(desc="Issue identifier (e.g., 'pkna-0')")
    all_scenes_summary: str = dspy.InputField(
        desc="Combined summary of all scenes with Uno in this issue"
    )
    all_uno_dialogues: list[str] = dspy.InputField(
        desc="All dialogue lines spoken by Uno in this issue (in Italian)"
    )
    character_interactions: str = dspy.InputField(
        desc="Summary of which characters Uno interacts with in this issue"
    )

    personality_patterns: list[str] = dspy.OutputField(
        desc=(
            "Observable personality traits as generalized patterns. "
            "Example: 'Shows protective concern for Paperinik's safety' not 'Warns PK in scene 3'"
        )
    )
    communication_patterns: list[str] = dspy.OutputField(
        desc=(
            "Speech patterns, linguistic markers, or communication style observations. "
            "Example: 'Uses 'socio' frequently when addressing Paperinik'"
        )
    )
    behavioral_patterns: list[str] = dspy.OutputField(
        desc=(
            "Recurring actions, decisions, or behavioral tendencies. "
            "Example: 'Monitors city-wide communications proactively'"
        )
    )
    relationship_insights: dict[str, str] = dspy.OutputField(
        desc=(
            "Key insights about relationships with specific characters. "
            "Format: {character_name: pattern_description}"
        )
    )
    best_dialogue_examples: list[dict] = dspy.OutputField(
        desc=(
            "1-2 most illustrative quotes from this issue. "
            "Format: [{'quote': 'Italian text', 'translation': 'English', 'reveals': 'What this shows about Uno'}]"
        )
    )
    capabilities_shown: list[str] = dspy.OutputField(
        desc=(
            "Technical abilities or knowledge demonstrated. "
            "Example: 'Can restructure tower architecture at will'"
        )
    )


class ProfileEdit(BaseModel):
    """A single search-and-replace edit operation."""

    search_text: str = Field(
        description=(
            "Exact text to find in the document. Must be unique. "
            "Can be a full section like 'To be developed...' or specific text to update."
        )
    )
    replace_text: str = Field(description="New text to replace the search_text with")
    reason: str = Field(description="Brief explanation of why this edit is being made")


@dataclass
class EditFailure:
    """Detailed information about why an edit failed."""

    edit: ProfileEdit
    failure_type: str  # "not_found", "not_unique", "broke_structure"
    failure_message: str
    timestamp: str


def create_profile_updater_signature() -> type[dspy.Signature]:
    """Create CondensedProfileUpdater signature with structure description injected.

    Returns:
        Dynamically created signature class with current structure
    """
    structure_desc = get_structure_description()

    # Build docstring with structure injected
    docstring = f"""Update condensed character profile with new insights using search-and-replace.

    CRITICAL INSTRUCTIONS:

    1. STRUCTURE PRESERVATION (MOST CRITICAL):
       - NEVER change section headers (##) or subsection headers (###)
       - NEVER add new top-level sections
       - NEVER merge or rename existing sections
       - ONLY replace content WITHIN existing sections

       {structure_desc}

    2. EDITING APPROACH - SEARCH AND REPLACE:
       - Find exact text in the current document
       - Replace it with updated/enhanced text
       - Each search_text must be UNIQUE in the document
       - Use multi-line text blocks for section replacements

    3. RETRY FEEDBACK:
       - If previous_failures is provided, those edits FAILED in the last attempt
       - DO NOT repeat the same edits that failed
       - Understand WHY they failed:
         * "not_found" - search text doesn't exist (maybe it was already edited)
         * "not_unique" - search text appears multiple times (be more specific)
         * "broke_structure" - edit removed a required section header (keep headers intact)
       - Adjust your approach based on the failure type
       - If content was already updated, skip redundant edits

    4. SIZE MANAGEMENT:
       - Current document is {{current_token_count}} tokens
       - Target max: {{target_max_tokens}} tokens
       - If approaching target, CONSOLIDATE:
         * Replace multiple similar traits with one generalized statement
         * Remove weaker examples, keep only the best
         * Merge redundant content WITHIN sections (not across)

    5. WHAT TO EDIT:
       - Replace "To be developed..." placeholders with actual content
       - Add new traits to existing sections (replace section content)
       - Update sections to add new insights
       - Consolidate redundant content

    6. EDIT EXAMPLES:

       Example 1 - Replace placeholder (preserves section header):
       search_text: "To be developed based on observed core facts and constraints."
       replace_text: "Uno is an artificial intelligence housed in Ducklair Tower. He has no off switch and is powered by ergogeo-dynamic flows from Earth's crust."

       Example 2 - Add to existing content (preserves section header):
       search_text: "**Protective:** Acts as a guardian for Paperinik."
       replace_text: "**Protective:** Acts as a guardian for Paperinik.\\n**Sarcastic:** Uses dry wit to deflate egos."

       Example 3 - Update subsection content (preserves subsection header):
       search_text: "To be developed with characteristic behaviors and capabilities."
       replace_text: "Monitors city-wide communications proactively\\nRestructures tower architecture at will\\nCreates personality backups as failsafes"

       FORBIDDEN - DO NOT DO THIS:
       search_text: "## Essential Identity"
       replace_text: "## Essential Identity & Personality"  # NEVER change headers!

    7. QUALITY OVER QUANTITY:
       - Add insights ONLY if they're truly new and distinctive
       - Better to have 30 well-defined traits than 50 vague ones
       - Each edit should meaningfully improve the profile

    8. LANGUAGE:
       - Write all descriptions in English
       - Preserve Italian dialogue examples with English translations in parentheses
    """

    class CondensedProfileUpdater(dspy.Signature):
        current_profile: str = dspy.InputField(desc="Current condensed profile content")
        current_token_count: int = dspy.InputField(
            desc="Current profile size in tokens"
        )
        target_max_tokens: int = dspy.InputField(
            desc="Target maximum tokens (typically 7000)"
        )
        issue_id: str = dspy.InputField(desc="Issue being processed")
        personality_patterns: list[str] = dspy.InputField(
            desc="New personality patterns from IssueInsightExtractor"
        )
        communication_patterns: list[str] = dspy.InputField(
            desc="New communication patterns"
        )
        behavioral_patterns: list[str] = dspy.InputField(desc="New behavioral patterns")
        relationship_insights: dict[str, str] = dspy.InputField(
            desc="New relationship insights"
        )
        best_dialogue_examples: list[dict] = dspy.InputField(
            desc="Best dialogue examples from issue"
        )
        capabilities_shown: list[str] = dspy.InputField(
            desc="Capabilities demonstrated"
        )
        previous_failures: str = dspy.InputField(
            desc="Description of edits that failed in previous attempts (empty if first attempt)"
        )

        edits: list[ProfileEdit] = dspy.OutputField(
            desc=(
                "List of search-and-replace edits to apply. "
                "Each search_text must be unique in the document. "
                "Empty list if no updates needed."
            )
        )
        insights_summary: str = dspy.OutputField(
            desc="Brief summary of what was added/updated/consolidated"
        )

    # Assign the dynamically generated docstring
    CondensedProfileUpdater.__doc__ = docstring

    return CondensedProfileUpdater


# ============================================================================
# Simple Document Manager with Search/Replace
# ============================================================================


class SimpleDocumentManager:
    """Manages document with simple search-and-replace operations."""

    def __init__(self, initial_content: str):
        self._content = initial_content

    def _validate_structure(self) -> bool:
        """Validate that document structure matches required sections."""
        for section in REQUIRED_SECTIONS:
            if section not in self._content:
                log.error(f"Structure validation failed: Missing section '{section}'")
                return False
        return True

    def apply_edit(self, edit: ProfileEdit) -> tuple[bool, EditFailure | None]:
        """Apply a search-and-replace edit.

        Returns:
            Tuple of (success: bool, failure: EditFailure | None)
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if edit.search_text not in self._content:
            failure_msg = f"Search text not found: '{edit.search_text[:100]}...'"
            log.warning(failure_msg)
            return False, EditFailure(
                edit=edit,
                failure_type="not_found",
                failure_message=failure_msg,
                timestamp=timestamp,
            )

        # Count occurrences to ensure uniqueness
        count = self._content.count(edit.search_text)
        if count > 1:
            failure_msg = f"Search text appears {count} times (not unique): '{edit.search_text[:100]}...'"
            log.warning(failure_msg)
            return False, EditFailure(
                edit=edit,
                failure_type="not_unique",
                failure_message=failure_msg,
                timestamp=timestamp,
            )

        # Save old content for rollback
        old_content = self._content

        # Apply replacement
        self._content = self._content.replace(edit.search_text, edit.replace_text)

        # Validate structure after edit
        if not self._validate_structure():
            failure_msg = (
                f"Edit broke document structure, rolling back: {edit.reason[:100]}"
            )
            log.error(failure_msg)
            self._content = old_content
            return False, EditFailure(
                edit=edit,
                failure_type="broke_structure",
                failure_message=failure_msg,
                timestamp=timestamp,
            )

        log.debug(f"Applied edit: {edit.reason}")
        return True, None

    def get_content(self) -> str:
        """Get the current document content."""
        return self._content

    def save(self, path: Path) -> None:
        """Save the document to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._content)


# ============================================================================
# Helper Functions
# ============================================================================


def natural_sort_key(path: Path) -> tuple:
    """Generate a sort key for natural/numeric sorting of issue directories.

    Examples:
        pkna-0 → ("pkna", 0)
        pkna-0-2 → ("pkna", 0, 2)
        pkna-10 → ("pkna", 10)
    """
    parts = path.name.split("-")
    key = []
    for part in parts:
        try:
            key.append(int(part))
        except ValueError:
            key.append(part)
    return tuple(key)


def extract_scenes_from_issue(issue_dir: Path) -> list[Scene]:
    """Extract all scenes containing Uno from an issue."""
    page_files = sorted(issue_dir.glob("page_*.json"))
    scenes = []
    current_scene_panels = []
    current_scene_pages = []

    for page_file in page_files:
        # Extract page number from filename (page_001.json -> 1)
        page_num = int(page_file.stem.split("_")[1])

        with open(page_file, encoding="utf-8") as f:
            page_data = json.load(f)

        panels = page_data.get("panels", [])
        if not panels:
            continue

        for panel in panels:
            # Check if this starts a new scene
            if panel.get("is_new_scene", False) and current_scene_panels:
                # Process the completed scene
                scene = create_scene_from_panels(
                    issue_dir.name, current_scene_pages, current_scene_panels
                )
                if scene:
                    scenes.append(scene)

                # Start new scene
                current_scene_panels = []
                current_scene_pages = []

            # Add panel to current scene
            current_scene_panels.append(panel)
            if page_num not in current_scene_pages:
                current_scene_pages.append(page_num)

    # Don't forget the last scene
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
    other_characters = set()

    for panel in panels:
        # Collect panel descriptions
        if desc := panel.get("description"):
            panel_descriptions.append(desc)

        # Collect dialogues
        for dialogue in panel.get("dialogues", []):
            character = dialogue.get("character", "").strip()
            line = dialogue.get("line", "").strip()

            if character.lower() == "uno":
                uno_dialogues.append(line)
            elif character:
                other_characters.add(character)

    # Only create scene if Uno has dialogue
    if not uno_dialogues:
        return None

    # Create summary from panel descriptions
    summary = " ".join(panel_descriptions)

    return Scene(
        issue=issue,
        page_numbers=page_numbers,
        summary=summary,
        uno_dialogues=uno_dialogues,
        panel_descriptions=panel_descriptions,
        other_characters=other_characters,
    )


def collect_scenes_by_issue() -> dict[str, list[Scene]]:
    """Collect all scenes grouped by issue."""
    log.info("Scanning for scenes containing Uno...")
    scenes_by_issue = defaultdict(list)

    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )

    for issue_dir in issue_dirs:
        scenes = extract_scenes_from_issue(issue_dir)
        if scenes:
            scenes_by_issue[issue_dir.name] = scenes
            log.info(f"Found {len(scenes)} scenes with Uno in {issue_dir.name}")

    total_scenes = sum(len(scenes) for scenes in scenes_by_issue.values())
    log.info(
        f"Total: {total_scenes} scenes across {len(scenes_by_issue)} issues with Uno"
    )

    return dict(scenes_by_issue)


def extract_issue_insights(
    extractor: dspy.Module, issue_id: str, scenes: list[Scene]
) -> dspy.Prediction:
    """Extract generalized insights from all scenes in an issue."""
    # Combine all scene summaries
    all_summaries = " | ".join(scene.summary for scene in scenes)

    # Collect all Uno dialogues
    all_dialogues = []
    for scene in scenes:
        all_dialogues.extend(scene.uno_dialogues)

    # Collect character interactions
    all_characters = set()
    for scene in scenes:
        all_characters.update(scene.other_characters)
    character_summary = ", ".join(sorted(all_characters)) if all_characters else "none"

    # Extract insights
    return extractor(
        issue_id=issue_id,
        all_scenes_summary=all_summaries,
        all_uno_dialogues=all_dialogues,
        character_interactions=f"Characters present: {character_summary}",
    )


def log_edit_failure(issue_id: str, failure: EditFailure) -> None:
    """Log edit failure to structured log file."""
    log_entry = {
        "timestamp": failure.timestamp,
        "issue_id": issue_id,
        "failure_type": failure.failure_type,
        "failure_message": failure.failure_message,
        "edit_reason": failure.edit.reason,
        "search_text_preview": failure.edit.search_text[:200],
        "replace_text_preview": failure.edit.replace_text[:200],
    }

    with open(FAILURES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def format_failures_feedback(failures: list[EditFailure]) -> str:
    """Format list of edit failures into feedback string for the model."""
    if not failures:
        return ""

    feedback_lines = ["Previous attempt had these failures:"]
    for i, failure in enumerate(failures, 1):
        feedback_lines.append(
            f"{i}. [{failure.failure_type}] {failure.failure_message}\n"
            f"   Reason: {failure.edit.reason}\n"
            f"   Search: {failure.edit.search_text[:150]}..."
        )

    return "\n".join(feedback_lines)


def check_token_reduction(
    old_tokens: int,
    new_tokens: int,
    target_tokens: int,
    issue_id: str,
) -> bool:
    """Check if token reduction is drastic and confirm with model if intentional.

    Returns True if reduction is acceptable, False if it seems wrong.
    """
    # Calculate how much we reduced
    reduction = old_tokens - new_tokens

    # If we're still far from target and reduced by more than 1000 tokens, check
    if new_tokens < target_tokens - 2000 and reduction > 1000:
        log.warning(
            f"{issue_id}: Large token reduction detected "
            f"({old_tokens} → {new_tokens}, -{reduction} tokens). "
            f"Still {target_tokens - new_tokens} tokens below target."
        )
        # This is suspicious - we should be growing toward target, not shrinking
        return False

    return True


def update_profile_with_retry(
    updater: dspy.Module,
    doc_manager: SimpleDocumentManager,
    issue_id: str,
    insights: dspy.Prediction,
) -> tuple[bool, str]:
    """Update profile with insights, with retry logic for failed edits."""
    current_content = doc_manager.get_content()
    current_tokens = count_tokens(current_content)
    previous_failures: list[EditFailure] = []

    for attempt in range(MAX_RETRIES):
        try:
            # Build feedback from previous failures
            failures_feedback = format_failures_feedback(previous_failures)

            # Get update edits from the model
            pred = updater(
                current_profile=current_content,
                current_token_count=current_tokens,
                target_max_tokens=TARGET_MAX_TOKENS,
                issue_id=issue_id,
                personality_patterns=insights.personality_patterns,
                communication_patterns=insights.communication_patterns,
                behavioral_patterns=insights.behavioral_patterns,
                relationship_insights=insights.relationship_insights,
                best_dialogue_examples=insights.best_dialogue_examples,
                capabilities_shown=insights.capabilities_shown,
                previous_failures=failures_feedback,
            )

            # Try to apply all edits
            all_successful = True
            current_failures: list[EditFailure] = []

            for edit in pred.edits:
                success, failure = doc_manager.apply_edit(edit)
                if not success and failure is not None:
                    all_successful = False
                    current_failures.append(failure)
                    # Log to file
                    log_edit_failure(issue_id, failure)

            if all_successful:
                new_tokens = count_tokens(doc_manager.get_content())

                # Check for drastic token reduction
                if not check_token_reduction(
                    current_tokens, new_tokens, TARGET_MAX_TOKENS, issue_id
                ):
                    log.error(
                        f"{issue_id}: Drastic token reduction detected - may indicate model error"
                    )

                log.info(
                    f"{issue_id}: Applied {len(pred.edits)} edits. "
                    f"Tokens: {current_tokens} → {new_tokens}. "
                    f"Summary: {pred.insights_summary[:100]}..."
                )
                return True, pred.insights_summary

            # If some edits failed and we have retries left
            if attempt < MAX_RETRIES - 1:
                log.warning(
                    f"{issue_id}: {len(current_failures)} edits failed, "
                    f"retrying (attempt {attempt + 2}/{MAX_RETRIES})"
                )
                # Update for retry
                current_content = doc_manager.get_content()
                current_tokens = count_tokens(current_content)
                previous_failures = current_failures
            else:
                log.error(
                    f"{issue_id}: {len(current_failures)} edits failed after "
                    f"{MAX_RETRIES} attempts"
                )
                return False, f"Partial update - {len(current_failures)} edits failed"

        except Exception as e:
            log.error(f"{issue_id}: Error during processing: {e}")
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Retrying (attempt {attempt + 2}/{MAX_RETRIES})")
            else:
                return False, f"Error: {str(e)}"

    return False, "Max retries exceeded"


def generate_and_save_diff(
    old_content: str, new_content: str, diff_path: Path, version_num: int
) -> None:
    """Generate unified diff and save to file using diff -u command.

    Args:
        old_content: Previous version content
        new_content: Current version content
        diff_path: Path where diff file should be saved
        version_num: Current version number
    """
    # Write contents to temporary files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as old_file:
        old_file.write(old_content)
        old_path = old_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as new_file:
        new_file.write(new_content)
        new_path = new_file.name

    try:
        # Run diff -u (returns exit code 1 when files differ, which is expected)
        result = subprocess.run(
            [
                "diff",
                "-u",
                "--label",
                f"a/checkpoint_{version_num - 1:03d}.md",
                "--label",
                f"b/checkpoint_{version_num:03d}.md",
                old_path,
                new_path,
            ],
            capture_output=True,
            text=True,
        )

        diff_output = result.stdout

        # Save diff to file
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(diff_output)

        log.debug(f"Saved diff to {diff_path}")
    finally:
        # Cleanup temp files
        Path(old_path).unlink()
        Path(new_path).unlink()


def save_checkpoint_with_diff(
    doc_manager: SimpleDocumentManager,
    checkpoint_num: int,
    previous_content: str | None,
) -> str:
    """Save checkpoint as both diff and full document (for last 3).

    Args:
        doc_manager: Document manager with current state
        checkpoint_num: Sequential checkpoint number (1, 2, 3, ...)
        previous_content: Content of previous version (None for first checkpoint)

    Returns:
        Current document content for next iteration
    """
    current_content = doc_manager.get_content()
    checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num:03d}.md"
    diff_path = DIFFS_DIR / f"checkpoint_{checkpoint_num:03d}.diff"

    # Always save the diff
    if previous_content is not None:
        generate_and_save_diff(
            previous_content, current_content, diff_path, checkpoint_num
        )
    else:
        # First checkpoint - diff from seed
        generate_and_save_diff(
            SEED_DOCUMENT, current_content, diff_path, checkpoint_num
        )

    # Always save full document (will be cleaned up later if not in last 3)
    doc_manager.save(checkpoint_path)

    # Cleanup old full documents (keep last 3)
    if checkpoint_num > 3:
        old_checkpoint = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num - 3:03d}.md"
        if old_checkpoint.exists():
            old_checkpoint.unlink()
            log.debug(f"Deleted old checkpoint: {old_checkpoint}")

    tokens = count_tokens(current_content)
    log.debug(f"Saved checkpoint {checkpoint_num}: {checkpoint_path} ({tokens} tokens)")

    return current_content


def main() -> None:
    """Main function to build condensed character profile."""
    console.print(
        f"\n[bold cyan]Condensed Character Profile Builder ({VERSION_TAG})[/bold cyan]\n"
    )

    # Setup
    configure_lm()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Save seed document
    seed_path = OUTPUT_DIR / "seed_document.md"
    with open(seed_path, "w", encoding="utf-8") as f:
        f.write(SEED_DOCUMENT)
    log.info(f"Saved seed document to {seed_path}")

    # Initialize document manager
    doc_manager = SimpleDocumentManager(SEED_DOCUMENT)

    # Initialize DSPy modules
    log.info("Initializing DSPy modules...")
    insight_extractor = dspy.ChainOfThought(IssueInsightExtractor)

    # Create profile updater with structure description injected
    CondensedProfileUpdater = create_profile_updater_signature()
    profile_updater = dspy.ChainOfThought(CondensedProfileUpdater)

    # Collect scenes grouped by issue
    scenes_by_issue = collect_scenes_by_issue()

    if not scenes_by_issue:
        log.error("No scenes found! Check INPUT_DIR path.")
        return

    # Open processing log file (JSONL format)
    log_path = OUTPUT_DIR / "processing_log.jsonl"

    # Process each issue
    successful_count = 0
    total_issues = len(scenes_by_issue)
    previous_content = None  # Track previous version for diffs

    with open(log_path, "w", encoding="utf-8") as log_file:
        with PROGRESS as progress:
            for i, (issue_id, scenes) in progress.track(
                enumerate(scenes_by_issue.items(), 1),
                total=total_issues,
                description="Building condensed profile...",
            ):
                log.info(
                    f"\nProcessing issue {i}/{total_issues}: {issue_id} ({len(scenes)} scenes)"
                )

                # Extract insights from all scenes in this issue
                insights = extract_issue_insights(insight_extractor, issue_id, scenes)

                # Update profile with insights
                success, summary = update_profile_with_retry(
                    profile_updater, doc_manager, issue_id, insights
                )

                if success:
                    successful_count += 1

                # Save checkpoint with diff and update previous_content
                previous_content = save_checkpoint_with_diff(
                    doc_manager, i, previous_content
                )

                # Log entry with issue information
                log_entry = {
                    "checkpoint": i,
                    "issue": issue_id,
                    "scene_count": len(scenes),
                    "success": success,
                    "summary": summary,
                    "tokens_after": count_tokens(doc_manager.get_content()),
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                log_file.flush()

    # Save final document
    final_path = OUTPUT_DIR / "uno_profile.md"
    doc_manager.save(final_path)

    final_tokens = count_tokens(doc_manager.get_content())
    final_words = len(doc_manager.get_content().split())

    # Print summary
    console.print()
    console.print("[bold green]✓ Profile Building Complete![/bold green]\n")
    console.print(f"Issues processed: {total_issues}")
    console.print(f"Successful updates: {successful_count}")
    console.print(f"Failed updates: {total_issues - successful_count}")
    console.print()
    console.print(f"[bold]Final profile:[/bold] {final_path}")
    console.print(f"  Size: {final_tokens:,} tokens (~{final_words:,} words)")
    console.print(f"  Target: {TARGET_MAX_TOKENS:,} tokens")
    if final_tokens <= TARGET_MAX_TOKENS:
        console.print("  [green]✓ Within target[/green]")
    else:
        overage = final_tokens - TARGET_MAX_TOKENS
        console.print(f"  [yellow]⚠ {overage:,} tokens over target[/yellow]")

    console.print(f"\nProcessing log: {log_path}")
    console.print(f"Checkpoints: {CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    main()
