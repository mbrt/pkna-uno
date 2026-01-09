#!/usr/bin/env python3

"""
Build a condensed character profile directly from scenes without intermediate bloat.

This script processes scenes grouped by issue and uses DSPy to extract generalized
patterns rather than scene-by-scene details. The result is a compact ~2k token
profile similar to Tier 2 quality, without needing a separate compression step.

Key differences from build_character_profile.py:
- Groups scenes by issue (~50 batches instead of ~hundreds of individual scenes)
- Extracts patterns and generalizations, not granular details
- Maintains size constraints throughout (targets ~2k tokens)
- Single script (no separate compression needed)
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
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
TARGET_MAX_TOKENS = 2000  # Tier 2 target
ENCODING_NAME = "cl100k_base"  # GPT-4 tokenizer as approximation
MAX_RETRIES = 3

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / "v4"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

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


# Condensed seed document with Behavioral Guidelines structure
CONDENSED_SEED_DOCUMENT = """# Uno - Character Profile

## Essential Identity
To be developed based on observed core facts and constraints.

## Core Personality
To be developed with 15-25 most distinctive traits (Tier 2 allows more than Tier 1).

## Communication Style
To be developed with speech patterns, linguistic markers, and visual interface details.

## Behavioral Guidelines
### What Uno Does:
To be developed with characteristic behaviors and capabilities.

### What Uno Doesn't Do:
To be developed with explicit constraints and limitations.

## Key Relationships
To be developed for major characters (Paperinik, Everett, Due, Lyla, etc.).
"""


# ============================================================================
# Document Structure Classes (Reused from build_character_profile.py)
# ============================================================================


@dataclass
class Line:
    """A single line of text in the document."""

    content: str


@dataclass
class Section:
    """A section in the document with optional subsections."""

    header: str  # Full header including markdown symbols (e.g., "## Personality")
    level: int  # Header level (1 for #, 2 for ##, etc.)
    lines: list[Line] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)


class DocumentStructure:
    """Manages hierarchical document structure with parsing and editing."""

    @staticmethod
    def parse_markdown(text: str) -> Section:
        """Parse markdown text into a hierarchical Section structure.

        Returns a root section (level 0) containing the document structure.

        Multi-line paragraphs (consecutive non-empty lines) are preserved as single
        Line objects. Empty lines separate paragraphs.
        """
        lines = text.split("\n")
        root = Section(header="", level=0)
        stack: list[Section] = [root]
        current_paragraph: list[str] = []

        def flush_paragraph() -> None:
            """Flush accumulated paragraph lines to current section."""
            if current_paragraph:
                # Join lines with newlines to preserve multi-line paragraphs
                content = "\n".join(current_paragraph)
                stack[-1].lines.append(Line(content=content))
                current_paragraph.clear()

        for line in lines:
            # Check if this is a header
            stripped = line.strip()
            if stripped.startswith("#"):
                # Flush any accumulated paragraph before starting new section
                flush_paragraph()

                # Count the number of # symbols
                level = 0
                for char in stripped:
                    if char == "#":
                        level += 1
                    else:
                        break

                # Create new section
                new_section = Section(header=stripped, level=level)

                # Pop stack until we find the parent (section with level < current)
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()

                # Add to parent's subsections
                stack[-1].subsections.append(new_section)
                stack.append(new_section)
            else:
                # Regular line - accumulate into paragraph or flush on empty line
                if stripped:
                    # Non-empty line - add to current paragraph
                    current_paragraph.append(stripped)
                else:
                    # Empty line - flush paragraph
                    flush_paragraph()

        # Don't forget to flush the final paragraph
        flush_paragraph()

        return root

    @staticmethod
    def to_markdown(root: Section) -> str:
        """Convert Section structure back to markdown text."""
        lines: list[str] = []

        def write_section(section: Section) -> None:
            # Write header (skip for root)
            if section.level > 0:
                lines.append(section.header)

            # Write lines
            for line in section.lines:
                lines.append(line.content)

            # Write subsections
            for subsection in section.subsections:
                write_section(subsection)

        write_section(root)

        # Join and clean up trailing newlines
        result = "\n\n".join(lines)
        return result.rstrip() + "\n" if result else ""

    @staticmethod
    def find_section(root: Section, section_name: str) -> Section | None:
        """Find a section by name anywhere in the document tree.

        The section name is matched case-insensitively against section headers
        (without the # symbols). Returns the first matching section found.
        """
        if not section_name:
            return root

        section_name_lower = section_name.strip().lower()

        def search_recursive(section: Section) -> Section | None:
            # Check current section's subsections
            for subsection in section.subsections:
                # Extract header text without # symbols
                header_text = subsection.header.lstrip("#").strip()
                if header_text.lower() == section_name_lower:
                    return subsection

                # Recursively search in subsections
                result = search_recursive(subsection)
                if result is not None:
                    return result

            return None

        return search_recursive(root)

    @staticmethod
    def find_line_in_section(section: Section, search: str) -> tuple[Line | None, bool]:
        """Find a line in a section by partial case-insensitive match.

        Returns: (Line | None, is_unique: bool)
        - (Line, True) if exactly one match found
        - (None, False) if no matches found
        - (Line, False) if multiple matches found (returns first)
        """
        search_lower = search.lower()
        matches: list[Line] = []

        for line in section.lines:
            if search_lower in line.content.lower():
                matches.append(line)

        if len(matches) == 0:
            return None, False
        elif len(matches) == 1:
            return matches[0], True
        else:
            # Multiple matches - not unique
            return matches[0], False


class EditOperation(str, Enum):
    """Types of edit operations on the document."""

    ADD_LINE = "add_line"
    REPLACE_LINE = "replace_line"
    DELETE_LINE = "delete_line"
    ADD_SUBSECTION = "add_subsection"


class DocumentEdit(BaseModel):
    """A single edit operation on the character document.

    Operations:
    - add_line: Add a new line to a section (at the end)
    - replace_line: Replace an existing line (found via partial search)
    - delete_line: Delete a line (found via partial search)
    - add_subsection: Add a new subsection to an existing section
    """

    operation: EditOperation = Field(description="Type of edit operation to perform")
    section_path: str = Field(
        description=(
            "Name of the target section. "
            "Example: 'Core Identity' or 'Core Personality' or 'With Paperinik'. "
            "Matching is case-insensitive and searches the entire document."
        )
    )
    search_text: str | None = Field(
        default=None,
        description=(
            "Partial text to find the target line (for replace_line and delete_line). "
            "Matching is case-insensitive and can match anywhere in the line. "
            "Must uniquely identify a single line."
        ),
    )
    new_content: str | None = Field(
        default=None,
        description=(
            "New content to add or replace with (for add_line, replace_line, add_subsection)"
        ),
    )
    subsection_header: str | None = Field(
        default=None,
        description=(
            "Header for new subsection (for add_subsection). "
            "Example: '### With Lyla' (include ## or ### symbols)"
        ),
    )


class DocumentManager:
    """Manages the character profile document using structured editing."""

    def __init__(self, initial_content: str):
        self._root = DocumentStructure.parse_markdown(initial_content)

    def apply_edit(self, edit: DocumentEdit) -> bool:
        """Apply a single edit to the document. Returns True if successful."""
        # Find the target section
        section = DocumentStructure.find_section(self._root, edit.section_path)
        if section is None:
            log.warning(f"Section not found: '{edit.section_path}'")
            return False

        if edit.new_content is not None:
            edit.new_content = edit.new_content.strip()

        if edit.operation == EditOperation.ADD_LINE:
            # Add a new line at the end of the section
            if edit.new_content is None:
                log.error("add_line requires new_content")
                return False
            section.lines.append(Line(content=edit.new_content))
            return True

        elif edit.operation == EditOperation.REPLACE_LINE:
            # Find and replace a line
            if not edit.search_text:
                log.error("replace_line requires search_text")
                return False
            if not edit.new_content:
                log.error("replace_line requires new_content")
                return False

            line, is_unique = DocumentStructure.find_line_in_section(
                section, edit.search_text
            )
            if line is None:
                log.warning(
                    f"Line not found in section '{edit.section_path}': "
                    f"'{edit.search_text[:50]}...'"
                )
                return False
            if not is_unique:
                log.warning(
                    f"Multiple lines match in section '{edit.section_path}': "
                    f"'{edit.search_text[:50]}...'. Please be more specific."
                )
                return False

            # Replace the line content
            line.content = edit.new_content
            return True

        elif edit.operation == EditOperation.DELETE_LINE:
            # Find and delete a line
            if not edit.search_text:
                log.error("delete_line requires search_text")
                return False

            line, is_unique = DocumentStructure.find_line_in_section(
                section, edit.search_text
            )
            if line is None:
                log.warning(
                    f"Line not found in section '{edit.section_path}': "
                    f"'{edit.search_text[:50]}...'"
                )
                return False
            if not is_unique:
                log.warning(
                    f"Multiple lines match in section '{edit.section_path}': "
                    f"'{edit.search_text[:50]}...'. Please be more specific."
                )
                return False

            # Remove the line
            section.lines.remove(line)
            return True

        elif edit.operation == EditOperation.ADD_SUBSECTION:
            # Add a new subsection
            if edit.subsection_header is None:
                log.error("add_subsection requires subsection_header")
                return False
            if edit.new_content is None:
                log.error("add_subsection requires new_content")
                return False

            # Determine the level of the new subsection
            level = section.level + 1
            if edit.subsection_header.startswith("#"):
                # Count # symbols in the provided header
                level = len(edit.subsection_header) - len(
                    edit.subsection_header.lstrip("#")
                )
                header = edit.subsection_header.strip()
            else:
                # Generate header with appropriate level
                header = "#" * level + " " + edit.subsection_header

            # Create new subsection with content
            new_section = Section(header=header, level=level)
            new_section.lines.append(Line(content=edit.new_content))
            section.subsections.append(new_section)
            return True

        log.error(f"Unknown operation: {edit.operation}")
        return False

    def get_content(self) -> str:
        """Get the current document content as markdown."""
        return DocumentStructure.to_markdown(self._root)

    def save(self, path: Path) -> None:
        """Save the document to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_content())


# ============================================================================
# Scene Data Structures (Reused from build_character_profile.py)
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
# NEW: DSPy Signatures for Condensed Profile Building
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


class CondensedProfileUpdater(dspy.Signature):
    """Update condensed character profile with new insights while maintaining compact size.

    CRITICAL INSTRUCTIONS:

    1. SIZE MANAGEMENT (MOST IMPORTANT):
       - Target max: 2000 tokens total
       - If current profile is approaching target, CONSOLIDATE before adding
       - Merge similar traits into generalized statements
       - Replace weaker examples with stronger ones (don't just accumulate)
       - If at token limit, DELETE less distinctive content to make room

    2. QUALITY OVER QUANTITY:
       - Add insights ONLY if they're truly new and distinctive
       - Each trait should be specific and actionable
       - Better to have 15 well-defined traits than 30 vague ones
       - Avoid redundancy across sections

    3. CONSOLIDATION EDITS:
       - Use replace_line to merge similar observations
       - Use delete_line to remove redundant or weak content
       - Combine multiple specific examples into pattern descriptions

    4. LANGUAGE:
       - Write all descriptions in English
       - Preserve Italian for selected dialogue examples (with translations)

    5. BEHAVIORAL GUIDELINES:
       - Maintain "What Uno Does" and "What Uno Doesn't Do" subsections
       - These are critical for character consistency

    6. EDIT OPERATIONS:
       - add_line: Add new content to a section
       - replace_line: Replace existing content (use for consolidation)
       - delete_line: Remove redundant/weak content
       - add_subsection: Create new character relationship subsection
    """

    current_profile: str = dspy.InputField(desc="Current condensed profile content")
    current_token_count: int = dspy.InputField(desc="Current profile size in tokens")
    target_max_tokens: int = dspy.InputField(
        desc="Target maximum tokens (typically 2000)"
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
    capabilities_shown: list[str] = dspy.InputField(desc="Capabilities demonstrated")

    edits: list[DocumentEdit] = dspy.OutputField(
        desc=(
            "List of edit operations to apply. "
            "Include consolidation edits (replace/delete) if approaching token limit. "
            "Empty list if no updates needed."
        )
    )
    insights_summary: str = dspy.OutputField(
        desc="Brief summary of what was added/updated/consolidated"
    )


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


def update_profile_with_retry(
    updater: dspy.Module,
    doc_manager: DocumentManager,
    issue_id: str,
    insights: dspy.Prediction,
) -> tuple[bool, str]:
    """Update profile with insights, with retry logic for failed edits."""
    current_content = doc_manager.get_content()
    current_tokens = count_tokens(current_content)

    for attempt in range(MAX_RETRIES):
        try:
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
            )

            # Try to apply all edits
            all_successful = True
            failed_edits = []

            for edit in pred.edits:
                if not doc_manager.apply_edit(edit):
                    all_successful = False
                    failed_edits.append(edit)

            if all_successful:
                new_tokens = count_tokens(doc_manager.get_content())
                log.info(
                    f"{issue_id}: Applied {len(pred.edits)} edits. "
                    f"Tokens: {current_tokens} → {new_tokens}. "
                    f"Summary: {pred.insights_summary[:100]}..."
                )
                return True, pred.insights_summary

            # If some edits failed and we have retries left
            if attempt < MAX_RETRIES - 1:
                log.warning(
                    f"{issue_id}: {len(failed_edits)} edits failed, "
                    f"retrying (attempt {attempt + 2}/{MAX_RETRIES})"
                )
            else:
                log.error(
                    f"{issue_id}: {len(failed_edits)} edits failed after "
                    f"{MAX_RETRIES} attempts"
                )
                return False, f"Partial update - {len(failed_edits)} edits failed"

        except Exception as e:
            log.error(f"{issue_id}: Error during processing: {e}")
            if attempt < MAX_RETRIES - 1:
                log.warning(f"Retrying (attempt {attempt + 2}/{MAX_RETRIES})")
            else:
                return False, f"Error: {str(e)}"

    return False, "Max retries exceeded"


def save_checkpoint(doc_manager: DocumentManager, issue_id: str) -> None:
    """Save checkpoint after processing an issue."""
    checkpoint_path = CHECKPOINTS_DIR / f"{issue_id}.md"
    doc_manager.save(checkpoint_path)

    tokens = count_tokens(doc_manager.get_content())
    log.debug(f"Saved checkpoint: {checkpoint_path} ({tokens} tokens)")


def main() -> None:
    """Main function to build condensed character profile."""
    console.print("\n[bold cyan]Condensed Character Profile Builder (v4)[/bold cyan]\n")

    # Setup
    configure_lm()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save seed document
    seed_path = OUTPUT_DIR / "seed_document.md"
    with open(seed_path, "w", encoding="utf-8") as f:
        f.write(CONDENSED_SEED_DOCUMENT)
    log.info(f"Saved seed document to {seed_path}")

    # Initialize document manager
    doc_manager = DocumentManager(CONDENSED_SEED_DOCUMENT)

    # Initialize DSPy modules
    log.info("Initializing DSPy modules...")
    insight_extractor = dspy.ChainOfThought(IssueInsightExtractor)
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

                # Save checkpoint
                save_checkpoint(doc_manager, issue_id)

                # Log entry
                log_entry = {
                    "issue": issue_id,
                    "issue_number": i,
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
    console.print(f"  Target: {TARGET_MAX_TOKENS:,} tokens (Tier 2)")
    if final_tokens <= TARGET_MAX_TOKENS:
        console.print("  [green]✓ Within target[/green]")
    else:
        overage = final_tokens - TARGET_MAX_TOKENS
        console.print(f"  [yellow]⚠ {overage:,} tokens over target[/yellow]")

    console.print(f"\nProcessing log: {log_path}")
    console.print(f"Checkpoints: {CHECKPOINTS_DIR}/")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print(
        "[dim]1. Review profile: cat output/character-profile/uno/v4/uno_profile.md[/dim]"
    )
    console.print(
        "[dim]2. Test quality: ./generate_from_character_profile.py --tier v4[/dim]"
    )
    console.print(
        "[dim]3. Compare to Tier 2: diff output/character-profile/uno/v3/uno_profile_tier2.md output/character-profile/uno/v4/uno_profile.md[/dim]"
    )


if __name__ == "__main__":
    main()
