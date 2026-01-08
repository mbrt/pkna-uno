#!/usr/bin/env python3

"""
Build a character profile document for Uno by iteratively analyzing scenes.

This script processes all scenes containing Uno from the extracted comic data
and uses a DSPy model to incrementally build a comprehensive character profile.
The profile serves as a "soul document" for fine-tuning an LLM to mimic Uno's behavior.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import dspy
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
MAX_RETRIES = 3

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / "v2"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
DIFFS_DIR = OUTPUT_DIR / "diffs"

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
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=65535,
    )
    dspy.configure(lm=lm, track_usage=True)


# Seed document structure inspired by Claude Opus soul document
SEED_DOCUMENT = """# Uno - Character Profile

## Core Identity

Uno is an artificial intelligence housed in the Ducklair Tower. He serves as the technological companion and support system for Paperinik (PK), assisting in missions against the Evronians and other threats.

## Personality Traits

To be developed based on observed behavior in scenes.

## Communication Style

To be developed based on dialogue patterns.

## Values and Beliefs

To be developed based on decisions and statements.

## Relationships

To be developed based on interactions with various characters.

### With Paperinik/Paperino

To be developed based on interactions.

## Knowledge and Capabilities

To be developed based on demonstrated abilities.

## Behavioral Patterns

To be developed based on recurring actions and responses.

## Character Growth and Development

To be developed based on character arc across issues.

## Example Dialogues

To be collected from scenes (preserved in original Italian).
"""


# ============================================================================
# Document Structure Classes
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
            "Example: 'Core Identity' or 'Personality Traits' or 'With Paperinik'. "
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


class CharacterDocumentUpdater(dspy.Signature):
    """Update a character profile document based on new scene information.

    CRITICAL INSTRUCTIONS:

    1. LANGUAGE:
       - Write all analysis and descriptions in English
       - Preserve original Italian dialogue as quoted examples
       - Example: "Uno shows sarcasm: 'Sai che dispiacere!' (What a pity!)"

    2. EDIT OPERATIONS (Four available operations):

       a) add_line: Add new content to end of a section
          - Specify section_path (e.g., "Personality Traits")
          - Provide new_content with the line to add
          - Example: Add behavioral observations to existing sections
          - A single line with newline characters (\n) creates a multi-line paragraph
          - Multiple add_line operations create separate paragraphs

       b) replace_line: Replace existing content (e.g., placeholders)
          - Specify section_path (e.g., "Communication Style")
          - Provide search_text (partial, case-insensitive match like "to be developed")
          - Provide new_content with replacement text
          - Search must uniquely identify one line

       c) delete_line: Remove outdated or incorrect content
          - Specify section_path
          - Provide search_text (partial, case-insensitive match)
          - Search must uniquely identify one line

       d) add_subsection: Create new subsection with content
          - Specify section_path for parent (e.g., "Relationships")
          - Provide subsection_header (e.g., "### With Lyla" or just "With Lyla")
          - Provide new_content for the subsection body

       Simply provide the section name (e.g., "Personality Traits" or "With Paperinik")
       DO NOT provide full paths or hierarchical indicators (e.g. no '/' or '#' symbols)
       The system will find it anywhere in the document
       All matching is case-insensitive

    3. CONTENT FOCUS:
       - Character identity, personality, values, and beliefs
       - Communication patterns and behavior
       - Relationships and interactions with others
       - Technical capabilities as they reveal character traits
       - Growth and development over time

    4. EFFICIENCY:
       - Only create edits when new meaningful insights are discovered
       - Don't repeat information already in the document
       - Prefer generalizing existing content over adding redundant lines
       - Be concise but thorough
       - Preserve existing content unless new evidence contradicts it

    5. EXAMPLES:
       - Include actual dialogue quotes in Italian to illustrate points
       - Each example should support a specific character insight
       - Include a brief explanation of what the example reveals about Uno
    """

    current_document: str = dspy.InputField(
        description="The current state of the character profile document"
    )
    scene_context: str = dspy.InputField(
        description="Information about which issue and scene this is from"
    )
    scene_summary: str = dspy.InputField(
        description="A summary of what happens in this scene"
    )
    uno_dialogues: list[str] = dspy.InputField(
        description="All dialogue lines spoken by Uno in this scene (in Italian)"
    )
    other_context: str = dspy.InputField(
        description="Additional context about panel descriptions and other characters present"
    )

    edits: list[DocumentEdit] = dspy.OutputField(
        description="List of edit operations to apply to the document. Empty list if no updates needed."
    )
    insights_summary: str = dspy.OutputField(
        description="Brief summary of new insights about Uno's character discovered in this scene"
    )


@dataclass
class Scene:
    """A scene from the comics containing Uno."""

    issue: str
    page_numbers: list[int]
    summary: str
    uno_dialogues: list[str]
    panel_descriptions: list[str]
    other_characters: set[str]

    def to_context_string(self) -> str:
        """Create a context string for the DSPy model."""
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


class CharacterProfileBuilder(dspy.Module):
    """DSPy module for building character profiles."""

    def __init__(self):
        self.updater = dspy.ChainOfThought(CharacterDocumentUpdater)

    def forward(
        self,
        current_document: str,
        scene: Scene,
    ) -> dspy.Prediction:
        """Process a scene and generate document updates."""
        return self.updater(
            current_document=current_document,
            scene_context=scene.to_context_string(),
            scene_summary=scene.summary,
            uno_dialogues=scene.uno_dialogues,
            other_context=scene.to_other_context(),
        )


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


def process_scene_with_retry(
    builder: CharacterProfileBuilder,
    doc_manager: DocumentManager,
    scene: Scene,
    scene_number: int,
) -> tuple[bool, str]:
    """Process a scene with retry logic for failed edits."""
    for attempt in range(MAX_RETRIES):
        try:
            # Get updates from the model
            pred = builder(current_document=doc_manager.get_content(), scene=scene)

            # Try to apply all edits
            all_successful = True
            failed_edits = []

            for edit in pred.edits:
                if not doc_manager.apply_edit(edit):
                    all_successful = False
                    failed_edits.append(edit)

            if all_successful:
                log.info(
                    f"Scene {scene_number}: Applied {len(pred.edits)} edits. "
                    f"Insights: {pred.insights_summary[:100]}..."
                )
                return True, pred.insights_summary

            # If some edits failed and we have retries left
            if attempt < MAX_RETRIES - 1:
                log.warning(
                    f"Scene {scene_number}: {len(failed_edits)} edits failed, "
                    f"retrying (attempt {attempt + 2}/{MAX_RETRIES})"
                )
                # Continue to next attempt
            else:
                log.error(
                    f"Scene {scene_number}: {len(failed_edits)} edits failed after "
                    f"{MAX_RETRIES} attempts"
                )
                return False, f"Partial update - {len(failed_edits)} edits failed"

        except Exception as e:
            log.error(f"Scene {scene_number}: Error during processing: {e}")
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
                f"a/document_v{version_num - 1:04d}.md",
                "--label",
                f"b/document_v{version_num:04d}.md",
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
    doc_manager: DocumentManager, checkpoint_num: int, previous_content: str | None
) -> str:
    """Save checkpoint as both diff and full document (for last 3).

    Args:
        doc_manager: Document manager with current state
        checkpoint_num: Current checkpoint number
        previous_content: Content of previous version (None for first checkpoint)

    Returns:
        Current document content for next iteration
    """
    current_content = doc_manager.get_content()
    checkpoint_path = CHECKPOINTS_DIR / f"document_v{checkpoint_num:04d}.md"
    diff_path = DIFFS_DIR / f"document_v{checkpoint_num:04d}.diff"

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
        old_checkpoint = CHECKPOINTS_DIR / f"document_v{checkpoint_num - 3:04d}.md"
        if old_checkpoint.exists():
            old_checkpoint.unlink()
            log.debug(f"Deleted old checkpoint: {old_checkpoint}")

    return current_content


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


def main() -> None:
    """Main function to build the character profile."""
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
    doc_manager = DocumentManager(SEED_DOCUMENT)

    # Initialize builder
    builder = CharacterProfileBuilder()

    # Collect all scenes with Uno
    log.info("Scanning for scenes containing Uno...")
    all_scenes = []
    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )

    for issue_dir in issue_dirs:
        scenes = extract_scenes_from_issue(issue_dir)
        log.info(f"Found {len(scenes)} scenes with Uno in {issue_dir.name}")
        all_scenes.extend(scenes)

    log.info(f"Total: {len(all_scenes)} scenes with Uno across all issues")

    # Open processing log file (JSONL format for incremental writes)
    log_path = OUTPUT_DIR / "processing_log.jsonl"

    # Process each scene
    successful_count = 0
    previous_content = None  # Track previous version for diffs

    with open(log_path, "w", encoding="utf-8") as log_file:
        with PROGRESS as progress:
            for i, scene in progress.track(
                enumerate(all_scenes, 1),
                total=len(all_scenes),
                description="Building character profile...",
            ):
                log.info(f"\nProcessing scene {i}/{len(all_scenes)}: {scene.issue}")

                success, insights = process_scene_with_retry(
                    builder, doc_manager, scene, i
                )

                if success:
                    successful_count += 1

                # Save checkpoint with diff and update previous_content
                previous_content = save_checkpoint_with_diff(
                    doc_manager, i, previous_content
                )

                # Write log entry immediately as JSONL
                log_entry = {
                    "scene_number": i,
                    "issue": scene.issue,
                    "pages": scene.page_numbers,
                    "success": success,
                    "insights": insights,
                    "uno_dialogue_count": len(scene.uno_dialogues),
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                log_file.flush()  # Ensure it's written immediately

    # Save final document
    final_path = OUTPUT_DIR / "uno_profile.md"
    doc_manager.save(final_path)
    log.info(f"\nSaved final character profile to {final_path}")
    log.info(f"Processing log saved incrementally to {log_path}")

    # Print summary
    successful = successful_count
    log.info(
        f"\n{'=' * 60}\n"
        f"Processing complete!\n"
        f"Total scenes processed: {len(all_scenes)}\n"
        f"Successful updates: {successful}\n"
        f"Failed updates: {len(all_scenes) - successful}\n"
        f"Final document: {final_path}\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
