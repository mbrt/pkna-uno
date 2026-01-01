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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / "v1"
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

### With Paperinik/Paperino
To be developed based on interactions.

### With Other Characters
To be developed based on interactions.

## Knowledge and Capabilities

To be developed based on demonstrated abilities.

## Behavioral Patterns

To be developed based on recurring actions and responses.

## Character Growth and Development

To be developed based on character arc across issues.

## Example Dialogue

To be collected from scenes (preserved in original Italian).
"""


class DocumentEdit(BaseModel):
    """A single edit operation on the character document."""

    operation: Literal["modify", "append_to_section"] = Field(
        description="Type of edit: modify existing text or append to a section"
    )
    search_text: str | None = Field(
        default=None,
        description="Text to find and replace (for modify operation). Must match exactly including whitespace.",
    )
    replacement_text: str = Field(
        description="New text to insert. For modify, this replaces search_text. For append_to_section, this is appended."
    )
    section_name: str | None = Field(
        default=None,
        description="Section header to append to (for append_to_section operation). E.g., '## Personality Traits'",
    )


class CharacterDocumentUpdater(dspy.Signature):
    """Update a character profile document based on new scene information.

    CRITICAL INSTRUCTIONS:

    1. LANGUAGE:
       - Write all analysis and descriptions in English
       - Preserve original Italian dialogue as quoted examples
       - Example: "Uno shows sarcasm: 'Sai che dispiacere!' (What a pity!)"

    2. EDIT OPERATIONS:
       - Use 'modify' to replace existing placeholder text (e.g., "To be developed...")
       - Use 'append_to_section' to add new insights to existing sections
       - For 'modify': search_text must match EXACTLY (including whitespace)
       - For 'append_to_section': specify the section header (e.g., "## Personality Traits")

    3. CONTENT FOCUS:
       - Character identity, personality, values, and beliefs
       - Communication patterns and behavior
       - Relationships and interactions with others
       - Technical capabilities as they reveal character traits
       - Growth and development over time

    4. EFFICIENCY:
       - Only create edits when new meaningful insights are discovered
       - Don't repeat information already in the document
       - Be concise but thorough
       - Preserve existing content unless new evidence contradicts it

    5. EXAMPLES:
       - Include actual dialogue quotes in Italian to illustrate points
       - Each example should support a specific character insight
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
        return f"Panel descriptions: {' | '.join(self.panel_descriptions[:3])}"


class DocumentManager:
    """Manages the character profile document and applies edits."""

    def __init__(self, initial_content: str):
        self.content = initial_content

    def apply_edit(self, edit: DocumentEdit) -> bool:
        """Apply a single edit to the document. Returns True if successful."""
        if edit.operation == "modify":
            if edit.search_text is None:
                log.error("modify operation requires search_text")
                return False
            if edit.search_text not in self.content:
                log.warning(f"Search text not found: {edit.search_text[:50]}...")
                return False
            self.content = self.content.replace(
                edit.search_text, edit.replacement_text, 1
            )
            return True

        elif edit.operation == "append_to_section":
            if edit.section_name is None:
                log.error("append_to_section operation requires section_name")
                return False

            # Find the section
            if edit.section_name not in self.content:
                log.warning(f"Section not found: {edit.section_name}")
                return False

            # Find the next section or end of document
            lines = self.content.split("\n")
            section_idx = None
            for i, line in enumerate(lines):
                if line.strip() == edit.section_name:
                    section_idx = i
                    break

            if section_idx is None:
                return False

            # Find the end of this section (next ## header or end of document)
            end_idx = len(lines)
            for i in range(section_idx + 1, len(lines)):
                if lines[i].startswith("## "):
                    end_idx = i
                    break

            # Insert the new content before the next section
            lines.insert(end_idx, "")
            lines.insert(end_idx + 1, edit.replacement_text)
            self.content = "\n".join(lines)
            return True

        return False

    def get_content(self) -> str:
        """Get the current document content."""
        return self.content

    def save(self, path: Path) -> None:
        """Save the document to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.content)


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

    log_file = open(log_path, "w", encoding="utf-8")
    with PROGRESS as progress:
        for i, scene in progress.track(
            enumerate(all_scenes, 1),
            total=len(all_scenes),
            description="Building character profile...",
        ):
            log.info(f"\nProcessing scene {i}/{len(all_scenes)}: {scene.issue}")

            success, insights = process_scene_with_retry(builder, doc_manager, scene, i)

            if success:
                successful_count += 1

            # Save checkpoint
            checkpoint_path = CHECKPOINTS_DIR / f"document_v{i:04d}.md"
            doc_manager.save(checkpoint_path)

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

    # Close log file explicitly
    log_file.close()

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
