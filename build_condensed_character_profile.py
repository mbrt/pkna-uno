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
- Uses simple search-and-replace edits instead of complex hierarchical editing
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
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
To be developed with 15-25 most distinctive traits.

## Communication Style
To be developed with speech patterns, linguistic markers, and visual interface details.

## Behavioral Guidelines

### What Uno Does
To be developed with characteristic behaviors and capabilities.

### What Uno Doesn't Do
To be developed with explicit constraints and limitations.

## Key Relationships
To be developed for major characters (Paperinik, Everett, Due, Lyla, etc.).
"""


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
    replace_text: str = Field(
        description="New text to replace the search_text with"
    )
    reason: str = Field(
        description="Brief explanation of why this edit is being made"
    )


class CondensedProfileUpdater(dspy.Signature):
    """Update condensed character profile with new insights using search-and-replace.

    CRITICAL INSTRUCTIONS:

    1. EDITING APPROACH - SEARCH AND REPLACE:
       - Find exact text in the current document
       - Replace it with updated/enhanced text
       - Each search_text must be UNIQUE in the document
       - Use multi-line text blocks for section replacements

    2. SIZE MANAGEMENT (MOST IMPORTANT):
       - Current document is {current_token_count} tokens
       - Target max: {target_max_tokens} tokens
       - If approaching target, CONSOLIDATE:
         * Replace multiple similar traits with one generalized statement
         * Remove weaker examples, keep only the best
         * Merge redundant sections

    3. WHAT TO EDIT:
       - Replace "To be developed..." placeholders with actual content
       - Add new traits to existing sections (replace the whole section)
       - Update sections to add new insights
       - Consolidate redundant content

    4. EDIT EXAMPLES:

       Example 1 - Replace placeholder:
       search_text: "To be developed based on observed core facts and constraints."
       replace_text: "Uno is an artificial intelligence housed in Ducklair Tower..."

       Example 2 - Add to existing section:
       search_text: "**Protective Caretaker:** Acts as a guardian for Paperinik."
       replace_text: "**Protective Caretaker:** Acts as a guardian for Paperinik.\\n**Sharp Sarcasm:** Uses dry wit to deflate egos."

       Example 3 - Consolidate traits:
       search_text: "**Trait A:** Description.\\n**Trait B:** Similar description.\\n**Trait C:** Also similar."
       replace_text: "**General Pattern:** Consolidated description covering A, B, and C."

    5. QUALITY OVER QUANTITY:
       - Add insights ONLY if they're truly new and distinctive
       - Better to have 15 well-defined traits than 30 vague ones
       - Each edit should meaningfully improve the profile

    6. LANGUAGE:
       - Write all descriptions in English
       - Preserve Italian dialogue examples with English translations in parentheses
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


# ============================================================================
# Simple Document Manager with Search/Replace
# ============================================================================


class SimpleDocumentManager:
    """Manages document with simple search-and-replace operations."""

    def __init__(self, initial_content: str):
        self._content = initial_content

    def apply_edit(self, edit: ProfileEdit) -> bool:
        """Apply a search-and-replace edit. Returns True if successful."""
        if edit.search_text not in self._content:
            log.warning(
                f"Search text not found: '{edit.search_text[:100]}...'"
            )
            return False

        # Count occurrences to ensure uniqueness
        count = self._content.count(edit.search_text)
        if count > 1:
            log.warning(
                f"Search text appears {count} times (not unique): '{edit.search_text[:100]}...'"
            )
            return False

        # Apply replacement
        self._content = self._content.replace(edit.search_text, edit.replace_text)
        log.debug(f"Applied edit: {edit.reason}")
        return True

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


def update_profile_with_retry(
    updater: dspy.Module,
    doc_manager: SimpleDocumentManager,
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
                # Update current_content for retry
                current_content = doc_manager.get_content()
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


def save_checkpoint(doc_manager: SimpleDocumentManager, issue_id: str) -> None:
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
    doc_manager = SimpleDocumentManager(CONDENSED_SEED_DOCUMENT)

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
