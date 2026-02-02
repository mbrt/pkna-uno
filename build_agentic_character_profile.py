#!/usr/bin/env python3

"""
Build a character profile using agentic document editing.

This script uses Google GenAI directly (no DSPy) and gives the model agentic
control over reading and editing the character profile document through
`read_document` and `edit_document` tools.

Key features:
- Automatic function calling with tool call logging
- Scene-by-scene processing like build_character_profile.py (v2)
- Token management with condensation prompts when over threshold
- Checkpoint/diff saving for incremental progress
"""

import json
import logging
import random
import re
import subprocess
import tempfile
import time
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
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

load_dotenv()

# Configure logging
console = Console(stderr=True)
# Silent root, use module loggers
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
# Explicit module logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Settings
MODEL_NAME = "gemini-3-flash-preview"
CHARACTER_NAME = "Uno"
TARGET_MAX_TOKENS = 7000
ENCODING_NAME = "cl100k_base"
VERSION_TAG = "v7"
MAX_TOOL_ITERATIONS = 64
MAX_CONDENSE_ITERATIONS = 5
MAX_STRUCTURE_FIX_ATTEMPTS = 3

# Retry settings for API calls
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0
API_TIMEOUT_SECONDS = 300  # 5 minutes per API call

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "dspy-extract-full" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / VERSION_TAG
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

# Seed document structure (matches build_condensed_character_profile.py)
SEED_DOCUMENT = """# Uno - Character Profile

## Essential Identity

To be developed based on observed core facts and constraints.

## Core Personality

To be developed with 30-40 most distinctive traits.

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


def extract_section_headers(content: str) -> set[str]:
    """Extract all section headers from markdown content using regex."""
    return set(re.findall(r"^#+\s+.+$", content, re.MULTILINE))


# Compute required sections from seed document
REQUIRED_SECTIONS = extract_section_headers(SEED_DOCUMENT)


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
# LineBasedDocument Class
# ============================================================================


class LineBasedDocument:
    """Stores document as list of lines for efficient editing."""

    def __init__(self, content: str):
        """Initialize from string content."""
        self._lines: list[str] = content.split("\n")

    def get_content(self) -> str:
        """Get full document as string."""
        return "\n".join(self._lines)

    def get_lines(self, offset: int = 1, limit: int | None = None) -> str:
        """Get numbered lines from the document.

        Args:
            offset: 1-indexed line number to start reading (default: 1)
            limit: Number of lines to read (default: all remaining)

        Returns:
            Numbered lines in format "1: content\\n2: content\\n..."
        """
        if offset < 1:
            offset = 1

        start_idx = offset - 1  # Convert to 0-indexed
        if start_idx >= len(self._lines):
            return ""

        if limit is None:
            end_idx = len(self._lines)
        else:
            end_idx = min(start_idx + limit, len(self._lines))

        result_lines = []
        for i in range(start_idx, end_idx):
            line_num = i + 1  # 1-indexed line number
            result_lines.append(f"{line_num}: {self._lines[i]}")

        return "\n".join(result_lines)

    def edit(self, old_text: str, new_text: str) -> tuple[bool, str]:
        """Apply a search-and-replace edit.

        Args:
            old_text: Exact text to find and replace
            new_text: Replacement text

        Returns:
            Tuple of (success: bool, message: str)
        """
        content = self.get_content()

        count = content.count(old_text)
        if count == 0:
            return False, "Error: Text not found in document."
        if count > 1:
            return (
                False,
                f"Error: Multiple occurrences found ({count} matches). "
                "Please use more specific text.",
            )

        # Apply the replacement
        new_content = content.replace(old_text, new_text)
        self._lines = new_content.split("\n")

        line_count = len(self._lines)
        token_count = count_tokens(new_content)

        return (
            True,
            f"Edit applied. Document now has {line_count} lines and {token_count} tokens.",
        )

    def save(self, path: Path) -> None:
        """Save the document to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_content())

    def validate_structure(self, required_sections: set[str]) -> tuple[bool, list[str]]:
        """Check that all required section headers are present.

        Args:
            required_sections: Set of section headers that must be present.

        Returns:
            Tuple of (is_valid, missing_sections).
        """
        current_sections = extract_section_headers(self.get_content())
        missing = required_sections - current_sections
        return len(missing) == 0, sorted(missing)


# ============================================================================
# Scene Data Structures (reused from build_character_profile.py)
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
# Checkpoint and Diff Management (reused from build_character_profile.py)
# ============================================================================


def generate_and_save_diff(
    old_content: str, new_content: str, diff_path: Path, version_num: int
) -> None:
    """Generate unified diff and save to file using diff -u command."""
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

        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(diff_output)
    finally:
        Path(old_path).unlink()
        Path(new_path).unlink()


def save_checkpoint_with_diff(
    document: LineBasedDocument, checkpoint_num: int, previous_content: str | None
) -> str:
    """Save checkpoint as both diff and full document (for last 3)."""
    current_content = document.get_content()
    checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num:03d}.md"
    diff_path = DIFFS_DIR / f"checkpoint_{checkpoint_num:03d}.diff"

    if previous_content is not None:
        generate_and_save_diff(
            previous_content, current_content, diff_path, checkpoint_num
        )
    else:
        generate_and_save_diff(
            SEED_DOCUMENT, current_content, diff_path, checkpoint_num
        )

    document.save(checkpoint_path)

    if checkpoint_num > 3:
        old_checkpoint = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_num - 3:03d}.md"
        if old_checkpoint.exists():
            old_checkpoint.unlink()

    tokens = count_tokens(current_content)
    log.debug(
        f"Saved checkpoint {checkpoint_num}: {path_str(checkpoint_path)} ({tokens} tokens)"
    )

    return current_content


def find_latest_checkpoint() -> tuple[int, str] | None:
    """Find the latest checkpoint and return its number and content.

    Returns:
        Tuple of (checkpoint_number, content) if a checkpoint exists,
        None otherwise.
    """
    if not CHECKPOINTS_DIR.exists():
        return None

    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("checkpoint_*.md"))
    if not checkpoint_files:
        return None

    # Get the latest checkpoint (highest number)
    latest = checkpoint_files[-1]

    # Extract checkpoint number from filename (checkpoint_XXX.md)
    try:
        checkpoint_num = int(latest.stem.split("_")[1])
    except (IndexError, ValueError):
        log.warning(f"Could not parse checkpoint number from {latest}")
        return None

    content = latest.read_text(encoding="utf-8")
    return checkpoint_num, content


# ============================================================================
# System Prompt
# ============================================================================


def get_system_prompt(target_tokens: int) -> str:
    """Generate system prompt for the agentic document editor."""
    return f"""You are building a character profile for "Uno", an AI character from the Italian comic book series PKNA (Paperinik New Adventures).

You will be given scenes one at a time. For each scene, you should:
1. Use read_document() to examine the current document
2. Use edit_document() to update sections with new insights from the scene
3. When you're done editing for this scene, respond with a one-line summary of what you updated

## Available Tools

- read_document(offset=1, limit=None): Read document lines. Returns numbered lines with a header showing total lines and tokens.
- edit_document(old_text, new_text): Replace exact text. The old_text must be unique in the document.

## Document Structure Guidelines

The document has sections marked with ## headers:
- Essential Identity
- Core Personality
- Communication Style
- Behavioral Guidelines (with subsections ### What Uno Does, ### What Uno Doesn't Do)
- Key Relationships

When editing:
- KEEP section headers intact
- Replace placeholder text like "To be developed..." with actual content
- Add new observations to appropriate sections
- Preserve existing content unless you're consolidating or correcting
- You can add new subsections if needed, especially for relationships, but keep the overall structure consistent

## Content Guidelines

- Write descriptions in ENGLISH
- Preserve Italian dialogue as quoted examples with translations
- Focus on personality traits, communication patterns, relationships
- Look for recurring patterns, not just single instances
- Include specific quotes that illustrate key traits
- Pay particular attention to interactions with Paperinik, Everett, Due

## Size Management

Target document size: {target_tokens} tokens

When the document approaches or exceeds this target:
- Consolidate similar traits into broader patterns
- Keep only the most illustrative examples
- Merge redundant content
- Remove weaker observations in favor of stronger ones
- Do not over-condense; aim for a 10-20% reduction per pass

If you're asked to condense the document, focus on preserving the most distinctive and well-supported insights.

## Language Handling

- All descriptions and analysis in English
- Italian dialogue preserved as: "Quote here" (English translation)
- Note speech patterns like nicknames, expressions, catchphrases

When you're finished updating the document for a scene, provide a brief summary of what you added or changed. If you made no changes (nothing new to add), say so.
"""


# ============================================================================
# Scene Processing with Agentic Editing
# ============================================================================


def format_scene_prompt(scene: Scene) -> str:
    """Format a scene as a prompt for the model."""
    dialogues_text = "\n".join(f"- {d}" for d in scene.uno_dialogues)
    context = scene.to_context_string()
    panel_context = scene.to_other_context()

    return f"""Analyze this scene and update the character profile document.

**Scene Context:** {context}

**Scene Summary:** {scene.summary}

**Uno's Dialogues (Italian):**
{dialogues_text}

**Additional Context:** {panel_context}

Use read_document() to see the current profile, then use edit_document() to add any new insights about Uno's character. When done, provide a brief summary of your updates."""


# ============================================================================
# SceneProcessor Class
# ============================================================================


def _str_info(a: str) -> str:
    return f"{count_tokens(a)} tokens, {len(a.splitlines())} lines"


class SceneProcessor:
    """Processes scenes and updates the character profile document.

    Encapsulates the document and provides tool methods as bound methods,
    eliminating the need for global state.
    """

    def __init__(self, client: genai.Client, document: LineBasedDocument):
        self._client = client
        self._document = document
        self._config = GenerateContentConfig(
            system_instruction=get_system_prompt(TARGET_MAX_TOKENS),
            temperature=0.8,
            top_p=0.95,
            tools=[self.read_document, self.edit_document],
        )

    def read_document(self, offset: int = 1, limit: int | None = None) -> str:
        """Read lines from the character profile document.

        Use this to examine the current document content before making edits.

        Args:
            offset: 1-indexed line number to start reading (default: 1)
            limit: Number of lines to read (default: all remaining)

        Returns:
            Document content with numbered lines and stats header:
            "Total: X lines, Y tokens\\n1: content\\n2: content\\n..."
        """
        log.debug(f"[read_document] offset={offset}, limit={limit}")

        content = self._document.get_content()
        total_lines = len(self._document._lines)
        total_tokens = count_tokens(content)

        header = f"Total: {total_lines} lines, {total_tokens} tokens\n"
        lines = self._document.get_lines(offset, limit)

        return header + lines

    def edit_document(self, old_text: str, new_text: str) -> str:
        """Edit the character profile document by replacing text.

        The old_text must be found exactly once in the document for the edit
        to succeed. If you need to replace text that appears multiple times,
        include more surrounding context to make it unique.

        Args:
            old_text: Exact text to find and replace (must be unique)
            new_text: Replacement text

        Returns:
            Success message with new document stats, or error message
        """
        log.debug(
            f"[edit_document] old_text='{_str_info(old_text)}', "
            f"new_text='{_str_info(new_text)}'"
        )

        ok, message = self._document.edit(old_text, new_text)
        if not ok:
            log.warning(f"[edit_document] Edit failed: {message}")
        return message

    def _generate_with_retry(
        self, conversation: list[Content]
    ) -> GenerateContentResponse | None:
        """Call generate_content with retry on resource exhausted and timeout errors.

        Returns:
            The response content on success, None if all retries failed.

        Raises:
            Exception: If a non-retryable error occurs.
        """
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=MODEL_NAME,
                    contents=conversation,  # type: ignore[arg-type]
                    config=self._config,
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

                # Add jitter: ±25% of backoff
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

    def process_scene(self, scene: Scene, scene_number: int) -> tuple[bool, str]:
        """Process a single scene using agentic editing.

        After processing the scene, checks if the document exceeds the token limit
        and runs condensation passes (continuing the same conversation) until
        it's within the target.

        Returns:
            Tuple of (success, summary_or_error)
        """
        scene_prompt = format_scene_prompt(scene)
        conversation: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=scene_prompt)])
        ]

        try:
            response = self._generate_with_retry(conversation)
            if response is None:
                return False, "API call failed after retries"

            # The SDK handles automatic function calling
            # The final response.text contains the model's summary
            summary = response.text or "No summary provided"

            # Update conversation with the model's response for potential continuation
            if response.candidates and response.candidates[0].content:
                conversation.append(response.candidates[0].content)

            # Validate structure after editing, retry with feedback if broken
            is_valid, missing = self._document.validate_structure(REQUIRED_SECTIONS)
            fix_attempt = 0

            while not is_valid and fix_attempt < MAX_STRUCTURE_FIX_ATTEMPTS:
                fix_attempt += 1
                log.warning(
                    f"Scene {scene_number}: Structure broken (attempt {fix_attempt}/"
                    f"{MAX_STRUCTURE_FIX_ATTEMPTS}). Missing: {missing}"
                )

                fix_prompt = f"""Your last edit broke the document structure. The following required section headers are missing:

{chr(10).join(f"- {s}" for s in missing)}

Please use read_document() to examine the current document, then use edit_document() to restore the missing sections. You must preserve the exact header format (e.g., "## Essential Identity", "### What Uno Does").

When done, confirm what you fixed."""

                conversation.append(
                    Content(role="user", parts=[Part.from_text(text=fix_prompt)])
                )

                fix_response = self._generate_with_retry(conversation)
                if fix_response is None:
                    log.error("Structure fix failed after retries")
                    return False, "Structure fix API call failed"

                if fix_response.candidates and fix_response.candidates[0].content:
                    conversation.append(fix_response.candidates[0].content)

                is_valid, missing = self._document.validate_structure(REQUIRED_SECTIONS)

            if not is_valid:
                log.error(
                    f"Scene {scene_number}: Structure still broken after "
                    f"{MAX_STRUCTURE_FIX_ATTEMPTS} fix attempts. Missing: {missing}"
                )
                return False, f"Structure validation failed: missing {missing}"

            log.info(f"Scene {scene_number}: {summary[:100]}...")

            # Condense if over token limit (continues the same conversation)
            self._condense_if_needed(conversation)

            return True, summary

        except Exception as e:
            log.error(f"Scene {scene_number}: Error - {e}")
            return False, str(e)

    def _condense_if_needed(self, conversation: list[Content]) -> None:
        """Condense the document if it exceeds the token limit.

        Continues the existing conversation with condensation prompts until
        the document is within TARGET_MAX_TOKENS or MAX_CONDENSE_ITERATIONS
        is reached.

        Args:
            conversation: The conversation history to continue.
        """
        for iteration in range(MAX_CONDENSE_ITERATIONS):
            current_tokens = count_tokens(self._document.get_content())
            if current_tokens <= TARGET_MAX_TOKENS:
                return

            log.info(
                f"Document at {current_tokens} tokens "
                f"(target: {TARGET_MAX_TOKENS}), condensing (pass {iteration + 1})..."
            )

            condense_prompt = f"""The document is currently {current_tokens} tokens, which is above the target of {TARGET_MAX_TOKENS} tokens.

Please condense the document by:
1. Consolidating similar traits into broader patterns
2. Removing redundant observations
3. Merging content where appropriate
4. Prioritize preserving communication style with the key relationships (Paperinik, Everett, Due)

Make sure to:
- Not over-condense; aim for a 10-20% reduction
- Not generalizing to the point of making the profile vague

Use read_document() to see the current content, then use edit_document() to make your changes. Aim to reduce the document to around {TARGET_MAX_TOKENS} tokens while preserving the most important character insights.

When done, provide a brief summary of what you condensed."""

            # Continue the conversation with the condense prompt
            conversation.append(
                Content(role="user", parts=[Part.from_text(text=condense_prompt)])
            )

            try:
                response = self._generate_with_retry(conversation)
                if response is None:
                    log.error("Condensation failed after retries")
                    return

                # Update conversation with the model's response
                if response.candidates and response.candidates[0].content:
                    conversation.append(response.candidates[0].content)

                summary = response.text or "No summary"
                new_tokens = count_tokens(self._document.get_content())
                log.info(
                    f"Condensation: {current_tokens} -> {new_tokens} tokens. {summary}"
                )

                # If no progress was made, stop trying
                if new_tokens >= current_tokens:
                    log.warning("Condensation made no progress, stopping.")
                    return

            except Exception as e:
                log.error(f"Condensation error: {e}")
                return

        # If we exit the loop, we've hit max iterations
        final_tokens = count_tokens(self._document.get_content())
        if final_tokens > TARGET_MAX_TOKENS:
            log.warning(
                f"Document still at {final_tokens} tokens after "
                f"{MAX_CONDENSE_ITERATIONS} condensation passes."
            )


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main function to build the character profile."""
    console.print(
        f"\n[bold cyan]Agentic Character Profile Builder ({VERSION_TAG})[/bold cyan]\n"
    )

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume from
    start_scene = 1
    previous_content: str | None = None

    checkpoint_info = find_latest_checkpoint()
    if checkpoint_info:
        checkpoint_num, checkpoint_content = checkpoint_info
        document = LineBasedDocument(checkpoint_content)
        start_scene = checkpoint_num + 1
        previous_content = checkpoint_content
        log.info(
            f"Resuming from checkpoint {checkpoint_num} "
            f"({count_tokens(checkpoint_content)} tokens)"
        )
    else:
        document = LineBasedDocument(SEED_DOCUMENT)
        # Save seed document only on fresh start
        seed_path = OUTPUT_DIR / "seed_document.md"
        with open(seed_path, "w", encoding="utf-8") as f:
            f.write(SEED_DOCUMENT)
        log.info(f"Starting fresh, saved seed document to {path_str(seed_path)}")

    client = genai.Client(http_options=HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000))
    processor = SceneProcessor(client, document)

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

    # Check if already complete
    if start_scene > len(all_scenes):
        log.info("All scenes already processed!")
        return

    remaining_scenes = len(all_scenes) - start_scene + 1
    log.info(
        f"Processing scenes {start_scene} to {len(all_scenes)} ({remaining_scenes} remaining)"
    )

    # Open processing log file (append mode for resume support)
    log_path = OUTPUT_DIR / "processing_log.jsonl"

    # Process each scene
    successful_count = 0

    with open(log_path, "a", encoding="utf-8") as log_file:
        with PROGRESS as progress:
            task = progress.add_task(
                "Building character profile...",
                total=len(all_scenes),
                completed=start_scene - 1,
            )
            for i in range(start_scene, len(all_scenes) + 1):
                scene = all_scenes[i - 1]  # Convert to 0-indexed
                log.info(f"\nProcessing scene {i}/{len(all_scenes)}: {scene.issue}")

                success, summary = processor.process_scene(scene, i)

                if success:
                    successful_count += 1

                # Save checkpoint with diff
                previous_content = save_checkpoint_with_diff(
                    document, i, previous_content
                )

                # Write log entry
                log_entry = {
                    "scene_number": i,
                    "issue": scene.issue,
                    "pages": scene.page_numbers,
                    "success": success,
                    "summary": summary,
                    "uno_dialogue_count": len(scene.uno_dialogues),
                    "tokens_after": count_tokens(document.get_content()),
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                log_file.flush()

                progress.update(task, completed=i)

    # Save final document
    final_path = OUTPUT_DIR / "uno_profile.md"
    document.save(final_path)

    final_tokens = count_tokens(document.get_content())
    final_words = len(document.get_content().split())

    # Print summary
    scenes_this_run = len(all_scenes) - start_scene + 1
    console.print()
    console.print("[bold green]Profile Building Complete![/bold green]\n")
    console.print(f"Total scenes: {len(all_scenes)}")
    console.print(f"Scenes processed this run: {scenes_this_run}")
    console.print(f"Successful updates: {successful_count}")
    console.print(f"Failed updates: {scenes_this_run - successful_count}")
    console.print()
    console.print(f"[bold]Final profile:[/bold] {path_str(final_path)}")
    console.print(f"  Size: {final_tokens:,} tokens (~{final_words:,} words)")
    console.print(f"  Target: {TARGET_MAX_TOKENS:,} tokens")
    if final_tokens <= TARGET_MAX_TOKENS:
        console.print("  [green]Within target[/green]")
    else:
        overage = final_tokens - TARGET_MAX_TOKENS
        console.print(f"  [yellow]{overage:,} tokens over target[/yellow]")

    console.print(f"\nProcessing log: {path_str(log_path)}")
    console.print(f"Checkpoints: {CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    main()
