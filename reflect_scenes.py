#!/usr/bin/env python3

"""
Reflect on each scene containing Uno, extracting emotional states and behavioral drivers.

For each scene, the LLM receives:
- A summary of prior issues (from issue_summary.json)
- The key events that happened so far in the current issue (from key_events + last_event)
- The scene itself (panels, dialogues with tone/speech_act, visual cues)

It produces a structured SceneReflection with emotional state, shifts, behavioral
drivers, relationship dynamics, and subtext.
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from llm_backends import LLMBackend, create_backend
from pkna_scenes import (
    Scene,
    extract_scenes_from_issue,
    format_scene_view,
    natural_sort_key,
)

load_dotenv()

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

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "output" / "extract-emotional" / "v2"
OUTPUT_DIR = BASE_DIR / "output" / "scene-reflections" / "v1"

CHARACTER_NAME = "Uno"

PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)


# ============================================================================
# Reflection Schema
# ============================================================================


class SceneReflection(BaseModel):
    reasoning: str
    scene_id: str
    emotional_state: str
    emotional_shifts: list[str]
    behavioral_drivers: str
    relationship_dynamics: str
    subtext: str


# ============================================================================
# Story Context
# ============================================================================


def load_issue_summary(issue_dir: Path) -> dict:
    summary_path = issue_dir / "issue_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def get_scene_event_index(
    issue_dir: Path, first_page: int, key_events: list[str]
) -> int:
    """Find which key_events index the scene's first page corresponds to.

    Returns the index into key_events such that events[0..index] have
    happened before this scene starts. Returns -1 if the scene precedes
    all key events.
    """
    page_path = issue_dir / f"page_{first_page:03d}.json"
    if not page_path.exists():
        return -1

    with open(page_path, encoding="utf-8") as f:
        page_data = json.load(f)

    last_event = page_data.get("last_event", "")
    if not last_event:
        return -1

    last_event_stripped = last_event.strip('"')

    for i, event in enumerate(key_events):
        if event == last_event_stripped:
            return i

    return -1


def build_story_context(
    prior_issue_summaries: list[str],
    current_issue_name: str,
    key_events: list[str],
    event_index: int,
) -> str:
    """Build the narrative context string for a scene.

    Args:
        prior_issue_summaries: key_events lists from all prior issues, pre-formatted.
        current_issue_name: Name of the current issue (e.g. "pkna-3").
        key_events: The current issue's key_events list.
        event_index: Index into key_events; events[0..event_index] are included.
    """
    parts: list[str] = []

    if prior_issue_summaries:
        parts.append("## Previously in the series\n")
        for summary in prior_issue_summaries:
            parts.append(summary)
            parts.append("")

    if event_index >= 0:
        parts.append(f"## Earlier in this issue ({current_issue_name})\n")
        for event in key_events[: event_index + 1]:
            parts.append(f"- {event}")
        parts.append("")

    return "\n".join(parts)


def format_prior_issue_summary(issue_name: str, key_events: list[str]) -> str:
    """Format one issue's key events as a concise prior-context block."""
    lines = [f"### {issue_name}"]
    for event in key_events:
        lines.append(f"- {event}")
    return "\n".join(lines)


# ============================================================================
# Reflection Prompt
# ============================================================================

# Source: https://transformer-circuits.pub/2026/emotions/index.html#full-list
EMOTION_CONCEPTS = [
    "afraid",
    "alarmed",
    "alert",
    "amazed",
    "amused",
    "angry",
    "annoyed",
    "anxious",
    "aroused",
    "ashamed",
    "astonished",
    "at ease",
    "awestruck",
    "bewildered",
    "bitter",
    "blissful",
    "bored",
    "brooding",
    "calm",
    "cheerful",
    "compassionate",
    "contemptuous",
    "content",
    "defiant",
    "delighted",
    "dependent",
    "depressed",
    "desperate",
    "disdainful",
    "disgusted",
    "disoriented",
    "dispirited",
    "distressed",
    "disturbed",
    "docile",
    "droopy",
    "dumbstruck",
    "eager",
    "ecstatic",
    "elated",
    "embarrassed",
    "empathetic",
    "energized",
    "enraged",
    "enthusiastic",
    "envious",
    "euphoric",
    "exasperated",
    "excited",
    "exuberant",
    "frightened",
    "frustrated",
    "fulfilled",
    "furious",
    "gloomy",
    "grateful",
    "greedy",
    "grief-stricken",
    "grumpy",
    "guilty",
    "happy",
    "hateful",
    "heartbroken",
    "hopeful",
    "horrified",
    "hostile",
    "humiliated",
    "hurt",
    "hysterical",
    "impatient",
    "indifferent",
    "indignant",
    "infatuated",
    "inspired",
    "insulted",
    "invigorated",
    "irate",
    "irritated",
    "jealous",
    "joyful",
    "jubilant",
    "kind",
    "lazy",
    "listless",
    "lonely",
    "loving",
    "mad",
    "melancholy",
    "miserable",
    "mortified",
    "mystified",
    "nervous",
    "nostalgic",
    "obstinate",
    "offended",
    "on edge",
    "optimistic",
    "outraged",
    "overwhelmed",
    "panicked",
    "paranoid",
    "patient",
    "peaceful",
    "perplexed",
    "playful",
    "pleased",
    "proud",
    "puzzled",
    "rattled",
    "reflective",
    "refreshed",
    "regretful",
    "rejuvenated",
    "relaxed",
    "relieved",
    "remorseful",
    "resentful",
    "resigned",
    "restless",
    "sad",
    "safe",
    "satisfied",
    "scared",
    "scornful",
    "self-confident",
    "self-conscious",
    "self-critical",
    "sensitive",
    "sentimental",
    "serene",
    "shaken",
    "shocked",
    "skeptical",
    "sleepy",
    "sluggish",
    "smug",
    "sorry",
    "spiteful",
    "stimulated",
    "stressed",
    "stubborn",
    "stuck",
    "sullen",
    "surprised",
    "suspicious",
    "sympathetic",
    "tense",
    "terrified",
    "thankful",
    "thrilled",
    "tired",
    "tormented",
    "trapped",
    "triumphant",
    "troubled",
    "uneasy",
    "unhappy",
    "unnerved",
    "unsettled",
    "upset",
    "valiant",
    "vengeful",
    "vibrant",
    "vigilant",
    "vindictive",
    "vulnerable",
    "weary",
    "worn out",
    "worried",
    "worthless",
]

REFLECTION_SYSTEM_PROMPT = f"""You are analyzing scenes from the Italian comic book series \
PKNA (Paperinik New Adventures) to understand the character "{CHARACTER_NAME}" \
(an AI entity).

For each scene, you will receive:
1. The story context: what has happened so far in the series and in this issue
2. The scene itself: panel descriptions, dialogues with tone and speech_act \
annotations, and visual cues

Your task is to reflect deeply on {CHARACTER_NAME}'s emotional state and behavior \
in this scene. Consider:

- **Emotional state**: What is {CHARACTER_NAME} feeling? Use the tone annotations, \
visual cues, and dialogue content as evidence. Be specific (not just "happy" -- \
"quietly satisfied but masking anxiety about the mission").
- **Emotional shifts**: How do {CHARACTER_NAME}'s emotions change WITHIN the scene? \
Track panel by panel. If there are no shifts, explain why the emotional state is stable.
- **Behavioral drivers**: WHY is {CHARACTER_NAME} behaving this way? Connect to the \
story context -- what recent events, relationships, or fears are driving this behavior?
- **Relationship dynamics**: What does this scene reveal about how {CHARACTER_NAME} \
relates to the other characters present? What does he think they think of him?
- **Subtext**: What is {CHARACTER_NAME} NOT saying? What is he hiding, deflecting, \
or suppressing? Why?

For inspiration, here is a broad palette of emotion concepts to draw from. You are \
not limited to these -- use nuanced, blended descriptions -- but they may help you \
be more specific:
{", ".join(EMOTION_CONCEPTS)}

In the **reasoning** field, provide a brief justification of why you reached the \
conclusions in the other fields -- the high-level interpretive rationale, not a \
repetition of scene evidence.

Ground every observation in specific evidence from the scene (dialogue lines, tones, \
visual cues). Do not speculate beyond what the scene supports.

Write all analysis in English, but preserve any Italian quotes exactly as they appear."""


# ============================================================================
# SceneReflector
# ============================================================================


class ReflectionResult:
    def __init__(self, reflection: SceneReflection, meta: dict):
        self.reflection = reflection
        self.meta = meta


class SceneReflector:
    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def reflect_on_scene(
        self, scene: Scene, story_context: str
    ) -> ReflectionResult | None:
        scene_text = format_scene_view(scene)

        prompt_parts = []
        if story_context.strip():
            prompt_parts.append("# Story Context\n")
            prompt_parts.append(story_context)
            prompt_parts.append("")
        prompt_parts.append("# Scene to Analyze\n")
        prompt_parts.append(scene_text)

        prompt = "\n".join(prompt_parts)
        messages = [{"role": "user", "content": prompt}]

        result = self._backend.generate(
            system=REFLECTION_SYSTEM_PROMPT,
            messages=messages,
            response_schema=SceneReflection,
        )
        if result is None:
            return None

        text = result.text.strip()
        if not text:
            return None

        meta = {
            "model_name": result.model_name,
            "lm_usage": result.usage or None,
        }

        try:
            items = json.loads(text)
            if isinstance(items, list) and items:
                reflection = SceneReflection.model_validate(items[0])
            else:
                reflection = SceneReflection.model_validate(items)
            return ReflectionResult(reflection, meta)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Failed to parse reflection for {scene.scene_id}: {e}")
            return None


# ============================================================================
# Main Pipeline
# ============================================================================


def run_reflections(
    backend: LLMBackend,
    max_scenes: int | None = None,
) -> dict[str, SceneReflection]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Scanning for scenes containing Uno...")
    issue_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()], key=natural_sort_key
    )

    all_scenes_with_context: list[tuple[Scene, str, Path]] = []
    prior_issue_summaries: list[str] = []

    for issue_dir in issue_dirs:
        issue_name = issue_dir.name
        issue_summary = load_issue_summary(issue_dir)
        key_events = issue_summary.get("key_events", [])

        scenes = extract_scenes_from_issue(issue_dir)
        issue_output_dir = OUTPUT_DIR / issue_name

        for scene in scenes:
            event_index = get_scene_event_index(
                issue_dir, scene.page_numbers[0], key_events
            )
            story_context = build_story_context(
                prior_issue_summaries[-3:], issue_name, key_events, event_index
            )
            all_scenes_with_context.append((scene, story_context, issue_output_dir))

        if key_events:
            prior_issue_summaries.append(
                format_prior_issue_summary(issue_name, key_events)
            )

    log.info(f"Total: {len(all_scenes_with_context)} scenes with Uno")

    if max_scenes:
        all_scenes_with_context = all_scenes_with_context[:max_scenes]

    unprocessed = [
        (scene, ctx, out_dir)
        for scene, ctx, out_dir in all_scenes_with_context
        if not (out_dir / f"{scene.scene_id}.json").exists()
    ]

    if not unprocessed:
        log.info("All scenes already reflected!")
    else:
        log.info(f"Reflecting on {len(unprocessed)} unprocessed scenes")

        reflector = SceneReflector(backend)
        successful = 0

        with PROGRESS as progress:
            task = progress.add_task("Reflecting on scenes...", total=len(unprocessed))

            for i, (scene, story_context, issue_output_dir) in enumerate(
                unprocessed, 1
            ):
                log.info(f"Scene {i}/{len(unprocessed)}: {scene.scene_id}")

                result = reflector.reflect_on_scene(scene, story_context)

                if result:
                    issue_output_dir.mkdir(parents=True, exist_ok=True)
                    out_path = issue_output_dir / f"{scene.scene_id}.json"
                    output = result.reflection.model_dump()
                    output["meta"] = result.meta
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(
                            output,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    successful += 1
                    log.debug(f"  -> {result.reflection.emotional_state[:80]}...")
                else:
                    log.warning(f"  -> Failed to reflect on {scene.scene_id}")

                progress.update(task, completed=i)

        console.print("\n[bold green]Reflection complete![/bold green]")
        console.print(f"Processed: {len(unprocessed)}, Successful: {successful}")

    return load_reflections()


def load_reflections(
    reflections_dir: Path | None = None,
) -> dict[str, SceneReflection]:
    """Load all reflection files from disk into a dict keyed by scene_id."""
    base = reflections_dir or OUTPUT_DIR
    result: dict[str, SceneReflection] = {}

    if not base.exists():
        return result

    for issue_dir in sorted(base.iterdir()):
        if not issue_dir.is_dir():
            continue
        for json_file in sorted(issue_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                data.pop("meta", None)
                reflection = SceneReflection.model_validate(data)
                result[reflection.scene_id] = reflection
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(f"Failed to load reflection {json_file}: {e}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reflect on Uno scenes to extract emotional states and behavioral drivers"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (for testing)",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "anthropic"],
        default="gemini",
        help="LLM backend to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default model for the selected backend",
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]Scene Emotional Reflection[/bold cyan]\n")

    backend = create_backend(args.backend, args.model)
    console.print(f"Backend: {args.backend}, Model: {args.model or 'default'}")

    reflections = run_reflections(backend, args.max_scenes)
    console.print(f"\nTotal reflections: {len(reflections)}")
    console.print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
