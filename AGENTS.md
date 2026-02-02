# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PKNA-Uno is a multi-stage ML pipeline for extracting structured data from Italian comic books (PKNA - Paperinik New Adventures), building a comprehensive character profile for "Uno" (an AI character), and enabling AI-powered character impersonation. The project uses DSPy for declarative LLM programming with Google Vertex AI (Gemini models), and `google.genai` directly.

## Commands

### Environment Setup

```bash
# Install dependencies (project uses uv for dependency management)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running Tests

```bash
make test
```

## Architecture

### Pipeline Stages

**Stage 1: Comic Data Extraction** (`dspy-extract-full.py`)
- Input: Comic book page images in `input/pkna/pkna-{N}/`
- Uses DSPy signatures: `PlotExtractor` (issue summary → characters/events) and `PageExtractor` (page image → panels)
- Maintains context across pages (previous panels, character names, key events)
- Parallel processing with ThreadPoolExecutor (max 8 workers)
- Resume capability: skips already-processed pages
- Output: `output/dspy-extract-full/v2/pkna-{N}/page_{XXX}.json`

**Stage 2: Scene Grouping** (`make-scenes-full.py`, `make-scenes-dataset.py`)
- Groups panels into scenes based on `is_new_scene` flag
- Filters for scenes containing target character (e.g., "Uno")
- Formats dialogues as conversation pairs (character = assistant, others = human)
- Output: `output/scenes/` (JSON) and `output/dataset/dataset-2.csv`

**Stage 3: Character Profile Building** (`build_character_profile.py`)
- Iteratively builds character profile using `CharacterDocumentUpdater` DSPy module
- Document structure: hierarchical markdown with sections/subsections
- Edit operations: `add_line`, `replace_line`, `delete_line`, `add_subsection`
- Checkpoint system: saves last 3 full documents + all incremental diffs
- Natural sorting for issues (pkna-0-2 before pkna-1)
- Output: `output/character-profile/uno/v2/uno_profile.md` (75k tokens, detailed profile)

**Stage 4: Profile Compression** (`compress_character_profile.py`)
- Compresses v2 profile (75k tokens) into 3 optimized tiers
- Uses DSPy with Gemini for intelligent summarization
- Preserves character essence while drastically reducing token count
- Output: `output/character-profile/uno/v3/`
  - **Tier 1 (Core)**: ~1.5k tokens (51x compression) - Essential traits, default for chat
  - **Tier 2 (Extended)**: ~2k tokens (38x compression) - More detail when needed
  - **Tier 3 (Full)**: 75k tokens (copy of v2) - Reference/fine-tuning

**Stage 5: Character Impersonation** (`generate_from_character_profile.py`)
- Loads character profile and creates enhanced system instructions
- System prompt includes:
  - Explicit constraints against hallucination
  - Language-aware response guidelines (inline translations)
  - Behavioral Do/Don't lists
  - Character consistency rules
- Supports interactive chat or automated testing
- Profile path as argument (defaults to Tier 1 Core profile)
- **Conversation annotations**: Prompts for quality notes after each session
  - Track hallucinations, character consistency, issues
  - Saved in JSON metadata for quality tracking over time
- Output: `output/test-conversations/` (JSON conversation logs with annotations)

### Configuration

- **Environment**: `.env` file with Vertex AI credentials
- **Model**: `vertex_ai/gemini-3-flash-preview` (configurable via `MODEL_NAME`)

## Development Patterns

### DSPy Usage

1. Define signatures with clear instructions and field descriptions
2. Use `dspy.ChainOfThought` for complex reasoning tasks
3. Use `dspy.Predict` for straightforward extraction
4. Enable usage tracking: `dspy.configure(lm=lm, track_usage=True)`
5. Save metadata in output JSON: `lm.usage` and `lm.history`

### Logging

All scripts use Rich console for beautiful CLI output:
```python
from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)]
)
```

### Progress Tracking

Use Rich Progress for long-running operations:
```python
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

PROGRESS = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)
```

### Resume Capability

All pipeline stages check for existing output files and skip already-processed items:
```python
output_path = OUT_ROOT / issue_id / f"page_{page_num:03d}.json"
if output_path.exists():
    log.info(f"Skipping {issue_id}/page_{page_num:03d} (already exists)")
    continue
```

## Important Notes

- **Resume Capability**: All stages support resuming interrupted runs (skip existing outputs)
- **Parallel Processing**: Use ThreadPoolExecutor for I/O-bound tasks (max 8 workers)
- **Incremental Logging**: Use JSONL format for long-running processes
- **Error Handling**: Retry logic with exponential backoff for LLM API calls
- **Version Control**: Pipeline outputs are versioned (v2, v3) in separate directories
