# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PKNA-Uno is a multi-stage ML pipeline for extracting structured data from Italian comic books (PKNA - Paperinik New Adventures), building a comprehensive character profile for "Uno" (an AI character), and enabling AI-powered character impersonation. The project uses DSPy for declarative LLM programming with Google Vertex AI (Gemini models).

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
# Run all tests
pytest

# Run specific test file
pytest tests/test_document_structure.py

# Run tests with verbose output
pytest -v

# Run specific test by name
pytest tests/test_document_structure.py::test_modify_line
```

### Main Pipeline Scripts

```bash
# Stage 1: Extract structured data from comic pages
./dspy-extract-full.py
# Processes: input/pkna/ → output/dspy-extract-full/v2/

# Stage 2: Group panels into scenes and create dataset
./make-scenes-full.py        # Creates individual scene JSON files
./make-scenes-dataset.py     # Creates CSV dataset for ML training

# Stage 3: Build character profile from scenes
./build_character_profile.py
# Processes scenes → output/character-profile/uno/v2/uno_profile.md

# Stage 4: Test character impersonation
./generate_from_character_profile.py
# Interactive chat using the generated profile
```

### Optimization and Review

```bash
# Optimize DSPy prompts using MIPROv2
./dspy-extract-optimize.py

# Launch review UI for extracted data (Streamlit)
streamlit run review-extract-eval.py
```

### Utility Commands

```bash
# Count tokens in files
./count_tokens_tiktoken.py <file_path>

# Test Vertex AI connection
./vertex-test.py

# Create backup of outputs
./backup.sh
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
- Output: `output/character-profile/uno/v2/uno_profile.md` (327KB final profile)

**Stage 4: Character Impersonation** (`generate_from_character_profile.py`)
- Loads character profile and creates system instructions for LLM
- Interactive chat loop using Google GenAI
- Output: `output/test-conversations/` (JSON conversation logs)

### Data Flow

```
input/pkna/                     # Comic page images (JPG)
input/wiki/fandom/              # Issue summaries (markdown)
  ↓ [dspy-extract-full.py]
output/dspy-extract-full/v2/    # Extracted panel data (JSON)
  ↓ [make-scenes-full.py]
output/scenes/                  # Scene-level JSON files
  ↓ [build_character_profile.py]
output/character-profile/uno/v2/uno_profile.md
  ↓ [generate_from_character_profile.py]
output/test-conversations/      # Chat logs
```

### Key Python Modules

- **DSPy Signatures**: Define LLM input/output schemas with instructions
  - `PlotExtractor`, `PageExtractor`, `CharacterDocumentUpdater`, `DialogueExtraction`
- **DSPy Modules**: Composable components (`dspy.ChainOfThought`, `dspy.Predict`)
- **Pydantic Models**: Type-safe data structures (`Panel`, `DialogueLine`, `Scene`)
- **Document Structure Manager**: Hierarchical markdown parser/editor (in `build_character_profile.py`)
- **Natural Sorting**: Custom key function for issue ordering

### Configuration

- **Environment**: `.env` file with Vertex AI credentials
  - `VERTEX_AI_CREDS`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
- **Model**: `vertex_ai/gemini-3-flash-preview` (configurable via `MODEL_NAME`)
- **LLM Parameters**: temperature=1.0, top_p=0.95, top_k=64, max_tokens=65535

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

### Document Editing Pattern

The character profile builder uses structured edits instead of regenerating the entire document:
1. Parse markdown into hierarchical structure (Section → Subsection → Line)
2. Generate edit operations (add/replace/delete lines, add subsections)
3. Apply edits with validation
4. Save checkpoint + diff after each scene
5. Retry on failure (max 3 attempts)

### Multi-line Paragraphs

Document structure preserves multi-line paragraphs as single Line objects:
```python
@dataclass
class Line:
    content: str  # May contain newlines for multi-line paragraphs
    indent: int = 0
```

### Checkpoint and Diff Management

- Keep last 3 full document checkpoints (rotating)
- Save all incremental diffs (never delete)
- Use Unix `diff -u` via subprocess for clean diffs
- JSONL logging for incremental progress tracking

## Language and Text Handling

- **Analysis**: Write in English for descriptions, summaries, metadata
- **Dialogue**: Preserve original Italian text in quotes
- **Text Normalization**:
  - Convert all-caps → normal caps
  - Merge hyphenated words at line breaks
  - Handle Italian accents correctly
- **Character Identification**: Use dialogue patterns and bubble styles to identify speakers

## Testing

- Use pytest with table-driven tests
- Tests located in `tests/` directory
- Key test files:
  - `test_document_structure.py`: Document editing logic (add/replace/delete lines, subsections)
  - `test_checkpoint_diff.py`: Diff generation and checkpoint management
  - `test_build_character_profile.py`: Natural sorting for issue names

## Important Notes

- **Resume Capability**: All stages support resuming interrupted runs (skip existing outputs)
- **Parallel Processing**: Use ThreadPoolExecutor for I/O-bound tasks (max 8 workers)
- **Incremental Logging**: Use JSONL format for long-running processes
- **Error Handling**: Retry logic with exponential backoff for LLM API calls
- **Version Control**: Pipeline outputs are versioned (v2, v3) in separate directories
