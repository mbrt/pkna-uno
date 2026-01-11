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

# Stage 4: Compress profile into tiered versions
./compress_character_profile.py
# Processes v2/uno_profile.md → v3/uno_profile_tier{1,2,3}.md

# Stage 5: Test character impersonation
./generate_from_character_profile.py
# Interactive chat or automated testing
```

### Character Impersonation Testing

```bash
# Interactive chat (default: Tier 1 Core profile)
./generate_from_character_profile.py

# Use different profile
./generate_from_character_profile.py output/character-profile/uno/v3/uno_profile_tier2.md
./generate_from_character_profile.py output/character-profile/uno/v4/uno_profile.md

# Automated testing with predefined questions
./generate_from_character_profile.py --test english   # English test set
./generate_from_character_profile.py --test italian   # Italian test set

# Custom test questions
./generate_from_character_profile.py --questions "Who are you?" "Do you need sleep?"

# Test with different profile
./generate_from_character_profile.py output/character-profile/uno/v3/uno_profile_tier3.md --test english

# Validate profile structure
./restructure_profile.py
```

### Wiki-Augmented Character Generation

The `generate_with_wiki.py` script extends character impersonation with wiki knowledge retrieval. The model can search the wiki (479 markdown files, 3.7MB) to verify facts about characters, events, and technology while maintaining character personality.

**Key Features**:
- Character profile + wiki tools (4 search/retrieval functions)
- Manual function calling with full tool call logging
- Conversations saved to `output/test-conversations/` with `wiki_enabled: true` metadata
- Gemini models only (no Hugging Face backend)

**Wiki Tools Available**:
1. `search_wiki_content(keywords)` - Fast keyword search across all wiki files
2. `list_wiki_categories()` - List available wiki categories and subdirectories
3. `get_wiki_file_summary(file_path)` - Get file summary (first pass, ~200 tokens)
4. `read_wiki_file(file_path, section)` - Get full file content (second pass)

```bash
# Interactive chat with wiki tools (default: Tier 1 Core profile)
./generate_with_wiki.py

# Use different profile
./generate_with_wiki.py output/character-profile/uno/v3/uno_profile_tier2.md

# Automated testing with predefined questions
./generate_with_wiki.py --test english   # English test set
./generate_with_wiki.py --test italian   # Italian test set

# Custom test questions
./generate_with_wiki.py --questions "Who is Xadhoom?" "What happened in PKNA issue 5?"

# Test with different profile
./generate_with_wiki.py output/character-profile/uno/v3/uno_profile_tier3.md --test english
```

**How Wiki Tools Work**:
- Model automatically searches wiki when uncertain about facts
- Uses ripgrep for fast keyword search (~10ms across 479 files)
- Two-stage retrieval: summaries first, full content on demand
- Security: path validation prevents accessing files outside wiki directory
- Manual function calling ensures all tool calls are logged

**Conversation JSON Format**:
Saved conversations include complete tool call logging:
```json
{
  "metadata": {
    "character": "Uno",
    "wiki_enabled": true,
    "tool_calls_count": 2,
    ...
  },
  "messages": [
    {
      "role": "user",
      "content": "Who is Xadhoom?",
      ...
    },
    {
      "role": "tool",
      "tool_name": "search_wiki_content",
      "arguments": {"keywords": "Xadhoom", "max_results": 5},
      "result": "Found 12 matches...",
      ...
    },
    {
      "role": "assistant",
      "content": "Ah, Xadhoom. An extraordinary being...",
      "tool_calls": [
        {
          "tool": "search_wiki_content",
          "arguments": {"keywords": "Xadhoom"},
          "result": "Found 12 matches..."
        }
      ],
      ...
    }
  ]
}
```

**Success Metrics for Wiki-Augmented Tests**:
- ✅ Model searches wiki when uncertain (e.g., for "Highclean" → searches → "Non lo so")
- ✅ Uses wiki facts accurately (e.g., "Xadhoom" → searches → provides factual answer)
- ✅ Character personality preserved when citing wiki information
- ✅ Proper Italian/English mixing maintained
- ✅ Tool calls logged in conversation JSON

**Predefined test questions** (for both English and Italian):
1. Identity: "Who are you?" / "Chi sei?"
2. Appearance: "Describe your appearance" / "Descrivi il tuo aspetto"
3. Sleep test: "Do you need sleep?" / "Hai bisogno di dormire?" (should say NO)
4. Paperinik: "What do you think of Paperinik?" / "Cosa pensi di Paperinik?"
5. **Hallucination test**: "Tell me about Highclean" / "Parlami della Highclean" (should say "Non lo so")
6. Everett: "What's your relationship with Everett Ducklair?" / "Qual è il tuo rapporto con Everett Ducklair?"

**Success metrics for test results**:
- ✅ No invented entities, companies, or scenarios
- ✅ Correct "Non lo so" response for Highclean (known hallucination from v2)
- ✅ No biological needs mentioned (no sleep, food, rest)
- ✅ Short responses (2-4 sentences typical)
- ✅ Character-consistent personality (sarcastic, witty, protective)
- ✅ Proper Italian expression usage with inline translations for English conversations
  - Short Italian kept: "socio", "ciao", "Non lo so"
  - Longer phrases translated: "Dormire? (Sleep?) What a primitive concept!"

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
- Output: `output/character-profile/uno/v2/uno_profile.md` (75k tokens, detailed profile)

**Stage 4: Profile Compression** (`compress_character_profile.py`)
- Compresses v2 profile (75k tokens) into 3 optimized tiers
- Uses DSPy with Gemini for intelligent summarization
- Preserves character essence while drastically reducing token count
- Output: `output/character-profile/uno/v3/`
  - **Tier 1 (Core)**: ~1.5k tokens (51x compression) - Essential traits, default for chat
  - **Tier 2 (Extended)**: ~2k tokens (38x compression) - More detail when needed
  - **Tier 3 (Full)**: 75k tokens (copy of v2) - Reference/fine-tuning

**Tier profile structure**:
- Essential Identity (cannot be turned off, no biological needs)
- Core Personality (top 10-25 distinctive traits)
- Communication Style (Italian expressions, "socio", avian puns)
- Behavioral Guidelines (explicit Do/Don't lists)
- Key Relationships (Paperinik, Everett, Due)

**Profile validation** (`restructure_profile.py`):
- Validates required sections present
- Checks behavioral guidelines exist
- Counts Italian dialogue examples
- Generates validation report

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

### Data Flow

```
input/pkna/                     # Comic page images (JPG)
input/wiki/fandom/              # Issue summaries (markdown)
  ↓ [dspy-extract-full.py]
output/dspy-extract-full/v2/    # Extracted panel data (JSON)
  ↓ [make-scenes-full.py]
output/scenes/                  # Scene-level JSON files
  ↓ [build_character_profile.py]
output/character-profile/uno/v2/uno_profile.md  # Full profile (75k tokens)
  ↓ [compress_character_profile.py]
output/character-profile/uno/v3/                # Tiered profiles
  ├── uno_profile_tier1.md      # Core (1.5k tokens)
  ├── uno_profile_tier2.md      # Extended (2k tokens)
  └── uno_profile_tier3.md      # Full (75k tokens)
  ↓ [generate_from_character_profile.py]
output/test-conversations/      # Chat logs (JSON)
```

### Key Python Modules

- **DSPy Signatures**: Define LLM input/output schemas with instructions
  - `PlotExtractor`, `PageExtractor`, `CharacterDocumentUpdater`, `DialogueExtraction`
  - `ProfileCompressor` (v3 compression)
- **DSPy Modules**: Composable components (`dspy.ChainOfThought`, `dspy.Predict`)
- **Pydantic Models**: Type-safe data structures (`Panel`, `DialogueLine`, `Scene`)
- **Document Structure Manager**: Hierarchical markdown parser/editor (in `build_character_profile.py`)
- **Natural Sorting**: Custom key function for issue ordering
- **Profile Compression**: DSPy-based intelligent summarization (in `compress_character_profile.py`)
- **Profile Validation**: Structure and content validation (in `restructure_profile.py`)
- **Test Runner**: Non-interactive testing mode with predefined questions (in `generate_from_character_profile.py`)

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

## Character Profile Optimization (v2 → v3)

### The Problem

The original full character profile (`v2/uno_profile.md`) had significant issues:
- **Too large**: 75,000 tokens (~49k words)
- **Caused hallucinations**: Model invented scenarios (Highclean, Flagstarr, Department 51), characters, and contradicted the profile
- **Poor context usage**: Profile dominated context window, leaving little room for conversation history
- **Over-elaborate responses**: Long, tangential responses instead of concise character-appropriate ones
- **Language mixing**: Untranslated Italian phrases in English conversations, English words in Italian responses

### The Solution (4-Phase Approach)

**Phase 1: Enhanced System Prompt** ✅
- Added explicit constraints against hallucination
- Clear behavioral guidelines (Do/Don't lists)
- Language-aware instructions (inline translations for longer Italian phrases)
- Response length constraints (2-4 sentences typical)

**Phase 2: Profile Compression** ✅
- Created tiered profiles using DSPy-powered summarization
- Tier 1 (Core): 1,456 tokens (51.4x compression) - default for chat
- Tier 2 (Extended): 1,959 tokens (38.2x compression) - more detail
- Tier 3 (Full): 74,802 tokens - preserved for fine-tuning
- Profile structure emphasizes patterns over individual examples

**Phase 3: Profile Restructuring** ✅
- Added "Behavioral Guidelines" with explicit Do/Don't sections
- Consolidated redundant traits across sections
- Pattern-focused descriptions over granular examples
- Validation script to ensure structure compliance

**Phase 4: RAG (Conditional)** - Not needed
- Retrieval-Augmented Generation considered but not implemented
- Phases 1-3 achieved success metrics without it

### Results

**Before optimization (v2)**:
- Profile: 75k tokens in context
- Hallucinations: Frequent (invented entities, contradictions)
- Responses: Over-elaborate, tangential
- Language: Poor mixing/translation

**After optimization (v3)**:
- Profile: 1.5k tokens in context (98% reduction)
- Hallucinations: Minimal (proper "Non lo so" responses)
- Responses: Short, focused, character-consistent
- Language: Proper inline translations

### Testing Character Quality

When testing character impersonation, look for:

**Red flags (hallucinations)**:
- ❌ Invented entities (Highclean, Flagstarr, etc.)
- ❌ Claiming to need sleep/rest/biological needs
- ❌ Contradicting profile facts
- ❌ Over-elaborate descriptions or scenarios
- ❌ Untranslated long Italian phrases in English conversations

**Success indicators**:
- ✅ "Non lo so" for unknown information
- ✅ Short responses (2-4 sentences typical)
- ✅ Character-consistent traits (sarcasm, wit, protective)
- ✅ Proper Italian usage: "socio", "ciao" (no translation) + "Dormire? (Sleep?)" (inline translation)
- ✅ Factual accuracy about character identity and constraints

## Important Notes

- **Resume Capability**: All stages support resuming interrupted runs (skip existing outputs)
- **Parallel Processing**: Use ThreadPoolExecutor for I/O-bound tasks (max 8 workers)
- **Incremental Logging**: Use JSONL format for long-running processes
- **Error Handling**: Retry logic with exponential backoff for LLM API calls
- **Version Control**: Pipeline outputs are versioned (v2, v3) in separate directories
- **Profile Tiers**: Always use Tier 1 (core) for chat unless testing requires more detail
- **Test Mode**: Use `--test` flag for automated quality verification before production use
