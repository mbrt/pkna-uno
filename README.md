# PKNA Uno

Multi-stage ML pipeline for extracting structured data from Italian comic books (PKNA - Paperinik New Adventures), building a comprehensive character profile for "Uno" (an AI character), and enabling AI-powered character impersonation using Google Vertex AI (Gemini models) with DSPy.

## Quick Start

### Character Impersonation (Interactive Chat)

```bash
# Interactive chat with Tier 1 (core) profile (default)
./generate_from_character_profile.py

# Try different profile tiers
./generate_from_character_profile.py --tier extended  # More detail
./generate_from_character_profile.py --tier full      # Full 75k token profile
```

### Automated Testing

```bash
# Run predefined English test questions
./generate_from_character_profile.py --test english

# Run Italian test questions
./generate_from_character_profile.py --test italian

# Run both English and Italian tests
./generate_from_character_profile.py --test both

# Custom test questions
./generate_from_character_profile.py --questions "Who are you?" "Do you need sleep?"
```

## Development

### Running Tests

The project uses `make` for common development tasks. All commands use `uv run` to execute tools.

```bash
# Run all checks (format check, lint, typecheck) and tests
make test

# Format all Python files
make format

# Run linters only
make lint

# Clean Python cache files
make clean
```

### Code Quality Tools

The project uses the following tools (all run via `uv run`):
- **ruff**: Code formatting and linting
- **ty**: Static type checking
- **pytest**: Test runner

All tools are configured in `pyproject.toml` and run automatically with `make test`.

## Pipeline Stages

### Stage 1: Comic Data Extraction
Extract structured panel data from comic page images.

```bash
./dspy-extract-full.py
```

**Input**: `input/pkna/pkna-{N}/` (page images)
**Output**: `output/dspy-extract-full/v2/pkna-{N}/page_*.json`

### Stage 2: Scene Grouping
Group panels into scenes and filter for character appearance.

```bash
./make-scenes-full.py         # Individual scene JSON files
./make-scenes-dataset.py      # CSV dataset for ML
```

**Input**: `output/dspy-extract-full/v2/`
**Output**: `output/scenes/` (JSON), `output/dataset/dataset-2.csv`

### Stage 3: Character Profile Building
Build comprehensive character profile from scenes.

```bash
./build_character_profile.py
```

**Input**: `output/scenes/`
**Output**: `output/character-profile/uno/v2/uno_profile.md` (75k tokens)

### Stage 4: Profile Compression (NEW)
Compress large profile into tiered versions optimized for chat.

```bash
./compress_character_profile.py
```

**Input**: `output/character-profile/uno/v2/uno_profile.md`
**Output**: Three tiered profiles in `output/character-profile/uno/v3/`:
- **Tier 1 (Core)**: ~1.5k tokens - Essential identity, core traits, key relationships
- **Tier 2 (Extended)**: ~2k tokens - More detail, additional nuances
- **Tier 3 (Full)**: 75k tokens - Original profile (for fine-tuning)

### Stage 5: Character Impersonation
Interactive chat using the generated profile.

```bash
./generate_from_character_profile.py
```

**Input**: `output/character-profile/uno/v3/uno_profile_tier1.md` (default)
**Output**: `output/test-conversations/*.json`

**Conversation annotations**: After each conversation (interactive or test), you'll be prompted to add notes about:
- Quality issues or hallucinations observed
- Character consistency
- Specific problems or successes
- Any other observations

Annotations are saved in the conversation JSON under `metadata.annotation`.

## Character Profile Optimization

### The Problem

The original full character profile (`v2/uno_profile.md`) was:
- **Too large**: 75,000 tokens (~49k words)
- **Caused hallucinations**: Model invented scenarios, characters, and contradicted the profile
- **Poor context usage**: Profile dominated context window, leaving little room for conversation

### The Solution

We implemented a **4-phase optimization** to reduce hallucinations while preserving character quality:

#### Phase 1: Enhanced System Prompt ✅
Added explicit constraints and behavioral guidelines:
- Never invent people, places, or events not in profile
- Never describe appearance beyond what's stated
- Keep responses short (2-4 sentences)
- Language-aware responses (translate Italian phrases in English conversations)

#### Phase 2: Profile Compression ✅
Created tiered profiles using DSPy-powered summarization:

| Tier | Tokens | Compression | Use Case |
|------|--------|-------------|----------|
| **Tier 1 (Core)** | 1,456 | 51.4x | Default for chat - essential traits only |
| **Tier 2 (Extended)** | 1,959 | 38.2x | More detail when needed |
| **Tier 3 (Full)** | 74,802 | 1x | Reference/fine-tuning |

**Profile structure includes**:
- Essential Identity (cannot be turned off, no biological needs)
- Core Personality (top 10-25 most distinctive traits)
- Communication Style (Italian expressions, "socio", avian puns)
- Behavioral Guidelines (explicit Do/Don't lists)
- Key Relationships (Paperinik, Everett, Due)

#### Phase 3: Profile Restructuring ✅
Reorganized profiles with:
- "What Uno Does" / "What Uno Doesn't Do" sections
- Pattern-focused descriptions over individual examples
- Consolidated redundant traits

#### Phase 4: RAG (Conditional)
Retrieval-Augmented Generation - only if Phases 1-3 don't meet success metrics.

### Results

**Before optimization**:
- Profile: 75k tokens
- Hallucinations: Frequent (invented companies, characters, scenarios)
- Responses: Over-elaborate, contradictory

**After optimization**:
- Profile: 1.5k tokens (98% reduction)
- Hallucinations: Minimal (properly says "Non lo so" for unknown entities)
- Responses: Short, focused, character-consistent

## Character Profile Builder (Legacy - Stage 3)

The profile builder script:
1. Scans all extracted comic data from `output/dspy-extract-full/v2/`
2. Identifies and groups scenes containing Uno's dialogue
3. Uses a DSPy model to iteratively update a character profile document
4. Employs efficient line-based edits (add, remove, modify) rather than regenerating the entire document
5. Saves checkpoints after each scene for resume capability

### Purpose

The generated character profile serves as a "soul document" or "constitutional AI" document that can be used for:
- Fine-tuning an LLM to mimic Uno's personality and behavior
- Understanding Uno's character arc across the PKNA series
- Reference material for character analysis

### Output Format

The document is structured with sections inspired by Claude Opus's soul document:
- **Core Identity**: Who Uno is and their role
- **Personality Traits**: Character traits observed across scenes
- **Communication Style**: How Uno speaks and interacts
- **Values and Beliefs**: What Uno values and believes in
- **Relationships**: Interactions with other characters
- **Knowledge and Capabilities**: Technical abilities and knowledge
- **Behavioral Patterns**: Recurring behaviors and responses
- **Character Growth**: Development over time
- **Example Dialogue**: Actual quotes from the comics (in Italian)

### Usage

#### Prerequisites

Ensure you have:
- Completed comic data extraction in `output/dspy-extract-full/v2/`
- Vertex AI credentials configured in `.env` file
- Python virtual environment with dependencies installed

#### Running the Script

```bash
cd PKNA
source exporter/.venv/bin/activate
./exporter/build-character-profile.py
```

Or with explicit Python:

```bash
python exporter/build-character-profile.py
```

#### Output Files

All outputs are saved to `output/character-profile/uno/`:

- `uno_profile.md` - The final character profile document
- `seed_document.md` - The initial template used to start the process
- `processing_log.jsonl` - Metadata about each scene processed (JSON Lines format, written incrementally)
- `checkpoints/document_v{N}.md` - Snapshot after each scene (for debugging/resume)

### Configuration

Key settings in the script:

- **MODEL_NAME**: `vertex_ai/gemini-2.0-flash-exp` (can be changed to other models)
- **MAX_RETRIES**: 3 (number of retry attempts for failed edits)
- **CHARACTER_NAME**: "Uno" (can be adapted for other characters)

### How It Works

#### Scene Extraction

The script:
1. Reads all `page_*.json` files from each issue directory
2. Groups panels into scenes using the `is_new_scene` flag
3. Filters for scenes where Uno has at least one dialogue line
4. Extracts Uno's dialogues, panel descriptions, and other character names

#### Document Updates

For each scene:
1. The current document state is fed to the DSPy model along with scene data
2. The model analyzes the scene and generates edit operations
3. Edits are applied using search/replace logic (not line numbers)
4. If edits fail, the script retries up to MAX_RETRIES times
5. A checkpoint is saved after each scene

#### Edit Operations

Two types of edits are supported:

1. **modify**: Replace existing text (e.g., placeholder "To be developed...")
   - Requires exact match of `search_text`
   - Replaces with `replacement_text`

2. **append_to_section**: Add new content to a section
   - Finds the section header (e.g., "## Personality Traits")
   - Appends content before the next section

#### Language Handling

- All analysis and descriptions are written in **English**
- Original Italian dialogue is preserved as quoted examples
- Example: "Uno shows sarcasm: 'Sai che dispiacere!' (What a pity!)"

### Resume Capability

If the script is interrupted, you can:
1. Check the last successful checkpoint in `checkpoints/`
2. Note the scene number from `processing_log.json`
3. Modify the script to start from that scene (future enhancement)

### Troubleshooting

#### Common Issues

**"Search text not found"**
- The model generated a search string that doesn't exactly match the document
- The script will retry automatically
- Check logs to see which edits failed

**"Section not found"**
- The model tried to append to a non-existent section
- Verify the seed document has all expected sections
- The script will retry automatically

**Rate limiting / API errors**
- The script may need to be run in smaller batches
- Check Vertex AI quotas
- Consider adding delay between scenes if needed

#### Logs

All processing is logged with Rich formatting. Look for:
- **INFO**: Normal progress messages
- **WARNING**: Failed edits that will be retried
- **ERROR**: Critical failures

### Customization

#### For Other Characters

To build profiles for other characters:

1. Change `CHARACTER_NAME` in the script
2. Update the filter in `create_scene_from_panels()` to check for the new character
3. Adjust the seed document template as needed

#### Adjusting Scene Selection

Modify `create_scene_from_panels()` to:
- Filter by specific issues
- Require minimum dialogue count
- Include scenes where character is mentioned but doesn't speak

#### Different Model

Change `MODEL_NAME` to use a different LLM:
- `vertex_ai/gemini-3-flash-preview`
- `openai/gpt-4o`
- Any model supported by DSPy

### Statistics

The script processes:
- ~50 comic issues (pkna-0 through pkna-49)
- Hundreds of scenes containing Uno
- Thousands of dialogue lines

Estimated runtime: 1-3 hours depending on model and scene count.

### Related Scripts

- `dspy-extract-full.py` - Extracts structured data from comic pages
- `make-scenes-full.py` - Groups panels into scenes and filters for character
- `make-scenes-dataset.py` - Creates datasets from scenes

## New Scripts

### `compress_character_profile.py`
Compress the full character profile into tiered versions.

```bash
./compress_character_profile.py
```

**Features**:
- Uses DSPy with Gemini for intelligent summarization
- Preserves character essence while reducing token count
- Generates 3 tiers: Core (1.5k), Extended (2k), Full (75k)
- Saves metadata with compression ratios and preserved traits

**Output**: `output/character-profile/uno/v3/`

### `restructure_profile.py`
Validate profile structure and organization.

```bash
./restructure_profile.py
```

**Features**:
- Validates required sections are present
- Checks for behavioral guidelines (Do/Don't lists)
- Counts Italian dialogue examples
- Generates validation report

**Output**: `output/character-profile/uno/v3/validation_report.md`

### `generate_from_character_profile.py`
Chat with the character using their profile.

**Interactive mode** (default):
```bash
./generate_from_character_profile.py [--tier core|extended|full]
```

**Test mode** (non-interactive):
```bash
# Predefined test questions
./generate_from_character_profile.py --test english
./generate_from_character_profile.py --test italian
./generate_from_character_profile.py --test both

# Custom test questions
./generate_from_character_profile.py --questions "Question 1" "Question 2"
```

**Predefined test questions**:
- English: Identity, appearance, sleep, Paperinik, Highclean (hallucination test), Everett
- Italian: Same questions in Italian

**Output**: `output/test-conversations/conversation_uno_*.json`

## Configuration

### Environment Variables

Create a `.env` file with your Vertex AI credentials:

```bash
VERTEX_AI_CREDS=/path/to/credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1  # or your preferred region
```

### Model Settings

All scripts use `gemini-3-flash-preview` by default. You can modify:
- `MODEL_NAME` in each script
- Or pass `--model` argument to `generate_from_character_profile.py`

## Testing Character Quality

### Hallucination Tests

The test questions include known hallucinations from earlier versions:
- **Highclean**: A company that was invented by the model (should say "Non lo so")
- **Sleep/Rest**: Uno claimed to need sleep (should say he doesn't have biological needs)
- **Appearance**: Over-elaborate descriptions (should be brief and match profile)

### Language Handling

**English conversations**:
- Short Italian: "socio", "ciao", "Non lo so" (no translation)
- Longer phrases: "Dormire? (Sleep?) What a primitive concept!"
- Avoids untranslated Italian sentences

**Italian conversations**:
- Entirely in Italian
- No English words mixed in

### Success Metrics

A good test run should show:
- ✅ No invented entities, companies, or scenarios
- ✅ Correct "Non lo so" for unknown information
- ✅ Short responses (2-4 sentences typical)
- ✅ Character-consistent personality (sarcastic, witty, protective)
- ✅ Proper Italian expression usage
- ✅ Inline translations for English conversations

### Tracking Quality with Annotations

After each conversation, you'll be prompted to add notes:

```
Conversation Annotation
Add notes about this conversation (quality, hallucinations, issues, etc.)
Press Enter on empty line to finish, or Ctrl+C to skip

> Good character consistency
> Properly said "Non lo so" for Highclean
> One minor hallucination about physical appearance
>
```

**Example annotations**:
- "Hallucinated company name 'XYZ Corp'"
- "Perfect character consistency, no issues"
- "Over-elaborate response on question 3"
- "Mixed English in Italian conversation"

Annotations are saved in the JSON metadata for easy tracking of quality improvements across versions.

## Project Structure

```
pkna-uno/
├── input/
│   ├── pkna/                        # Comic page images
│   └── wiki/fandom/                 # Issue summaries
├── output/
│   ├── dspy-extract-full/v2/        # Extracted panel data
│   ├── scenes/                      # Grouped scenes
│   ├── dataset/                     # ML datasets
│   ├── character-profile/
│   │   └── uno/
│   │       ├── v2/                  # Full profile (75k tokens)
│   │       │   └── uno_profile.md
│   │       └── v3/                  # Tiered profiles (NEW)
│   │           ├── uno_profile_tier1.md  # Core (1.5k)
│   │           ├── uno_profile_tier2.md  # Extended (2k)
│   │           ├── uno_profile_tier3.md  # Full (75k)
│   │           └── *_metadata.json       # Tier metadata
│   └── test-conversations/          # Chat logs
├── build_character_profile.py       # Stage 3: Build profile
├── compress_character_profile.py    # Stage 4: Compress profile (NEW)
├── restructure_profile.py           # Validation script (NEW)
├── generate_from_character_profile.py  # Stage 5: Chat with character
├── dspy-extract-full.py             # Stage 1: Extract comic data
├── make-scenes-full.py              # Stage 2: Group scenes
└── make-scenes-dataset.py           # Stage 2: Create dataset
```

## Troubleshooting

### "Profile not found" Error

If you get tier profile not found errors:
```bash
# Generate tiered profiles first
./compress_character_profile.py
```

### Poor Character Quality

Try different tiers:
- **Tier 1 (core)**: Best for avoiding hallucinations, minimal profile
- **Tier 2 (extended)**: More detail, still compact
- **Tier 3 (full)**: Maximum detail, but may cause hallucinations

### API Errors

Check your `.env` file and Vertex AI credentials:
```bash
# Test connection
./vertex-test.py
```

## Future Enhancements

### Implemented ✅
- ✅ Tiered profile system with compression
- ✅ Non-interactive test mode
- ✅ Language-aware translations
- ✅ Behavioral guidelines and constraints
- ✅ Profile validation

### Potential Improvements
- RAG (Retrieval-Augmented Generation) for dynamic context
- Fine-tuning with the full profile
- Multi-character profiles
- Automated hallucination detection in test results
- Conversation quality scoring
