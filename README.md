# PKNA Uno

## Character Profile Builder

This script builds a comprehensive character profile document for Uno by iteratively analyzing all scenes from the PKNA comics where Uno appears.

### Overview

The script:
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

### Future Enhancements

Potential improvements:
- Resume from specific scene number via CLI argument
- Parallel processing of scenes (with careful document merging)
- Interactive mode to review/approve edits
- Comparison mode to track profile changes over time
- Multi-character profiles in a single run
