# AGENTS.md

## Project Overview

PKNA-Uno is a multi-stage ML pipeline for extracting structured data from Italian comic books (PKNA - Paperinik New Adventures), building a comprehensive character profile for "Uno" (an AI character), and enabling AI-powered character impersonation.

## Environment Setup

```bash
# Install dependencies (project uses uv for dependency management)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Running Tests

```bash
make test
```

## Development Patterns

Two possibilities:

* DSPy models (see e.g. `dspy-extract-full.py`)
* Google `genai` package (see `build_agentic_character_profile.py`)

Common patterns:

* Standalone self-contained scripts.
* Use Rich console for CLI output and logging.
* Use Rich Progress for long-running operations
* All pipeline stages check for existing output files and skip already-processed items
* Resume Capability: All stages support resuming interrupted runs (skip existing outputs)
* Incremental Logging: Use JSONL format for long-running processes
* Error Handling: Retry logic with exponential backoff for LLM API calls
* Version Control: Pipeline outputs are versioned (v2, v3) in separate directories
