#!/usr/bin/env bash
#
# Run the full SFT datagen pipeline: generate prompts, run trace generation
# with a real LLM, quality-filter the traces, and assemble the HF dataset.
#
# Every stage is resumable -- re-running after an interruption skips
# already-processed items.
#
# Environment variables:
#   DATAGEN_MODEL   LLM model for datagen + filtering (default: gemini-3-flash)
#   DATAGEN_BACKEND LLM backend name (default: gemini)
#   SFT_MODEL       Tokenizer model for dataset assembly (default: Qwen/Qwen3.5-4B)
#
# Usage:
#   ./scripts/run_datagen_pipeline.sh              # full pipeline
#   ./scripts/run_datagen_pipeline.sh --skip-gen   # skip prompt generation (reuse existing)

set -euo pipefail

BACKEND="${DATAGEN_BACKEND:-gemini}"
MODEL="${DATAGEN_MODEL:-gemini-3-flash-preview}"
SFT_MODEL="${SFT_MODEL:-Qwen/Qwen3.5-4B}"

CORPUS="output/datagen/memory_corpus.jsonl"
PROMPTS="output/datagen/prompts.jsonl"
TRACES="output/datagen/traces.jsonl"
SCORED="output/datagen/traces_scored.jsonl"
FILTERED="output/datagen/traces_filtered.jsonl"
DATASET="output/sft/dataset"

SKIP_GEN=false
for arg in "$@"; do
    case "$arg" in
        --skip-gen) SKIP_GEN=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

# ------------------------------------------------------------------
# Stage 0: Generate memory corpus
# ------------------------------------------------------------------
if [ "$SKIP_GEN" = true ]; then
    banner "Stage 0: Generate memory corpus [SKIPPED]"
else
    banner "Stage 0: Generate memory corpus (seed banks + LLM)"
    uv run python datagen/generate_memory_corpus.py \
        --output "$CORPUS" \
        --backend "$BACKEND" \
        --model "$MODEL"
fi

# ------------------------------------------------------------------
# Stage 1: Generate prompts
# ------------------------------------------------------------------
if [ "$SKIP_GEN" = true ]; then
    banner "Stage 1: Generate prompts [SKIPPED]"
else
    banner "Stage 1: Generate prompts (manual + scene + LLM-generated)"
    uv run python datagen/generate_prompts.py \
        --output "$PROMPTS" \
        --include-generated \
        --backend "$BACKEND" \
        --model "$MODEL"
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Stage 2: Run datagen (trace generation with real LLM)
# ------------------------------------------------------------------
banner "Stage 2: Run datagen (backend=$BACKEND, model=$MODEL)"
uv run python datagen/run_datagen.py \
    --prompts "$PROMPTS" \
    --output "$TRACES" \
    --corpus "$CORPUS" \
    --backend "$BACKEND" \
    --model "$MODEL"

# ------------------------------------------------------------------
# Stage 3: Quality filtering
# ------------------------------------------------------------------
banner "Stage 3: Quality filtering"
uv run python datagen/filter_traces.py \
    --input "$TRACES" \
    --scored-output "$SCORED" \
    --filtered-output "$FILTERED" \
    --backend "$BACKEND" \
    --model "$MODEL"

# ------------------------------------------------------------------
# Stage 4: Assemble HF Dataset
# ------------------------------------------------------------------
banner "Stage 4: Assemble SFT dataset (tokenizer=$SFT_MODEL)"
uv run python training/assemble_sft.py \
    --input "$FILTERED" \
    --output "$DATASET" \
    --model "$SFT_MODEL"

banner "Done"
echo "  Corpus:   $CORPUS"
echo "  Prompts:  $PROMPTS"
echo "  Traces:   $TRACES"
echo "  Scored:   $SCORED"
echo "  Filtered: $FILTERED"
echo "  Dataset:  $DATASET"
