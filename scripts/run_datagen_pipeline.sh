#!/usr/bin/env bash
#
# Run the full datagen pipeline: generate prompts, run trace generation with
# a real LLM, quality-filter the traces, assemble the SFT dataset, and sample
# distillation prompts from Tulu3.
#
# Every stage is resumable -- re-running after an interruption skips
# already-processed items.
#
# Environment variables:
#   DATAGEN_MODEL   LLM model for datagen + filtering (default: gemini-3-flash)
#   DATAGEN_BACKEND LLM backend name (default: gemini)
#
# Usage:
#   ./scripts/run_datagen_pipeline.sh              # full pipeline
#   ./scripts/run_datagen_pipeline.sh --skip-gen   # skip prompt generation (reuse existing)
#   ./scripts/run_datagen_pipeline.sh --mini       # integration test with 5 items per stage
#   ./scripts/run_datagen_pipeline.sh --mini 3     # integration test with 3 items per stage

set -euo pipefail

BACKEND="${DATAGEN_BACKEND:-gemini}"
MODEL="${DATAGEN_MODEL:-gemini-3-flash-preview}"

CORPUS="output/datagen/memory_corpus.jsonl"
LEDGER="results/ledger_filtered.json"
PROMPTS="output/datagen/prompts.jsonl"
TRACES="output/datagen/traces.jsonl"
SCORED="output/datagen/traces_scored.jsonl"
FILTERED="output/datagen/traces_filtered.jsonl"
DATASET="output/sft/dataset"
DISTILL_PROMPTS="output/distillation/prompts"

EVAL_PROMPTS="output/evals/prompts"
EVAL_TRACES="output/evals/traces"
EVAL_SCORED="output/evals/scored"

SKIP_GEN=false
MINI=0

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-gen) SKIP_GEN=true; shift ;;
        --mini)
            MINI=5
            if [ $# -gt 1 ] && [[ "$2" =~ ^[0-9]+$ ]]; then
                MINI="$2"; shift
            fi
            shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MAX_ITEMS_FLAG=()
if [ "$MINI" -gt 0 ]; then
    MAX_ITEMS_FLAG=(--max-items "$MINI")
fi

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
elif [ "$MINI" -gt 0 ]; then
    banner "Stage 1: Generate prompts (manual + claims, max=$MINI)"
    uv run python datagen/generate_prompts.py \
        --output "$PROMPTS" \
        --ledger "$LEDGER" \
        "${MAX_ITEMS_FLAG[@]}"
else
    banner "Stage 1: Generate prompts (manual + scene + claims + LLM-generated)"
    uv run python datagen/generate_prompts.py \
        --output "$PROMPTS" \
        --ledger "$LEDGER" \
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
    --model "$MODEL" \
    "${MAX_ITEMS_FLAG[@]}"

# ------------------------------------------------------------------
# Stage 3: Quality filtering
# ------------------------------------------------------------------
banner "Stage 3: Quality filtering"
uv run python datagen/filter_traces.py \
    --input "$TRACES" \
    --scored-output "$SCORED" \
    --filtered-output "$FILTERED" \
    --backend "$BACKEND" \
    --model "$MODEL" \
    "${MAX_ITEMS_FLAG[@]}"

# ------------------------------------------------------------------
# Stage 4: Assemble HF Dataset
# ------------------------------------------------------------------
banner "Stage 4: Assemble SFT dataset"
uv run python training/assemble_sft.py \
    --input "$FILTERED" \
    --output "$DATASET"

# ------------------------------------------------------------------
# Stage 5: Generate distillation prompts (Tulu3 sampling, no LLM)
# ------------------------------------------------------------------
banner "Stage 5: Generate distillation prompts"
uv run python distillation/generate_prompts.py \
    --output "$DISTILL_PROMPTS" \
    "${MAX_ITEMS_FLAG[@]}"

banner "Done"
echo "  Corpus:            $CORPUS"
echo "  Prompts:           $PROMPTS"
echo "  Traces:            $TRACES"
echo "  Scored:            $SCORED"
echo "  Filtered:          $FILTERED"
echo "  SFT dataset:       $DATASET"
echo "  Distill prompts:   $DISTILL_PROMPTS"
if [ "$MINI" -gt 0 ]; then
    echo "  Eval prompts:      $EVAL_PROMPTS"
    echo "  Eval traces:       $EVAL_TRACES"
    echo "  Eval scored:       $EVAL_SCORED"
fi
