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
#   ./scripts/run_datagen_pipeline.sh --mini       # integration test with 5 items per stage
#   ./scripts/run_datagen_pipeline.sh --mini 3     # integration test with 3 items per stage

set -euo pipefail

BACKEND="${DATAGEN_BACKEND:-gemini}"
MODEL="${DATAGEN_MODEL:-gemini-3-flash-preview}"
SFT_MODEL="${SFT_MODEL:-Qwen/Qwen3.5-4B}"

CORPUS="output/datagen/memory_corpus.jsonl"
LEDGER="results/ledger_filtered.json"
PROMPTS="output/datagen/prompts.jsonl"
TRACES="output/datagen/traces.jsonl"
SCORED="output/datagen/traces_scored.jsonl"
FILTERED="output/datagen/traces_filtered.jsonl"
DATASET="output/sft/dataset"

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
if [ "$SKIP_GEN" = true ] || [ "$MINI" -gt 0 ]; then
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
banner "Stage 4: Assemble SFT dataset (tokenizer=$SFT_MODEL)"
uv run python training/assemble_sft.py \
    --input "$FILTERED" \
    --output "$DATASET" \
    --model "$SFT_MODEL"

# ------------------------------------------------------------------
# Stage 5: Eval inference (mini mode only)
# ------------------------------------------------------------------
if [ "$MINI" -gt 0 ]; then
    banner "Stage 5: Eval inference (backend=$BACKEND, model=$MODEL)"
    uv run python evals/generate_eval_prompts.py \
        --output-dir "$EVAL_PROMPTS" \
        --suites personality,tool_use
    uv run python evals/run_eval_inference.py \
        --prompts-dir "$EVAL_PROMPTS" \
        --output-dir "$EVAL_TRACES" \
        --backend "$BACKEND" \
        --model "$MODEL" \
        "${MAX_ITEMS_FLAG[@]}"

    # ------------------------------------------------------------------
    # Stage 6: Eval scoring (mini mode only)
    # ------------------------------------------------------------------
    banner "Stage 6: Eval scoring"
    uv run python evals/score_eval_traces.py \
        --traces-dir "$EVAL_TRACES" \
        --prompts-dir "$EVAL_PROMPTS" \
        --output-dir "$EVAL_SCORED" \
        --backend "$BACKEND" \
        --model "$MODEL"
fi

banner "Done"
echo "  Corpus:   $CORPUS"
echo "  Prompts:  $PROMPTS"
echo "  Traces:   $TRACES"
echo "  Scored:   $SCORED"
echo "  Filtered: $FILTERED"
echo "  Dataset:  $DATASET"
if [ "$MINI" -gt 0 ]; then
    echo "  Eval prompts: $EVAL_PROMPTS"
    echo "  Eval traces:  $EVAL_TRACES"
    echo "  Eval scored:  $EVAL_SCORED"
fi
