#!/usr/bin/env bash
#
# Run the eval pipeline against a local or HuggingFace model/adapter.
#
# Three stages:
#   1. Generate eval prompts (deterministic, no LLM needed)
#   2. Run eval inference (model under test via --backend)
#   3. Score traces (judge model via --judge-backend)
#
# Prerequisites:
#   - GPU available for --backend local
#   - GOOGLE_API_KEY set for --judge-backend gemini (default)
#
# Usage:
#   ./scripts/run_evals.sh --model mbrt/uno-sft-adapter
#   ./scripts/run_evals.sh --model output/sft/lora_adapter --backend local
#   ./scripts/run_evals.sh --model mbrt/uno-distill-adapter --suites personality,tool_use
#   ./scripts/run_evals.sh --model gemini-3-flash --backend gemini
#   ./scripts/run_evals.sh --traces-dir output/evals/traces/run-20260424  # score only
#   ./scripts/run_evals.sh --4bit --model output/sft/lora_adapter         # 4-bit quantized inference
#   ./scripts/run_evals.sh --resume output/evals/run-lora-adapter-20260427  # resume interrupted run
#   ./scripts/run_evals.sh --mini --model output/sft/lora_adapter        # quick run, 5 items

set -euo pipefail

MODEL=""
BACKEND="local"
JUDGE_BACKEND="gemini"
JUDGE_MODEL=""
SUITES=""
MAX_ITEMS=0
MAX_TOKENS=0
OUTPUT_BASE="output/evals"
TRACES_DIR=""
SKIP_PROMPTS=false
SKIP_INFERENCE=false
LOAD_4BIT=false
SIM_BACKEND=""
SIM_MODEL=""
RESUME_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --model MODEL           Model name or adapter path (required unless --traces-dir)
    --backend BACKEND       Inference backend: local, vllm, gemini, anthropic (default: local)
    --judge-backend BACKEND Judge backend: gemini, anthropic (default: gemini)
    --judge-model MODEL     Judge model name (default: backend default)
    --suites SUITES         Comma-separated suites (default: all)
    --max-items N           Max prompts per suite (default: 0 = unlimited)
    --max-tokens N          Max output tokens per generation (default: backend default)
    --output-base DIR       Base output directory (default: $OUTPUT_BASE)
    --resume DIR            Resume an interrupted run (reuse existing run directory)
    --traces-dir DIR        Score existing traces (skip prompts + inference)
    --skip-prompts          Reuse existing prompts (skip stage 1)
    --4bit                  Load local model in 4-bit quantization (faster, less VRAM)
    --simulator-backend B   Backend for user simulator (default: judge backend)
    --simulator-model M     Model for user simulator (default: backend default)
    --mini [N]              Quick run with N prompts per suite (default: 5)
    -h, --help              Show this help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --judge-backend) JUDGE_BACKEND="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --suites) SUITES="$2"; shift 2 ;;
        --max-items) MAX_ITEMS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        --resume) RESUME_DIR="$2"; shift 2 ;;
        --traces-dir) TRACES_DIR="$2"; SKIP_INFERENCE=true; SKIP_PROMPTS=true; shift 2 ;;
        --skip-prompts) SKIP_PROMPTS=true; shift ;;
        --4bit) LOAD_4BIT=true; shift ;;
        --simulator-backend) SIM_BACKEND="$2"; shift 2 ;;
        --simulator-model) SIM_MODEL="$2"; shift 2 ;;
        --mini)
            if [ "$MAX_ITEMS" -eq 0 ]; then MAX_ITEMS=5; fi
            if [ $# -gt 1 ] && [[ "$2" =~ ^[0-9]+$ ]]; then
                MAX_ITEMS="$2"; shift
            fi
            shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] && [ "$SKIP_INFERENCE" = false ]; then
    echo "ERROR: --model is required (unless using --traces-dir)" >&2
    exit 1
fi

# Resolve run directory: resume existing or create new
if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo "ERROR: resume directory does not exist: $RESUME_DIR" >&2
        exit 1
    fi
    RUN_DIR="$RESUME_DIR"
else
    RUN_ID=$(date +%Y%m%d-%H%M%S)
    if [ -n "$MODEL" ]; then
        MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9]|-|g' | tr '[:upper:]' '[:lower:]')
        RUN_DIR="$OUTPUT_BASE/run-${MODEL_SHORT}-${RUN_ID}"
    else
        RUN_DIR="$OUTPUT_BASE/run-score-${RUN_ID}"
    fi
fi

PROMPTS_DIR="$RUN_DIR/prompts"
if [ -z "$TRACES_DIR" ]; then
    TRACES_DIR="$RUN_DIR/traces"
fi
SCORED_DIR="$RUN_DIR/scored"

mkdir -p "$RUN_DIR"

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

banner "Eval Pipeline"
echo "  Model:         ${MODEL:-N/A}"
echo "  Backend:       $BACKEND"
echo "  4-bit:         $LOAD_4BIT"
echo "  Simulator:     ${SIM_BACKEND:-$JUDGE_BACKEND} ${SIM_MODEL:-${JUDGE_MODEL:-(default)}}"
echo "  Judge:         $JUDGE_BACKEND"
echo "  Suites:        ${SUITES:-all}"
echo "  Output:        $RUN_DIR"
echo ""

# ------------------------------------------------------------------
# Stage 1: Generate eval prompts
# ------------------------------------------------------------------
if [ "$SKIP_PROMPTS" = false ] && [ ! -d "$PROMPTS_DIR" ]; then
    banner "Stage 1: Generate Eval Prompts"

    PROMPT_FLAGS=("--output-dir" "$PROMPTS_DIR")
    if [ -n "$SUITES" ]; then
        PROMPT_FLAGS+=("--suites" "$SUITES")
    fi

    uv run python evals/generate_eval_prompts.py "${PROMPT_FLAGS[@]}"
else
    echo "Skipping prompt generation (prompts dir exists, --skip-prompts, or --traces-dir)"
    if [ ! -d "$PROMPTS_DIR" ]; then
        # Use default prompts dir when scoring existing traces
        PROMPTS_DIR="$OUTPUT_BASE/prompts"
        if [ ! -d "$PROMPTS_DIR" ]; then
            banner "Stage 1: Generate Eval Prompts (for scoring)"
            PROMPT_FLAGS=("--output-dir" "$PROMPTS_DIR")
            if [ -n "$SUITES" ]; then
                PROMPT_FLAGS+=("--suites" "$SUITES")
            fi
            uv run python evals/generate_eval_prompts.py "${PROMPT_FLAGS[@]}"
        fi
    fi
fi

# ------------------------------------------------------------------
# Stage 2: Run eval inference
# ------------------------------------------------------------------
if [ "$SKIP_INFERENCE" = false ]; then
    banner "Stage 2: Run Eval Inference (backend=$BACKEND, model=$MODEL)"

    INFER_FLAGS=(
        "--prompts-dir" "$PROMPTS_DIR"
        "--output-dir" "$TRACES_DIR"
        "--backend" "$BACKEND"
        "--model" "$MODEL"
    )
    if [ -n "$SUITES" ]; then
        INFER_FLAGS+=("--suites" "$SUITES")
    fi
    if [ "$MAX_ITEMS" -gt 0 ]; then
        INFER_FLAGS+=("--max-items" "$MAX_ITEMS")
    fi
    if [ "$MAX_TOKENS" -gt 0 ]; then
        INFER_FLAGS+=("--max-tokens" "$MAX_TOKENS")
    fi
    if [ "$LOAD_4BIT" = true ]; then
        INFER_FLAGS+=("--4bit")
    fi
    INFER_FLAGS+=("--simulator-backend" "${SIM_BACKEND:-$JUDGE_BACKEND}")
    if [ -n "${SIM_MODEL:-$JUDGE_MODEL}" ]; then
        INFER_FLAGS+=("--simulator-model" "${SIM_MODEL:-$JUDGE_MODEL}")
    fi

    uv run python evals/run_eval_inference.py "${INFER_FLAGS[@]}"
else
    echo "Skipping inference (--traces-dir provided)"
fi

# ------------------------------------------------------------------
# Stage 3: Score traces
# ------------------------------------------------------------------
banner "Stage 3: Score Traces (judge=$JUDGE_BACKEND)"

SCORE_FLAGS=(
    "--traces-dir" "$TRACES_DIR"
    "--prompts-dir" "$PROMPTS_DIR"
    "--output-dir" "$SCORED_DIR"
    "--backend" "$JUDGE_BACKEND"
)
if [ -n "$JUDGE_MODEL" ]; then
    SCORE_FLAGS+=("--model" "$JUDGE_MODEL")
fi
if [ -n "$SUITES" ]; then
    SCORE_FLAGS+=("--suites" "$SUITES")
fi

uv run python evals/score_eval_traces.py "${SCORE_FLAGS[@]}"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
banner "Done"
echo "  Run directory: $RUN_DIR"
echo "  Prompts:       $PROMPTS_DIR"
echo "  Traces:        $TRACES_DIR"
echo "  Scored:        $SCORED_DIR"
echo "  Report:        $SCORED_DIR/report.json"

if [ -f "$SCORED_DIR/report.json" ]; then
    echo ""
    echo "  Report summary:"
    uv run python -c "
import json, sys
report = json.load(open('$SCORED_DIR/report.json'))
for suite, r in sorted(report['suites'].items()):
    print(f'    {suite}: mean={r[\"mean_score\"]:.2f} (n={r[\"n\"]})')
flagged = report.get('flagged_traces', [])
if flagged:
    print(f'    Flagged traces (score <= 2): {len(flagged)}')
"
fi
