#!/usr/bin/env bash
#
# Run eval tracing (stages 1+2 only, no grading) using vLLM.
#
# Detects whether the model is a LoRA adapter (by checking for
# adapter_config.json on HuggingFace Hub), merges if needed, starts a
# vLLM server, generates eval prompts, and runs inference.
#
# The same model is used for both the model under test and the user
# simulator backend.
#
# Prerequisites:
#   - GPU available
#   - uv installed
#   - HF_TOKEN set (if model is gated)
#
# Usage:
#   ./scripts/run_eval_tracing.sh --model Qwen/Qwen3.5-4B
#   ./scripts/run_eval_tracing.sh --model mbrt/uno-sft-adapter --max-model-len 8192
#   ./scripts/run_eval_tracing.sh --model mbrt/uno-distill-adapter --output-dir output/evals

set -euo pipefail

MODEL=""
MAX_MODEL_LEN=16384
OUTPUT_BASE="output/evals"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --model MODEL           HuggingFace model ID or adapter (required)
    --max-model-len N       Max model context length for vLLM (default: $MAX_MODEL_LEN)
    --output-dir DIR        Base output directory (default: $OUTPUT_BASE)
    -h, --help              Show this help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --output-dir) OUTPUT_BASE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9]|-|g' | tr '[:upper:]' '[:lower:]')
RUN_DIR="$OUTPUT_BASE/run-${MODEL_SHORT}-$(date +%Y%m%d-%H%M%S)"
PROMPTS_DIR="$RUN_DIR/prompts"
TRACES_DIR="$RUN_DIR/traces"

mkdir -p "$RUN_DIR"

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

banner "Eval Tracing Pipeline"
echo "  Model:          $MODEL"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  Output:         $RUN_DIR"
echo ""

# ------------------------------------------------------------------
# Step 1: Detect LoRA adapter and merge if needed
# ------------------------------------------------------------------
banner "Step 1: Detect model type"

SERVE_MODEL="$MODEL"

IS_ADAPTER=$(uv run python -c "
import os
from pathlib import Path

model = '$MODEL'
if os.path.isdir(model):
    # Local path: check for adapter_config.json on disk
    print('true' if (Path(model) / 'adapter_config.json').exists() else 'false')
else:
    # HuggingFace Hub: check remote repo
    from huggingface_hub import hf_hub_url, get_hf_file_metadata
    try:
        url = hf_hub_url(model, 'adapter_config.json')
        get_hf_file_metadata(url)
        print('true')
    except Exception:
        print('false')
")

if [ "$IS_ADAPTER" = "true" ]; then
    echo "Model is a LoRA adapter. Merging into base model..."
    MERGED_DIR="output/merged-models/$MODEL_SHORT"
    mkdir -p "$(dirname "$MERGED_DIR")"

    uv run python -c "
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    '$MODEL',
    max_seq_length=$MAX_MODEL_LEN,
    load_in_4bit=True,
    load_in_16bit=False,
    full_finetuning=False,
)
model.save_pretrained_merged(
    '$MERGED_DIR',
    tokenizer,
    save_method='merged_16bit',
)
print('Merge complete: $MERGED_DIR')
"
    SERVE_MODEL="$MERGED_DIR"
    echo "Will serve merged model from: $SERVE_MODEL"
else
    echo "Model is a full model. Will serve directly from HuggingFace."
fi

# ------------------------------------------------------------------
# Step 2: Start vLLM server
# ------------------------------------------------------------------
banner "Step 2: Start vLLM server"

VLLM_LOG="$RUN_DIR/vllm.log"

uv tool run --from vllm vllm serve "$SERVE_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --dtype auto \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup_vllm() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup_vllm EXIT

echo "vLLM server starting (PID $VLLM_PID), log: $VLLM_LOG"
echo "Waiting for server to become healthy..."

MAX_WAIT=600
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server exited unexpectedly. Check $VLLM_LOG" >&2
        exit 1
    fi
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is healthy (took ${ELAPSED}s)."
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server did not become healthy within ${MAX_WAIT}s" >&2
    echo "Last lines of vLLM log:"
    tail -20 "$VLLM_LOG"
    exit 1
fi

# ------------------------------------------------------------------
# Step 3: Generate eval prompts
# ------------------------------------------------------------------
banner "Step 3: Generate eval prompts"

uv run python evals/generate_eval_prompts.py --output-dir "$PROMPTS_DIR"

# ------------------------------------------------------------------
# Step 4: Run eval inference (tracing only)
# ------------------------------------------------------------------
banner "Step 4: Run eval inference"

MAX_TOKENS=4096

uv run python evals/run_eval_inference.py \
    --prompts-dir "$PROMPTS_DIR" \
    --output-dir "$TRACES_DIR" \
    --backend vllm \
    --model "$SERVE_MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --simulator-backend vllm \
    --simulator-model "$SERVE_MODEL"

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
banner "Done"
echo "  Run directory: $RUN_DIR"
echo "  Prompts:       $PROMPTS_DIR"
echo "  Traces:        $TRACES_DIR"
