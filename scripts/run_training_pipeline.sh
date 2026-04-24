#!/usr/bin/env bash
#
# Run the training pipeline: SFT personality fine-tuning followed by
# on-policy distillation for behavior recovery. Requires a GPU.
#
# Expects the datagen pipeline to have already produced:
#   - output/sft/dataset              (SFT training data)
#   - output/distillation/prompts     (distillation prompt set)
#
# Environment variables:
#   TRAIN_MODEL   Base model (default: Qwen/Qwen3.5-4B)
#   EXPORT_GGUF   GGUF quantization method for final export (default: empty = skip)
#
# Usage:
#   ./scripts/run_training_pipeline.sh                   # full pipeline
#   ./scripts/run_training_pipeline.sh --sft-only        # SFT only, skip distillation
#   ./scripts/run_training_pipeline.sh --distill-only    # distillation only (needs existing adapter)
#   ./scripts/run_training_pipeline.sh --export q4_k_m   # export GGUF after distillation

set -euo pipefail

MODEL="${TRAIN_MODEL:-unsloth/Qwen3.5-4B}"
GGUF="${EXPORT_GGUF:-}"

SFT_DATASET="output/sft/dataset"
SFT_ADAPTER="output/sft/lora_adapter"
DISTILL_PROMPTS="output/distillation/prompts"
DISTILL_ADAPTER="output/distillation/lora_adapter"

RUN_SFT=true
RUN_DISTILL=true

while [ $# -gt 0 ]; do
    case "$1" in
        --sft-only) RUN_DISTILL=false; shift ;;
        --distill-only) RUN_SFT=false; shift ;;
        --export)
            GGUF="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
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
# Stage 1: SFT (personality fine-tuning)
# ------------------------------------------------------------------
if [ "$RUN_SFT" = true ]; then
    if [ ! -d "$SFT_DATASET" ]; then
        echo "ERROR: SFT dataset not found at $SFT_DATASET" >&2
        echo "       Run scripts/run_datagen_pipeline.sh first." >&2
        exit 1
    fi

    banner "Stage 1: SFT Training (model=$MODEL)"
    uv run python training/run_sft.py \
        --dataset "$SFT_DATASET" \
        --output "$SFT_ADAPTER" \
        --model "$MODEL"
fi

# ------------------------------------------------------------------
# Stage 2: On-policy distillation (behavior recovery)
# ------------------------------------------------------------------
if [ "$RUN_DISTILL" = true ]; then
    if [ ! -d "$SFT_ADAPTER" ]; then
        echo "ERROR: SFT adapter not found at $SFT_ADAPTER" >&2
        echo "       Run SFT first (remove --distill-only)." >&2
        exit 1
    fi
    if [ ! -d "$DISTILL_PROMPTS" ]; then
        echo "ERROR: Distillation prompts not found at $DISTILL_PROMPTS" >&2
        echo "       Run scripts/run_datagen_pipeline.sh first." >&2
        exit 1
    fi

    GGUF_FLAG=()
    if [ -n "$GGUF" ]; then
        GGUF_FLAG=(--export-gguf "$GGUF")
    fi

    banner "Stage 2: On-Policy Distillation (model=$MODEL)"
    uv run python training/run_distillation.py \
        --dataset "$DISTILL_PROMPTS" \
        --adapter "$SFT_ADAPTER" \
        --output "$DISTILL_ADAPTER" \
        --model "$MODEL" \
        "${GGUF_FLAG[@]}"
fi

banner "Done"
if [ "$RUN_SFT" = true ]; then
    echo "  SFT adapter:       $SFT_ADAPTER"
fi
if [ "$RUN_DISTILL" = true ]; then
    echo "  Distill adapter:   $DISTILL_ADAPTER"
    if [ -n "$GGUF" ]; then
        echo "  GGUF export:       ${DISTILL_ADAPTER}-gguf"
    fi
fi
