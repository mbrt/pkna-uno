#!/usr/bin/env bash
#
# Upload a locally cached HuggingFace model to S3 in the format expected
# by training and eval stacks (s3://BUCKET/models/<org>/<model>/).
#
# The script resolves the model files from either:
#   1. A flat local directory (e.g. downloaded with snapshot_download local_dir=)
#   2. The default HuggingFace cache (~/.cache/huggingface/hub) — takes the
#      most recent snapshot.
#
# Usage:
#   ./infra/upload-model-to-s3.sh --model unsloth/Qwen3-4B --bucket my-bucket
#   ./infra/upload-model-to-s3.sh --model unsloth/Qwen3-4B --bucket my-bucket \
#       --local-dir /path/to/flat/model

set -euo pipefail

BUCKET=""
MODEL=""
LOCAL_DIR=""
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --model MODEL       HuggingFace model ID, e.g. unsloth/Qwen3-4B (required)
    --bucket NAME       S3 bucket name (required)
    --local-dir DIR     Path to a flat model directory; if omitted the HF
                        cache at $HF_CACHE is used
    --region REGION     AWS region (default: $REGION)
    -h, --help          Show this help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --bucket) BUCKET="$2"; shift 2 ;;
        --local-dir) LOCAL_DIR="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi
if [ -z "$BUCKET" ]; then
    echo "ERROR: --bucket is required" >&2
    exit 1
fi

# Resolve source directory
if [ -n "$LOCAL_DIR" ]; then
    SRC_DIR="$LOCAL_DIR"
    if [ ! -d "$SRC_DIR" ]; then
        echo "ERROR: --local-dir '$SRC_DIR' does not exist" >&2
        exit 1
    fi
else
    # Look in the HF hub cache: models--org--name/snapshots/<hash>
    CACHE_KEY="models--$(echo "$MODEL" | sed 's|/|--|g')"
    SNAPSHOTS_DIR="$HF_CACHE/$CACHE_KEY/snapshots"
    if [ ! -d "$SNAPSHOTS_DIR" ]; then
        echo "ERROR: No HF cache found for '$MODEL' at $SNAPSHOTS_DIR" >&2
        echo "       Download the model first, or pass --local-dir." >&2
        exit 1
    fi
    # Use the most recent snapshot
    SRC_DIR="$SNAPSHOTS_DIR/$(ls -1t "$SNAPSHOTS_DIR" | head -1)"
fi

S3_DEST="s3://$BUCKET/models/$MODEL/"

echo "================================================================"
echo "  Upload model to S3"
echo "================================================================"
echo ""
echo "  Model:    $MODEL"
echo "  Source:   $SRC_DIR"
echo "  Dest:     $S3_DEST"
echo "  Region:   $REGION"
echo ""

aws s3 sync "$SRC_DIR/" "$S3_DEST" --region "$REGION" --size-only

echo ""
echo "Upload complete."
echo ""
echo "Verify:"
echo "  aws s3 ls $S3_DEST --region $REGION"
echo ""
echo "Use in training/eval:"
echo "  ./infra/launch-training.sh --model $MODEL --model-cache-bucket $BUCKET ..."
echo "  ./infra/launch-eval.sh     --model $MODEL --model-cache-bucket $BUCKET ..."
