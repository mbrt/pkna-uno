#!/usr/bin/env bash
#
# Local reproduction test for the S3-cached model merge flow.
#
# Simulates what aws_eval.sh does:
#   1. Copy adapter + base model from HF cache to a temp dir
#   2. Patch adapter_config.json to point to local base model
#   3. Export HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
#   4. Run the same LoRA merge as run_eval_tracing.sh
#
# This lets us verify the merge works without any internet access,
# and iterate on fixes without deploying to AWS.
#
# Usage:
#   ./infra/test-local-model-merge.sh
#   ./infra/test-local-model-merge.sh --adapter mbrt/uno-Qwen3.5-4B --max-model-len 8192

set -euo pipefail

ADAPTER="mbrt/uno-Qwen3.5-4B"
MAX_MODEL_LEN=8192
HF_CACHE="${HOME}/.cache/huggingface/hub"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --adapter MODEL     LoRA adapter model ID (default: $ADAPTER)
    --max-model-len N   Max context length (default: $MAX_MODEL_LEN)
    -h, --help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --adapter) ADAPTER="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -h|--help) usage ;;
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
# Resolve HF cache paths
# ------------------------------------------------------------------
hf_cache_path() {
    local model_id="$1"
    local cache_key
    cache_key="models--$(echo "$model_id" | sed 's|/|--|g')"
    local snap_dir="$HF_CACHE/$cache_key/snapshots"
    if [ ! -d "$snap_dir" ]; then
        echo "ERROR: model not in HF cache: $model_id" >&2
        echo "  Run: uv run huggingface-cli download $model_id" >&2
        exit 1
    fi
    local snap
    snap=$(ls "$snap_dir" | head -1)
    echo "$snap_dir/$snap"
}

ADAPTER_CACHE=$(hf_cache_path "$ADAPTER")
echo "Adapter cache: $ADAPTER_CACHE"

# Read base model from adapter_config.json
BASE_MODEL=$(python3 -c "
import json
with open('$ADAPTER_CACHE/adapter_config.json') as f:
    print(json.load(f)['base_model_name_or_path'])
")
echo "Base model:    $BASE_MODEL"

BASE_CACHE=$(hf_cache_path "$BASE_MODEL")
echo "Base cache:    $BASE_CACHE"

# ------------------------------------------------------------------
# Set up temp dir (simulates the EC2 /home/ubuntu/model-cache)
# ------------------------------------------------------------------
TMPDIR=$(mktemp -d -t model-merge-test-XXXXXX)
trap 'echo "Cleaning up $TMPDIR..."; rm -rf "$TMPDIR"' EXIT

ADAPTER_LOCAL="$TMPDIR/model-cache/$ADAPTER"
BASE_LOCAL="$TMPDIR/model-cache/$BASE_MODEL"
MERGED_DIR="$TMPDIR/merged"

banner "Step 1: Copy models to temp dir (simulates S3 sync)"

mkdir -p "$ADAPTER_LOCAL" "$BASE_LOCAL"
cp -L "$ADAPTER_CACHE"/. "$ADAPTER_LOCAL"/ 2>/dev/null || \
    cp -LR "$ADAPTER_CACHE"/. "$ADAPTER_LOCAL"/
cp -LR "$BASE_CACHE"/. "$BASE_LOCAL"/
echo "Copied adapter to:    $ADAPTER_LOCAL"
echo "Copied base model to: $BASE_LOCAL"

banner "Step 2: Patch adapter_config.json (simulates aws_eval.sh)"

ADAPTER_CFG="$ADAPTER_LOCAL/adapter_config.json"
python3 -c "
import json
with open('$ADAPTER_CFG') as f:
    cfg = json.load(f)
original = cfg['base_model_name_or_path']
cfg['base_model_name_or_path'] = '$BASE_LOCAL'
with open('$ADAPTER_CFG', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Patched: {original!r} -> {cfg[\"base_model_name_or_path\"]!r}')
"

banner "Step 3: Block HF Hub access (simulates aws_eval.sh export)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

banner "Step 4: Run LoRA merge (same as run_eval_tracing.sh)"

mkdir -p "$MERGED_DIR"

# Capture any network-related errors as a clear failure
uv run python -c "
import os
from unsloth import FastLanguageModel

print('Loading model from local path:', '$ADAPTER_LOCAL')
print('HF_HUB_OFFLINE =', os.environ.get('HF_HUB_OFFLINE'))
print('TRANSFORMERS_OFFLINE =', os.environ.get('TRANSFORMERS_OFFLINE'))

model, tokenizer = FastLanguageModel.from_pretrained(
    '$ADAPTER_LOCAL',
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
print()
print('SUCCESS: Merge complete, no internet access needed.')
print('Merged model at:', '$MERGED_DIR')
"
