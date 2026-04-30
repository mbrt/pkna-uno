#!/usr/bin/env bash
#
# AWS eval entrypoint — runs as the ubuntu user.
# Called by the CloudFormation UserData bootstrap after cloning the repo.
#
# Thin wrapper that handles AWS-specific setup (uv install, HF token from
# SSM), then delegates to the portable scripts/run_eval_tracing.sh.
#
# Required environment variables:
#   EVAL_MODEL          HuggingFace model ID or adapter
#   MAX_MODEL_LEN       Max context length for vLLM
#   S3_BUCKET           S3 bucket for results upload
#   REGION              AWS region
#   HF_TOKEN_SSM_PATH   SSM path for the HuggingFace token

set -euo pipefail

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Installing dependencies ==="
uv sync

# -------------------------------------------------------------------
# Fetch HuggingFace token from SSM SecureString
# -------------------------------------------------------------------
echo "=== Fetching HF token from SSM ==="
HF_TOKEN=$(aws ssm get-parameter \
    --name "$HF_TOKEN_SSM_PATH" \
    --with-decryption \
    --query Parameter.Value \
    --output text \
    --region "$REGION" 2>/dev/null) || true
export HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF token not found at $HF_TOKEN_SSM_PATH in $REGION. Gated models will fail."
fi

# -------------------------------------------------------------------
# Fetch model from S3 cache (if configured)
# -------------------------------------------------------------------
if [ -n "${MODEL_CACHE_BUCKET:-}" ]; then
    echo "=== Fetching model from S3 cache: $EVAL_MODEL ==="
    LOCAL_MODEL_DIR="/home/ubuntu/model-cache/$EVAL_MODEL"
    mkdir -p "$LOCAL_MODEL_DIR"
    aws s3 sync "s3://$MODEL_CACHE_BUCKET/models/$EVAL_MODEL/" "$LOCAL_MODEL_DIR/" \
        --region "$REGION"

    # If it's a LoRA adapter, also fetch the base model and patch
    # adapter_config.json to point to the local base model path.
    ADAPTER_CFG="$LOCAL_MODEL_DIR/adapter_config.json"
    if [ -f "$ADAPTER_CFG" ]; then
        BASE_MODEL=$(python3 -c "
import json
with open('$ADAPTER_CFG') as f:
    print(json.load(f).get('base_model_name_or_path', ''))
")
        if [ -n "$BASE_MODEL" ]; then
            echo "=== Fetching base model from S3 cache: $BASE_MODEL ==="
            BASE_LOCAL_DIR="/home/ubuntu/model-cache/$BASE_MODEL"
            mkdir -p "$BASE_LOCAL_DIR"
            aws s3 sync "s3://$MODEL_CACHE_BUCKET/models/$BASE_MODEL/" "$BASE_LOCAL_DIR/" \
                --region "$REGION"
            python3 -c "
import json
with open('$ADAPTER_CFG') as f:
    cfg = json.load(f)
cfg['base_model_name_or_path'] = '$BASE_LOCAL_DIR'
with open('$ADAPTER_CFG', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Patched adapter_config.json: base_model_name_or_path -> $BASE_LOCAL_DIR')
"
        fi
    fi

    export EVAL_MODEL="$LOCAL_MODEL_DIR"
    echo "Using local model: $EVAL_MODEL"
fi

# -------------------------------------------------------------------
# Run the portable eval tracing script
# -------------------------------------------------------------------
echo "=== Running eval tracing ==="
./scripts/run_eval_tracing.sh \
    --model "$EVAL_MODEL" \
    --max-model-len "$MAX_MODEL_LEN"

echo "=== Eval tracing complete at $(date -u) ==="
