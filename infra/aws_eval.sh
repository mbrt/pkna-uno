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
# Run the portable eval tracing script
# -------------------------------------------------------------------
echo "=== Running eval tracing ==="
./scripts/run_eval_tracing.sh \
    --model "$EVAL_MODEL" \
    --max-model-len "$MAX_MODEL_LEN"

echo "=== Eval tracing complete at $(date -u) ==="
