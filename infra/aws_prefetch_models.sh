#!/usr/bin/env bash
#
# AWS model pre-fetch entrypoint — runs as the ubuntu user.
# Called by the CloudFormation UserData bootstrap after cloning the repo.
#
# Required environment variables:
#   MODELS            Comma-separated HuggingFace model IDs to pre-fetch
#   S3_BUCKET         S3 bucket for model storage
#   REGION            AWS region
#   HF_TOKEN_SSM_PATH SSM path for the HuggingFace token

set -euo pipefail

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

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
    echo "WARNING: HF token not found at $HF_TOKEN_SSM_PATH. Gated models will fail."
fi

# -------------------------------------------------------------------
# Build model list from comma-separated MODELS env var
# -------------------------------------------------------------------
IFS=',' read -ra _RAW_MODELS <<< "$MODELS"
MODEL_ARGS=()
for _m in "${_RAW_MODELS[@]}"; do
    _m="$(echo "$_m" | xargs)"
    [ -n "$_m" ] && MODEL_ARGS+=("$_m")
done

echo "=== Pre-fetching ${#MODEL_ARGS[@]} model(s): ${MODEL_ARGS[*]} ==="

# -------------------------------------------------------------------
# Download models to local cache
# -------------------------------------------------------------------
# Use --with to avoid a full uv sync: this script only needs huggingface_hub.
uv run --with "huggingface_hub>=0.25" python infra/prefetch_models.py \
    --cache-dir model-cache \
    "${MODEL_ARGS[@]}"

# -------------------------------------------------------------------
# Sync each model directory (org/model, depth 2) to S3
# -------------------------------------------------------------------
while IFS= read -r -d '' _dir; do
    _rel="${_dir#model-cache/}"
    echo "=== Syncing $_rel to S3 ==="
    aws s3 sync "$_dir/" "s3://$S3_BUCKET/models/$_rel/" --region "$REGION"
done < <(find model-cache -mindepth 2 -maxdepth 2 -type d -print0)

echo "=== Pre-fetch complete at $(date -u) ==="
