#!/usr/bin/env bash
#
# AWS training entrypoint — runs as the ubuntu user.
# Called by the CloudFormation UserData bootstrap after cloning the repo.
#
# Required environment variables:
#   TRAIN_MODEL         HuggingFace model name
#   EXPORT_GGUF         GGUF quantization method (empty = skip)
#   RUN_DISTILL         "true" or "false"
#   S3_BUCKET           S3 bucket for results upload
#   REGION              AWS region
#   HF_TOKEN_SSM_PATH   SSM path for the HuggingFace token
#   HF_REPO_PREFIX      HuggingFace Hub repo prefix
#   GIT_REF             Git ref that was checked out

set -euo pipefail

MLFLOW_DIR="mlflow"

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
    echo "WARNING: HF token not found at $HF_TOKEN_SSM_PATH in $REGION. HF Hub upload will be skipped."
fi

# -------------------------------------------------------------------
# Download datasets from HuggingFace
# -------------------------------------------------------------------
echo "=== Downloading datasets from HuggingFace ==="
uv run python -c "
from datasets import load_dataset
ds = load_dataset('mbrt/uno-sft-dataset', split='train')
ds.save_to_disk('output/sft/dataset')
print('SFT dataset:', len(ds), 'examples')
"

uv run python -c "
from datasets import load_dataset
ds = load_dataset('mbrt/uno-distill-prompts', split='train')
ds.save_to_disk('output/distillation/prompts')
print('Distillation prompts:', len(ds), 'examples')
"

# -------------------------------------------------------------------
# Start MLflow tracking server (localhost only)
# -------------------------------------------------------------------
echo "=== Starting MLflow server ==="
mkdir -p "$MLFLOW_DIR/artifacts"
uv run mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri "sqlite:///$PWD/$MLFLOW_DIR/mlflow.db" \
    --default-artifact-root "$PWD/$MLFLOW_DIR/artifacts" &
MLFLOW_PID=$!
sleep 3
export MLFLOW_TRACKING_URI="http://localhost:5000"

# -------------------------------------------------------------------
# GPU benchmark
# -------------------------------------------------------------------
echo "=== Running GPU benchmark ==="
uv run python training/benchmark.py --model "$TRAIN_MODEL"

# -------------------------------------------------------------------
# Run training pipeline
# -------------------------------------------------------------------
PIPELINE_FLAGS=""
if [ "$RUN_DISTILL" = "false" ]; then
    PIPELINE_FLAGS="--sft-only"
fi
if [ -n "$EXPORT_GGUF" ]; then
    PIPELINE_FLAGS="$PIPELINE_FLAGS --export $EXPORT_GGUF"
fi

echo "=== Running training pipeline ==="
# shellcheck disable=SC2086
./scripts/run_training_pipeline.sh $PIPELINE_FLAGS

# -------------------------------------------------------------------
# Upload results to S3
# -------------------------------------------------------------------
MODEL_KEY="${TRAIN_MODEL##*/}"

echo "=== Uploading results to S3 (prefix=$MODEL_KEY) ==="
if [ -d output/sft/lora_adapter ]; then
    aws s3 sync output/sft/lora_adapter "s3://$S3_BUCKET/$MODEL_KEY/sft/lora_adapter/" --region "$REGION"
fi
if [ -d output/distillation/lora_adapter ]; then
    aws s3 sync output/distillation/lora_adapter "s3://$S3_BUCKET/$MODEL_KEY/distillation/lora_adapter/" --region "$REGION"
fi
if [ -n "$EXPORT_GGUF" ] && [ -d output/distillation/lora_adapter-gguf ]; then
    aws s3 sync output/distillation/lora_adapter-gguf "s3://$S3_BUCKET/$MODEL_KEY/distillation/lora_adapter-gguf/" --region "$REGION"
fi

# -------------------------------------------------------------------
# Upload adapters to HuggingFace Hub
# -------------------------------------------------------------------
if [ -n "$HF_TOKEN" ]; then
    echo "=== Uploading adapters to HuggingFace Hub ==="
    HF_REVISION="$GIT_REF"
    if [ -d output/sft/lora_adapter ]; then
        uv tool run hf upload \
            "$HF_REPO_PREFIX-sft-adapter" \
            output/sft/lora_adapter \
            --revision "$HF_REVISION" \
            --commit-message "SFT adapter from $GIT_REF (model=$TRAIN_MODEL)" || true
    fi
    if [ -d output/distillation/lora_adapter ]; then
        uv tool run hf upload \
            "$HF_REPO_PREFIX-distill-adapter" \
            output/distillation/lora_adapter \
            --revision "$HF_REVISION" \
            --commit-message "Distill adapter from $GIT_REF (model=$TRAIN_MODEL)" || true
    fi
    if [ -n "$EXPORT_GGUF" ] && [ -d output/distillation/lora_adapter-gguf ]; then
        uv tool run hf upload \
            "$HF_REPO_PREFIX-distill-adapter-gguf" \
            output/distillation/lora_adapter-gguf \
            --revision "$HF_REVISION" \
            --commit-message "GGUF ($EXPORT_GGUF) from $GIT_REF (model=$TRAIN_MODEL)" || true
    fi
else
    echo "=== Skipping HuggingFace Hub upload (no token) ==="
fi

# Stop MLflow server and upload its DB to S3
kill "$MLFLOW_PID" 2>/dev/null || true
sleep 2

echo "=== Uploading MLflow data to S3 ==="
if [ -d "$MLFLOW_DIR" ]; then
    aws s3 sync "$MLFLOW_DIR" "s3://$S3_BUCKET/$MODEL_KEY/mlflow/" --region "$REGION"
fi

echo "=== Training complete at $(date -u) ==="
