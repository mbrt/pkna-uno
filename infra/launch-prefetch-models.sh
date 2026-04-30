#!/usr/bin/env bash
#
# Launch a CloudFormation model cache stack to pre-fetch HuggingFace models
# into a persistent S3 bucket for fast retrieval by training and eval jobs.
#
# Uses `aws cloudformation deploy` (idempotent create-or-update) so the same
# stack can be re-run with new models to add them to the existing bucket.
#
# Prerequisites:
#   - AWS CLI v2 configured with appropriate credentials
#   - Session Manager plugin installed (for SSM access)
#
# Usage:
#   ./infra/launch-prefetch-models.sh \
#       --bucket-name my-model-cache \
#       --models unsloth/Qwen3.5-4B
#   ./infra/launch-prefetch-models.sh \
#       --bucket-name my-model-cache \
#       --models "unsloth/Qwen3.5-4B,unsloth/Qwen3.5-0.8B"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/model-cache-stack.yaml"

# Defaults
BUCKET_NAME=""
MODELS=""
INSTANCE_TYPE="c5n.xlarge"
GIT_REPO="https://github.com/mbrt/pkna-uno.git"
GIT_REF="main"
# Fixed stack name: reused across runs so all prefetches share one bucket.
STACK_NAME="uno-model-cache"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
HF_TOKEN_SSM_PATH="/pkna-uno/hf-token"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --bucket-name NAME      S3 bucket name for the model cache (required)
    --models COMMA_LIST     Comma-separated HuggingFace model IDs to fetch (required)
    --instance-type TYPE    EC2 instance type (default: $INSTANCE_TYPE)
    --git-repo URL          Git repository URL (default: $GIT_REPO)
    --git-ref REF           Git branch or tag (default: $GIT_REF)
    --hf-token-ssm-path P   SSM path for HF token (default: $HF_TOKEN_SSM_PATH)
    --stack-name NAME       CloudFormation stack name (default: $STACK_NAME)
    --region REGION         AWS region (default: $REGION)
    -h, --help              Show this help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --bucket-name) BUCKET_NAME="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --git-repo) GIT_REPO="$2"; shift 2 ;;
        --git-ref) GIT_REF="$2"; shift 2 ;;
        --hf-token-ssm-path) HF_TOKEN_SSM_PATH="$2"; shift 2 ;;
        --stack-name) STACK_NAME="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$BUCKET_NAME" ]; then
    echo "ERROR: --bucket-name is required" >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

if [ -z "$MODELS" ]; then
    echo "ERROR: --models is required" >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template not found at $TEMPLATE" >&2
    exit 1
fi

echo "================================================================"
echo "  Uno Model Cache Pre-fetch"
echo "================================================================"
echo ""
echo "  Stack name:     $STACK_NAME"
echo "  Region:         $REGION"
echo "  Instance type:  $INSTANCE_TYPE"
echo "  Bucket name:    $BUCKET_NAME"
echo "  Models:         $MODELS"
echo "  Git ref:        $GIT_REF"
echo ""

aws cloudformation deploy \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --template-file "$TEMPLATE" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
        "BucketName=$BUCKET_NAME" \
        "Models=$MODELS" \
        "InstanceType=$INSTANCE_TYPE" \
        "GitRepo=$GIT_REPO" \
        "GitRef=$GIT_REF" \
        "HfTokenSsmPath=$HF_TOKEN_SSM_PATH"

# Retrieve outputs
INSTANCE_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
    --output text)

echo ""
echo "================================================================"
echo "  Stack deployed — prefetch instance is running"
echo "================================================================"
echo ""
echo "  Instance ID:    $INSTANCE_ID"
echo "  S3 Bucket:      $BUCKET_NAME"
echo ""
echo "  Shell access (SSM):"
echo "    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo ""
echo "  Tail prefetch log (SSM):"
echo "    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "    # then: tail -f /home/ubuntu/prefetch.log"
echo ""
echo "  Check cached models after completion:"
echo "    aws s3 ls s3://$BUCKET_NAME/models/ --region $REGION"
echo ""
echo "  Use the cache in training/eval runs:"
echo "    ./infra/launch-training.sh --model-cache-bucket $BUCKET_NAME ..."
echo "    ./infra/launch-eval.sh --model-cache-bucket $BUCKET_NAME ..."
echo ""
