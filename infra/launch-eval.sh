#!/usr/bin/env bash
#
# Launch a CloudFormation eval tracing stack for a single HuggingFace model.
#
# Deploys the eval-stack.yaml template, waits for the EC2 instance
# to start, and prints SSM commands for monitoring.
#
# Prerequisites:
#   - AWS CLI v2 configured with appropriate credentials
#   - Session Manager plugin installed (for SSM access)
#
# Usage:
#   ./infra/launch-eval.sh --model Qwen/Qwen3.5-4B
#   ./infra/launch-eval.sh --model mbrt/uno-sft-adapter --instance-type g6e.4xlarge
#   ./infra/launch-eval.sh --model Qwen/Qwen3.5-9B --max-model-len 8192
#   ./infra/launch-eval.sh --model mbrt/uno-distill-adapter --stack-name my-eval-01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/eval-stack.yaml"

# Defaults
MODEL=""
INSTANCE_TYPE="g6e.4xlarge"
MAX_MODEL_LEN=16384
GIT_REPO="https://github.com/mbrt/pkna-uno.git"
GIT_REF="main"
STACK_NAME=""
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
HF_TOKEN_SSM_PATH="/pkna-uno/hf-token"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --model MODEL           HuggingFace model ID or adapter (required)
    --instance-type TYPE    EC2 instance type (default: $INSTANCE_TYPE)
    --max-model-len N       Max context length for vLLM (default: $MAX_MODEL_LEN)
    --git-repo URL          Git repository URL (default: $GIT_REPO)
    --git-ref REF           Git branch or tag (default: $GIT_REF)
    --hf-token-ssm-path P  SSM path for HF token (default: $HF_TOKEN_SSM_PATH)
    --stack-name NAME       CloudFormation stack name (auto-generated if omitted)
    --region REGION         AWS region (default: $REGION)
    -h, --help              Show this help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --git-repo) GIT_REPO="$2"; shift 2 ;;
        --git-ref) GIT_REF="$2"; shift 2 ;;
        --hf-token-ssm-path) HF_TOKEN_SSM_PATH="$2"; shift 2 ;;
        --stack-name) STACK_NAME="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required" >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template not found at $TEMPLATE" >&2
    exit 1
fi

# Generate stack name from model and timestamp if not provided
if [ -z "$STACK_NAME" ]; then
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9]|-|g' | tr '[:upper:]' '[:lower:]')
    STACK_NAME="uno-eval-${MODEL_SHORT}-$(date +%Y%m%d-%H%M%S)"
fi

echo "================================================================"
echo "  Uno Eval Tracing Stack"
echo "================================================================"
echo ""
echo "  Stack name:     $STACK_NAME"
echo "  Region:         $REGION"
echo "  Instance type:  $INSTANCE_TYPE"
echo "  Model:          $MODEL"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  Git ref:        $GIT_REF"
echo ""

aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --template-body "file://$TEMPLATE" \
    --capabilities CAPABILITY_IAM \
    --parameters \
        "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE" \
        "ParameterKey=Model,ParameterValue=$MODEL" \
        "ParameterKey=MaxModelLen,ParameterValue=$MAX_MODEL_LEN" \
        "ParameterKey=GitRepo,ParameterValue=$GIT_REPO" \
        "ParameterKey=GitRef,ParameterValue=$GIT_REF" \
        "ParameterKey=HfTokenSsmPath,ParameterValue=$HF_TOKEN_SSM_PATH"

echo ""
echo "Stack creation initiated. Waiting for instance to launch..."

aws cloudformation wait stack-create-complete \
    --stack-name "$STACK_NAME" \
    --region "$REGION"

# Retrieve outputs
INSTANCE_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
    --output text)

BUCKET=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" \
    --output text)

echo ""
echo "================================================================"
echo "  Stack deployed successfully"
echo "================================================================"
echo ""
echo "  Instance ID:    $INSTANCE_ID"
echo "  S3 Bucket:      $BUCKET"
echo ""
echo "  Shell access (SSM):"
echo "    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo ""
echo "  Tail eval log (SSM):"
echo "    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "    # then: tail -f /home/ubuntu/eval.log"
echo ""
MODEL_KEY=$(echo "$MODEL" | sed 's|.*/||')
echo "  Download results after completion:"
echo "    aws s3 sync s3://$BUCKET/$MODEL_KEY/ ./eval-results/ --region $REGION"
echo ""
echo "  Delete stack (bucket is retained):"
echo "    aws cloudformation delete-stack --stack-name $STACK_NAME --region $REGION"
echo ""
