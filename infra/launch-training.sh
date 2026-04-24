#!/usr/bin/env bash
#
# Launch a CloudFormation training stack for Uno SFT + distillation.
#
# Deploys the training-stack.yaml template, waits for the EC2 instance
# to start, and prints SSM commands for monitoring.
#
# Prerequisites:
#   - AWS CLI v2 configured with appropriate credentials
#   - Session Manager plugin installed (for SSM port forwarding)
#
# Usage:
#   ./infra/launch-training.sh
#   ./infra/launch-training.sh --model Qwen/Qwen3.5-9B --instance-type g6e.2xlarge
#   ./infra/launch-training.sh --export-gguf q4_k_m
#   ./infra/launch-training.sh --sft-only
#   ./infra/launch-training.sh --stack-name my-run-01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/training-stack.yaml"

# Defaults matching the CloudFormation template
MODEL="unsloth/Qwen3.5-4B"
INSTANCE_TYPE="g6e.2xlarge"
EXPORT_GGUF=""
GIT_REPO="https://github.com/mbrt/pkna-uno.git"
GIT_REF="main"
RUN_DISTILL="true"
STACK_NAME=""
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
HF_REPO_PREFIX="mbrt/uno"
HF_TOKEN_SSM_PATH="/pkna-uno/hf-token"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --model MODEL           Base model (default: $MODEL)
  --instance-type TYPE    EC2 instance type (default: $INSTANCE_TYPE)
  --export-gguf METHOD    GGUF quantization method (e.g. q4_k_m)
  --git-repo URL          Git repository URL
  --git-ref REF           Git branch or tag (default: $GIT_REF)
  --sft-only              Skip distillation (SFT only)
  --hf-repo-prefix PFX    HuggingFace repo prefix (default: $HF_REPO_PREFIX)
  --hf-token-ssm-path P   SSM path for HF token (default: $HF_TOKEN_SSM_PATH)
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
    --export-gguf) EXPORT_GGUF="$2"; shift 2 ;;
    --git-repo) GIT_REPO="$2"; shift 2 ;;
    --git-ref) GIT_REF="$2"; shift 2 ;;
    --sft-only) RUN_DISTILL="false"; shift ;;
    --hf-repo-prefix) HF_REPO_PREFIX="$2"; shift 2 ;;
    --hf-token-ssm-path) HF_TOKEN_SSM_PATH="$2"; shift 2 ;;
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: Template not found at $TEMPLATE" >&2
  exit 1
fi

# Generate stack name from model and timestamp if not provided
if [ -z "$STACK_NAME" ]; then
  MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9]|-|g' | tr '[:upper:]' '[:lower:]')
  STACK_NAME="uno-train-${MODEL_SHORT}-$(date +%Y%m%d-%H%M%S)"
fi

echo "================================================================"
echo "  Uno Training Stack"
echo "================================================================"
echo ""
echo "  Stack name:     $STACK_NAME"
echo "  Region:         $REGION"
echo "  Instance type:  $INSTANCE_TYPE"
echo "  Model:          $MODEL"
echo "  Git ref:        $GIT_REF"
echo "  Distillation:   $RUN_DISTILL"
echo "  GGUF export:    ${EXPORT_GGUF:-none}"
echo "  HF repo prefix: $HF_REPO_PREFIX"
echo ""

aws cloudformation create-stack \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --template-body "file://$TEMPLATE" \
  --capabilities CAPABILITY_IAM \
  --parameters \
    "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE" \
    "ParameterKey=Model,ParameterValue=$MODEL" \
    "ParameterKey=ExportGguf,ParameterValue=$EXPORT_GGUF" \
    "ParameterKey=GitRepo,ParameterValue=$GIT_REPO" \
    "ParameterKey=GitRef,ParameterValue=$GIT_REF" \
    "ParameterKey=RunDistillation,ParameterValue=$RUN_DISTILL" \
    "ParameterKey=HfRepoPrefix,ParameterValue=$HF_REPO_PREFIX" \
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
echo "  MLflow UI (port forward via SSM):"
echo "    aws ssm start-session \\"
echo "      --target $INSTANCE_ID \\"
echo "      --document-name AWS-StartPortForwardingSession \\"
echo "      --parameters portNumber=5000,localPortNumber=5000 \\"
echo "      --region $REGION"
echo "    Then browse: http://localhost:5000"
echo ""
echo "  Shell access (SSM):"
echo "    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo ""
echo "  Tail training log (SSM):"
echo "    aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "    # then: tail -f /home/ubuntu/training.log"
echo ""
MODEL_KEY=$(echo "$MODEL" | sed 's|.*/||')
echo "  Download results after completion:"
echo "    aws s3 sync s3://$BUCKET/$MODEL_KEY/ ./training-results/ --region $REGION"
echo ""
echo "  Delete stack (bucket is retained):"
echo "    aws cloudformation delete-stack --stack-name $STACK_NAME --region $REGION"
echo ""
