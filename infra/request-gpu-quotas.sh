#!/usr/bin/env bash
#
# Request EC2 GPU (G/VT) service quota increases for a given region.
#
# Submits increase requests for both On-Demand and Spot G/VT instance
# vCPU quotas, which default to 0 in newly-enabled regions.
#
# Usage:
#   ./infra/request-gpu-quotas.sh --region eu-central-1
#   ./infra/request-gpu-quotas.sh --region eu-central-1 --vcpus 16

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
DESIRED_VCPUS=48

# G and VT instance family quotas (measured in vCPUs)
ONDEMAND_QUOTA="L-DB2E81BA"  # Running On-Demand G and VT instances
SPOT_QUOTA="L-3819A6DF"      # All G and VT Spot Instance Requests

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --region REGION   AWS region (default: $REGION)
    --vcpus N         Desired vCPU limit for each quota (default: $DESIRED_VCPUS)
    -h, --help        Show this help
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --region) REGION="$2"; shift 2 ;;
        --vcpus) DESIRED_VCPUS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

request_quota() {
    local code="$1"
    local name="$2"

    echo "--- $name ($code) ---"

    current=$(aws service-quotas get-service-quota \
        --service-code ec2 \
        --quota-code "$code" \
        --region "$REGION" \
        --query "Quota.Value" \
        --output text 2>/dev/null || echo "unknown")

    echo "  Current value: $current vCPUs"

    if [ "$current" != "unknown" ]; then
        at_least=$(printf '%.0f' "$current")
        if [ "$at_least" -ge "$DESIRED_VCPUS" ]; then
            echo "  Already >= $DESIRED_VCPUS vCPUs, skipping."
            return
        fi
    fi

    echo "  Requesting increase to $DESIRED_VCPUS vCPUs..."
    aws service-quotas request-service-quota-increase \
        --service-code ec2 \
        --quota-code "$code" \
        --desired-value "$DESIRED_VCPUS" \
        --region "$REGION"

    echo "  Request submitted."
}

echo "Region: $REGION"
echo "Desired vCPUs: $DESIRED_VCPUS"
echo ""

request_quota "$ONDEMAND_QUOTA" "Running On-Demand G and VT instances"
echo ""
request_quota "$SPOT_QUOTA" "All G and VT Spot Instance Requests"

echo ""
echo "Track request status:"
echo "  aws service-quotas list-requested-service-quota-changes-in-history \\"
echo "    --service-code ec2 --region $REGION"
