#!/usr/bin/env bash
set -euo pipefail

AWS_ACCOUNT_ID="861208159576"
AWS_REGION="us-east-1"
ECR_REPO="fifthseason/rag-api"
IMAGE_TAG="${1:-latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

# Move to project root (one level up from deploy/)
cd "$(dirname "$0")/.."

echo "==> Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Building image with Dockerfile.lite..."
docker build -t "${ECR_REPO}:${IMAGE_TAG}" -f Dockerfile.lite .

echo "==> Tagging image..."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

echo "==> Pushing to ECR..."
docker push "${ECR_URI}:${IMAGE_TAG}"

echo "==> Done! Image pushed: ${ECR_URI}:${IMAGE_TAG}"