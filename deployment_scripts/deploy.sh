#!/bin/bash
# deploy.sh — subsequent deployment to EKS (run after setup.sh has been run once)
#
# What it does:
#   1. Builds a new Docker image tagged with the current git SHA
#   2. Pushes it to ECR
#   3. Updates app-deployment.yaml with the new tag
#   4. Applies the manifest — ArgoCD or kubectl picks up the rollout
#   5. Waits for the rollout to complete
#   6. Optionally resets eval_user to seed state (--reset-eval)
#
# Usage:
#   ./deployment_scripts/deploy.sh                # deploy current HEAD
#   ./deployment_scripts/deploy.sh --reset-eval   # also reset eval_user after deploy

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

RESET_EVAL=false
for arg in "$@"; do
  [[ "$arg" == "--reset-eval" ]] && RESET_EVAL=true
done

# ── Config ────────────────────────────────────────────────
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-south-1
CLUSTER_NAME=portfolio-cluster
ECR_REGISTRY=$AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com
IMAGE_NAME=portfolio-app
IMAGE_TAG=$(git rev-parse --short HEAD)
FULL_IMAGE=$ECR_REGISTRY/$IMAGE_NAME:$IMAGE_TAG

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
MANIFEST=$SCRIPT_DIR/k8s/app-deployment.yaml

echo -e "${GREEN}Deploying Portfolio Agent — $IMAGE_TAG${NC}"
echo "  Image : $FULL_IMAGE"
echo "  Region: $REGION / Cluster: $CLUSTER_NAME"

# ── Step 1: Build and push ────────────────────────────────
echo -e "\n${YELLOW}Step 1: Building and pushing image...${NC}"

aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

docker buildx build \
    --platform linux/amd64 \
    --push \
    -t $FULL_IMAGE \
    -t $ECR_REGISTRY/$IMAGE_NAME:latest \
    "$PROJECT_ROOT"

echo -e "${GREEN}Image pushed: $FULL_IMAGE${NC}"

# ── Step 2: Update manifest with new image tag ────────────
echo -e "\n${YELLOW}Step 2: Updating app-deployment.yaml...${NC}"

# Replace any existing ECR image line (any tag) with the new SHA tag
sed -i.bak \
    "s|$ECR_REGISTRY/$IMAGE_NAME:.*|$FULL_IMAGE|g" \
    "$MANIFEST"
rm -f "$MANIFEST.bak"

echo "  Image tag in manifest → $IMAGE_TAG"

# ── Step 3: Apply to cluster ──────────────────────────────
echo -e "\n${YELLOW}Step 3: Configuring kubectl and applying manifest...${NC}"

aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

kubectl apply -f "$MANIFEST"

# ── Step 4: Wait for rollout ──────────────────────────────
echo -e "\n${YELLOW}Step 4: Waiting for rollout...${NC}"

kubectl rollout status deployment/portfolio-app --timeout=300s

echo -e "${GREEN}Rollout complete${NC}"

# ── Step 5: Smoke test ────────────────────────────────────
echo -e "\n${YELLOW}Step 5: Smoke test...${NC}"

URL=$(kubectl get service portfolio-app-service \
    -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

if [ -z "$URL" ]; then
    echo -e "${RED}Load balancer URL not available — skipping smoke test${NC}"
else
    sleep 5
    if curl -sf "http://$URL/" > /dev/null; then
        echo -e "${GREEN}Smoke test passed — http://$URL${NC}"
    else
        echo -e "${RED}Smoke test failed${NC}"
        exit 1
    fi
fi

# ── Step 6: Optional eval_user reset ─────────────────────
if [ "$RESET_EVAL" = true ]; then
    echo -e "\n${YELLOW}Step 6: Resetting eval_user to seed state...${NC}"

    # Run reset script via a one-off k8s job using the app image
    kubectl run reset-eval-user \
        --image=$FULL_IMAGE \
        --restart=Never \
        --rm -it \
        --env="DATABASE_URL=$(kubectl get secret app-secret \
            -o jsonpath='{.data.database-url}' 2>/dev/null | base64 -d || \
            echo $DATABASE_URL)" \
        -- python -m db.reset_eval_user || true

    echo -e "${GREEN}eval_user reset${NC}"
fi

# ── Done ──────────────────────────────────────────────────
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}DEPLOY COMPLETE — $IMAGE_TAG${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "To run the eval suite against the live environment:"
echo "  python -m eval.run_agent_eval --url http://$URL --user eval_user"
echo ""
echo "To reset eval_user before running evals:"
echo "  python -m db.reset_eval_user"
echo ""
echo "To check pod status:"
echo "  kubectl get pods -l app=portfolio-app"
echo -e "${GREEN}════════════════════════════════════════${NC}"
