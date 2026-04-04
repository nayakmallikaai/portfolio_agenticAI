#!/bin/bash
# cluster-up.sh — recreate EKS cluster for testing
# Assumes: ECR image already exists (no rebuild needed).
# Secrets are re-entered interactively.
#
# Usage: ./deployment_scripts/cluster-up.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-south-1
CLUSTER_NAME=portfolio-cluster
ECR_REGISTRY=$AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo -e "${GREEN}Bringing up EKS cluster '${CLUSTER_NAME}'...${NC}"

# ── Step 1: Secrets ───────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 1: Enter secrets...${NC}"
read -s -p "ANTHROPIC_API_KEY: "    ANTHROPIC_API_KEY;    echo
read -s -p "OPENAI_API_KEY: "       OPENAI_API_KEY;       echo
read -s -p "POSTGRES_PASSWORD: "    POSTGRES_PASSWORD;    echo
read -s -p "LANGCHAIN_API_KEY: "    LANGCHAIN_API_KEY;    echo
read    -p "LANGCHAIN_PROJECT: "    LANGCHAIN_PROJECT;    echo
read    -p "LANGCHAIN_TRACING_V2: " LANGCHAIN_TRACING_V2; echo
read    -p "LANGCHAIN_ENDPOINT: "   LANGCHAIN_ENDPOINT;   echo

# ── Step 2: Create EKS cluster ───────────────────────────────────────────────
echo -e "\n${YELLOW}Step 2: Creating EKS cluster (10-15 mins)...${NC}"

eksctl create cluster \
    --name $CLUSTER_NAME \
    --region $REGION \
    --nodegroup-name standard-workers \
    --node-type t3.medium \
    --nodes 2 \
    --nodes-min 2 \
    --nodes-max 5 \
    --managed \
    --with-oidc \
    --timeout 40m

echo -e "${GREEN}Cluster ready${NC}"

# ── Step 3: Configure kubectl ─────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 3: Configuring kubectl...${NC}"

aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

kubectl wait --for=condition=ready node --all --timeout=300s
kubectl get nodes

# ── Step 4: Secrets ───────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 4: Creating k8s secrets...${NC}"

kubectl create secret generic postgres-secret \
    --from-literal=password="$POSTGRES_PASSWORD"

kubectl create secret generic app-secret \
    --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
    --from-literal=openai-api-key="$OPENAI_API_KEY" \
    --from-literal=langchain-api-key="$LANGCHAIN_API_KEY" \
    --from-literal=langchain-project="$LANGCHAIN_PROJECT" \
    --from-literal=langchain-tracing="$LANGCHAIN_TRACING_V2" \
    --from-literal=langchain-endpoint="$LANGCHAIN_ENDPOINT"

# ── Step 5: EBS CSI driver ────────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 5: Installing EBS CSI driver...${NC}"

eksctl create addon \
    --name aws-ebs-csi-driver \
    --cluster $CLUSTER_NAME \
    --region $REGION \
    --force

for i in {1..24}; do
    COUNT=$(kubectl get pods -n kube-system -l app=ebs-csi-node --no-headers 2>/dev/null | grep -c . || true)
    [ "$COUNT" -gt 0 ] && break
    echo "  waiting ($i/24)..."; sleep 10
done

kubectl wait --for=condition=ready pod -l app=ebs-csi-node -n kube-system --timeout=120s

# ── Step 6: Deploy app ────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 6: Deploying app...${NC}"

kubectl apply -f "$SCRIPT_DIR/k8s/postgres-pvc.yaml"
kubectl apply -f "$SCRIPT_DIR/k8s/postgres-deployment.yaml"

for i in {1..12}; do
    COUNT=$(kubectl get pods -l app=postgres --no-headers 2>/dev/null | grep -c . || true)
    [ "$COUNT" -gt 0 ] && break
    echo "  waiting for postgres pod ($i/12)..."; sleep 10
done

kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s

kubectl apply -f "$SCRIPT_DIR/k8s/app-deployment.yaml"
kubectl rollout status deployment/portfolio-app --timeout=300s

# ── Step 7: Get URL ───────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 7: Waiting for LoadBalancer URL...${NC}"

for i in {1..24}; do
    URL=$(kubectl get service portfolio-app-service \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
    [ -n "$URL" ] && break
    sleep 5
done

sleep 10
curl -sf "http://$URL/" > /dev/null && \
    echo -e "${GREEN}Smoke test passed${NC}" || \
    echo -e "${RED}Smoke test failed — check: kubectl get pods${NC}"

echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}CLUSTER UP${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo "App live at: http://$URL"
echo ""
echo "When done testing, run:"
echo "  ./deployment_scripts/cluster-down.sh"
echo -e "${GREEN}════════════════════════════════════════${NC}"
