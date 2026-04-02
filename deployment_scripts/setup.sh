#!/bin/bash
# setup.sh — run once to set up everything on AWS
# Usage: ./setup.sh

set -e  # stop on any error

# ── Colors for output ──────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

echo -e "${GREEN}Starting AWS setup for Portfolio Agent...${NC}"

# ── Variables ──────────────────────────────────────────
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-south-1
CLUSTER_NAME=portfolio-cluster
ECR_REGISTRY=$AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# resolve paths relative to this script regardless of where it is invoked from
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

echo "Account: $AWS_ACCOUNT"
echo "Region:  $REGION"
echo "Cluster: $CLUSTER_NAME"

# ── Step 1: Secrets ───────────────────────────────────
# (commented out — keys hardcoded below for now)
 echo -e "\n${YELLOW}Step 1: Collecting secrets...${NC}"
 read -s -p "ANTHROPIC_API_KEY: "    ANTHROPIC_API_KEY;    echo
 read -s -p "OPENAI_API_KEY: "       OPENAI_API_KEY;       echo
 read -s -p "POSTGRES_PASSWORD: "    POSTGRES_PASSWORD;    echo
 read -s -p "LANGCHAIN_API_KEY: "    LANGCHAIN_API_KEY;    echo
 read    -p "LANGCHAIN_PROJECT: "    LANGCHAIN_PROJECT;    echo
 read    -p "LANGCHAIN_TRACING_V2: " LANGCHAIN_TRACING_V2; echo
 read    -p "LANGCHAIN_ENDPOINT: "   LANGCHAIN_ENDPOINT;   echo


# ── Step 2: Create ECR Repository ─────────────────────
echo -e "\n${YELLOW}Step 2: Creating ECR repository...${NC}"

aws ecr create-repository \
    --repository-name portfolio-app \
    --region $REGION \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "portfolio-app repo already exists"

echo -e "${GREEN}ECR repository ready${NC}"

# ── Step 3: Build and Push Docker Image ───────────────
echo -e "\n${YELLOW}Step 3: Building and pushing Docker image...${NC}"

aws ecr get-login-password --region $REGION | \
    docker login --username AWS \
    --password-stdin $ECR_REGISTRY

docker buildx build \
    --platform linux/amd64 \
    --push \
    -t $ECR_REGISTRY/portfolio-app:latest \
    "$PROJECT_ROOT"

echo -e "${GREEN}Image pushed to ECR${NC}"

# ── Step 4: Create EKS Cluster ────────────────────────
echo -e "\n${YELLOW}Step 4: Creating EKS cluster (15-20 mins)...${NC}"

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
    --timeout 40m \
    2>/dev/null || echo "Cluster already exists, continuing..."

echo -e "${GREEN}Cluster ready${NC}"

# ── Step 5: Update kubeconfig ─────────────────────────
echo -e "\n${YELLOW}Step 5: Configuring kubectl...${NC}"

aws eks update-kubeconfig \
    --name $CLUSTER_NAME \
    --region $REGION

# wait for nodes to be ready (kubeconfig must be set first)
echo "Waiting for nodes to be ready..."
kubectl wait --for=condition=ready node \
    --all --timeout=300s

kubectl get nodes
echo -e "${GREEN}kubectl configured${NC}"

# ── Step 6: Create k8s Secrets ────────────────────────
echo -e "\n${YELLOW}Step 6: Creating k8s secrets...${NC}"

kubectl create secret generic postgres-secret \
    --from-literal=password="$POSTGRES_PASSWORD" \
    2>/dev/null || kubectl create secret generic postgres-secret \
    --from-literal=password="$POSTGRES_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic app-secret \
    --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
    --from-literal=openai-api-key="$OPENAI_API_KEY" \
    --from-literal=langchain-api-key="$LANGCHAIN_API_KEY" \
    --from-literal=langchain-project="$LANGCHAIN_PROJECT" \
    --from-literal=langchain-tracing="$LANGCHAIN_TRACING_V2" \
    --from-literal=langchain-endpoint="$LANGCHAIN_ENDPOINT" \
    2>/dev/null || kubectl create secret generic app-secret \
    --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
    --from-literal=openai-api-key="$OPENAI_API_KEY" \
    --from-literal=langchain-api-key="$LANGCHAIN_API_KEY" \
    --from-literal=langchain-project="$LANGCHAIN_PROJECT" \
    --from-literal=langchain-tracing="$LANGCHAIN_TRACING_V2" \
    --from-literal=langchain-endpoint="$LANGCHAIN_ENDPOINT" \
    --dry-run=client -o yaml | kubectl apply -f -

echo -e "${GREEN}Secrets created${NC}"

# ── Step 7: Update k8s manifests with ECR URL ─────────
echo -e "\n${YELLOW}Step 7: Updating k8s manifests...${NC}"

# idempotent — handles placeholder and already-replaced account IDs
sed -i.bak \
    -e "s|YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com|$ECR_REGISTRY|g" \
    -e "s|[0-9]\{12\}.dkr.ecr.ap-south-1.amazonaws.com|$ECR_REGISTRY|g" \
    "$SCRIPT_DIR/k8s/app-deployment.yaml"

echo -e "${GREEN}Manifests updated${NC}"

# ── Step 8: Install EBS CSI Driver (required for PVCs on EKS 1.23+) ──────────
echo -e "\n${YELLOW}Step 8: Installing EBS CSI driver...${NC}"

eksctl create addon \
    --name aws-ebs-csi-driver \
    --cluster $CLUSTER_NAME \
    --region $REGION \
    --force \
    2>/dev/null || echo "EBS CSI driver already installed"

# wait for ebs-csi-node pods to appear then become ready
echo "Waiting for EBS CSI pods to appear..."
for i in {1..24}; do
    COUNT=$(kubectl get pods -n kube-system -l app=ebs-csi-node --no-headers 2>/dev/null | grep -c . || true)
    if [ "$COUNT" -gt 0 ]; then break; fi
    echo "  not yet ($i/24)..."
    sleep 10
done

kubectl wait --for=condition=ready pod \
    -l app=ebs-csi-node \
    -n kube-system \
    --timeout=120s

echo -e "${GREEN}EBS CSI driver ready${NC}"

# ── Step 9: Deploy to EKS ─────────────────────────────
echo -e "\n${YELLOW}Step 9: Deploying to EKS...${NC}"

kubectl apply -f "$SCRIPT_DIR/k8s/postgres-pvc.yaml"
kubectl apply -f "$SCRIPT_DIR/k8s/postgres-deployment.yaml"

# wait for postgres pod to appear then become ready
echo "Waiting for postgres pod to appear..."
for i in {1..12}; do
    COUNT=$(kubectl get pods -l app=postgres --no-headers 2>/dev/null | grep -c . || true)
    if [ "$COUNT" -gt 0 ]; then break; fi
    echo "  not yet ($i/12)..."
    sleep 10
done

kubectl wait --for=condition=ready pod \
    -l app=postgres --timeout=300s || {
    echo -e "${RED}Postgres pod not ready — diagnostics:${NC}"
    kubectl describe pod -l app=postgres | tail -20
    kubectl logs -l app=postgres --tail=30 2>/dev/null || true
    exit 1
}

# deploy app — tables created automatically on startup
kubectl apply -f "$SCRIPT_DIR/k8s/app-deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/k8s/hpa.yaml"

kubectl rollout status deployment/portfolio-app --timeout=300s

echo -e "${GREEN}App deployed${NC}"

# ── Step 10: Install Metrics Server (needed for HPA) ───
echo -e "\n${YELLOW}Step 10: Installing metrics server...${NC}"

kubectl apply -f \
    https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

echo -e "${GREEN}Metrics server installed${NC}"

# ── Step 11: Install ArgoCD ───────────────────────────
echo -e "\n${YELLOW}Step 11: Installing ArgoCD...${NC}"

kubectl create namespace argocd 2>/dev/null || true

kubectl apply -n argocd -f \
    https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# wait for ArgoCD server pod to appear then become ready
echo "Waiting for ArgoCD server pod to appear..."
for i in {1..18}; do
    COUNT=$(kubectl get pods -n argocd -l app.kubernetes.io/name=argocd-server --no-headers 2>/dev/null | grep -c . || true)
    if [ "$COUNT" -gt 0 ]; then break; fi
    echo "  not yet ($i/18)..."
    sleep 10
done

kubectl wait --for=condition=ready pod \
    -l app.kubernetes.io/name=argocd-server \
    -n argocd \
    --timeout=180s

# wait for ArgoCD CRDs to be fully registered
kubectl wait --for=condition=established \
    crd/applications.argoproj.io --timeout=60s

# apply ArgoCD app config — retry in case API server lags behind CRD registration
for i in {1..6}; do
    kubectl apply -f "$SCRIPT_DIR/argocd/application.yaml" && break
    echo "  CRD not ready yet, retrying in 10s ($i/6)..."
    sleep 10
done

echo -e "${GREEN}ArgoCD installed${NC}"

# ── Step 12: Get Public URL ───────────────────────────
echo -e "\n${YELLOW}Step 12: Getting public URL...${NC}"

echo "Waiting for load balancer URL (up to 2 mins)..."
for i in {1..24}; do
    URL=$(kubectl get service portfolio-app-service \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
    if [ -n "$URL" ]; then break; fi
    sleep 5
done

# ── Step 13: Smoke Test ───────────────────────────────
echo -e "\n${YELLOW}Step 13: Running smoke test...${NC}"

if [ -z "$URL" ]; then
    echo -e "${RED}Load balancer URL not available — skipping smoke test${NC}"
    echo "Check: kubectl get service portfolio-app-service"
else
    sleep 10
    curl -f http://$URL/ && \
        echo -e "${GREEN}Smoke test passed${NC}" || \
        echo -e "${RED}Smoke test failed — check: kubectl get pods${NC}"
fi

# ── Step 14: Done ─────────────────────────────────────
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}SETUP COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "Your app is live at:"
echo "  http://$URL"
echo ""
echo "Add these to GitHub Secrets (Settings → Secrets → Actions):"
echo "────────────────────────────────────────"
echo "AWS_ACCOUNT:    $AWS_ACCOUNT"
echo "AWS_REGION:     $REGION"
echo "CLUSTER_NAME:   $CLUSTER_NAME"
echo "────────────────────────────────────────"
echo "Also add: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,"
echo "  ANTHROPIC_API_KEY, OPENAI_API_KEY, LANGCHAIN_API_KEY,"
echo "  LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT"
echo ""
echo "ArgoCD UI:"
echo "  kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "  open https://localhost:8080  (username: admin)"
ARGOCD_PASS=$(kubectl -n argocd get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" | base64 -d)
echo "  password: $ARGOCD_PASS"
echo ""
echo "Retrieve postgres password:"
echo "  kubectl get secret postgres-secret -o jsonpath='{.data.password}' | base64 -d"
echo ""
echo "Rotate postgres password:"
echo "  kubectl create secret generic postgres-secret --from-literal=password=NEWPASSWORD --dry-run=client -o yaml | kubectl apply -f -"
echo "  kubectl rollout restart deployment/postgres deployment/portfolio-app"
echo ""
echo "Delete cluster:"
echo "  eksctl delete cluster --name $CLUSTER_NAME --region $REGION"
echo -e "${GREEN}════════════════════════════════════════${NC}"
