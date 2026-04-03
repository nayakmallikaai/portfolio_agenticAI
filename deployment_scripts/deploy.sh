#!/bin/bash
# deploy.sh — deploy a new image to EKS and run DB migrations
#
# What it does:
#   1. Builds a new Docker image tagged with the current git SHA
#   2. Pushes it to ECR
#   3. Updates app-deployment.yaml with the new tag
#   4. Runs DB migrations as a one-off k8s Job (before the app rolls out)
#   5. Applies the updated deployment — rolling restart
#   6. Waits for rollout to complete
#   7. Smoke tests the live URL
#   8. Optionally resets eval_user to seed state (--reset-eval)
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

# ── Config ────────────────────────────────────────────────────────────────────
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
echo "  Image  : $FULL_IMAGE"
echo "  Region : $REGION"
echo "  Cluster: $CLUSTER_NAME"

# ── Step 1: Build and push image ──────────────────────────────────────────────
echo -e "\n${YELLOW}Step 1: Building and pushing Docker image...${NC}"

aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

docker buildx build \
    --platform linux/amd64 \
    --push \
    -t $FULL_IMAGE \
    -t $ECR_REGISTRY/$IMAGE_NAME:latest \
    "$PROJECT_ROOT"

echo -e "${GREEN}Image pushed: $FULL_IMAGE${NC}"

# ── Step 2: Update manifest ───────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 2: Updating app-deployment.yaml with new image tag...${NC}"

sed -i.bak \
    "s|$ECR_REGISTRY/$IMAGE_NAME:.*|$FULL_IMAGE|g" \
    "$MANIFEST"
rm -f "$MANIFEST.bak"

echo "  Image tag in manifest → $IMAGE_TAG"

# ── Step 3: Configure kubectl ─────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 3: Configuring kubectl...${NC}"

aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

# ── Step 4: Run DB migrations ─────────────────────────────────────────────────
# Runs migrate_db() from the new image as a one-off Job before the app rolls out.
# This is safe to run while the old app is still serving — all migrations use
# ALTER TABLE ... ADD COLUMN IF NOT EXISTS (additive, non-breaking).
# The Job deletes itself after completion (ttlSecondsAfterFinished: 120).
echo -e "\n${YELLOW}Step 4: Running database migrations...${NC}"

# Fetch DATABASE_URL from the running pod's env (avoids storing it in script)
DB_URL=$(kubectl exec deployment/portfolio-app -- \
    sh -c 'echo $DATABASE_URL' 2>/dev/null || true)

if [ -z "$DB_URL" ]; then
    echo -e "${RED}Could not read DATABASE_URL from running pod.${NC}"
    echo "Falling back — migrations will run on app startup instead."
    SKIP_MIGRATION_JOB=true
else
    SKIP_MIGRATION_JOB=false
fi

if [ "$SKIP_MIGRATION_JOB" = false ]; then
    # Delete any leftover job from a prior deploy
    kubectl delete job db-migrate --ignore-not-found

    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migrate
spec:
  ttlSecondsAfterFinished: 120
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: $FULL_IMAGE
          command: ["python", "-c", "from db.engine import engine, migrate_db; migrate_db(); print('Migrations complete.')"]
          env:
            - name: DATABASE_URL
              value: "$DB_URL"
EOF

    echo "  Waiting for migration job to complete..."
    kubectl wait --for=condition=complete job/db-migrate --timeout=120s

    MIGRATE_LOG=$(kubectl logs job/db-migrate 2>/dev/null | tail -5)
    echo "  Migration output: $MIGRATE_LOG"
    echo -e "${GREEN}DB migrations complete${NC}"
fi

# ── Step 5: Roll out new app pods ─────────────────────────────────────────────
echo -e "\n${YELLOW}Step 5: Applying deployment and waiting for rollout...${NC}"

kubectl apply -f "$MANIFEST"
kubectl rollout status deployment/portfolio-app --timeout=300s

echo -e "${GREEN}Rollout complete${NC}"

# ── Step 6: Verify migrations ran in new pods ─────────────────────────────────
echo -e "\n${YELLOW}Step 6: Verifying migrations in new pod...${NC}"

VERIFY=$(kubectl exec deployment/portfolio-app -- \
    python -c "
from db.engine import engine
from sqlalchemy import text, inspect
with engine.connect() as c:
    cols = [r[0] for r in c.execute(text(\"SELECT column_name FROM information_schema.columns WHERE table_name='portfolios' ORDER BY column_name\"))]
    print('portfolios columns:', cols)
" 2>/dev/null || echo "verification skipped")

echo "  $VERIFY"

# Check buy_price column exists
if echo "$VERIFY" | grep -q "buy_price"; then
    echo -e "${GREEN}Schema verified — buy_price column present${NC}"
else
    echo -e "${YELLOW}Warning: could not verify schema — check pod logs if issues occur${NC}"
fi

# ── Step 7: Smoke test ────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Step 7: Smoke test...${NC}"

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

# ── Step 8: Optional eval_user reset ─────────────────────────────────────────
if [ "$RESET_EVAL" = true ]; then
    echo -e "\n${YELLOW}Step 8: Resetting eval_user to seed state...${NC}"

    kubectl delete job reset-eval-user --ignore-not-found

    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: reset-eval-user
spec:
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: reset
          image: $FULL_IMAGE
          command: ["python", "-m", "db.reset_eval_user"]
          env:
            - name: DATABASE_URL
              value: "$DB_URL"
EOF

    kubectl wait --for=condition=complete job/reset-eval-user --timeout=60s
    kubectl logs job/reset-eval-user 2>/dev/null || true
    echo -e "${GREEN}eval_user reset${NC}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}DEPLOY COMPLETE — $IMAGE_TAG${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
[ -n "$URL" ] && echo "App live at: http://$URL"
echo ""
echo "Useful commands:"
echo "  kubectl get pods -l app=portfolio-app"
echo "  kubectl logs deployment/portfolio-app --tail=50"
echo "  python -m db.reset_eval_user"
echo "  python -m eval.run_agent_eval --url http://$URL --user eval_user"
echo -e "${GREEN}════════════════════════════════════════${NC}"
