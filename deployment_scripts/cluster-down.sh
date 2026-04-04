#!/bin/bash
# cluster-down.sh — tear down EKS cluster to $0 cost
# ECR images and secrets are NOT deleted — bring-up is fast.
# WARNING: Postgres EBS volume is deleted with the cluster.
#          All DB data will be lost unless you snapshot first.
#
# Usage: ./deployment_scripts/cluster-down.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REGION=ap-south-1
CLUSTER_NAME=portfolio-cluster

echo -e "${YELLOW}Bringing down EKS cluster '${CLUSTER_NAME}'...${NC}"
echo -e "${RED}WARNING: All Postgres data will be lost.${NC}"
read -p "Continue? (yes/no): " CONFIRM
[[ "$CONFIRM" != "yes" ]] && echo "Aborted." && exit 0

# Delete the LoadBalancer service first so AWS cleans up the ELB
echo -e "\n${YELLOW}Deleting LoadBalancer service...${NC}"
kubectl delete service portfolio-app-service --ignore-not-found

echo "Waiting 30s for ELB to be released..."
sleep 30

# Delete the cluster (nodes, control plane, security groups, etc.)
echo -e "\n${YELLOW}Deleting EKS cluster (5-15 mins)...${NC}"
eksctl delete cluster --name $CLUSTER_NAME --region $REGION

echo -e "\n${GREEN}Cluster deleted. Cost is now \$0.${NC}"
echo "ECR images are retained — run cluster-up.sh to restore."
