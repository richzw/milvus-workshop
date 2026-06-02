#!/usr/bin/env bash
set -euo pipefail

OPERATOR_NS="milvus-operator"

echo "[1/5] Create namespace ${OPERATOR_NS}"
kubectl create namespace ${OPERATOR_NS} --dry-run=client -o yaml | kubectl apply -f -

echo "[2/5] Add Milvus Operator helm repo"
helm repo add zilliztech-milvus-operator https://zilliztech.github.io/milvus-operator/ || true

echo "[3/5] Update helm repo"
helm repo update zilliztech-milvus-operator

echo "[4/5] Install or upgrade Milvus Operator"
# CDC 需要 Operator >= v1.3.4
OPERATOR_VERSION="${OPERATOR_VERSION:-1.3.4}"
helm upgrade --install milvus-operator \
  zilliztech-milvus-operator/milvus-operator \
  -n ${OPERATOR_NS} \
  --version "${OPERATOR_VERSION}" \
  --wait

echo "[5/5] Check operator status"
kubectl rollout status -n ${OPERATOR_NS} deploy/milvus-operator --timeout=300s
kubectl get pods -n ${OPERATOR_NS}

echo "Milvus Operator installed/upgraded successfully."