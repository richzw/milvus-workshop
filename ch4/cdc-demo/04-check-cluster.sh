#!/usr/bin/env bash
set -euo pipefail

MILVUS_NS="milvus"

echo "========== Milvus CR =========="
kubectl get milvus -n ${MILVUS_NS}

echo
echo "========== Pods =========="
kubectl get pods -n ${MILVUS_NS}

echo
echo "========== Services =========="
kubectl get svc -n ${MILVUS_NS}

echo
echo "========== Source CDC Pod =========="
kubectl get pods -n ${MILVUS_NS} | grep source-cluster-milvus-cdc || true

echo
echo "========== Not Running Pods =========="
kubectl get pods -n ${MILVUS_NS} --no-headers | awk '$3!="Running" && $3!="Completed" {print}'