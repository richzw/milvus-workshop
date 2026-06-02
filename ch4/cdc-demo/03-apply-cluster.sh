#!/usr/bin/env bash
set -euo pipefail

MILVUS_NS="milvus"

echo "[1/3] Create namespace ${MILVUS_NS}"
kubectl create namespace ${MILVUS_NS} --dry-run=client -o yaml | kubectl apply -f -

echo "[2/3] Apply source cluster"
kubectl apply -f 01-source-cluster.yaml

echo "[3/3] Apply target cluster"
kubectl apply -f 02-target-cluster.yaml

echo "Milvus source and target cluster manifests applied."