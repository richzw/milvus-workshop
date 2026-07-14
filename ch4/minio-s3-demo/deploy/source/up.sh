#!/usr/bin/env bash
# 拉起源 Milvus + MinIO，等待健康。
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d
echo ">> 等待 Milvus proxy 健康 (最多 ~3min)..."
for i in $(seq 1 36); do
  if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then
    echo "源 Milvus 就绪: localhost:19530  (MinIO console: http://localhost:9001 minioadmin/minioadmin)"
    exit 0
  fi
  sleep 5
done
echo "等待超时，查看日志: docker compose -f $(pwd)/docker-compose.yml logs milvus" >&2
exit 1
