#!/usr/bin/env bash
# 停目标环境。加 --purge 同时删本地卷(etcd/milvus 元数据)。S3 桶数据不动。
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--purge" ]]; then
  docker compose down -v
  rm -rf ./volumes
  echo "目标环境已删除（含本地卷；S3 数据未动）"
else
  docker compose down
  echo "目标环境已停止（本地卷保留，--purge 可清空）"
fi
