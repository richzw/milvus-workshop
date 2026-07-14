#!/usr/bin/env bash
# 停源环境。加 --purge 同时删数据卷。
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--purge" ]]; then
  docker compose down -v
  rm -rf ./volumes
  echo "源环境已删除（含数据卷）"
else
  docker compose down
  echo "源环境已停止（数据卷保留，--purge 可清空）"
fi
