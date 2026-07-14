#!/usr/bin/env bash
# 拉起目标 Milvus（连 S3）。先渲染 user.yaml，再等待健康。
# 前置：S3 桶 ${S3_BUCKET} 已存在；env.sh 已填好 region/桶/密钥(或 IAM)。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

bash "${HERE}/render-user-yaml.sh"

cd "${HERE}"
docker compose up -d
echo ">> 等待目标 Milvus proxy 健康 (最多 ~3min)..."
for i in $(seq 1 36); do
  if curl -sf http://localhost:9092/healthz >/dev/null 2>&1; then
    echo "目标 Milvus 就绪: localhost:19531 (etcd 干净, 存储=S3)"
    echo "提示: 把 env.sh 的 DST_MILVUS_HOST 设为 localhost、DST_MILVUS_PORT 设为 19531"
    exit 0
  fi
  sleep 5
done
echo "等待超时，查看日志: docker compose -f ${HERE}/docker-compose.yml logs milvus" >&2
echo "常见原因: S3 桶不存在 / 密钥无权限 / region 错误。" >&2
exit 1
