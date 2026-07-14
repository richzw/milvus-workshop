#!/usr/bin/env bash
# 端到端驱动：装工具 -> 灌测试数据+索引 -> MinIO 备份到 S3 -> 目标 Milvus 从 S3 恢复 -> 验证。
#
# 用法：
#   1. 改 env.sh 里的地址/桶/密钥。
#   2. 确保源 Milvus(+MinIO) 在跑；目标 Milvus(连 S3, etcd 干净) 在跑。
#   3. ./run-all.sh
#
# 注意：第 4 步「部署目标 Milvus」无法在此脚本里替你完成（依赖 K8s/Helm 或
# docker-compose 环境），需在跑 03 之前手动部署好，详见 README.md。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

echo "===== [0/4] 安装 milvus-backup ====="
bash "${HERE}/00-install-backup.sh"

echo "===== [1/4] 源 Milvus 灌入测试数据 + 建索引 ====="
python3 "${HERE}/01-seed-data.py"

echo "===== [2/4] MinIO -> S3 备份 ====="
bash "${HERE}/02-backup.sh"

cat <<MSG

===== 暂停：请确认目标 Milvus 已就绪 =====
目标 Milvus (${DST_MILVUS_HOST}:${DST_MILVUS_PORT}) 必须：
  - 全新部署、etcd 干净、collection 列表为空
  - 对象存储指向 S3 桶 ${S3_BUCKET} 的 rootPath=${S3_DATA_ROOTPATH}
就绪后按回车继续恢复。
MSG
read -r _

echo "===== [3/4] 目标 Milvus 从 S3 恢复 ====="
bash "${HERE}/03-restore.sh"

echo "===== [4/4] 验证数据完整性 ====="
python3 "${HERE}/04-verify.py"

echo "全部完成。"
