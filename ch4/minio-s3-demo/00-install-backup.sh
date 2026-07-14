#!/usr/bin/env bash
# 步骤 0：安装 milvus-backup 并下载配置模板。
set -euo pipefail
source "$(dirname "$0")/env.sh"

cd "${WORKDIR}"
mkdir -p configs logs

PKG="milvus-backup_${BACKUP_OS}_${BACKUP_ARCH}.tar.gz"
echo ">> 下载 ${PKG} (${BACKUP_VERSION})"
curl -fL -o milvus-backup.tar.gz \
  "https://github.com/zilliztech/milvus-backup/releases/download/${BACKUP_VERSION}/${PKG}"

tar -xzf milvus-backup.tar.gz
chmod +x milvus-backup

echo ">> 校验"
"${BACKUP_BIN}" --help >/dev/null
echo "milvus-backup 安装完成: ${BACKUP_BIN}"
