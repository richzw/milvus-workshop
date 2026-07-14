#!/usr/bin/env bash
# 步骤 3：在连接 S3 的全新目标 Milvus 上恢复（含索引）。
# 前置：目标 Milvus 已部署、etcd 干净、对象存储指向 S3 的 ${S3_DATA_ROOTPATH}。
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "${WORKDIR}"

if [[ ! -f "${BACKUP_NAME_FILE}" ]]; then
  echo "找不到备份名文件 ${BACKUP_NAME_FILE}，先跑 02-backup.sh 或手动 export BACKUP_NAME" >&2
  exit 1
fi
BACKUP_NAME="$(cat "${BACKUP_NAME_FILE}")"

# 目标 Milvus 存储密钥块（IAM 时留空 -> useIAM:true）
USE_IAM="true"
S3_KEYS=""
if [[ -n "${S3_AK}" ]]; then
  USE_IAM="false"
  S3_KEYS=$(printf '  accessKeyID: "%s"\n  secretAccessKey: "%s"' "${S3_AK}" "${S3_SK}")
fi
BK_KEYS=""
if [[ -n "${S3_AK}" ]]; then
  BK_KEYS=$(printf '  backupAccessKeyID: "%s"\n  backupSecretAccessKey: "%s"' "${S3_AK}" "${S3_SK}")
fi

cat > configs/restore-s3.yaml <<YAML
log:
  level: info
  console: true
  file:
    filename: "logs/restore.log"

milvus:
  address: ${DST_MILVUS_HOST}
  port: ${DST_MILVUS_PORT}
  user: "${DST_MILVUS_USER}"
  password: "${DST_MILVUS_PASS}"
  tlsMode: 0

minio:
  # ---- 目标 Milvus 当前存储：S3 (数据路径) ----
  storageType: "aws"
  address: "${S3_ENDPOINT}"
  region: "${S3_REGION}"
  port: ${S3_PORT}
  useSSL: true
  useIAM: ${USE_IAM}
  bucketName: "${S3_BUCKET}"
  rootPath: "${S3_DATA_ROOTPATH}"
${S3_KEYS}

  # ---- 备份文件位置：同一 S3 (备份路径) ----
  backupStorageType: "aws"
  backupAddress: "${S3_ENDPOINT}"
  backupRegion: "${S3_REGION}"
  backupPort: ${S3_PORT}
  backupUseSSL: true
  backupBucketName: "${S3_BUCKET}"
  backupRootPath: "${S3_BACKUP_ROOTPATH}"
${BK_KEYS}

  # 目标与备份同一存储，无需跨存储
  crossStorage: false

backup:
  parallelism:
    copydata: 128
    restoreCollection: 2
    importJob: 256
YAML

CFG="configs/restore-s3.yaml"
echo ">> 连通性检查"
"${BACKUP_BIN}" check --config "${CFG}"

echo ">> 恢复备份: ${BACKUP_NAME} (含索引)"
"${BACKUP_BIN}" restore --config "${CFG}" -n "${BACKUP_NAME}" --restore_index

echo "恢复完成。目标数据应出现在: s3://${S3_BUCKET}/${S3_DATA_ROOTPATH}/"
