#!/usr/bin/env bash
# 步骤 2：从源 MinIO 备份到目标 S3（crossStorage）。
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "${WORKDIR}"

# 可选静态密钥块（IAM 角色时留空）
S3_KEYS=""
if [[ -n "${S3_AK}" ]]; then
  S3_KEYS=$(printf '  backupAccessKeyID: "%s"\n  backupSecretAccessKey: "%s"' "${S3_AK}" "${S3_SK}")
fi

cat > configs/backup-minio-to-s3.yaml <<YAML
log:
  level: info
  console: true
  file:
    filename: "logs/backup.log"

milvus:
  address: ${SRC_MILVUS_HOST}
  port: ${SRC_MILVUS_PORT}
  user: "${SRC_MILVUS_USER}"
  password: "${SRC_MILVUS_PASS}"
  tlsMode: 0

minio:
  # ---- 源 Milvus 当前存储：MinIO ----
  storageType: "minio"
  address: ${MINIO_HOST}
  port: ${MINIO_PORT}
  accessKeyID: ${MINIO_AK}
  secretAccessKey: ${MINIO_SK}
  useSSL: false
  useIAM: false
  bucketName: "${MINIO_BUCKET}"
  rootPath: "${MINIO_ROOTPATH}"

  # ---- 备份目标：AWS S3 ----
  backupStorageType: "aws"
  backupAddress: "${S3_ENDPOINT}"
  backupRegion: "${S3_REGION}"
  backupPort: ${S3_PORT}
  backupUseSSL: true
  backupBucketName: "${S3_BUCKET}"
  backupRootPath: "${S3_BACKUP_ROOTPATH}"
${S3_KEYS}

  # MinIO -> S3 跨对象存储拷贝，必须开启
  crossStorage: true

backup:
  parallelism:
    copydata: 128
    backupCollection: 4
    backupSegment: 1024
    restoreCollection: 2
    importJob: 256
YAML

CFG="configs/backup-minio-to-s3.yaml"
echo ">> 连通性检查"
"${BACKUP_BIN}" check --config "${CFG}"

BACKUP_NAME="minio_to_s3_$(date +%Y%m%d_%H%M%S)"
echo "${BACKUP_NAME}" > "${BACKUP_NAME_FILE}"
echo ">> 创建备份: ${BACKUP_NAME} (含索引/rbac)"
"${BACKUP_BIN}" create --config "${CFG}" -n "${BACKUP_NAME}" --rebuild_index --rbac

echo ">> 备份列表"
"${BACKUP_BIN}" list --config "${CFG}"
echo "备份已写入: s3://${S3_BUCKET}/${S3_BACKUP_ROOTPATH}/${BACKUP_NAME}/"
