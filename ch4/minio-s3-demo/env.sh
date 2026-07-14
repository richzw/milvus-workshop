#!/usr/bin/env bash
# 所有脚本共用的变量。按实际环境修改后 source 即可。
set -euo pipefail

# ---- milvus-backup 工具 ----
export BACKUP_VERSION="${BACKUP_VERSION:-v0.5.16}"   # Milvus>=2.6.9 需 >=0.5.11
export BACKUP_OS="${BACKUP_OS:-Linux}"               # Linux / Darwin
export BACKUP_ARCH="${BACKUP_ARCH:-x86_64}"          # x86_64 / arm64
export WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export BACKUP_BIN="${WORKDIR}/milvus-backup"

# ---- 源 Milvus + MinIO ----
export SRC_MILVUS_HOST="${SRC_MILVUS_HOST:-milvus-source.example.com}"
export SRC_MILVUS_PORT="${SRC_MILVUS_PORT:-19530}"
export SRC_MILVUS_USER="${SRC_MILVUS_USER:-root}"
export SRC_MILVUS_PASS="${SRC_MILVUS_PASS:-Milvus}"

export MINIO_HOST="${MINIO_HOST:-minio-source.example.com}"
export MINIO_PORT="${MINIO_PORT:-9000}"
export MINIO_AK="${MINIO_AK:-minioadmin}"
export MINIO_SK="${MINIO_SK:-minioadmin}"
export MINIO_BUCKET="${MINIO_BUCKET:-milvus-bucket}"
export MINIO_ROOTPATH="${MINIO_ROOTPATH:-file}"        # 必须与源 Milvus 实际配置一致

# ---- 目标 Milvus + AWS S3 ----
export DST_MILVUS_HOST="${DST_MILVUS_HOST:-milvus-target.example.com}"
export DST_MILVUS_PORT="${DST_MILVUS_PORT:-19530}"
export DST_MILVUS_USER="${DST_MILVUS_USER:-root}"
export DST_MILVUS_PASS="${DST_MILVUS_PASS:-Milvus}"

export S3_ENDPOINT="${S3_ENDPOINT:-s3.us-east-1.amazonaws.com}"
export S3_REGION="${S3_REGION:-us-east-1}"
export S3_PORT="${S3_PORT:-443}"
export S3_BUCKET="${S3_BUCKET:-my-milvus-prod-bucket}"
export S3_BACKUP_ROOTPATH="${S3_BACKUP_ROOTPATH:-milvus-backup}"   # 备份文件路径
export S3_DATA_ROOTPATH="${S3_DATA_ROOTPATH:-milvus-data}"         # 目标 Milvus 数据路径
# 静态密钥（留空则走 IAM 角色 / 实例 profile）
export S3_AK="${S3_AK:-}"
export S3_SK="${S3_SK:-}"

# ---- 测试数据 ----
export TEST_COLLECTION="${TEST_COLLECTION:-migration_demo}"
export TEST_DIM="${TEST_DIM:-768}"
export TEST_ROWS="${TEST_ROWS:-10000}"

# ---- 备份名（生成一次，后续步骤复用）----
export BACKUP_NAME_FILE="${WORKDIR}/.backup_name"
