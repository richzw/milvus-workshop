#!/usr/bin/env bash
# 从 env.sh 生成目标 Milvus 的 S3 user.yaml。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../../env.sh"

# 静态密钥 vs IAM 角色
if [[ -n "${S3_AK}" ]]; then
  USE_IAM="false"
  KEYS=$(printf '  accessKeyID: "%s"\n  secretAccessKey: "%s"' "${S3_AK}" "${S3_SK}")
else
  USE_IAM="true"
  KEYS="  # useIAM=true：走实例 IAM 角色，无需明文密钥"
fi

cat > "${HERE}/user.yaml" <<YAML
# 目标 Milvus 对象存储 = AWS S3。bucketName/rootPath 必须与 env.sh 一致。
minio:
  address: ${S3_ENDPOINT}
  port: ${S3_PORT}
  useSSL: true
  useIAM: ${USE_IAM}
  cloudProvider: aws
  region: ${S3_REGION}
  bucketName: ${S3_BUCKET}        # = env.sh S3_BUCKET
  rootPath: ${S3_DATA_ROOTPATH}   # = env.sh S3_DATA_ROOTPATH
${KEYS}
YAML

echo "已生成 ${HERE}/user.yaml (bucket=${S3_BUCKET} rootPath=${S3_DATA_ROOTPATH} useIAM=${USE_IAM})"
