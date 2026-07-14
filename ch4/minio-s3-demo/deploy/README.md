# 部署脚本：源 Milvus+MinIO / 目标 Milvus+S3

docker-compose 拉起两套环境，可并存（不同端口、独立 etcd）。

| 环境 | proxy | 对象存储 | etcd |
|------|-------|----------|------|
| `source/` | localhost:**19530** | 容器内 MinIO (9000/9001) | 独立卷 |
| `target/` | localhost:**19531** | 外部 AWS S3 | 独立卷（干净） |

两者都用挂载的 `user.yaml` 声明对象存储，bucket/rootPath 与上层 `env.sh` 对齐。

## 源 Milvus + MinIO

```bash
cd deploy/source
./up.sh        # 拉起，等健康
# MinIO console: http://localhost:9001  (minioadmin/minioadmin)
./down.sh             # 停（留数据）
./down.sh --purge     # 停并删数据卷
```

bucket/rootPath 改 `source/user.yaml`，须与 `env.sh` 的 `MINIO_BUCKET`/`MINIO_ROOTPATH` 一致。Milvus 启动会自动建 MinIO bucket。

## 目标 Milvus + S3

前置：
- **S3 桶必须先建好**（Milvus 不会替你建 S3 桶）。
- 在 `env.sh` 填 `S3_ENDPOINT/S3_REGION/S3_BUCKET/S3_DATA_ROOTPATH`，以及 `S3_AK/S3_SK`（留空则容器走 IAM 角色，仅在 EC2/EKS 等带角色的环境可行）。

```bash
cd deploy/target
./up.sh        # 先从 env.sh 渲染 user.yaml，再拉起，等健康
./down.sh --purge   # 删本地 etcd/元数据卷；S3 数据不动
```

`up.sh` 自动跑 `render-user-yaml.sh` 生成 `target/user.yaml`。

## 接上迁移流程

docker 本地跑时，改 `env.sh` 让上层脚本指到这两套环境：

```bash
# 源
export SRC_MILVUS_HOST=localhost
export SRC_MILVUS_PORT=19530
export MINIO_HOST=localhost MINIO_PORT=9000
export MINIO_BUCKET=milvus-bucket MINIO_ROOTPATH=file
# 目标
export DST_MILVUS_HOST=localhost
export DST_MILVUS_PORT=19531
```

> 注意：`02-backup.sh` 在宿主机跑，连 `MINIO_HOST=localhost:9000` 没问题；备份目标是真实 AWS S3。若想纯本地零成本验证，可把 S3 也换成第二个 MinIO（S3 兼容），此时 `backupStorageType` 用 `minio`、`crossStorage` 仍为 true。

完整顺序：

```bash
cd deploy/source && ./up.sh && cd -
cd deploy/target && ./up.sh && cd -   # 需先建好 S3 桶 + env.sh 填好
# 改好 env.sh 的 localhost/端口后：
./00-install-backup.sh
python3 01-seed-data.py
./02-backup.sh
./03-restore.sh
python3 04-verify.py
```
