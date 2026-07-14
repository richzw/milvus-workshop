# Milvus MinIO → S3 迁移实操脚本

整合两篇指南脚本，端到端跑通：**建集合 → 灌测试数据+索引 → MinIO 备份到 S3 → 新 Milvus 从 S3 恢复 → 验证完整性**。

核心原理：`milvus-backup` 是 collection 级**逻辑备份/恢复**，不是桶到桶拷文件。"迁移存储" = "在新存储上把集合恢复出来"。**必须恢复到 etcd 干净的全新 Milvus**，否则残留元数据指向不存在的文件会损坏集合。

## 文件

| 文件 | 作用 |
|------|------|
| `env.sh` | 所有变量（地址/桶/密钥/测试参数）。先改这里。 |
| `00-install-backup.sh` | 下载 milvus-backup 工具 |
| `01-seed-data.py` | 源 Milvus 建集合、灌 10000 行、建 IVF_FLAT 索引、flush |
| `02-backup.sh` | 渲染 `backup-minio-to-s3.yaml`（crossStorage:true），check + create（`--rebuild_index --rbac`） |
| `03-restore.sh` | 渲染 `restore-s3.yaml`，目标 Milvus restore（`--restore_index`） |
| `04-verify.py` | 对比源/目标集合数、行数、分区、索引、字段，做检索冒烟，PASS/FAIL |
| `run-all.sh` | 串起全部步骤 |
| `deploy/source/` | docker-compose 拉起源 Milvus + MinIO |
| `deploy/target/` | docker-compose 拉起目标 Milvus + S3 |

部署细节见 [`deploy/README.md`](deploy/README.md)。

## 前置

- Python: `pip install pymilvus`
- 源 Milvus + MinIO 在跑
- **目标 Milvus 已部署、连 S3、etcd 干净**（脚本不替你部署，见下）
- AWS 鉴权：IAM 角色（推荐，密钥留空）或 `env.sh` 填 `S3_AK/S3_SK`

## 跑

```bash
vim env.sh          # 改地址/桶/密钥
./run-all.sh        # 跑到备份后会暂停，等你确认目标 Milvus 就绪
```

或分步：

```bash
source env.sh
./00-install-backup.sh
python3 01-seed-data.py
./02-backup.sh
# --- 此处部署目标 Milvus（见下）---
./03-restore.sh
python3 04-verify.py
```

## 步骤 4：部署连 S3 的目标 Milvus（手动）

脚本无法替你拉起集群。二选一：

**Helm:**
```yaml
# values-s3.yaml
cluster: { enabled: true }
minio: { enabled: false }
externalS3:
  enabled: true
  host: "s3.us-east-1.amazonaws.com"
  port: "443"
  useSSL: true
  bucketName: "my-milvus-prod-bucket"
  rootPath: "milvus-data"      # 必须 = env.sh 的 S3_DATA_ROOTPATH
```
```bash
helm upgrade --install milvus-s3 milvus/milvus -n milvus --create-namespace -f values-s3.yaml
```

**Docker Compose:** 在 `milvus.yaml` 的 `minio:` 段填 S3，并确认 `docker-compose.yaml` 没有 `MINIO_ADDRESS` 等环境变量覆盖。

> `rootPath` 必须等于 `env.sh` 的 `S3_DATA_ROOTPATH`（默认 `milvus-data`），与备份路径 `S3_BACKUP_ROOTPATH`（默认 `milvus-backup`）分开。

## 常见坑

- 不要把现有 Milvus 直接重启指向空 S3 桶 —— 元数据/文件不一致会损坏集合。永远恢复到干净实例。
- `bucketName`/`rootPath` 必须与对应 Milvus 实际配置一致。
- 默认不重建索引，靠 `--restore_index`（03 已带）。
- 跨不同 S3 服务时手动开 `crossStorage`；类型相同 ≠ 服务相同。
- 正式迁移先冻结写入再备份，否则备份后新写入不会同步到目标。
