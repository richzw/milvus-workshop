#!/usr/bin/env python3
"""步骤 1：在源 Milvus 创建 collection、灌入测试数据、建索引、flush。

通过环境变量读取配置（见 env.sh）。运行：
    python3 01-seed-data.py
"""
import os
import random

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

HOST = os.environ.get("SRC_MILVUS_HOST", "localhost")
PORT = os.environ.get("SRC_MILVUS_PORT", "19530")
USER = os.environ.get("SRC_MILVUS_USER", "root")
PASS = os.environ.get("SRC_MILVUS_PASS", "Milvus")
NAME = os.environ.get("TEST_COLLECTION", "migration_demo")
DIM = int(os.environ.get("TEST_DIM", "768"))
ROWS = int(os.environ.get("TEST_ROWS", "10000"))

connections.connect(alias="default", host=HOST, port=PORT, user=USER, password=PASS)

if utility.has_collection(NAME):
    print(f"collection {NAME} 已存在，先删除重建")
    utility.drop_collection(NAME)

schema = CollectionSchema(
    fields=[
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("label", DataType.INT64),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
    ],
    description="MinIO->S3 migration demo",
)
coll = Collection(NAME, schema=schema)
print(f"创建 collection: {NAME} (dim={DIM})")

# 分批灌入，避免单次请求过大
BATCH = 2000
inserted = 0
while inserted < ROWS:
    n = min(BATCH, ROWS - inserted)
    ids = list(range(inserted, inserted + n))
    labels = [random.randint(0, 9) for _ in range(n)]
    vecs = [[random.random() for _ in range(DIM)] for _ in range(n)]
    coll.insert([ids, labels, vecs])
    inserted += n
    print(f"已灌入 {inserted}/{ROWS}")

print(">> flush")
coll.flush()

print(">> 建索引 (IVF_FLAT / L2)")
coll.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
)
utility.wait_for_index_building_complete(NAME)

coll.load()
print(f"完成。{NAME} num_entities={coll.num_entities} indexes={[i.index_name for i in coll.indexes]}")
