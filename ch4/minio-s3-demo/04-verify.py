#!/usr/bin/env python3
"""步骤 4：验证数据完整性 —— 对比源/目标 Milvus 的 collection、行数、索引，并做检索冒烟。

运行：
    python3 04-verify.py
"""
import os
import sys

from pymilvus import connections, utility, Collection

DIM = int(os.environ.get("TEST_DIM", "768"))
TARGET = os.environ.get("TEST_COLLECTION", "migration_demo")

SRC = dict(
    host=os.environ.get("SRC_MILVUS_HOST", "localhost"),
    port=os.environ.get("SRC_MILVUS_PORT", "19530"),
    user=os.environ.get("SRC_MILVUS_USER", "root"),
    password=os.environ.get("SRC_MILVUS_PASS", "Milvus"),
)
DST = dict(
    host=os.environ.get("DST_MILVUS_HOST", "localhost"),
    port=os.environ.get("DST_MILVUS_PORT", "19530"),
    user=os.environ.get("DST_MILVUS_USER", "root"),
    password=os.environ.get("DST_MILVUS_PASS", "Milvus"),
)


def snapshot(alias, conn):
    connections.connect(alias=alias, **conn)
    out = {}
    for name in utility.list_collections(using=alias):
        c = Collection(name, using=alias)
        c.load()
        out[name] = {
            "entities": c.num_entities,
            "partitions": len(c.partitions),
            "indexes": sorted(i.index_name for i in c.indexes),
            "fields": sorted(f.name for f in c.schema.fields),
        }
    return out


def main():
    src = snapshot("src", SRC)
    dst = snapshot("dst", DST)

    print("=== 源 Milvus ===")
    for k, v in src.items():
        print(f"  {k}: {v}")
    print("=== 目标 Milvus ===")
    for k, v in dst.items():
        print(f"  {k}: {v}")

    ok = True
    # 1) collection 集合一致
    if set(src) != set(dst):
        print(f"[FAIL] collection 列表不一致: 仅源 {set(src)-set(dst)} / 仅目标 {set(dst)-set(src)}")
        ok = False

    # 2) 每个 collection 行数/分区/索引/字段一致
    for name in set(src) & set(dst):
        s, d = src[name], dst[name]
        for key in ("entities", "partitions", "indexes", "fields"):
            if s[key] != d[key]:
                print(f"[FAIL] {name}.{key}: 源={s[key]} 目标={d[key]}")
                ok = False

    # 3) 目标检索冒烟
    if TARGET in dst:
        c = Collection(TARGET, using="dst")
        c.load()
        res = c.search(
            data=[[0.0] * DIM],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=5,
        )
        hits = len(res[0])
        print(f"[search] {TARGET} 返回 {hits} 条")
        if hits == 0:
            print(f"[FAIL] {TARGET} 检索返回 0 条")
            ok = False

    print("\n结果:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
