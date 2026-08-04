# 14 — Milvus 3.0 Native Capabilities

Status: implementation-ready · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md)

## 1. Purpose

本文固定 Milvus 3.0 adapter、lifecycle 与 ingestion 的实现边界。Milvus 是 production adapter；`InMemoryHybridRetriever` 与内存 Memory/Cache store 保留相同的公开语义，测试和无服务 workshop 路径不得依赖网络。

## 2. Adapter contract

### 2.1 ORDER BY

`search(..., order_by, order_mode)` 只接受 allow-list 字段 `updated_at`、`priority` 和方向 `asc|desc`：

- `order_mode=relevance` 为默认值。ANN 先召回，hybrid score 是主排序，`order_by` 仅作稳定 tie-break；这与当前 local fallback 行为一致。
- `order_mode=scalar` 是显式业务排序。dense 与 BM25 search 都把解析后的 `order_by_fields=[{"field": ..., "order": ...}]` 交给 Milvus，融合后的候选再用相同 scalar-primary key 排序；`chunk_id` 是最终稳定 tie-break。
- 调用方未显式选择 `scalar` 时，不得因为升级 SDK 而改变相关性顺序。非法字段、方向或 mode 在任何 I/O 前失败。

### 2.2 Facets / aggregation

Facet 的统计域是一次检索保留的、去重后的 `chunk_id` 集合，不是整个 collection。Milvus Search Aggregation 与 Hybrid Search 不可组合，因此 adapter 先完成 hybrid recall，再用一个 bounded Query Aggregation 请求：

```text
filter = chunk_id in <candidate ids>
group_by_fields = <requested public fields>
output_fields = <fields> + [count(*)]
```

允许字段为 `source_type`、`doc_type`、`department`、`has_image_vector`、`doc_version`、`is_current`；候选最多 64 个。组合 group 由 adapter marginalize 成现有 `{field: {value: count}}` 返回结构。空候选不访问 Milvus。缺字段、非法 count 或候选外数据 fail closed。Local fallback 对同一候选集用确定性 `Counter`，输出必须逐项相等。

### 2.3 BM25 Function and synonyms

`kb_chunks` 新增存储字段 `retrieval_text`（VarChar, analyzer enabled），由 title、heading path、section 与正文按固定 recipe 构成。`bm25_function` 使用 `FunctionType.BM25`，输入 `retrieval_text`，输出 `sparse_vector`。插入时客户端写 `retrieval_text`，不得写 function output；sparse search 直接提交规范化 query 文本，metric 为 `BM25`。

Analyzer 使用 standard tokenizer、lowercase 与 reviewable inline synonym dictionary。首版只包含稳定技术同义词（例如 `object storage/S3/MinIO`、`vector database/vector db`、`full text/BM25`）；entity catalog 仍独立拥有业务词语消歧和授权边界。Synonym 扩展不得替代 entity resolution，也不得把未授权 scope 引入 filter。

### 2.4 Sparse index

Milvus 3.0 默认路径为 `SPARSE_INVERTED_INDEX + BM25`，index params 不再固定 `DAAT_MAXSCORE`，由 server 选择默认 SINDI。只有显式 compatibility mode 才生成 `{"inverted_index_algo": "DAAT_MAXSCORE"}`；默认和 CI dry-run 必须验证该键不存在。切换算法不改变公开 `SearchResult`、filters、facets 或 local fallback contract。

## 3. Lifecycle contract

`conversation_memory`、`memory_events`、`memory_facts` 与 `grounded_response_cache` 的 `expires_at` 改为 nullable/required `TIMESTAMPTZ`，collection property 固定 `ttl_field=expires_at`。Python domain model 继续使用 UTC epoch milliseconds；唯一的 storage codec 负责 epoch-ms 与 canonical ISO-8601 UTC 字符串互转。所有 Milvus write、read 和显式 expiry predicate 必须经过该 codec。

Server TTL 是主过滤边界；现有显式 `expires_at is null or expires_at > <now>` 继续存在，用于确定性测试、cleanup eligibility 与旧/新 adapter parity。TIMESTAMPTZ literal 必须 JSON quote，naive datetime、bool、负值和无法解析的 server value fail closed。

同名字段不能原地改变类型。本 demo 对已有 Int64 collection 的迁移是受控 recreation：dry-run 报告受影响 collection，只有已有的 destructive confirmation flow 才能 drop/recreate/re-ingest；schema evolution 命令不得伪装成无损 alter。

## 4. Server-side MinHash (DIDO)

`doc_dedup_signatures.normalized_text` 是 raw VarChar input，启用 analyzer；`minhash_function` 使用 `FunctionType.MINHASH`，输出 8192-bit `BINARY_VECTOR`（256 hashes × 32 bits）。参数固定：`num_hashes=256`、`shingle_size=3`、`seed=1234`、word-level tokenization。

Index 使用 `MINHASH_LSH`、metric `MHJACCARD`、`mh_element_bit_width=32`、`mh_lsh_band=128`、`with_raw_data=true`。Ingestion 只提交 raw text 和 exact SHA-256，不在客户端生成 `minhash_signature`。Local fallback 保留 exact checksum 与纯函数近重复估计用于测试，但该估计不是持久化 schema，也不声称与 server bit pattern 相同。

## 5. Evaluation snapshots

联网 eval 必须绑定一个 named snapshot 和一个独立 target collection：

1. flush source；若 snapshot 不存在则创建，若存在则验证 source collection；
2. 若 target 不存在则 restore，并用有界 timeout 轮询 job；若 target 已存在则通过 completed restore job 验证 snapshot/target lineage；
3. load target，后续 retriever 只读取 target；eval 的 `--sparse-field` 默认继承 `MILVUS_SPARSE_FIELD`，保证 snapshot eval 与线上 reader cutover 使用同一 BM25 output；报告记录 snapshot name、source、target 和 restore job；
4. eval 不自动 drop snapshot 或 target，避免下一次运行观察到不同 corpus。

CLI 参数必须 all-or-none；默认离线 eval 不连接 Milvus。原始 server exception、token 和 URI credential 不写入报告。

## 6. Schema evolution and backfill

Schema evolution 是 allow-list、dry-run-first、可重入操作，仅支持：

- 为 `kb_chunks` 增加 nullable `retrieval_text`、`SparseFloatVector` 或指定维度的 `FloatVector` 字段；
- 原子增加一个全新 sparse output field、BM25 Function 与默认 SINDI index；请求必须把 `AlterCollectionSchemaRequest.AddRequest.do_physical_backfill` 设为 `true`。PyMilvus 3.0.1 的 `add_function_field` wrapper 未传递该 protobuf flag，因此 adapter 使用版本保护的兼容路径并在 SDK shape 不匹配时 fail closed；既有同名 output 不允许拆分 alter；
- 对外部 embedding 字段以 primary-key bounded batches 执行 `partial_update` backfill，先验证字段、维度、finite values 与批内 ID 唯一性；调用方的版本化 manifest 拥有全 collection coverage；
- add-field 后 describe 再次验证完整字段 shape；revalidation 必须兼容 PyMilvus 3.0.1 wire round-trip（例如 analyzer boolean 的 canonical string），但只接受严格的 `true/false` 表示。BM25 apply 后验证 output type、Function input/output/type 和 index field/type/metric/SINDI default。已有对象只有完全兼容才 no-op，类型、维度、analyzer 或 Function identity 不同都 fail closed。

BM25 migration 不隐式切换线上读流量。成功报告返回 `MILVUS_SPARSE_FIELD=<new field>` activation contract；部署方完成 backfill validation 后显式设置该变量，retrieval adapter 才将 sparse search 的 `anns_field` 切到新字段。默认值仍是 `sparse_vector`，便于回滚及旧 collection 兼容。

字段名只允许 `[a-z][a-z0-9_]{0,63}`；禁止 drop、rename、类型/维度变更和 dynamic field。Apply 前输出 plan；需要写操作时必须显式 `--apply`。批次失败不得把 migration 标记 complete，report 仅记录计数、schema identity 和 registered error code，不记录向量或敏感 payload。

## 7. Acceptance criteria

- Fake-client contract tests 证明 ORDER BY、Query Aggregation、BM25 text query、SINDI default、TTL codec/property、MINHASH function/index、snapshot state machine 与 schema evolution 参数确实送到 server。
- Local/Milvus adapter parity fixtures 覆盖 relevance 与 scalar order、facets、expiry 和失败策略。
- Collection/index dry-run 可复制；所有新增命令默认 non-mutating。
- deterministic unit suite、golden eval、Ruff 和 strict mypy 通过；真实 Milvus integration 缺少服务时明确 skip，不得伪报通过。

## 8. Cross-references

- Data fields and migration: [`10-data-model.md`](./10-data-model.md)
- Offline write path: [`11-ingestion.md`](./11-ingestion.md)
- Evaluation gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- Engineering phase: [`91-impl-plan.md`](./91-impl-plan.md)
- Decisions: [`99-key-decisions.md`](./99-key-decisions.md)
