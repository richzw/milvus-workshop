# 14 — Milvus 3.0 Native Capabilities

Status: implemented; StructArray activation remains eval-gated and default-disabled · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md)

## 1. Purpose

本文固定 Milvus 3.0 adapter、lifecycle 与 ingestion 的实现边界。Milvus 是 production adapter；`InMemoryHybridRetriever` 与内存 Memory/Cache store 保留相同的公开语义，测试和无服务 workshop 路径不得依赖网络。

## 2. Adapter contract

### 2.1 ORDER BY

`search(..., order_by, order_mode)` 只接受 allow-list 字段 `updated_at`、`priority` 和方向 `asc|desc`：

- `order_mode=relevance` 为默认值。ANN 先召回，hybrid score 是主排序，`order_by` 仅作稳定 tie-break；这与当前 local fallback 行为一致。
- `order_mode=scalar` 是显式业务排序。dense 与 BM25 search 都把解析后的 `order_by_fields=[{"field": ..., "order": ...}]` 交给 Milvus，融合后的候选再用相同 scalar-primary key 排序；`chunk_id` 是最终稳定 tie-break。
- 调用方未显式选择 `scalar` 时，不得因为升级 SDK 而改变相关性顺序。非法字段、方向或 mode 在任何 I/O 前失败。

### 2.2 Facets / aggregation

Facet 的统计域是一次检索保留的、去重后的 `chunk_id` 集合，不是整个 collection。Milvus Search Aggregation 与 Hybrid Search 不可组合，因此 adapter 先完成 hybrid recall，再对 Milvus 支持 group-by 的字段使用 bounded Query Aggregation：

```text
filter = chunk_id in <candidate ids>
group_by_fields = <requested non-Bool public fields>
output_fields = <group_by_fields> + [count(*)]
```

允许字段为 `source_type`、`doc_type`、`department`、`has_image_vector`、`doc_version`、`is_current`；候选最多 64 个。Milvus 3.0 Query Aggregation 不支持 Bool group-by，因此 `has_image_vector` 与 `is_current` 不得出现在 `group_by_fields`，而是直接从已经授权、去重的 retained candidates 计数；其余 VarChar 字段继续 server-side aggregation。组合 group 由 adapter marginalize 成现有 `{field: {value: count}}` 返回结构。空候选不访问 Milvus。缺字段、非法 count 或候选外数据 fail closed。Local fallback 对同一候选集用确定性 `Counter`，输出必须逐项相等。

### 2.3 BM25 Function and synonyms

`kb_chunks` 新增存储字段 `retrieval_text`（VarChar, analyzer enabled），由 title、heading path、section 与正文按固定 recipe 构成。`bm25_function` 使用 `FunctionType.BM25`，输入 `retrieval_text`，输出 `sparse_vector`。插入时客户端写 `retrieval_text`，不得写 function output；sparse search 直接提交规范化 query 文本，metric 为 `BM25`。

Analyzer 使用 standard tokenizer、lowercase 与 reviewable inline synonym dictionary。首版只包含稳定技术同义词（例如 `object storage/S3/MinIO`、`vector database/vector db`、`full text/BM25`）；entity catalog 仍独立拥有业务词语消歧和授权边界。Synonym 扩展不得替代 entity resolution，也不得把未授权 scope 引入 filter。

Inline dictionary 的多词 term 必须按 Milvus/Tantivy 语法转义空格（Python
字符串中例如 `r"object\ storage"`），synonym filter 固定 `expand=true` 以保留原
token；未转义规则不得进入 collection schema 或 additive evolution。实现和迁移
共用同一 analyzer 常量，并以 `run_analyzer` 加实际 collection-create smoke test
验证 server 接收的 JSON，而不只验证 PyMilvus 对象能序列化。

### 2.4 Sparse index

Milvus 3.0 默认路径为 `SPARSE_INVERTED_INDEX + BM25`，index params 不再固定 `DAAT_MAXSCORE`，由 server 选择默认 SINDI。只有显式 compatibility mode 才生成 `{"inverted_index_algo": "DAAT_MAXSCORE"}`；默认和 CI dry-run 必须验证该键不存在。切换算法不改变公开 `SearchResult`、filters、facets 或 local fallback contract。

### 2.5 StructArray search and filter contract

The authoritative source for server semantics is the Milvus 3.0 [StructArray overview](https://milvus.io/docs/array-of-structs.md), [limits](https://milvus.io/docs/structarray-limits.md), [EmbeddingList strategy guide](https://milvus.io/docs/choose-an-embeddinglist-search-strategy.md), and [hybrid-search contract](https://milvus.io/docs/hybrid-search-with-structarray.md). The Workshop applies those capabilities to `kb_documents.passages` without changing the stable evidence model in spec 10.

`STRUCT_ARRAY_RETRIEVAL` is explicit and accepts `disabled | struct_element | struct_two_stage | struct_fused`, default `disabled`:

| Mode | Retrieval behavior | Result allowed into evidence grading |
| --- | --- | --- |
| `disabled` | existing `kb_chunks` dense+BM25 path | normalized flat chunk |
| `struct_element` | one query vector searches `passages[element_vector]` | resolved passage with parent id, offset and stable `chunk_id` |
| `struct_two_stage` | 2–3 aspect vectors form an EmbeddingList parent shortlist, then element searches run inside that authorized/versioned shortlist | only second-stage resolved passages; parent hits are routing hints |
| `struct_fused` | `kb_chunks` BM25 candidates and StructArray element-dense candidates are fused application-side by stable passage identity before the existing reranker | resolved passages only |

Explicit non-disabled mode validates collection/schema/index/embedding fingerprints at workflow construction and fails closed on mismatch. It does not silently route part of a query through an unverified projection. The default/offline path remains `disabled`; local parity tests may emulate the same logical result shapes without claiming native execution.

The runtime contract is intentionally small and strict:

| Variable | Default | Validation |
| --- | --- | --- |
| `STRUCT_ARRAY_RETRIEVAL` | `disabled` | one of the four registered modes |
| `MILVUS_STRUCT_ARRAY_COLLECTION_NAME` | `kb_documents` | non-empty Milvus collection name |
| `STRUCT_ARRAY_PROJECTION_FINGERPRINT` | unset | required for a non-disabled mode; exactly 64 lowercase hex characters |
| `STRUCT_ARRAY_PARENT_TOP_K` | `8` | integer in `[1, 64]` |

Workflow construction for a non-disabled mode verifies the collection, both named vector indexes, the text-embedding fingerprint, projection fingerprint, repeated declared parent/passage counts and physical parent/passage counts before serving a query. A failed check raises a sanitized configuration error. There is no per-request downgrade to `flat_hybrid`; operators switch back by explicitly setting `STRUCT_ARRAY_RETRIEVAL=disabled`.

#### 2.5.1 Filtering at the correct scope

- Permission, `department`, source, document type, `doc_version` and `is_current` are parent predicates and must be applied before any result is released.
- `element_filter(passages, ...)` limits the same elements that participate in element-level ANN. Multiple `$[subfield]` conditions must bind to one offset; values are constructed only from registered tool policy and validated query constraints.
- `MATCH_ANY/ALL/LEAST/MOST/EXACT` evaluates a same-element predicate and then admits or rejects the parent. The primary RAG path uses `MATCH_ANY` only when it needs a parent shortlist satisfying a passage-local requirement; the other quantifiers remain an explicit teaching lab until a query intent owns their semantics.
- Parallel-array approximations are forbidden because they cannot prove that, for example, `record_type` and `section` matched on the same passage.
- `$[...]` appears only inside StructArray operators. JSON, text-match, array-container, GIS and TIMESTAMPTZ expressions are not passed into element predicates.

#### 2.5.2 Element-level evidence

Element search submits one validated 1024-d query vector to `passages[element_vector]` with `COSINE`. Each raw hit must provide parent primary key and element offset. The adapter reads only bounded identity/provenance subfields, resolves every retained `passages[offset].chunk_id`, then performs one bounded batch exact `kb_chunks` lookup under the same permission/version predicates to rehydrate authoritative text and live checksum before normalizing hits to the existing evidence candidate shape:

```python
{
    "doc_id": str,
    "doc_version": str,
    "chunk_id": str,
    "page_no": int | None,
    "section": str | None,
    "text": str,
    "checksum": str,
    "retrieval_granularity": "element",
    "struct_field": "passages",
    "element_offset": int,
}
```

The adapter rejects an out-of-range offset, empty/duplicate `chunk_id`, mismatched parent/version, checksum drift, unexpected embedding fingerprint or permission/version predicate violation. Downstream merge, rerank, grade, generation and cache use `chunk_id`, never offset. One parent may appear multiple times because distinct passages are distinct evidence candidates.

The native request asks for `document_key`, `doc_id`, `doc_version`, `department`, `is_current`, `text_embedding_fingerprint`, `projection_fingerprint` and `passages`. PyMilvus 3.0.1 returns the selected element offset on the top-level hit as `offset`; the adapter accepts only a non-negative integer there, then indexes the reconstructed `passages` list. A missing offset is a document-level/collapsed result and is rejected from evidence normalization. Rehydration uses batches of at most 16 `chunk_id` values because that is the existing authoritative lookup bound.

#### 2.5.3 EmbeddingList as a routing stage

`struct_two_stage` is legal only for a retrieval plan with two or three independent `aspect`/`primary` queries under the same tool, permission and version scope. Their vectors form one `EmbeddingList` query against `passages[embedding_list_vector]` with `MAX_SIM_COSINE`. A one-vector question uses element search; step-back background and primary items are not bundled unless both are explicitly required document-coverage aspects.

EmbeddingList returns parent entities and no citeable offset. The result may only bound a subsequent element-level search. Every required aspect runs element search within the shortlisted `document_key` set, and each selected citation must come from those resolved element hits or the separately fused flat BM25 lane. If the second stage finds no citeable passages, the parent match cannot satisfy evidence grading.

The initial index strategy is quality-first TokenANN, with `emb_list_rerank=true` and provisional `retrieval_ann_ratio=3.0`. Phase 0 records exact-MAX_SIM reference recall, latency and index size before these values become a baseline. MUVERA is considered only after TokenANN violates an accepted resource budget. LEMUR is out of the initial Workshop scope because it adds a corpus-specific training lifecycle.

#### 2.5.4 Fusion, grouping and collapse

Milvus hybrid result identity depends on the mixed search scopes. All element-level requests under the same `passages` field may retain `(document_key, offset)`. Mixing an element request with a top-level vector, an EmbeddingList request or another StructArray field collapses to parent identity; collapse operates only over the element hits already returned by that ANN request, so its `limit` changes the score.

The citation-producing path therefore does not mix an entity-level request and an element-level request in one server hybrid call. `struct_fused` executes flat BM25 and StructArray element ANN separately, maps both to stable `chunk_id`, deduplicates exact passage identities and applies a deterministic, fingerprinted application-side fusion before the existing complete-pool reranker. If a future document-ranking panel uses native collapse, it must declare `max | sum | avg | topk_sum | topk_avg`, compatible metric family, positive `topk` where required and sub-search limit; the collapsed result remains a document candidate and never a citation.

The first registered fusion recipe is `struct-rrf-v1`:

```text
score(chunk) = 0.65 / (60 + element_rank)
             + 0.35 / (60 + bm25_rank)
```

A missing lane contributes zero. Exact `chunk_id` is the deduplication key and final deterministic tie-break. The normalized result retains both present retrieval paths and the element offset when the StructArray lane contributed it; BM25-only results have no offset. The recipe id is part of evaluation and cache provenance.

### 2.6 StructArray index and schema contract

`kb_documents` owns two physically separate vector subfields because each vector subfield accepts one index:

| Field | Index / metric | Purpose |
| --- | --- | --- |
| `passages[embedding_list_vector]` | HNSW + `MAX_SIM_COSINE` | EmbeddingList entity shortlist |
| `passages[element_vector]` | HNSW + `COSINE` | element-level passage evidence |

The exact HNSW build/search parameters are Phase 0 outputs, not hard-coded product truth. Frequently used scalar subfields may receive supported scalar indexes only after a filter benchmark. Every index path uses bracket syntax such as `passages[element_vector]`, never dotted syntax.

The adapter enforces current Milvus boundaries: Struct exists only inside Array; `max_capacity` is required; all elements share one fixed schema; nested Struct/Array/ArrayOfStruct/JSON, sparse vector, Text, TIMESTAMPTZ and field functions are excluded; vector subfields must be indexed before search. The Workshop does not depend on partially nullable element records: projection assembly emits the non-null sentinel values fixed in spec 10 and rejects a shape the target server/SDK cannot round-trip. A subfield-schema change recreates `kb_documents` and revalidates the complete projection.

### 2.7 StructArray capability probe

Before implementation enables a native mode against the pinned Milvus/PyMilvus version, a disposable Phase 0 exercise must prove:

1. schema create/describe round trip, required limits and both index descriptions;
2. natural nested insert plus exact offset/subfield reconstruction;
3. `MATCH_ANY` same-offset correctness and `element_filter` participation correctness, including a parallel-array false-positive counterexample;
4. element search offset output, repeated-parent hits and grouping behavior;
5. EmbeddingList `MAX_SIM_COSINE`, strategy params, rerank toggle and candidate-ratio behavior against exact reference ranking;
6. same-StructArray element hybrid identity and mixed-granularity collapse behavior, including the fact that collapse sees only returned sub-search hits;
7. all unsupported schema/filter/search combinations fail with registered, sanitized reason codes.

The report records server/SDK version, mode, schema/index fingerprints, corpus/list-length distribution, query-list length, recall/nDCG, P50/P95 latency, index bytes and peak build resource observation. A failed probe keeps `STRUCT_ARRAY_RETRIEVAL=disabled`; it never weakens citation, permission or version isolation.

## 3. Lifecycle contract

`conversation_memory`、`memory_events`、`memory_facts` 与 `grounded_response_cache` 的 `expires_at` 改为 nullable/required `TIMESTAMPTZ`，collection property 固定 `ttl_field=expires_at`。Python domain model 继续使用 UTC epoch milliseconds；唯一的 storage codec 负责 epoch-ms 与 canonical ISO-8601 UTC 字符串互转。所有 Milvus write、read 和显式 expiry predicate 必须经过该 codec。

Server TTL 是主过滤边界；现有显式 `expires_at is null or expires_at > <now>` 继续存在，用于确定性测试、cleanup eligibility 与旧/新 adapter parity。写入值使用 canonical ISO-8601 UTC 字符串，但 expression literal 必须使用 Milvus 的 `ISO '2026-08-05T09:55:31.426Z'` 语法；普通 JSON quoted string 会被解析为 `VarChar`，不得用于和 `TIMESTAMPTZ` 比较。naive datetime、bool、负值和无法解析的 server value fail closed。

`TIMESTAMPTZ` 不得复用通用 scalar 的 `INVERTED` index；所有 `expires_at`
固定使用 Milvus 支持的 `STL_SORT`。Index builder 的 scalar entry 因此允许显式
声明 `field_name/index_type/params`，未声明类型的普通 scalar 仍默认
`INVERTED`，非法 entry fail closed。

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
- 从已验证 `kb_chunks` manifest 创建或整体重建 `kb_documents`；不允许向已有 StructArray 追加或原地变更 subfield；
- 原子增加一个全新 sparse output field、BM25 Function 与默认 SINDI index；请求必须把 `AlterCollectionSchemaRequest.AddRequest.do_physical_backfill` 设为 `true`。PyMilvus 3.0.1 的 `add_function_field` wrapper 未传递该 protobuf flag，因此 adapter 使用版本保护的兼容路径并在 SDK shape 不匹配时 fail closed；既有同名 output 不允许拆分 alter；
- 对外部 embedding 字段以 primary-key bounded batches 执行 `partial_update` backfill，先验证字段、维度、finite values 与批内 ID 唯一性；调用方的版本化 manifest 拥有全 collection coverage；
- add-field 后 describe 再次验证完整字段 shape；revalidation 必须兼容 PyMilvus 3.0.1 wire round-trip（例如 analyzer boolean 的 canonical string），但只接受严格的 `true/false` 表示。BM25 apply 后验证 output type、Function input/output/type 和 index field/type/metric/SINDI default。已有对象只有完全兼容才 no-op，类型、维度、analyzer 或 Function identity 不同都 fail closed。

Embedding provider/model 变更不属于 schema evolution，而是全量 re-ingest。启动时 flat retriever 抽样读取已存 chunk 的 `text_embedding_fingerprint`，与当前配置不一致即 fail closed，不允许两个向量空间同时服务；成本与迁移计划见 [`15-retrieval-tier-selection.md § 7`](./15-retrieval-tier-selection.md#7-embedding-model-lifecycle)。

BM25 migration 不隐式切换线上读流量。成功报告返回 `MILVUS_SPARSE_FIELD=<new field>` activation contract；部署方完成 backfill validation 后显式设置该变量，retrieval adapter 才将 sparse search 的 `anns_field` 切到新字段。默认值仍是 `sparse_vector`，便于回滚及旧 collection 兼容。

字段名只允许 `[a-z][a-z0-9_]{0,63}`；禁止 drop、rename、类型/维度变更和 dynamic field。Apply 前输出 plan；需要写操作时必须显式 `--apply`。批次失败不得把 migration 标记 complete，report 仅记录计数、schema identity 和 registered error code，不记录向量或敏感 payload。

## 7. Acceptance criteria

- Fake-client contract tests 证明 ORDER BY、Query Aggregation、BM25 text query、SINDI default、TTL codec/property、MINHASH function/index、snapshot state machine 与 schema evolution 参数确实送到 server。
- StructArray fake-client tests pin bracket field paths, two index families, nested insert shape, element filters, EmbeddingList query shape, offset normalization and collapse params; disposable real-server tests satisfy § 2.7 before activation.
- Every StructArray evidence candidate round-trips to one live `kb_chunks` citation identity; parent-only and collapsed hits are rejected by evidence grading.
- Local/Milvus adapter parity fixtures 覆盖 relevance 与 scalar order、facets、expiry 和失败策略。
- Collection/index dry-run 可复制；所有新增命令默认 non-mutating。
- deterministic unit suite、golden eval、Ruff 和 strict mypy 通过；真实 Milvus integration 缺少服务时明确 skip，不得伪报通过。

## 8. Cross-references

- Data fields and migration: [`10-data-model.md`](./10-data-model.md)
- Offline write path: [`11-ingestion.md`](./11-ingestion.md)
- Evaluation gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- Retrieval tier and embedding-migration cost: [`15-retrieval-tier-selection.md`](./15-retrieval-tier-selection.md)
- Engineering phase: [`91-impl-plan.md`](./91-impl-plan.md)
- Decisions: [`99-key-decisions.md § D48`](./99-key-decisions.md#d48--structarray-is-a-derived-document-projection-chunk-identity-remains-authoritative), [`99-key-decisions.md § D49`](./99-key-decisions.md#d49--search-granularity-is-explicit-and-entity-hits-are-not-citation-evidence)
