UI Demo 的最终 MVP 方案

# UI Demo 关键决策更新

| 项目             | 决策                                                                       |
| -------------- | ------------------------------------------------------------------------ |
| UI 查询          | 只做查询，不做在线 ingestion                                                      |
| Ingestion      | 离线 demo / notebook / CLI                                                 |
| S3             | MinIO / mock S3                                                          |
| MFS            | 作为架构扩展，不进入 MVP 主链路                                                       |
| 图片检索           | 进阶功能                                                                     |
| Agent Workflow | LangGraph                                                                |
| classify_query | **保留**，用于展示 Agentic RAG 流程                                               |
| max_retry      | **3 次**                                                                  |
| reranker       | **需要引入**                                                                 |
| 权限控制           | 不需要                                                                      |
| Citation       | chunk / page 级                                                           |
| Streaming      | 答案 streaming，trace 完成后展示                                                 |
| Milvus UI 展示   | hybrid search、metadata filter、order by、aggregation、nullable image_vector |
| image_vector   | schema 中包含，MVP 大多数为 null，少量图片样例有值                                        |

---

# 1. 更新后的 LangGraph Workflow

引入 reranker 和 max_retry=3 后，推荐 workflow 变成：

```text id="ui_flow_001"
START
  ↓
classify_query
  ↓
rewrite_query
  ↓
milvus_hybrid_retrieve
  ↓
rerank_evidence
  ↓
grade_evidence
  ├── enough evidence → generate_answer_streaming
  └── not enough evidence && retry_count < 3
          ↓
        rewrite_query_retry
          ↓
        milvus_hybrid_retrieve
          ↓
        rerank_evidence
          ↓
        grade_evidence
  ↓
END
```

和之前相比，多了一个明确的 `rerank_evidence` 节点。这样 workshop 里可以很好地解释：

```text id="rerank_reason_001"
Milvus 负责高召回：
  dense vector + keyword/BM25 + metadata filter + order_by

Reranker 负责高精排：
  根据 query 和候选 chunk 重新排序，提升最终上下文质量

Evidence grader 负责判断：
  rerank 后的证据是否足够回答问题
```

---

# 2. 推荐节点设计

## 2.1 classify_query

保留，但不要做复杂。

职责：

```text id="classify_001"
判断问题类型：
  - architecture
  - policy
  - product
  - general
  - unknown

判断是否需要检索：
  - need_retrieval: true / false
```

MVP 中，大多数企业内部问题都走检索。

示例输出：

```json id="classify_out_001"
{
  "query_type": "architecture",
  "need_retrieval": true,
  "reason": "Question asks about internal S3 sync design."
}
```

---

## 2.2 rewrite_query

职责：

```text id="rewrite_001"
生成更适合检索的 query：
  - 原始中文 query
  - 英文技术表达
  - 企业术语扩展
  - keyword query
```

示例：

```json id="rewrite_out_001"
{
  "rewritten_queries": [
    "S3 文档同步流程",
    "S3 document sync pipeline",
    "object storage ingestion architecture",
    "MinIO document ingestion flow"
  ]
}
```

---

## 2.3 milvus_hybrid_retrieve

职责：

```text id="retrieve_001"
调用 Milvus：
  - dense vector search
  - keyword / BM25 search
  - metadata filter
  - order by updated_at / priority
  - top_k candidate recall
```

注意：因为后面有 reranker，这里的 `top_k` 可以稍微大一些。

建议：

```text id="topk_001"
milvus_top_k = 20
reranker_top_k = 8
answer_context_top_k = 5
```

这样逻辑是：

```text id="topk_pipeline_001"
Milvus 召回 20 条
  ↓
Reranker 精排到 8 条
  ↓
Evidence grader 判断
  ↓
Answer 使用前 5 条作为主要上下文
```

---

## 2.4 rerank_evidence

这是新加入的关键节点。

职责：

```text id="rerank_001"
输入：
  user_query
  rewritten_queries
  retrieved_chunks

输出：
  reranked_chunks
  rerank_score
  selected_context
```

可以选择两种实现路线。

### MVP 推荐：轻量 reranker

使用一个通用 reranker 模型或 API：

```text id="rerank_model_001"
bge-reranker
cohere rerank
jina reranker
cross-encoder reranker
```

如果为了 workshop 复现稳定，可以先封装一个统一接口：

```python id="reranker_interface_001"
class Reranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        ...
```

后续可以替换实现。

### Demo fallback：规则 reranker

如果不想让 demo 强依赖额外模型，可以准备 fallback：

```text id="rerank_fallback_001"
final_score =
  0.6 * hybrid_score
  + 0.2 * keyword_overlap
  + 0.1 * recency_score
  + 0.1 * priority_score
```

这不如真正 reranker，但现场 demo 更稳。

我的建议是：

```text id="rerank_reco_001"
主实现：
  cross-encoder / bge-reranker

fallback：
  rule-based reranker

UI trace 中展示当前使用哪个 reranker。
```

---

## 2.5 grade_evidence

职责：

```text id="grade_001"
判断 rerank 后证据是否足够：
  - enough_evidence
  - missing_aspects
  - weak_sources
  - suggested_retry_query
```

max_retry=3 后，grade 节点要避免无限循环。

建议判断条件：

```text id="grade_condition_001"
enough_evidence = true，当：
  - 至少 2 条 high relevance chunk
  - rerank_score 超过阈值
  - evidence 覆盖问题核心意图
  - 没有明显冲突

retry，当：
  - relevant_chunks < 2
  - top rerank_score 低
  - citation 不足
  - query 太模糊
```

示例输出：

```json id="grade_out_001"
{
  "enough_evidence": false,
  "reason": "Retrieved chunks mention S3 but do not describe the sync pipeline steps.",
  "missing_aspects": ["pipeline steps", "scheduler", "metadata extraction"],
  "suggested_retry_query": "S3 ingestion pipeline scheduler metadata extraction",
  "retry_count": 1
}
```

---

## 2.6 generate_answer_streaming

职责：

```text id="answer_001"
使用 reranked context 生成答案，并带 citation。
```

注意：因为你要求 **答案 streaming，trace 完成后展示**，所以推荐流程是：

```text id="streaming_001"
1. LangGraph 先完成 retrieve / rerank / grade
2. 如果 evidence 足够，开始 streaming final answer
3. answer stream 结束后，把 trace / evidence / aggregation 渲染出来
```

也就是 trace 不需要边跑边展示。

---

# 3. 更新后的 AgentState

```python id="agent_state_001"
class AgentState(TypedDict):
    user_query: str

    # classify
    query_type: str
    need_retrieval: bool

    # rewrite
    rewritten_queries: list[str]

    # retrieval config
    search_mode: str
    search_filters: dict
    search_order_by: list[str]
    milvus_top_k: int

    # retrieval results
    retrieved_chunks: list[dict]

    # rerank
    reranker_name: str
    reranker_top_k: int
    reranked_chunks: list[dict]

    # grading / retry
    enough_evidence: bool
    evidence_grade: dict
    retry_count: int
    max_retry: int
    retry_queries: list[str]

    # final answer
    answer: str
    citations: list[dict]

    # UI
    aggregations: dict
    metrics: dict
    trace: dict
```

默认参数：

```python id="agent_defaults_001"
DEFAULTS = {
    "max_retry": 3,
    "milvus_top_k": 20,
    "reranker_top_k": 8,
    "answer_context_top_k": 5,
    "search_mode": "hybrid",
    "order_by": ["updated_at desc", "priority desc"]
}
```

---

# 4. Reranker 对 UI 的影响

既然要引入 reranker，UI 里应该明确展示 **rerank 前后对比**。这会让 demo 更有教学价值。

## Evidence Tab 建议改成两段

### 4.1 Milvus Recall Results

展示 Milvus 原始召回：

| rank | hybrid_score | source                | page/chunk | updated_at | snippet |
| ---: | -----------: | --------------------- | ---------- | ---------- | ------- |
|    1 |         0.82 | s3_sync_design.md     | chunk-003  | 2026-05-10 | ...     |
|    2 |         0.79 | architecture_notes.md | chunk-008  | 2026-04-22 | ...     |

### 4.2 Reranked Results

展示 reranker 后排序：

| rerank | rerank_score | old_rank | source                  | page/chunk | selected |
| -----: | -----------: | -------: | ----------------------- | ---------- | -------- |
|      1 |         0.91 |        3 | rag_architecture_v1.pdf | page 3     | yes      |
|      2 |         0.88 |        1 | s3_sync_design.md       | chunk-003  | yes      |
|      3 |         0.76 |        6 | minio_ingestion.md      | chunk-002  | yes      |

这样能清楚说明：

```text id="rerank_ui_reason_001"
Milvus 负责召回候选；
reranker 负责把最适合回答当前问题的证据排到前面。
```

---

# 5. Agent Trace 更新

Trace 应该包含 reranker 和 retry 信息。

建议展示：

```json id="trace_001"
{
  "original_query": "我们 S3 文档同步流程是怎么设计的？",
  "classify_query": {
    "query_type": "architecture",
    "need_retrieval": true
  },
  "query_rewrite_rounds": [
    {
      "round": 0,
      "queries": [
        "S3 文档同步流程",
        "S3 document sync pipeline",
        "object storage ingestion architecture"
      ]
    },
    {
      "round": 1,
      "queries": [
        "S3 ingestion scheduler metadata extraction Milvus indexing"
      ]
    }
  ],
  "milvus_search": {
    "mode": "hybrid",
    "top_k": 20,
    "filters": {
      "source_type": ["local", "s3"]
    },
    "order_by": ["updated_at desc", "priority desc"]
  },
  "reranker": {
    "name": "bge-reranker",
    "input_candidates": 20,
    "output_top_k": 8
  },
  "evidence_grading": {
    "enough_evidence": true,
    "retry_count": 1,
    "max_retry": 3,
    "relevant_chunks": 5
  }
}
```

---

# 6. Schema 更新确认

MVP schema 包含 `image_vector`，大多数为 null，少量图片样例有值。

推荐 collection schema 字段：

```text id="schema_001"
id
doc_id
chunk_id
source_uri
source_type          # local / s3
doc_type             # markdown / pdf / text / image
title
department
page                 # nullable
section              # nullable
text
text_vector
image_vector         # nullable
has_image_vector
updated_at
priority
```

其中：

```text id="schema_notes_001"
text_vector:
  普通文本 chunk 必填

image_vector:
  普通文本 chunk 为 null
  图片样例 record 有值

has_image_vector:
  用于 UI 展示和后续 image retrieval filter

page:
  PDF chunk 使用
  markdown/text 可为空

chunk_id:
  citation 使用
```

---

# 7. 图片样例建议

因为图片检索是进阶功能，但 schema 要展示 image_vector，建议准备 3-5 个图片样例：

```text id="image_samples_001"
sample_data/local_docs/images/
  s3_sync_flow.png
  rag_architecture.png
  milvus_hybrid_search.png
  agentic_rag_workflow.png
  ingestion_pipeline.png
```

这些图片可以作为：

```text id="image_usage_001"
MVP：
  展示 schema 中 image_vector 非空的样例
  Evidence table 中展示 has_image_vector = true

进阶：
  开启 image retrieval
  文本 query → 图片证据
  图片 query → 相似图片 / 相关文档
```

---

# 8. 最终 UI Demo 页面结构更新

## Tab 1：Chat

```text id="tab_chat_001"
- 用户输入
- streaming answer
- chunk/page citation
- source cards
```

## Tab 2：Evidence

```text id="tab_evidence_001"
- Milvus recalled candidates
- Reranked candidates
- selected answer context
- source snippets
```

## Tab 3：Agent Trace

```text id="tab_trace_001"
- classify_query
- rewrite_query rounds
- retry_count / max_retry
- Milvus search params
- reranker params
- evidence grading result
```

## Tab 4：Milvus Features

```text id="tab_milvus_001"
- hybrid search mode
- metadata filters
- order_by
- aggregation by source/doc_type/department
- schema view
- nullable image_vector examples
```

---

# 9. 最终 MVP 定义更新版

可以把最终定义写成：

```text id="final_mvp_001"
UI Demo 使用 Streamlit 实现查询型企业 Agent Chat。

用户输入问题后，LangGraph 执行：
classify_query → rewrite_query → Milvus hybrid search → reranker →
evidence grading → retry up to 3 times → streaming answer generation。

UI 在答案完成后展示：
chunk/page 级 citation、source cards、Milvus 召回结果、rerank 前后对比、
Agent trace、aggregation 统计，以及包含 nullable image_vector 的 Milvus schema。

Ingestion 作为离线 notebook/CLI demo；
S3 使用 MinIO/mock S3；
MFS 作为架构扩展；
图片检索作为进阶功能，但 MVP schema 和少量样例数据保留 image_vector。
```

---

# 10. 现在的实现优先级

## P0：必须实现

```text id="p0_001"
1. Streamlit Chat UI
2. LangGraph workflow
3. classify_query
4. query rewrite
5. Milvus hybrid search
6. reranker
7. evidence grading
8. max_retry = 3
9. streaming answer
10. chunk/page citation
11. Evidence table：Milvus recall + reranked results
12. Trace panel
```

## P1：Milvus 3.0 展示

```text id="p1_001"
1. metadata filter
2. order_by
3. aggregation
4. nullable image_vector schema
5. 少量 has_image_vector=true 的样例数据
```

## P2：进阶扩展

```text id="p2_001"
1. image retrieval
2. MFS connector
3. more document types
4. online ingestion
5. production auth / ACL
```

这个版本已经可以进入项目设计和代码骨架阶段了。

------
