下面是基于最新资料整理出来的 workshop 方向。Milvus 3.0 目前公开 release notes 显示为 **v3.0-beta，发布日期 2026-05-09**；核心方向非常适合包装成“公司内部 Agent Chat / RAG 升级”的项目，因为 3.0 强化了 lake integration、多模态/多向量检索、生命周期治理、文本检索可定制化和运维控制。([Milvus][1])

## 1. Workshop 题目建议

**基于 Milvus 3.0 构建企业内部 Agent Chat：从传统 RAG 到 Agentic Multimodal RAG**

副标题可以是：

**连接云文档、S3 与本地资料，构建可搜索、可追溯、可治理的企业知识助手**

这个题目比“RAG 升级”更明确，突出三个卖点：

1. **企业内部资料异构分布**：本地文档、S3、云文档、代码仓库、Slack/Gmail/Issue 等。
2. **多模态 RAG**：文本 + 图片 + PDF 截图/图表/架构图。
3. **Agentic RAG**：不是固定 retrieve-then-generate，而是带查询规划、检索路由、证据校验、重写和引用的 Agent Chat。

---

## 2. Milvus 3.0 新特性与 workshop 可用点

| Milvus 3.0 特性                              | Release notes 含义                                                                                                                                       | 在 workshop 中怎么用                                                                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **External Collection**                    | Milvus Collection 可以引用 Parquet、Lance、Iceberg 等外部 lake/object storage 文件，Milvus 只管理 schema、index 和 query execution，支持 incremental refresh。([Milvus][1]) | 对应“公司资料在 S3 / data lake，不想全部复制进向量库”。可以设计一个 S3 文档/embedding 数据集，用 External Collection 演示 zero-copy 或准 zero-copy 检索架构。               |
| **Snapshot**                               | 创建 point-in-time 只读视图，batch job 可以在 MVCC-style isolation 下读取，live collection 继续写入。([Milvus][1])                                                        | 用于“离线评测 / A/B embedding 模型评估 / 回滚”。workshop 可以演示：对同一批企业文档创建 snapshot，然后比较旧 embedding 与新 embedding 检索效果。                            |
| **Query/Search Order By**                  | Search 和 Query 支持多字段排序，并 push down 到 Milvus kernel，避免应用层 over-fetch 后重排。([Milvus][1])                                                                  | 企业问答经常不只看相似度，还要按 `updated_at DESC`、`doc_priority DESC`、`department`、`security_level` 排序。可以演示“相似度 + 最新版本优先”。                        |
| **Query Aggregation**                      | 支持 `group_by_fields` 和 `count/sum/avg/min/max` 等 server-side aggregation。([Milvus][1])                                                                 | 用于 UI 里的“检索结果分布”：来源于 S3 多少、本地文档多少、不同部门/文档类型命中多少；也可做数据质量 dashboard。                                                                 |
| **Null Vector**                            | vector 字段可为 NULL；搜索会跳过 NULL vector；支持在线扩展 nullable vector field。([Milvus][1])                                                                          | 非常适合多模态数据。不是所有文档都有图片，不是所有图片都有 OCR 文本。可以设计 schema：`text_vector`、`image_vector`、`layout_vector`，允许部分为空。                              |
| **Custom Dictionary & Synonym Dictionary** | FileResource 支持注册 tokenizer dictionary、synonym、stop words、decompounder rules，可用于 BM25、analyzer、Text Match。([Milvus][1])                                | 企业内部术语、缩写、项目代号非常多。可以演示 synonym：`PTO = paid time off`、`SRE = site reliability engineering`、`Milvus 3 = Milvus 3.0`。尤其适合中文/中英混合内部文档。 |
| **Entity TTL**                             | 支持 per-entity TTL，适合 right-to-be-forgotten、session data、bounded conversation history。([Milvus][1])                                                     | 用于 Agent Chat 的会话记忆、临时上传文档、过期政策文档。可以演示“临时知识 24 小时后自动失效”。                                                                           |
| **MinHash DIDO**                           | 服务端 MinHash function，可用于近重复检测、指纹、抄袭检测。([Milvus][1])                                                                                                    | 企业文档经常有重复版本、复制粘贴、历史草稿。可以做 ingestion 阶段去重：避免 RAG 返回多个几乎相同的 chunk。                                                                   |
| **EmbList + DISKANN**                      | 支持一个 entity 存可变长度 vector list，适合长文档、多 chunk、ColBERT/late interaction、多视角多模态；DISKANN 降低内存压力。([Milvus][1])                                               | 这是 workshop 的亮点。传统“一 chunk 一向量”可以升级成“一份文档/一页 PDF/一个知识对象含多个 vectors”：文本 chunk vectors + 图片 vectors + 页面区域 vectors。                  |
| **Storage V3**                             | manifest-based columnar storage，数据和 metadata 位于 S3-compatible object storage，支撑 External Collection、Snapshot 和 lake integration。([Milvus][1])          | 用于讲企业级架构：Milvus 不只是“向量 DB”，而是和 object storage / lakehouse 结合，适合大规模内部知识库。                                                           |
| **Force Merge**                            | 支持手动触发 segment compaction，降低碎片导致的延迟抖动和存储膨胀。([Milvus][1])                                                                                               | 放在“生产化运维”章节：大规模增量同步后，在低峰期 force merge。                                                                                             |

---

## 3. MFS 在项目里的定位

MFS 的定位非常贴合你的主题：它把代码、memory、skills、docs、消息、SaaS、数据库、对象存储等上下文统一成一个 file-like workspace，Agent 可以用统一方式搜索和浏览。MFS README 里明确列了 Docs/knowledge、Slack/Gmail、Jira/Linear、Postgres/Mongo/BigQuery/Snowflake/S3/Drive 等来源。([GitHub][2])

MFS 的核心交互模型也很适合拿来做 workshop：

```text
mfs add postgres://reports
mfs search "churn assumptions" --all
mfs cat <hit> --range 40:80
```

它强调“先 search 找候选，再 browse/cat 打开精确内容”，这正好可以作为 Agentic RAG 的可靠性原则：**检索结果不是证据，重新打开原文才是证据**。([Zilliz][3])

MFS 底层是 Rust CLI 调 Python server，server 负责 connectors、indexing、retrieval、metadata、cache，并把内容索引进 Milvus 做 hybrid search；它既可以本地离线跑，也可以接 production-scale 的 Milvus/Zilliz 集群。([Zilliz][3])

建议 workshop 中把 MFS 放在 **数据接入层 / context harness 层**：

```text
异构数据源
  ├── 本地 docs
  ├── S3 docs / parquet / images
  ├── Google Drive / Notion / Confluence
  ├── Slack / Gmail / Jira
  └── Code repo

        ↓ MFS connector / ingest

统一 file-like namespace
        ↓
chunk / OCR / image extraction / metadata enrichment
        ↓
Milvus 3.0
        ↓
Agentic RAG
        ↓
Agent Chat UI / Notebook / Vibe Coding Demo
```

---

## 4. DINOv3 在多模态 RAG 里的定位

DINOv3 更适合作为 **图片/视觉特征 embedding**，而不是文本 embedding。Hugging Face 文档里说明 DINOv3 可以输出整张图片的 embedding，通常是 CLS token，适合 classification 和 retrieval；也可以使用 patch-level local embeddings 做更细粒度视觉任务。([Hugging Face][4])

所以你的多模态设计可以是：

```text
文本内容：
  text → text embedding model → text_vector

图片内容：
  image / PDF figure / screenshot → DINOv3 → image_vector

PDF / PPT / Wiki 页面：
  OCR text → text_vector
  embedded image / chart / architecture diagram → image_vector
  metadata: source, page, section, updated_at, ACL, department
```

推荐不要把 DINOv3 讲成“文本+图片统一 embedding”。更稳妥的说法是：

> DINOv3 用于企业文档中的图片、截图、架构图、产品图、白板图等视觉内容检索；文本仍使用文本 embedding 模型。Milvus 3.0 的 Null Vector、EmbList、多字段 schema 可以把文本向量和图片向量组织在同一个知识对象里。

---

## 5. Agent 开发技巧：适合 workshop 的 Agentic RAG 模式

LangGraph 官方 Agentic RAG 教程强调：Agent 可以决定什么时候调用 retriever tool，而不是每次都固定检索。它的基本流程包括文档预处理、语义索引、创建 retriever tool，然后构建能自主判断是否检索的 RAG agent。([LangChain Docs][5])

LlamaIndex 的 agentic strategies 也强调在已有 RAG query engines 上部署 agent loop，用于 query planning 和更高级的决策。([Developer Documentation][6])

可以把 workshop 的 Agent 技巧设计成 7 个递进模块：

### 5.1 Query Router：判断是否需要检索、检索哪个源

用户问题先分意图，再判断主题：

```text
1. 普通闲聊 / 通用解释
2. 私有知识问答
3. 对比分析
4. 操作任务
5. 权限敏感问题
```

Agent 决策：

```text
判断是否需要检索
  -> 私有知识先检查权限
  -> 选择最相关的知识工具
  -> 不默认搜索所有知识域
```

### 5.2 Tool Selection：Agent 自主选择知识域

```text
search_policy_docs
search_product_docs
search_meeting_notes
search_code_docs
get_user_permission
summarize_document
```

`source_type`、`doc_type`、`department` 是 tool-owned metadata filters，
只在 Agent Trace 中解释，不作为 UI Search Controls 交给用户。

### 5.3 Query Rewrite / Decomposition：企业术语扩展与问题拆解

把用户口语化问题改写成更适合检索的 query：

```text
用户：今年假期咋算？
改写：
- vacation policy 2026
- PTO policy
- annual leave
- paid time off
- 假期政策
```

这里可以结合 Milvus 3.0 的 Custom Dictionary / Synonym Dictionary，把企业术语、缩写、项目代号沉到检索层，而不是只靠 prompt rewrite。([Milvus][1])

复杂问题最多拆成三个带 tool 与依赖关系的 subqueries。对比问题可以并行检索，
多跳问题可以用第一跳证据细化第二跳 query。

### 5.4 Hybrid Retrieval：向量 + BM25 + metadata filter

企业 RAG 不应该只做 dense vector search。建议 workshop 展示：

```text
dense vector search:
  语义召回

BM25 / Text Match:
  精确术语、代码名、产品名、错误码

metadata filter:
  department = "engineering"
  security_level <= user_clearance
  updated_at > 2025-01-01
```

Milvus 3.0 的 synonym dictionary、BM25/analyzer/Text Match 定制能力很适合放在这一节。([Milvus][1])

### 5.5 Evidence Grading：检索结果质量判断

Agent 不应该拿到 top-k 就直接回答。可以加一个 evaluator：

```text
输入：
  user question
  retrieved chunks

输出：
  relevant / partially_relevant / irrelevant
  missing_info
  need_more_search: true/false
```

如果证据不足，进入 query rewrite 或多跳检索。LangChain 的 self-reflective RAG 实践也强调 document grading、retry queries、提升 generation quality。([LangChain][7])

### 5.6 Multi-hop Retrieval：多源迭代检索

例如用户问：

> “我们现在 Milvus 迁移方案里，S3 上的数据同步和权限控制是怎么设计的？”

Agent 可能需要：

```text
Step 1: 搜索架构设计文档
Step 2: 搜索 S3 数据同步脚本或配置
Step 3: 搜索权限/ACL 文档
Step 4: 汇总并引用来源
```

这可以和 MFS 的 search → cat 精确读取模式结合：先找候选，再打开原文。MFS 文档也强调 search results 只是起点，可靠做法是重新打开精确 source 后再引用、编辑或行动。([Zilliz][3])

### 5.7 Answer with Citations + Self-check：必须带出处

企业内部 Agent Chat 的关键不是“像人一样回答”，而是：

```text
答案
证据来源
文档版本
更新时间
命中的页码 / 行号 / 文件路径
置信度
不确定信息
```

这个环节可以在 UI demo 中做成“答案 + Sources + Debug Trace”。

---

## 6. Workshop 项目架构建议

### 6.1 Demo 架构

```text
Streamlit UI
  ├── Chat
  ├── Evidence
  └── Agent Trace

Agent Layer
  ├── intent / retrieval decision
  ├── permission tool
  ├── tool selection
  ├── query rewrite / decomposition
  ├── multi-hop retriever tools
  ├── evidence grader
  ├── answer generator
  └── citation self-check

Retrieval Layer
  ├── Milvus 3.0 hybrid search
  ├── text vector search
  ├── image vector search via DINOv3
  ├── BM25 / Text Match
  ├── metadata filter
  └── order by updated_at / priority

Ingestion Layer
  ├── local files
  ├── S3 documents
  ├── MFS connector
  ├── OCR / PDF parsing
  ├── image extraction
  ├── chunking
  ├── dedup via MinHash
  └── metadata enrichment

Storage
  ├── Milvus 3.0
  ├── S3-compatible object storage
  └── optional SQLite/Postgres for app state
```

---

## 7. 数据构造建议

### 7.1 本地文档数据

构造一个模拟公司内部知识库：

```text
company_docs/
  hr/
    vacation_policy_2026.md
    reimbursement_policy.pdf
    remote_work_policy.md

  engineering/
    milvus_migration_design.md
    rag_architecture_v1.pdf
    incident_review_2026_q2.md
    service_slo.md

  product/
    product_roadmap.md
    feature_spec_agent_chat.md

  security/
    data_access_policy.md
    pii_redaction_guideline.md

  images/
    rag_architecture.png
    milvus_storage_v3_diagram.png
    s3_sync_flow.png
```

### 7.2 S3 数据

S3 里放两类数据：

```text
s3://internal-agent-chat-demo/raw-docs/
  onboarding.pdf
  architecture/
  policy/
  screenshots/

s3://internal-agent-chat-demo/lake/
  documents.parquet
  chunks.parquet
  image_assets.parquet
  metadata.parquet
```

这样可以对应 Milvus 3.0 的 External Collection / Storage V3 叙事：文档和 metadata 原本就在 object storage / lake 中，Milvus 负责 schema、index 和检索。([Milvus][1])

### 7.3 示例问题

```text
1. 我今年还有多少种假期可以申请？远程办公政策里有没有限制？
2. Milvus 迁移方案里，为什么要把对象存储作为底层数据源？
3. Agent Chat 的 RAG 架构图里，MFS 在哪一层？
4. 找一下和 S3 同步流程相关的设计图。
5. 我们对临时上传文档的保留时间是多久？
6. RAG 检索结果为什么要按更新时间重排？
7. 哪些文档提到了 PII redaction？
8. 这张架构图和哪份设计文档最相关？
```

---

## 8. 三类受众的 workshop 形式设计

你原来的三类受众划分很好，可以设计成 **同一个项目，三种入口**。

### 8.1 面向开箱即用用户：UI Demo

目标：让用户直观看到“企业内部 Agent Chat”的价值。

演示流程：

```text
1. 打开 Chat UI
2. 输入自然语言问题
3. Agent 自动选择知识工具
4. Agent 返回答案
5. 展示 Sources
6. 展示检索 trace：
   - intent / permission
   - query plan
   - selected tools
   - top-k docs
   - evidence grading
   - supplementary retrieval
   - citation self-check
   - final answer
7. 切换一个问题，演示图片检索 / 架构图检索
```

UI 重点展示：

```text
答案区域
Sources 区域
命中文档分布
图片/图表命中
更新时间排序
debug trace
```

可以把 Milvus 3.0 的 Query Aggregation 用在 “命中来源统计”：

```text
Source distribution:
  local docs: 4
  S3 docs: 7
  engineering: 5
  security: 2
```

### 8.2 面向开发者：可运行 demo + Jupyter Notebook

Notebook 可以拆成 8 个章节：

```text
00_setup.ipynb
  启动 Milvus 3.0 / Milvus Lite / docker compose
  准备 S3 mock / MinIO

01_load_documents.ipynb
  加载本地文档
  加载 S3 文档
  解析 PDF / Markdown / 图片

02_build_embeddings.ipynb
  文本 embedding
  DINOv3 image embedding
  schema 设计：text_vector / image_vector / nullable vector

03_ingest_to_milvus.ipynb
  insert / upsert
  metadata
  MinHash 去重
  TTL 字段

04_hybrid_search.ipynb
  dense search
  BM25 / Text Match
  metadata filter
  order by updated_at

05_agentic_rag.ipynb
  query router
  query rewrite
  retriever tool
  evidence grader
  answer with citations

06_multimodal_rag.ipynb
  根据问题检索图片
  根据图片找相关文档
  文本 + 图片证据融合

07_eval_and_observability.ipynb
  golden QA
  recall@k
  citation correctness
  latency
  token usage
```

### 8.3 面向 Vibe Coding 用户：实操步骤

可以给一个“从自然语言到可运行 demo”的流程，让用户用 Cursor / Claude Code / Codex / Copilot 之类工具跟做。

建议步骤：

```text
Step 1: 让 coding agent 生成项目骨架
Prompt:
“创建一个企业内部 Agent Chat demo，使用 Streamlit 直接运行 Agent workflow，
使用 Milvus 3.0 做 hybrid retrieval，支持本地 docs 和 S3 docs ingestion。”

Step 2: 生成 ingestion pipeline
Prompt:
“实现 docs ingestion：支持 markdown/pdf/png/jpg，
抽取文本、图片和 metadata，文本使用 text embedding，图片使用 DINOv3 embedding。”

Step 3: 生成 Milvus schema
Prompt:
“设计 Milvus collection schema，包含 doc_id、chunk_id、source_uri、department、
updated_at、security_level、text、text_vector、image_vector，其中 image_vector 可为空。”

Step 4: 生成 retriever
Prompt:
“实现 hybrid retriever：支持 text vector search、BM25、metadata filter、
按 updated_at 和 priority 重排，返回 source citation。”

Step 5: 生成 Agentic RAG workflow
Prompt:
“实现 intent classification、permission gate、tool selection、query
decomposition、multi-hop retrieval、evidence grading、answer synthesis 和
citation self-check。如果证据不足，最多补充检索 3 次。”

Step 6: 生成 UI
Prompt:
“实现 Streamlit Chat UI，展示答案、sources、debug trace、命中文档分布和图片命中。”

Step 7: 生成测试数据
Prompt:
“生成一套模拟公司内部文档，包括 HR、Engineering、Security、Product，
并生成 10 个 QA 测试用例。”
```

---

## 9. 推荐 workshop 详细大纲

### Part 0：开场，为什么企业 RAG 需要升级

```text
0.1 传统 RAG 的问题
  - 数据源分散
  - 只支持文本
  - chunk 粒度粗糙
  - 没有权限/生命周期治理
  - 检索失败后容易幻觉
  - 缺少证据链

0.2 Agent Chat 的目标
  - 能连接多源数据
  - 能检索文本和图片
  - 能判断是否需要检索
  - 能多轮检索和自我修正
  - 能给出处
  - 能生产化治理
```

### Part 1：Milvus 3.0 能力速览

```text
1.1 External Collection：企业 lake / S3 数据不必重复搬运
1.2 Storage V3：object storage-native 的数据组织方式
1.3 Snapshot：离线评测、回滚、A/B 检索实验
1.4 EmbList + DISKANN：长文档、多向量、多模态检索
1.5 Null Vector：文本/图片字段不完整时的优雅建模
1.6 Custom Dictionary / Synonym：企业术语和中文检索优化
1.7 MinHash：重复文档和近重复 chunk 清理
1.8 Entity TTL：会话记忆和临时文档自动过期
1.9 Query Aggregation / Order By：生产级检索排序和统计
```

### Part 2：企业内部资料接入

```text
2.1 数据源类型
  - local docs
  - S3 docs
  - S3 parquet/lake table
  - cloud docs
  - code repo
  - chat / issue / email

2.2 MFS 作为 context harness
  - add source
  - search source
  - cat exact source
  - search → browse → cite

2.3 文档处理 pipeline
  - parse markdown/pdf/docx/html
  - extract images
  - OCR
  - chunking
  - metadata enrichment
  - ACL/security metadata
  - dedup
```

### Part 3：多模态 Embedding 与 Milvus Schema

```text
3.1 文本 embedding
3.2 图片 embedding：DINOv3
3.3 PDF 中的图片、图表、截图如何处理
3.4 schema 设计
  - doc_id
  - chunk_id
  - source_uri
  - source_type
  - department
  - security_level
  - updated_at
  - ttl_at
  - text
  - text_vector
  - image_vector nullable
  - minhash_vector
3.5 一文多向量：EmbList vs chunk-level collection
3.6 metadata filter 与权限控制
```

### Part 4：从 RAG 到 Agentic RAG

```text
4.1 Naive RAG baseline
  - retrieve top-k
  - stuff context
  - generate answer

4.2 Agentic RAG workflow
  - classify intent / decide retrieval
  - check permission
  - choose knowledge tools
  - rewrite / decompose
  - one or more retrieval calls
  - grade evidence
  - targeted supplementary / multi-hop retrieval
  - synthesize answer
  - citation self-check

4.3 Self-correction
  - 检索结果不相关怎么办
  - 证据不足怎么办
  - 结果冲突怎么办
  - 用户问题太模糊怎么办

4.4 企业级约束
  - 不越权
  - 不回答无来源内容
  - 不泄露敏感信息
  - 不使用过期文档
```

### Part 5：UI Demo

```text
5.1 输入企业问题
5.2 展示答案
5.3 展示 sources
5.4 展示图片证据
5.5 展示 debug trace
5.6 展示命中统计
5.7 展示检索失败后的 query rewrite
```

### Part 6：开发者实操

```text
6.1 启动 Milvus 3.0
6.2 准备本地和 S3 数据
6.3 跑 ingestion
6.4 构建文本和图片 embedding
6.5 写入 Milvus
6.6 实现 hybrid search
6.7 实现 Agent workflow
6.8 启动 UI
6.9 跑评测
```

### Part 7：Vibe Coding 实操

```text
7.1 用自然语言生成项目骨架
7.2 让 Agent 补 ingestion pipeline
7.3 让 Agent 补 Milvus schema
7.4 让 Agent 补 retriever
7.5 让 Agent 补 UI
7.6 用 notebook 验证每一步
7.7 给 Agent 增加 eval case
```

### Part 8：生产化与扩展

```text
8.1 权限控制：user → source ACL → metadata filter
8.2 数据更新：incremental refresh / upsert / TTL
8.3 数据质量：MinHash dedup / missing vector / stale docs
8.4 检索质量：hybrid search / rerank / synonym dictionary
8.5 评测：retrieval recall、answer correctness、citation correctness
8.6 成本：DINOv3 离线批处理、DISKANN、Snapshot eval
8.7 运维：Force Merge、监控、慢查询、segment fragmentation
```

---

## 10. 最终建议的 workshop 结构

如果是 2.5 到 3 小时，可以这样安排：

|          时间 | 内容                                                    |
| ----------: | ----------------------------------------------------- |
| 0:00 - 0:15 | 企业 RAG 痛点与 Agent Chat 目标                              |
| 0:15 - 0:40 | Milvus 3.0 新特性与项目映射                                   |
| 0:40 - 1:05 | 数据接入：本地文档、S3、MFS                                      |
| 1:05 - 1:35 | 多模态 embedding：文本 + DINOv3 图片                          |
| 1:35 - 2:05 | Agentic RAG：intent、tools、multi-hop、grading、self-check |
| 2:05 - 2:25 | UI Demo：企业知识问答、图片检索、debug trace                       |
| 2:25 - 2:50 | Notebook 实操：ingest → search → answer                  |
| 2:50 - 3:00 | Vibe Coding 实操路线与扩展作业                                 |

---

## 11. 一个更聚焦的 MVP 版本

为了避免 workshop 太大，建议 MVP 先做这 5 个功能：

```text
1. 本地 docs + S3 docs ingestion
2. 文本 embedding + DINOv3 图片 embedding
3. Milvus 3.0 schema：text_vector + image_vector nullable + metadata
4. Hybrid retrieval：vector + keyword + metadata filter + order by updated_at
5. Agentic RAG：tool selection + query plan + multi-hop + evidence grading + citation self-check
```

Milvus 3.0 的高级特性可以分层展示：

```text
必做：
  Null Vector
  Custom Dictionary / Synonym
  Query Order By
  Query Aggregation

进阶：
  EmbList + DISKANN
  MinHash
  Entity TTL

架构讲解：
  External Collection
  Snapshot
  Storage V3
  Force Merge
```

这样更适合 workshop：既能跑起来，又能讲出 Milvus 3.0 的技术纵深。

[1]: https://milvus.io/docs/release_notes.md "Release Notes | Milvus Documentation"
[2]: https://github.com/zilliztech/mfs "GitHub - zilliztech/mfs: A context harness for AI agents: all your scattered context — code, memory, docs, databases, SaaS — in one searchable, browsable, file-like interface. · GitHub"
[3]: https://zilliztech.github.io/mfs/ "Overview - MFS"
[4]: https://huggingface.co/docs/transformers/model_doc/dinov3?utm_source=chatgpt.com "DINOv3 · Hugging Face"
[5]: https://docs.langchain.com/oss/python/langgraph/agentic-rag?utm_source=chatgpt.com "Build a custom RAG agent with LangGraph - Docs by LangChain"
[6]: https://developers.llamaindex.ai/python/framework/optimizing/agentic_strategies/agentic_strategies/?utm_source=chatgpt.com "Agentic strategies | Developer Documentation - LlamaIndex"
[7]: https://www.langchain.com/blog/agentic-rag-with-langgraph?utm_source=chatgpt.com "Self-Reflective RAG with LangGraph - LangChain Blog"
