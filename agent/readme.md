# Milvus 3.0 Agentic RAG Workshop

这是一个可以运行、观察和评测的 Agentic RAG Workshop。项目以“企业内部知识助手”为场景，把本地文档和模拟 S3 数据导入 Milvus，通过 LangGraph 编排问题理解、工具选择、混合检索、证据评估、答案校验和多轮 Memory，并在 Streamlit 中动态展示整个 Agent 执行过程。

它适合第一次接触 RAG、Milvus 或 Agent workflow 的开发者，也适合希望把传统 `retrieve → generate` 流程升级为可追踪、可重试、带引用的 Agentic RAG 的工程师。

## 你将构建什么

Workshop 最终产物是一个具备以下界面的知识问答应用：

- **Chat**：连续多轮问答和经过校验的答案；
- **Evidence**：Milvus 召回结果、Rerank 结果、引用和文档版本；
- **Agent Trace**：动态展示意图识别、工具调用、重试、证据评估等步骤；
- **Memory**：展示当前 session 的召回、写入、TTL 和清除状态。

完整流程图见 [`flow.md`](./flow.md)。

```text
用户问题
  → Session Memory 召回
  → Intent / Query Type 分类
  → 专业术语与文档版本解析
  → Permission Check
  → Tool Selection
  → Query Rewrite / Decompose
  → Milvus Hybrid Retrieval
  → Merge / Rerank / Evidence Grading
  → 不足时补充检索，达到上限则 Abstain
  → Answer Generation
  → Citation / Version Self-check
  → Streaming Answer
  → Persist Turn Memory
  → Final Snapshot
```

## 主要功能

### 1. Agentic RAG

这里的 Agent 不会对每个问题机械地执行一次向量搜索，而是根据问题选择路径：

- 判断是普通对话、内部知识、版本比较、操作请求还是 Memory 请求；
- 在搜索私有知识前执行演示用 Permission Check；
- 从 policy、product、meeting、code 等知识工具中选择最小相关集合；
- 将复杂问题拆成最多三个有依赖关系的 subqueries；
- 支持并行对比和基于第一跳证据继续检索的 multi-hop 流程；
- 对证据覆盖度进行评分，缺失时最多执行三轮定向补充检索；
- 证据仍不足时明确 abstain，不用模型补全未知事实。

### 2. 专业术语理解

项目通过预定义 entity catalog 处理行业词、缩写和同义词。例如：

```yaml
entity: GO按钮
aliases:
  - 跳转按钮
  - 领取按钮
comment: 表示触发页面跳转或领取动作的按钮
```

Entity 在 query rewrite 和生成 prompt 中帮助理解问题，但不能作为知识证据，也不能生成 citation。跨行业含义无法确定时，Agent 会要求用户补充场景。

### 3. 文档版本隔离

每个知识 chunk 都带有稳定的：

- `doc_id`
- `doc_version`
- `is_current`
- `chunk_id`

默认问题只检索当前版本；指定版本只检索对应 edition；只有明确的版本比较才允许同时使用多个版本。版本信息会一直保留到 Evidence、Citation 和最终答案，避免不同版本内容交叉显示。

### 4. Milvus Hybrid Retrieval

Milvus-backed 路径组合使用：

- Dense vector search：语义召回；
- Sparse/BM25：产品名、错误码和专业词精确匹配；
- Metadata filters：部门、文档类型、当前版本等范围约束；
- 多工具结果合并和 `chunk_id` 去重；
- Reranker：重新计算相关性并选择生成上下文。

本地 CLI 和测试提供 deterministic fallback，不需要 Milvus 或 API key；Streamlit 使用真实 Milvus collection。

### 5. 引用与答案校验

检索到 Top-K 并不代表可以直接回答。生成前后还会执行：

- Evidence coverage 检查；
- Selected context 上限控制；
- `[C1]`、`[C2]` 等 request-local citation 映射；
- Citation 是否来自本轮 selected chunks 的检查；
- 文档版本一致性检查；
- Abstention 是否包含无证据结论的检查。

答案只有通过校验后才会以 `answer_delta` 分块输出。因此当前 streaming 是 **validated-buffered streaming**，不是直接透传未经验证的模型 token。

### 6. 多轮 Agent Memory

每个成功完成的 turn 可以写入：

- 用户 `short_term`
- 助手 `short_term`
- `session_summary`
- 显式“请记住……”产生的 `task_state`

Memory 按 `session_id` 隔离，并使用显式 `expires_at` 过滤。它可以帮助理解“它有哪些步骤？”之类的指代问题，但不能：

- 成为 KB citation；
- 绕过 Permission Check；
- 授权工具；
- 修改 tool-owned filters；
- 把不足的 KB evidence 变成充分证据。

如果用户中断 stream、没有请求最终结果，本轮内容不会写入 Memory。

### 7. 可观察的 Agent Trace

UI 会动态展示经过脱敏的执行事件，包括：

- Memory recall 状态；
- Intent 和 Query Type；
- Entity 与版本解析；
- Permission decision；
- Tool selection 和 tool calls；
- Retrieval、Rerank 和 Evidence Grade；
- Retry / supplementary retrieval；
- Generation 和 Citation verification；
- 各阶段耗时与最终状态。

Trace 不展示 chain-of-thought、完整 prompt、文档正文、Memory 内容、凭据或原始依赖错误。

## Milvus 3.0 学习重点

Workshop 不只是调用一次 vector search，而是用完整数据链路理解 Milvus 在 Agent 系统中的位置。

| 主题 | Workshop 中的实践 |
| --- | --- |
| Collection schema | 创建 `kb_chunks`、`conversation_memory` 和 `doc_dedup_signatures` |
| Dense + Sparse | 组合语义检索和 BM25/Sparse 检索 |
| Nullable vector | `image_vector` 可为空，为多模态知识对象预留空间 |
| Scalar filter | 按部门、文档类型、版本和当前 edition 约束结果 |
| Vector/scalar index | 通过独立脚本创建并验证索引 |
| Conversation Memory | 使用向量召回、session filter 和显式 TTL |
| Data lifecycle | 安全清理旧 collection、重建 schema、重新导入数据 |
| Dedup | 生成 checksum 和实验性的 MinHash-style signature |
| Observability | 将 Milvus recall 与 Rerank、Evidence Grade 分开展示 |

部分 Milvus 3.0 扩展能力仍属于后续实验，而不是当前 MVP 已验证的能力，例如 External Collection、Snapshot、原生 Entity TTL、服务端 MinHash、EmbList 和 DISKANN。具体边界见 [`specs/93-improvements-review.md`](./specs/93-improvements-review.md)。

## 新手可以获得什么

完成 Workshop 后，你应该能够回答以下问题：

1. Embedding、chunk、vector index 和 similarity search 分别解决什么问题？
2. Dense search、BM25 和 metadata filter 为什么需要组合使用？
3. 为什么传统 RAG 容易出现证据不足、版本污染和错误引用？
4. Agent 如何决定是否检索、使用哪个工具以及何时重试？
5. Query rewrite、multi-hop retrieval、rerank 和 evidence grading 有什么区别？
6. 如何让模型只能引用本轮真正提供的知识 chunk？
7. 如何设计 session-scoped Memory，同时避免它越过证据与权限边界？
8. 如何通过 Trace 和离线评测定位召回、排序、生成中的问题？
9. Milvus 在 Agentic RAG 中为什么不仅是“存向量的数据库”？

## 项目结构

```text
.
├── readme.md                 # Workshop 首页
├── flow.md                   # Agentic RAG 精简版与详细版流程图
├── specs/                    # 权威设计、质量标准和实现计划
└── demo/
    ├── config/               # Entity catalog 等配置
    ├── eval/                 # Golden questions 与 expected answers
    ├── notebooks/            # Workshop 分阶段 notebook
    ├── sample_data/          # 本地文档、mock S3 和版本 manifest
    ├── scripts/              # Collection、Index、Ingestion、Cleanup、Eval
    ├── src/                  # Agentic RAG 实现
    └── tests/                # Deterministic test suite
```

建议阅读顺序：

1. [`flow.md`](./flow.md)：先理解系统运行路径；
2. [`specs/00-prd.md`](./specs/00-prd.md)：了解目标和边界；
3. [`specs/10-data-model.md`](./specs/10-data-model.md)：理解 Milvus 数据模型；
4. [`specs/12-agent-workflow.md`](./specs/12-agent-workflow.md)：理解 Agent 节点；
5. [`specs/20-ui-demo.md`](./specs/20-ui-demo.md)：理解 UI 和 Trace；
6. [`demo/README.md`](./demo/README.md)：执行完整安装和运行步骤。

## 快速开始：离线模式

离线模式不需要 Milvus、OpenAI API key 或对象存储服务，适合新手先理解流程。

需要 Python 3.10 或更高版本，推荐 Python 3.11～3.13。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demo/requirements.txt
```

运行 CLI：

```bash
python -m agent_workshop_demo.cli \
  "我们 S3 文档同步流程是怎么设计的？"
```

运行测试：

```bash
python -m unittest discover demo/tests -v
```

运行 Golden QA 评测：

```bash
python demo/scripts/run_eval.py
```

## 运行 Milvus-backed Demo

先复制配置：

```bash
cp demo/.env.example demo/.env
```

默认本地配置：

```dotenv
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION_NAME=kb_chunks
MILVUS_MEMORY_COLLECTION_NAME=conversation_memory
MEMORY_TOP_K=3
MEMORY_TTL_SECONDS=86400
```

Milvus 启动后，依次执行：

```bash
python demo/scripts/create_collections.py
python demo/scripts/create_indexes.py
python demo/scripts/ingest_demo.py
```

启动 Streamlit：

```bash
python -m streamlit run demo/src/agent_workshop_demo/streamlit_app.py
```

通常可以通过 `http://localhost:8501` 访问。

推荐测试问题：

```text
我们 S3 文档同步流程是怎么设计的？
RAG 架构里 Milvus 负责哪一层？
领取按钮会把用户带到哪里？
GO按钮 v1 和 v2 有什么区别？
请记住我叫张三
你还记得我叫什么吗？
它有哪些步骤？
```

如果 collection 来自旧 schema，先预览固定清理范围：

```bash
python demo/scripts/cleanup_milvus.py
```

确认只包含三个 Workshop collection 后再执行：

```bash
python demo/scripts/cleanup_milvus.py --confirm-drop-demo-data
python demo/scripts/create_collections.py
python demo/scripts/create_indexes.py
python demo/scripts/ingest_demo.py
```

清理脚本只允许操作固定的 Workshop collection，不会接受任意 collection 名称。

## 可选 OpenAI 配置

默认 deterministic provider 不会因为环境中存在 API key 而自动发起网络请求。需要使用 OpenAI 时，在 `demo/.env` 中显式配置：

```dotenv
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

ANSWER_GENERATOR=openai
OPENAI_MODEL=your-enabled-model-id
```

Embedding 固定请求 1,024 维以匹配当前 schema。同一个 collection 不能混用不同 provider/model/dimension 的 vector；切换 embedding provider 后需要重新导入数据。

## 当前边界

这是一个使用 synthetic/curated 数据的本地 Workshop，不是生产系统：

- Permission Check 用于演示节点顺序，不是生产身份认证或 ACL；
- Streamlit 没有生产级登录和租户隔离；
- 不应导入真实公司机密、个人数据或生产凭据；
- Memory TTL、清除和 session isolation 不等同于完整的隐私合规方案；
- deterministic embedding、reranker 和 generator 是可复现 fallback；
- 真实 Milvus/OpenAI/Zilliz Cloud 部署仍需要独立的安全、容量和性能设计。

## 深入阅读

- [`specs/index.md`](./specs/index.md)：全部 specs 和阅读顺序；
- [`specs/10b-conversation-memory.md`](./specs/10b-conversation-memory.md)：多轮 Memory；
- [`specs/13-llm-answer-generation.md`](./specs/13-llm-answer-generation.md)：答案生成与 citation guard；
- [`specs/70-quality-and-evaluation.md`](./specs/70-quality-and-evaluation.md)：测试和评测标准；
- [`demo/README.md`](./demo/README.md)：完整运行、配置和故障排查。
