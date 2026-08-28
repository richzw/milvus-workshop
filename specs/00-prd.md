# PRD — Milvus 3.0 Enterprise Agent Chat Workshop

Status: draft v8 · Owner: workshop author · Last updated: 2026-08-24

## 1. Problem

企业内部资料分散在本地文件、S3/对象存储等异构数据源中，文本、PDF 和图片混合存在。普通 RAG 示例通常只展示“切分—向量检索—回答”，难以解释真实系统里的混合检索、证据筛选、失败重试、引用追踪和数据治理。

Workshop 需要同时满足三类学习者：想直接体验效果的用户、希望理解实现的开发者，以及希望通过 vibe coding 完成实践的参与者。现有草稿描述了这些功能，但产品范围、数据模型与实现顺序混在一起，缺少可验证的交付合同。

## 2. Vision

参与者可以运行一个本地、可复现的企业 Agent Chat：离线导入带文档版本 metadata 的本地与 MinIO 文档后，先用多组切块配置的检索与生成评测选择语料配置；在 Streamlit 只输入自然语言问题，LangGraph 通过受限结构化 LLM classification 理解意图并在失败时回退到 rule-based classifier，随后使用预定义词语实体消歧行业术语，执行权限检查，自主选择知识工具，在 identity、rewrite、step-back 或至多三个子问题的 decompose 中选择一种受限转换策略，经版本隔离的一次或多次 Milvus 混合检索、reranker 与 evidence grader 汇总证据，必要时最多补充检索 3 轮。对长文档，Agent 可在 Phase 0 能力验证后使用 Milvus 3.0 StructArray，在一个 document entity 内搜索带稳定 `chunk_id`、页码和章节的 passages；单向量问题保留 element offset 定位局部证据，多方面问题可先用 EmbeddingList 找到覆盖多个局部匹配的文档，再回到 element-level 命中生成可引用证据。对超过预算的 selected context，生成前可按原文 provenance 做选择性压缩，最后生成并自检带 chunk/page/version citation 的答案。界面同时解释“问题如何分类与转换、术语如何理解、为什么选择这些工具和文档版本、检索命中的是文档还是其中 passage、上下文是否被压缩、为什么证据足够或不足”。

```text
问题：我们 S3 文档同步流程是怎么设计的？
结果：流式答案 + [C1]/[C2] 引用
解释：原始召回 → rerank 后排序 → evidence grade → retry rounds
```

## 3. Goals

| # | Goal | Measure |
| --- | --- | --- |
| G1 | 提供最小可运行的端到端 Agentic RAG demo | 一条 golden question 能从预置数据检索、精排、回答并返回至少 1 个可解析 citation |
| G2 | 让 Agent 决策可教学、可观察 | Chat/Evidence/Agent Trace 三个核心 tab 能展示 intent、permission、tool selection、query plan、recall、rerank、grade、retry 与 answer self-check，且 trace 与本次回答使用同一 `query_id` |
| G3 | 展示 Milvus 在企业 RAG 中的核心价值 | 经 Phase 0 验证后，demo 展示 dense+sparse/hybrid、metadata filter，以及至少一种排序或聚合能力 |
| G4 | 支持不同学习路径 | 提供 UI demo、可运行代码和按依赖排列的 notebook/实践步骤 |
| G5 | 保持现场演示可恢复 | reranker 或 OpenAI answer generation 不可用时，显式 deterministic fallback 仍能完成 golden path；失败路径不伪装成模型成功 |
| G6 | 避免行业术语误解和文档版本串答 | 术语 fixture 使用命中的预定义实体完成正确改写；非版本对比问题的 selected context 不混合同一文档的多个版本 |
| G7 | 让自然语言会话历史问题得到确定性回答 | “查找下我最近的三个问题是什么”等 same-session 请求只返回仍有效的最近 3 个 user turns，从近到远排列，不调用 KB search tools |
| G8 | 建立可维护、可行动的 eval metric set | active registry 同时覆盖 goal、guardrail、operational 三类；每个 metric 都绑定来源、grader、owner、成本、阈值或观察预算、触发动作和退休条件，行为质量指标先由 30–50 条 trace 的 error analysis 验证 |
| G9 | 展示局部粒度检索与整体文档身份可同时保留 | StructArray fixture 同时证明 same-element filter、element-level offset 命中和 EmbeddingList entity shortlist；所有最终 citation 仍解析到稳定 `chunk_id/doc_version` |

## 4. Non-goals

- MVP 不提供在线 ingestion；导入通过 notebook 或 CLI 离线完成。
- MVP 不接入真实企业 S3、云 Doc 或 MFS；S3 使用 MinIO/mock S3，MFS 仅作为扩展设计。
- MVP 不提供生产级认证、ACL、租户隔离、审计或 secret 管理，因此不得部署为真实内部知识系统。
- 图片检索、conversation memory、MinHash 近重复检测不是首个垂直切片的硬依赖。
- 本 spec 不承诺未经 Phase 0 验证的 Milvus API、索引类型或精确参数。
- StructArray 不在首个垂直切片中取代 `kb_chunks`，不把 element offset 当作持久 citation id，也不在本阶段引入 ColBERT/ColPali 模型训练或 LEMUR 训练管线。

## 5. Users

- **体验者（primary）**：希望通过 UI 理解企业 Agent Chat 能做什么，以及答案证据从哪里来。
- **开发者（primary）**：希望通过 notebook 与代码掌握 ingestion、schema、hybrid search 和 LangGraph 工作流。
- **Vibe Coding 学习者（secondary）**：需要一条按阶段、有验收点的实践路径。
- **反向 persona**：需要生产级安全、实时同步或大规模 SLA 的平台团队；本 Workshop 只能作为概念验证。

## 6. MVP scope

### P0 — 可运行主链路

- Streamlit 只提供自然语言输入，以及 Chat、Evidence、Agent Trace 三个核心 tab；不暴露 source/doc/department search controls。P2 的 Memory tab 由 [`20-ui-demo.md § Memory`](./20-ui-demo.md#44-memory) 定义，不属于 P0 主链路。
- LangGraph：classify intent → resolve terminology → decide retrieval → check permission → select tools → identity/rewrite/step-back/decompose → version-scoped one-or-more retrieve → rerank → grade → bounded supplementary retrieval → optional context compression → answer/abstain → citation/self-check。
- Query classification 使用固定 intent/topic/retrieval-goal 枚举；OpenAI Structured Outputs 为可配置主实现，明确 memory/operation action 与所有 provider/output failures 使用可观察的 rule-based safe path。
- `source_type`、`doc_type`、`department` 是 search tool 的内部过滤参数，由 Agent 选择并写入 trace，不由用户手动指定。
- 查询转换只能从 `identity | rewrite | step_back | decompose` 中选择一个主策略；终端可执行查询总数仍不超过 3，每个派生查询必须继承原始意图、permission、tool allow-list、实体和 version scope。
- 预定义词语实体 catalog 为 query understanding 提供领域词、别名和解释，例如 `GO按钮` 表示“跳转/领取按钮”；只把与当前问题匹配的实体注入术语解析/改写 prompt，并在 trace 中记录解析结果。
- 每个 `kb_chunks` 记录必须携带 `doc_version` 与 `is_current`。默认检索仅使用 current version；只有用户明确要求指定版本或版本对比时，工具才改变 version scope。
- 答案经 citation guard 校验后分块输出；完成后展示 citation、source cards、evidence 与 trace。
- OpenAI 基于 selected source chunks 的原文或 provenance-safe projection 合成答案；配置缺失或 provider 失败时使用可观察的 deterministic fallback。
- Context compression 是可关闭、有阈值的 generation 前优化；默认优先保留可在原 chunk 中精确复核的 verbatim spans，任何无法验证的压缩结果回退原 selected chunks。
- 同一 session 的 exact 或高置信语义等价 KB 问题可复用三天内、权限与 KB evidence 仍有效的 grounded response；cache hit 保留原 citations。
- 主知识库来自离线预置的本地与 MinIO 文档。

### P1 — Milvus 教学能力

- metadata filter、排序/聚合展示，以及 nullable `image_vector` 样例；metadata filters 只作为工具执行细节展示。
- 对有清晰文档 parent 和可变长 passages 的长文档构建 `kb_documents` StructArray 检索投影；展示 `MATCH_*`、`element_filter`、element-level search 和 EmbeddingList search 的结果粒度差异。
- Agentic RAG 只接收规范化的 passage evidence。EmbeddingList 的 entity-level 结果必须再经 element-level 定位；混合检索 collapse 丢失 offset 时不得直接产生 citation。
- 每项能力以 Phase 0 的实际 SDK/服务验证结果为准。

### P2 — 扩展实验

- 图片检索、selective dual-speed conversation memory、文档去重、MFS、更多文档类型、在线 ingestion、生产级 auth/ACL。
- Memory 以 append-only episode lineage、Selection Gate、versioned durable facts 和 working-state projection 演进；Milvus Decay Ranker 只承担软遗忘，逻辑失效与物理删除保持独立。

## 7. Success metrics

本节定义产品目标与硬约束，不等同于要求长期展示的 metric dashboard。哪些信号进入 active eval、由什么 evaluator 计算、变化后采取什么动作，以 [`70-quality-and-evaluation.md § Metric portfolio contract`](./70-quality-and-evaluation.md#40-metric-portfolio-contract) 的小型 registry 为准；没有决策用途的诊断字段不得因为“已经能采集”就升级为 KPI。

- 所有 golden questions 都产生结构合法的 answer-or-abstain 结果，且 citation 只指向本次检索到的记录。
- 术语实体命中可复现；无法按 topic/domain 消歧时不静默猜测，而是返回 clarification-required 结果。
- 除显式版本对比外，同一答案不得引用同一 `doc_id` 的多个 `doc_version`；版本对比答案必须明确标注每组证据版本。
- retry 永不超过 3 次；无足够证据时明确 abstain。
- 新环境按文档完成 setup 后，可以运行预置数据的 golden path；实际耗时基线在 Phase 0 记录后写入质量门槛。
- 切块评测对至少三组 small/medium/large token 配置使用同一语料、问题、retriever、reranker 和 generator，独立报告 retrieval/selected recall、citation、required-fact coverage、faithfulness、answer relevancy、latency 与成本；不用单一混合总分选型。
- 查询转换不得丢失产品/功能/版本/否定词，step-back 背景证据不得单独满足具体问题；context compression 不得改变 selected chunk ids、version scope、evidence grade 或 citation map。
- Workshop 结束时，参与者能通过 trace 解释 Milvus recall、reranker 和 evidence grader 的职责差异。
- 重复/等价问题命中 cache 时不调用 search tools、reranker 或 answer generator；过期、权限/版本/checksum/revision 不匹配时 100% 回到正常 RAG。
- Selective Memory fixtures 中，用户纠正必须 100% supersede 旧 active fact 且保留 source lineage；expired、tombstoned 或跨 session Memory 的可见率为 0。
- Decay recall 不得让近期无关 episode 压过精确 active task/fact；仅发生“被召回”不得刷新生命周期。
- 最近问题 recall 对中文/英文受支持表达使用同一 deterministic action detector；返回数量、顺序、session/TTL scope 与“不包含当前 recall command”全部由 contract tests 固定。
- 模型、prompt、检索策略或功能显著变化后抽样复核 trace；新失败先进入 error analysis，长期无信息量的非 guardrail metric 可退休，但其仍有价值的 deterministic regression fixture 不随 metric 一起删除。

## 8. Naming conventions (binding)

- Milvus collections：当前基线使用 `kb_chunks`、`conversation_memory`、`grounded_response_cache`、`doc_dedup_signatures`；P1 新增可重建 `kb_documents` StructArray 检索投影；Selective Memory 迁移新增 `memory_events`、`memory_facts` 与 session-private 的 `memory_consolidation_journal`，完成 cutover 前不得静默复用旧 schema。这八个名字是 repository-owned demo collection 的完整集合，与 `demo/scripts/cleanup_milvus.py` 的固定目标一致。
- 查询关联键：`query_id`；会话键：`session_id`；文档/证据键：`doc_id`、`doc_version`、`chunk_id`。
- 时间统一为 UTC epoch milliseconds；citation 面向用户显示为 `[C1]`，内部必须保留 `chunk_id`、`doc_version` 与可选 `page_no`。
- `M0…Mn` 表示用户可见里程碑；`Phase 0…n` 表示工程依赖顺序，两者不得混用。
- StructArray 知识投影 collection 名为 `kb_documents`，parent field 名为 `passages`；持久证据身份仍使用 `chunk_id`，`offset` 只是某次 StructArray 结果的执行定位。

## 9. Requirements traceability

| Requirement | Authoritative spec |
| --- | --- |
| 数据源、离线导入、MinIO | [`11-ingestion.md`](./11-ingestion.md) |
| collection 与字段不变量 | [`10-data-model.md`](./10-data-model.md) |
| StructArray 文档/passage 建模、搜索粒度与适用边界 | [`10-data-model.md`](./10-data-model.md), [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md) |
| 预定义词语实体与文档版本合同 | [`10-data-model.md`](./10-data-model.md), [`12-agent-workflow.md`](./12-agent-workflow.md) |
| Grounded response cache、TTL 与 freshness | [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md) |
| Selective Memory、consolidation、projection 与 decay forgetting | [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md) |
| LangGraph、reranker、retry、citation | [`12-agent-workflow.md`](./12-agent-workflow.md) |
| Rule/LLM query classification 与 fallback | [`12a-query-classification.md`](./12a-query-classification.md) |
| OpenAI answer generation、citation guard 与 fallback | [`13-llm-answer-generation.md`](./13-llm-answer-generation.md) |
| UI tab 布局与 question-only 交互 | [`20-ui-demo.md`](./20-ui-demo.md) |
| Error analysis、metric portfolio、RAG eval、Min-Max Chunking、性能/安全边界 | [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md) |
| 查询转换策略与上下文压缩 | [`12-agent-workflow.md`](./12-agent-workflow.md), [`13-llm-answer-generation.md`](./13-llm-answer-generation.md) |
| 交付与 notebook 顺序 | [`90-roadmap.md`](./90-roadmap.md), [`91-impl-plan.md`](./91-impl-plan.md) |
