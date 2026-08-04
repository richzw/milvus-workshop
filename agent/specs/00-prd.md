# PRD — Milvus 3.0 Enterprise Agent Chat Workshop

Status: draft v5 · Owner: workshop author · Last updated: 2026-07-30

## 1. Problem

企业内部资料分散在本地文件、S3/对象存储等异构数据源中，文本、PDF 和图片混合存在。普通 RAG 示例通常只展示“切分—向量检索—回答”，难以解释真实系统里的混合检索、证据筛选、失败重试、引用追踪和数据治理。

Workshop 需要同时满足三类学习者：想直接体验效果的用户、希望理解实现的开发者，以及希望通过 vibe coding 完成实践的参与者。现有草稿描述了这些功能，但产品范围、数据模型与实现顺序混在一起，缺少可验证的交付合同。

## 2. Vision

参与者可以运行一个本地、可复现的企业 Agent Chat：离线导入带文档版本 metadata 的本地与 MinIO 文档后，在 Streamlit 只输入自然语言问题；LangGraph 通过受限结构化 LLM classification 理解意图并在失败时回退到 rule-based classifier，随后使用预定义词语实体消歧行业术语，执行权限检查，自主选择知识工具，将复杂问题拆为至多三个子问题，经版本隔离的一次或多次 Milvus 混合检索、reranker 与 evidence grader 汇总证据，必要时最多补充检索 3 轮，最后生成并自检带 chunk/page/version citation 的答案。界面同时解释“问题如何分类、术语如何理解、为什么选择这些工具、为什么使用这些文档版本、为什么证据足够或不足”。

```text
问题：我们 S3 文档同步流程是怎么设计的？
结果：流式答案 + [C1]/[C2] 引用
解释：原始召回 → rerank 后排序 → evidence grade → retry rounds
```

## 3. Goals

| # | Goal | Measure |
| --- | --- | --- |
| G1 | 提供最小可运行的端到端 Agentic RAG demo | 一条 golden question 能从预置数据检索、精排、回答并返回至少 1 个可解析 citation |
| G2 | 让 Agent 决策可教学、可观察 | UI 能展示 intent、permission、tool selection、query plan、recall、rerank、grade、retry 与 answer self-check，且 trace 与本次回答使用同一 `query_id` |
| G3 | 展示 Milvus 在企业 RAG 中的核心价值 | 经 Phase 0 验证后，demo 展示 dense+sparse/hybrid、metadata filter，以及至少一种排序或聚合能力 |
| G4 | 支持不同学习路径 | 提供 UI demo、可运行代码和按依赖排列的 notebook/实践步骤 |
| G5 | 保持现场演示可恢复 | reranker 或 OpenAI answer generation 不可用时，显式 deterministic fallback 仍能完成 golden path；失败路径不伪装成模型成功 |
| G6 | 避免行业术语误解和文档版本串答 | 术语 fixture 使用命中的预定义实体完成正确改写；非版本对比问题的 selected context 不混合同一文档的多个版本 |
| G7 | 让自然语言会话历史问题得到确定性回答 | “查找下我最近的三个问题是什么”等 same-session 请求只返回仍有效的最近 3 个 user turns，从近到远排列，不调用 KB search tools |

## 4. Non-goals

- MVP 不提供在线 ingestion；导入通过 notebook 或 CLI 离线完成。
- MVP 不接入真实企业 S3、云 Doc 或 MFS；S3 使用 MinIO/mock S3，MFS 仅作为扩展设计。
- MVP 不提供生产级认证、ACL、租户隔离、审计或 secret 管理，因此不得部署为真实内部知识系统。
- 图片检索、conversation memory、MinHash 近重复检测不是首个垂直切片的硬依赖。
- 本 spec 不承诺未经 Phase 0 验证的 Milvus API、索引类型或精确参数。

## 5. Users

- **体验者（primary）**：希望通过 UI 理解企业 Agent Chat 能做什么，以及答案证据从哪里来。
- **开发者（primary）**：希望通过 notebook 与代码掌握 ingestion、schema、hybrid search 和 LangGraph 工作流。
- **Vibe Coding 学习者（secondary）**：需要一条按阶段、有验收点的实践路径。
- **反向 persona**：需要生产级安全、实时同步或大规模 SLA 的平台团队；本 Workshop 只能作为概念验证。

## 6. MVP scope

### P0 — 可运行主链路

- Streamlit 只提供自然语言输入，以及 Chat、Evidence、Agent Trace 三个 tab；不暴露 source/doc/department search controls。
- LangGraph：classify intent → resolve terminology → decide retrieval → check permission → select tools → rewrite/decompose → version-scoped one-or-more retrieve → rerank → grade → bounded supplementary retrieval → answer/abstain → citation/self-check。
- Query classification 使用固定 intent/topic/retrieval-goal 枚举；OpenAI Structured Outputs 为可配置主实现，明确 memory/operation action 与所有 provider/output failures 使用可观察的 rule-based safe path。
- `source_type`、`doc_type`、`department` 是 search tool 的内部过滤参数，由 Agent 选择并写入 trace，不由用户手动指定。
- 预定义词语实体 catalog 为 query understanding 提供领域词、别名和解释，例如 `GO按钮` 表示“跳转/领取按钮”；只把与当前问题匹配的实体注入术语解析/改写 prompt，并在 trace 中记录解析结果。
- 每个 `kb_chunks` 记录必须携带 `doc_version` 与 `is_current`。默认检索仅使用 current version；只有用户明确要求指定版本或版本对比时，工具才改变 version scope。
- 答案经 citation guard 校验后分块输出；完成后展示 citation、source cards、evidence 与 trace。
- OpenAI 基于 selected chunks 合成答案；配置缺失或 provider 失败时使用可观察的 deterministic fallback。
- 同一 session 的 exact 或高置信语义等价 KB 问题可复用三天内、权限与 KB evidence 仍有效的 grounded response；cache hit 保留原 citations。
- 主知识库来自离线预置的本地与 MinIO 文档。

### P1 — Milvus 教学能力

- metadata filter、排序/聚合展示，以及 nullable `image_vector` 样例；metadata filters 只作为工具执行细节展示。
- 每项能力以 Phase 0 的实际 SDK/服务验证结果为准。

### P2 — 扩展实验

- 图片检索、selective dual-speed conversation memory、文档去重、MFS、更多文档类型、在线 ingestion、生产级 auth/ACL。
- Memory 以 append-only episode lineage、Selection Gate、versioned durable facts 和 working-state projection 演进；Milvus Decay Ranker 只承担软遗忘，逻辑失效与物理删除保持独立。

## 7. Success metrics

- 所有 golden questions 都产生结构合法的 answer-or-abstain 结果，且 citation 只指向本次检索到的记录。
- 术语实体命中可复现；无法按 topic/domain 消歧时不静默猜测，而是返回 clarification-required 结果。
- 除显式版本对比外，同一答案不得引用同一 `doc_id` 的多个 `doc_version`；版本对比答案必须明确标注每组证据版本。
- retry 永不超过 3 次；无足够证据时明确 abstain。
- 新环境按文档完成 setup 后，可以运行预置数据的 golden path；实际耗时基线在 Phase 0 记录后写入质量门槛。
- Workshop 结束时，参与者能通过 trace 解释 Milvus recall、reranker 和 evidence grader 的职责差异。
- 重复/等价问题命中 cache 时不调用 search tools、reranker 或 answer generator；过期、权限/版本/checksum/revision 不匹配时 100% 回到正常 RAG。
- Selective Memory fixtures 中，用户纠正必须 100% supersede 旧 active fact 且保留 source lineage；expired、tombstoned 或跨 session Memory 的可见率为 0。
- Decay recall 不得让近期无关 episode 压过精确 active task/fact；仅发生“被召回”不得刷新生命周期。
- 最近问题 recall 对中文/英文受支持表达使用同一 deterministic action detector；返回数量、顺序、session/TTL scope 与“不包含当前 recall command”全部由 contract tests 固定。

## 8. Naming conventions (binding)

- Milvus collections：当前基线使用 `kb_chunks`、`conversation_memory`、`grounded_response_cache`、`doc_dedup_signatures`；Selective Memory 迁移新增 `memory_events` 与 `memory_facts`，完成 cutover 前不得静默复用旧 schema。
- 查询关联键：`query_id`；会话键：`session_id`；文档/证据键：`doc_id`、`doc_version`、`chunk_id`。
- 时间统一为 UTC epoch milliseconds；citation 面向用户显示为 `[C1]`，内部必须保留 `chunk_id`、`doc_version` 与可选 `page_no`。
- `M0…Mn` 表示用户可见里程碑；`Phase 0…n` 表示工程依赖顺序，两者不得混用。

## 9. Requirements traceability

| Requirement | Authoritative spec |
| --- | --- |
| 数据源、离线导入、MinIO | [`11-ingestion.md`](./11-ingestion.md) |
| collection 与字段不变量 | [`10-data-model.md`](./10-data-model.md) |
| 预定义词语实体与文档版本合同 | [`10-data-model.md`](./10-data-model.md), [`12-agent-workflow.md`](./12-agent-workflow.md) |
| Grounded response cache、TTL 与 freshness | [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md) |
| Selective Memory、consolidation、projection 与 decay forgetting | [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md) |
| LangGraph、reranker、retry、citation | [`12-agent-workflow.md`](./12-agent-workflow.md) |
| Rule/LLM query classification 与 fallback | [`12a-query-classification.md`](./12a-query-classification.md) |
| OpenAI answer generation、citation guard 与 fallback | [`13-llm-answer-generation.md`](./13-llm-answer-generation.md) |
| 三个 UI tab 与 question-only 交互 | [`20-ui-demo.md`](./20-ui-demo.md) |
| RAG eval、Min-Max Chunking、性能/安全边界 | [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md) |
| 交付与 notebook 顺序 | [`90-roadmap.md`](./90-roadmap.md), [`91-impl-plan.md`](./91-impl-plan.md) |
