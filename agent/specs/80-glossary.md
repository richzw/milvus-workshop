# 80 — Glossary

Status: draft · Last updated: 2026-08-27

## Agent Chat

本项目中的查询型企业知识助手。MVP 只读取离线构建的知识库，不包含在线 ingestion 或生产级权限系统。

## Agentic RAG

由可观察、可分支的步骤组成的 RAG：意图判断、权限检查、受限工具选择、查询改写/step-back/拆解、多次或多跳检索、精排、证据判断、有限补充检索、可选上下文压缩、回答与自检。它不是“让 LLM 自由调用任意工具”的同义词。

## Tool

Agent 可选择的受限能力。Search tool 封装知识域与 metadata filter policy；UI 不直接构造这些 filters。Tool call 必须记录名称、子查询、安全参数摘要和结果计数。

## Query plan

当前问题的一到三个检索子任务。每项绑定一个注册工具，标记 primary/background/aspect/hop 角色，可声明对前序子任务的依赖。Parallel decomposition 同时覆盖多个方面；multi-hop plan 用第一跳证据细化后续查询。

## Query transformation

在检索前将原问题映射为受限检索计划。本项目每次只选择 `identity`、`rewrite`、`step_back` 或 `decompose` 一个主策略，且派生查询必须继承原意图、产品/版本/否定约束、permission、tool allow-list 和 version scope。

## Step-back query

为具体问题派生一个更宽泛的原理或架构背景问题，并与保留原始主体的 primary query 共同检索。背景证据只能补充解释，不能单独证明具体功能、版本或操作结论。

## Query classifier

只在固定枚举内输出 `intent`、`query_type` 与 `retrieval_goal` 的可替换组件。`LLMQueryClassifier` 负责理解自然语言变体；`RuleBasedQueryClassifier` 负责明确 action、离线复现和 fallback。它不授予权限、不选择工具、不构造 metadata filters。

## Grounded response cache

同一 session 内复用已经过 citation/self-check 的 KB 回答。它保存完整 citations 与 evidence version/checksum snapshot，并在当前权限和 freshness 验证后才能命中。它不同于 Conversation Memory：后者补全多轮语境，前者仍是可引用的 KB-grounded response。

## Episode

一次具体经历的不可变 Memory event，例如用户纠正、任务状态变化、检索失败或成功策略。Episode 记录“发生了什么”，不等于系统当前相信的事实；其召回价值可随时间 decay。

## Selection Gate

Episode capture 后的低成本筛选边界。它根据显式记忆、纠正、任务变化、失败严重度、新颖性、复发和未来效用，输出 retention class、salience 与 decay profile。LLM 只可处理规则得分的模糊区间。

## Consolidated Memory

由一组 source episodes 缓慢归纳出的 versioned durable fact。它必须保留 lineage，并以 active、superseded、disputed 或 tombstoned 表达当前有效性；它不是 Response Cache。

## Working-state projection

从 event lineage 和 fact revisions 重建的当前任务、偏好、决策、纠正、经验与冲突视图。Projection 可重建，不作为不可追溯的 mutable truth。

## MemoryPack

`recall_memory` 返回的 bounded typed context，包含 working state、durable facts、recent episodes、conflicts 与 provenance。Grounded-response cache candidate 是独立 workflow state，不属于 MemoryPack。

## Soft vs logical vs physical forgetting

Soft forgetting 使用 Milvus decay 降低旧 Memory 排名；logical forgetting 通过 expiry、supersession、dispute 或 tombstone 阻止使用；physical forgetting 由清理流程移除 payload/vector。Decay 不能代替后两者。

## Predefined entity

受版本控制的领域词语定义，包含稳定 `entity_id`、canonical `entity`、别名、`comment` 和适用 domains。它帮助 intent/query rewrite 理解“GO按钮”等产品词或游戏行业黑话，但不是知识证据，不能独立支撑答案或 citation。

## Document version scope

每次检索对同一逻辑文档可见的 edition 范围。默认 `current` 只查 `is_current=true`；`exact` 查询用户指定的 `doc_version`；只有明确的版本对比问题使用 `comparison`。普通答案不得混合多个 edition。

## Permission gate

任何私有知识检索前执行的允许/拒绝决策。MVP 只实现合成数据上的教学 gate，不等价于身份认证、ACL 或生产授权。

## Supplementary retrieval

Evidence grader 发现明确缺口后发起的定向检索。它保留已有证据，只补充 missing aspects；不同于重复执行同一个宽泛查询。

## Recall vs rerank

Recall 是 Milvus 从知识库中尽量找全候选；rerank 是针对当前 query 重新排列这批候选。前者偏召回率，后者偏上下文精度。

## Evidence vs selected context

Evidence 是本次检索和精排产生的候选记录；selected context 是其中真正发送给 answer generator 的子集。只有 selected context 可以生成 citation。

## Context compression

证据充分性确定后，将 selected original chunks 投影为更小的 generation context。Selective 保留可精确对回原文的句段；summary/extraction 是派生内容，每项都必须绑定原文 support spans。压缩不改变 evidence grade、source chunk 或 citation identity。

## Evidence grader

判断当前证据是否覆盖问题、是否足以回答以及是否需要重试的节点。它不替代 reranker，也不直接生成答案。

## Answer generator

只把 selected context 合成为用户可读答案的可替换模块。OpenAI 是主实现；deterministic generator 是明确标注的 fallback。它不负责选择证据或创建 citation identity。

## Citation

答案中面向用户的 `[C1]` 标记及其结构化来源。内部至少绑定 `chunk_id` 与 `doc_version`，PDF 可额外绑定 `page_no`，Markdown 可绑定 `section`。

## Answer self-check

生成后、终态前的结构化验证：inline markers 与结构化 citations 必须解析到本次 selected context，grounded answer 至少有一项证据，abstain 不得包含无依据的具体结论。它不暴露 chain-of-thought。

## Trace

一次查询的教学级执行记录，包括节点输入摘要、参数、计数、retry 和 latency。Streamlit session state 中的 Trace 是临时数据，不是生产审计日志。

## Error analysis

人工阅读一批有代表性的 output/trace，先用 `overall_pass` 与自由文本 `review_note` 描述实际问题，再把 notes 聚类为命名 failure categories 的过程。它用于发现产品真实的 generalization failures；不是从通用 metric catalog 复制一组分数。

## Goal metric

衡量系统在当前产品目标上是否改善的 active metric，例如 expected-source retrieval recall。它必须绑定 owner 和实验接受、回滚或调查动作；产品 north star 改变时可以被修改或退休。

## Guardrail metric

监控不能破坏的 hard constraint 或已知 incident class 的 active metric，例如 citation 必须解析、permission 前不得检索。长期满分不构成删除理由；只有约束被正式移除时才可退休。

## Operational metric

从 trace 或 runtime 自动采集的 latency、tokens/cost、provider calls 或 throughput。它解释资源与容量，不代表答案质量；没有固定 runtime/provider/dataset/concurrency profile 时不得与 baseline 比较。

## Metric registry

active eval metrics 的版本化决策清单。每项记录 role、问题、证据来源、grader、dataset segment、owner、threshold/budget、触发动作、运行成本、cadence 与 lifecycle。runner 能输出的 diagnostic 字段不等于 registry 中需要长期维护的 metric。

## Hybrid search

经 Phase 0 验证后采用的 dense 与 sparse/BM25 候选组合方式。具体 SDK API、融合策略和分数语义以验证结果为准。

## Retrieval tier

检索复杂度阶梯上的一级：T0 lexical only、T1 lexical + bounded query transformation、T2 pre-embedded hybrid（当前默认）、T3 on-the-fly embedding、T4 hot/cold embedding tiers、T5 full pre-embedding。Tier 由 freshness、corpus churn、query pattern、scale/latency 和 team capability 五项输入选择，升级必须有记录在案的 failure mode 和对照报告。它与 `flat_hybrid`/`struct_element`/`struct_two_stage`/`struct_fused` 不是同一维度：后者是 T2 内部的 retrieval profile。

## Lexical-only baseline

只使用 `kb_chunks` BM25 Function 的对照 arm（T0）。它不参与选型竞争，作用是给出 dense lane 在当前语料上的增量分母；任何 tier 对照报告缺少该 arm 即为 `evaluation_incomplete`。

## Corpus churn

语料中每天/每月被新增或修改的文档比例。高 churn（日 > 10%）使预嵌入的 re-ingest 成本压过其收益；低 churn（月 < 5%）才让 T2/T5 的预嵌入摊销成立。它是 tier 选择输入，不是质量指标。

## On-the-fly embedding

查询时才对候选 shortlist 生成向量、不持久化 chunk 向量的 T3 做法。freshness 永远最新，模型切换只是一次调用点修改，代价是每次查询的 embedding 延迟。本仓库未实现，仅作为高 churn 或模型下线场景的记录出口。

## Hot/cold embedding tiers

按访问频次把语料分为预嵌入的 hot 子集与按需嵌入的 cold 尾部的 T4 做法（Pareto 假设：约 20% 文档承载约 80% 流量）。模型更新只需重嵌 hot 子集。本仓库未实现，仅作为记录出口。

## Embedding fingerprint gate

启动时校验 chunk metadata 中记录的 embedding provider/model/dimension 指纹与当前配置是否一致。不一致必须直接失败，而不是混用两个向量空间；它是 T2 下模型迁移成本可见的机制。

## StructArray

Milvus 中的 `ARRAY<STRUCT>`：一个 parent entity 内保存有序、可变长且共享预定义 schema 的 elements。本项目只用它表示 `kb_documents.passages`，使 passage scalar/vector subfields 在同一 offset 上可相关过滤和搜索；它不是任意 nested JSON。

## Parent entity vs Struct element

Parent entity 是业务授权、版本和展示的整体文档；Struct element 是其中可搜索的 passage。Element-level hit 的身份是 parent primary key 加 offset，但本项目解析后仍以稳定 `chunk_id` 作为 evidence/citation identity。

## EmbeddingList search

查询和 parent 都用一组 vectors 表示，通过 `MAX_SIM*` 匹配并返回 entity-level 结果。在本 Agentic RAG 中它只用作多方面问题的文档 shortlist，必须再做 element-level search 才能得到可引用 passage。

## Element-level search

一个普通 query vector 让 StructArray 中每个 element vector 独立参与 ANN，返回 parent 和 element offset。`element_filter` 会约束同一 element 上的 scalar 条件；同一 parent 可以因多个 passage 命中而出现多次。

## Collapse

当 hybrid search 混合 element 与 entity 粒度时，将已返回的 element hits 聚合成 parent score 的过程。Collapse 不会重新扫描 parent 的所有 elements，因此 ANN sub-search `limit` 会影响结果；collapsed parent 在本项目中不是可引用证据。

## Min-Max Chunking

通过最小/最大 token 长度、硬语义边界与 overlap 规则约束 chunk 的实验策略。它是待经隔离索引的端到端检索/生成评测选型的方案，不是已选择的第三方库。

## MFS

Milvus 社区相关的数据源集成候选。当前只作为架构扩展，未进入 MVP 主链路，也未在本仓库验证。

## Milestone vs Phase

Milestone（M0…Mn）按用户可见能力组织；Phase（Phase 0…n）按工程依赖顺序组织。一个 Phase 可以支撑多个 Milestone，一个 Milestone 也可能跨多个 Phase。
