# 80 — Glossary

Status: draft · Last updated: 2026-07-27

## Agent Chat

本项目中的查询型企业知识助手。MVP 只读取离线构建的知识库，不包含在线 ingestion 或生产级权限系统。

## Agentic RAG

由可观察、可分支的步骤组成的 RAG：意图判断、权限检查、受限工具选择、查询改写/拆解、多次或多跳检索、精排、证据判断、有限补充检索、回答与自检。它不是“让 LLM 自由调用任意工具”的同义词。

## Tool

Agent 可选择的受限能力。Search tool 封装知识域与 metadata filter policy；UI 不直接构造这些 filters。Tool call 必须记录名称、子查询、安全参数摘要和结果计数。

## Query plan

当前问题的一到三个检索子任务。每项绑定一个注册工具，可声明对前序子任务的依赖。Parallel decomposition 同时覆盖多个方面；multi-hop plan 用第一跳证据细化后续查询。

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

## Hybrid search

经 Phase 0 验证后采用的 dense 与 sparse/BM25 候选组合方式。具体 SDK API、融合策略和分数语义以验证结果为准。

## Min-Max Chunking

通过最小/最大长度、边界与 overlap 规则约束 chunk 的实验策略。它是待评估方案，不是已选择的第三方库。

## MFS

Milvus 社区相关的数据源集成候选。当前只作为架构扩展，未进入 MVP 主链路，也未在本仓库验证。

## Milestone vs Phase

Milestone（M0…Mn）按用户可见能力组织；Phase（Phase 0…n）按工程依赖顺序组织。一个 Phase 可以支撑多个 Milestone，一个 Milestone 也可能跨多个 Phase。
