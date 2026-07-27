# 91 — Implementation Plan: Dependency-Ordered Build

Status: draft v3 · Audience: implementers · Last updated: 2026-07-27

## 0. Readiness assessment

第一版 deterministic demo、依赖清单和测试框架已经存在，但真实 Milvus round trip 与 OpenAI answer generation 尚未完成端到端验证。以下能力仍需 Phase 0 证据：目标 Milvus/SDK 的 hybrid/BM25、ordering、aggregation、nullable vectors、TTL、BinaryVector/MinHash，以及已配置 OpenAI model 的 Responses 调用、超时和输出合同。

## 1. Why dependency order differs from feature order

- 用户最先看到 Chat，但 citation 依赖稳定的 `doc_id/chunk_id`、seed corpus 和检索结果契约，所以数据合同先于 UI。
- retry 是 UI 可见能力，但它依赖 reranker 与 grader 的结构化输出；先写循环会把临时 dict 固化成状态接口。
- tool routing 依赖明确的权限、tool policy 和 query-plan shape；先做 UI filters 会把 Agent 决策错误地下放给用户。
- 术语解析和版本隔离会影响 rewrite、filter、citation 与 eval，所以 entity catalog 和 chunk version contract 必须先于 workflow consumer。
- image/vector、memory 和 MinHash 看起来适合 feature demo，但它们的 Milvus 表示尚未验证，不能阻塞核心 grounded Q&A。

## 2. Estimated total effort

M0–M2 暂估 10–15 focused developer days，假设一名熟悉 Python/RAG 的开发者、模型可用、Milvus/MinIO 环境已准备。新增的 entity catalog、领域消歧、多版本 ingestion/filter 与串版评测约占 2–3 days。Phase 0 完成后更新数字。并行工作仅适用于稳定合同后的 UI 与 notebook 文档；数据模型和 workflow contract 不能并行分叉。

## 3. Phase 0 — Risk retirement

| # | Deliverable | Lands in | Effort |
| --- | --- | --- | --- |
| 0.1 | 固定 Milvus/SDK 版本，验证 hybrid/BM25 与 filter 的最小 round trip | research memo + 10/70 | 0.5–1d |
| 0.2 | 验证 ordering、aggregation、nullable vector 的实际语义和 fallback | research memo + 10/20/70 | 0.5–1d |
| 0.3 | 用小样本验证 OpenAI text embedding 的 1024 维响应/成本，以及候选 image embedding 与 reranker | research memo + 10a/11/12 | 0.5–1d |
| 0.4 | 验证 TTL 和 MinHash representation；若不成立，确定 P2 替代存储 | research memo + 10 | 0.5d |
| 0.5 | 比较两组 Min-Max Chunking 配置，建立初始 eval/latency baseline | research memo + 70 | 0.5–1d |
| 0.6 | 固定 OpenAI SDK 版本并验证 Responses 调用、timeout 和 citation-valid smoke | research memo + 13/70 | 0.5d |

研究 memo 应进入未来的 `docs/research/`，记录版本、可执行命令、观察结果和决策；必要时先使用 `research` skill。

**Exit gate**: 所有核心能力有可执行证据或明确 fallback；模型和维度已配置化；00/10/11/12/20/70 已按结果更新。未通过则不进入生产代码。

## 4. Phase 1 — Foundation and fixtures

Maps to: starts M0 and M2.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 1.1 | 创建 demo 项目结构、依赖与可重复本地配置文档 | 00, 70 | 0.5d |
| 1.2 | 先写 schema/record validation、`demo/config/predefined_entities.yaml` loader 与 version invariant tests | 10, 70 | 1–1.5d |
| 1.3 | 建立含术语别名和多版本文档的 seed corpus、questions 与 golden answers，加入完整性测试 | 10, 70 | 1–1.5d |
| 1.4 | 实现 collection/index setup 和 insert/search round trip test | 10 | 0.5–1d |

**Exit criteria**: formatter/linter/tests pass；entity catalog 可校验；seed records 可插入并按 `chunk_id/doc_version` 回读；每个文档族恰有一个 current edition；fixture 引用全部解析；标准启动/测试命令已写入 demo README。

## 5. Phase 2 — Offline ingestion

Maps to: advances M2.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 2.1 | 先写 local/MinIO adapter contract tests | 11 | 0.5d |
| 2.2 | 先写 OpenAI embedding request/config/output-validation fake-client tests | 10a, 70 | 0.5d |
| 2.3 | 在既有 `dense_vector()` seam 实现 OpenAI provider，保持 schema 与调用点不变 | 10a | 0.5d |
| 2.4 | 实现 parse → stable/version-aware id → chunk → embed → validate pipeline | 11 | 1.5–2d |
| 2.5 | 通过 `document_versions.json` 实现 idempotent full-corpus rebuild、单一 current-edition 校验与 contextual failure report | 10, 11 | 0.5–1d |
| 2.6 | 编写 `01`–`04` notebooks，复用实现模块 | 11, 70 | 1d |

**Exit criteria**: local golden path 与 MinIO exercise 可运行；OpenAI fake-client contract 与默认 no-network tests 通过；写入/查询共享 provider 与维度；重复 ingestion 保持逻辑记录集合不变；current edition 切换不产生双 current 且历史版本保留；PDF page/version citation 通过测试；notebooks 从干净 kernel 顺序执行。

## 6. Phase 3 — Retrieval and Agent workflow

Maps to: closes M0 backend and most of M1.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 3.1 | 先写 hybrid retrieval adapter 与 normalized result tests | 12 | 0.5d |
| 3.2 | 实现主 reranker + deterministic fallback contract | 12, 70 | 0.5–1d |
| 3.3 | 定义 permission/search/summarize tool contracts 与 policy-owned filters | 12, 70 | 0.5d |
| 3.4 | 先写 intent、entity resolution、permission、tool selection、version scope、decomposition、multi-hop、retry/exhausted graph tests | 12 | 1–1.5d |
| 3.5 | 实现 bounded query plan、entity-aware rewrite、多工具执行、版本隔离、结果合并与定向补充检索 | 12 | 1.5–2d |
| 3.6 | 实现 OpenAI answer generator、citation guard、terminal self-check 与 deterministic fallback | 12, 13, 70 | 0.5–1d |
| 3.7 | 加入 tool/plan/coverage/self-check metrics、trace 与 RAG eval runner | 12, 13, 70 | 0.5–1d |

**Exit criteria**: 所有图路径有 terminal result；术语命中可追踪且歧义不静默选择；私有检索前权限 gate 通过；初始计划不超过 3 个子查询且只调用注册工具；current/exact/comparison version fixtures 无非预期串版；multi-tool/multi-hop fixture 通过；retry 不超过 3；citation subset 与 answer self-check invariant 通过；无 API key 的默认测试仍可重复。

## 7. Phase 4 — Streamlit teaching UI

Maps to: closes M0 and M1.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 4.1 | 移除 metadata Search Controls，实现 question-only Chat 与 permission/abstain/error states | 20 | 0.5–1d |
| 4.2 | 实现 tool recall/rerank/coverage Evidence table | 20 | 0.5d |
| 4.3 | 实现 intent、entity/version resolution、permission、tool plan、supplementary retrieval 与 self-check Trace | 20 | 0.5–1d |
| 4.4 | 加入 query snapshot consistency/UI smoke tests | 20, 70 | 0.5d |
| 4.5 | 定义并实现 query-local `trace_event → validated answer_delta → final` streaming contract，LangGraph/local runtime 保持相同行为 | 12, 13, 20 | 1d |
| 4.6 | 用柔和的动态 timeline 替代主界面 raw JSON，session 内保存事件用于完成后回放，并加入顺序/安全/UI 测试 | 20, 70 | 0.5–1d |

**Exit criteria**: UI 不提供 metadata filters；三个 tab 共享 `query_id`；local/LangGraph stream 在节点完成时实时发出连续、安全的 trace events；tool call 和 retry 可动态观察；grounded answer delta 只在 self-check 成功后出现；timeline 与最终 evidence/trace 一致；raw JSON 降级到 advanced expander；fallback 和所有错误态可辨识；M0/M1 roadmap criteria 全部满足。

## 8. Phase 5 — Evaluation, rehearsal, and optional labs

Maps to: closes M2; optionally starts M3.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 5.1 | 跑 chunking/RAG/latency baseline，锁定质量阈值 | 70 | 0.5–1d |
| 5.2 | 从干净 checkout 演练 setup、notebooks、UI workshop flow | 70, 90 | 0.5d |
| 5.3 | 实现 local/Milvus ConversationMemory store、显式 TTL、session isolation、upsert/list/delete | 10, 10b | 1–1.5d |
| 5.4 | 把 recall/persist、memory-only intent 和 bounded follow-up context 接入 local/LangGraph workflow | 10b, 12, 13 | 1–1.5d |
| 5.5 | 实现 Streamlit 多轮 history、Memory tab、degraded state 和 active-session clear | 10b, 20 | 0.5–1d |
| 5.6 | 完成 Memory adapter/workflow/UI parity、安全和生命周期测试；保持 RAG eval | 10b, 70 | 0.5–1d |
| 5.7 | 按独立验收实现 image/dedup/MFS lab（可选） | 10, 11, 90 | separately estimated |

**Exit criteria**: 可重复 eval 报告和硬件/版本信息已提交；clean-run rehearsal 通过；Memory 的 local/Milvus store、local/LangGraph workflow 和 UI contract 一致；follow-up、remember/recall、TTL、session isolation、clear、failure degradation 均有测试；Memory 不污染 citation/evidence；每个其余可选 lab 不改变 M0/M1 的稳定合同。

## 9. What makes this order correct

1. **先合同后消费者**：identity、citation、normalized result 先稳定，再让 graph、UI 和 notebooks 消费。
2. **先证伪高风险假设**：Milvus/model 能力失败时只改 Phase 0/spec，而不是返工所有调用点。
3. **每阶段保持可测试**：先写行为测试，再实现最小路径；组件错误显式进入 terminal state。

## 10. Cross-references

- Stakeholder milestones: [`90-roadmap.md`](./90-roadmap.md)
- Quality gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- Decisions: [`99-key-decisions.md`](./99-key-decisions.md)
