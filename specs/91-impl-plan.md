# 91 — Implementation Plan: Dependency-Ordered Build

Status: draft v8 · Audience: implementers · Last updated: 2026-08-24

## 0. Readiness assessment

第一版 deterministic demo、依赖清单和测试框架已经存在，但真实 Milvus round trip 与 OpenAI provider 尚未完成端到端验证。以下能力仍需 Phase 0 证据：目标 Milvus/SDK 的 hybrid/BM25、ordering、aggregation、nullable vectors、TTL、Decay Ranker、BinaryVector/MinHash，以及已配置 OpenAI model 的 Responses 调用、超时和输出合同。离线代码现已包含 `chunking-experiment-v2`、bounded identity/rewrite/step-back/decompose query transformation、生成前 provenance-safe context compression，以及 `eval-metric-registry-v1` 驱动的 `rag-eval-v3` decision surface。Phase 5 的 code contract 与 compatible baseline 已落地；30–50 条真实 trace 的人工 bootstrap error-analysis artifact 仍待 workshop author 提交。StructArray 的 server/SDK probe 已在 Milvus 3.0.0 + PyMilvus 3.0.1 上验证 schema、same-element filter、element offset、EmbeddingList MaxSim、element hybrid identity 与 mixed-granularity collapse；可复制证据和安全结论见 [`docs/research/milvus-3-structarray-probe.md`](../docs/research/milvus-3-structarray-probe.md)。容量、真实 corpus 质量与资源预算仍由 Phase 9 eval 决定，默认模式保持 `disabled`。

## 1. Why dependency order differs from feature order

- 用户最先看到 Chat，但 citation 依赖稳定的 `doc_id/chunk_id`、seed corpus 和检索结果契约，所以数据合同先于 UI。
- retry 是 UI 可见能力，但它依赖 reranker 与 grader 的结构化输出；先写循环会把临时 dict 固化成状态接口。
- tool routing 依赖明确的权限、tool policy 和 query-plan shape；先做 UI filters 会把 Agent 决策错误地下放给用户。
- 术语解析和版本隔离会影响 rewrite、filter、citation 与 eval，所以 entity catalog 和 chunk version contract 必须先于 workflow consumer。
- image/vector、memory 和 MinHash 看起来适合 feature demo，但它们的 Milvus 表示尚未验证，不能阻塞核心 grounded Q&A。
- StructArray 会改变 schema、ingestion、search result identity 和 eval；必须先锁定稳定 `chunk_id` citation 与 flat baseline，再建 parent projection、双索引和 Agent profile，否则 entity hit 容易被误当成证据。

## 2. Estimated total effort

M0–M2 原估 10–15 focused developer days。在当前 runnable demo 之上，chunking v2、query-transformation policy、context compression 与可维护 metric portfolio 预留额外 3–5 focused days；StructArray 投影、双粒度 adapter、Agent 接入和隔离评测再预留 3–5 focused days。假设一名熟悉 Python/RAG 的开发者、模型可用、Milvus/MinIO 环境已准备。Phase 0 完成后用真实 provider latency/cost/throughput 更新数字。并行工作仅适用于稳定合同后的 UI 与 notebook 文档；数据模型、metric registry 和 workflow contract 不能并行分叉。

## 3. Phase 0 — Risk retirement

| # | Deliverable | Lands in | Effort |
| --- | --- | --- | --- |
| 0.1 | 固定 Milvus/SDK 版本，验证 hybrid/BM25 与 filter 的最小 round trip | research memo + 10/70 | 0.5–1d |
| 0.2 | 验证 ordering、aggregation、nullable vector 的实际语义和 fallback | research memo + 10/20/70 | 0.5–1d |
| 0.3 | 用小样本验证 OpenAI text embedding 的 1024 维响应/成本，以及候选 image embedding 与 reranker | research memo + 10a/11/12 | 0.5–1d |
| 0.4 | 验证 TTL 和 MinHash representation；若不成立，确定 P2 替代存储 | research memo + 10 | 0.5d |
| 0.5 | 用隔离 chunk set/index 比较至少三组 small/medium/large Min-Max 配置，跑通 retrieval→rerank→generation→citation，完成 faithfulness/relevancy grader 校准或明确标记 incomplete | research memo + 11/70 | 1–1.5d |
| 0.6 | 固定 OpenAI SDK 版本并验证 Responses 调用、timeout 和 citation-valid smoke | research memo + 13/70 | 0.5d |
| 0.7 | 验证 Responses `text.format` strict JSON schema classification 与非法输出 fallback | 12a/70 | 0.5d |
| 0.8 | 在目标 Milvus/SDK 验证 decay `gauss/exp/linear`、毫秒单位、offset/scale/decay 分数点、单 numeric field/grouping 限制、hybrid composition 和 application fallback parity | research memo + 10d/70 | 0.5–1d |
| 0.9 | 验证 OpenAI strict query-transformation 与 context-compression support-span schema、单请求上限、provider latency/token/cost 和 deterministic/original-context fallback | research memo + 12/13/70 | 0.5–1d |
| 0.10 | 在目标 Milvus/PyMilvus 用 disposable collection 验证 StructArray schema/limits、same-element MATCH/element_filter、element offset、EmbeddingList MAX_SIM_COSINE、TokenANN 参数和混合 collapse identity；对 exact MaxSim 记录 quality/cost baseline | research memo + 10/12/14/70 | 1–1.5d |

研究 memo 进入 `docs/research/`（目前只有 StructArray probe 一篇），记录版本、可执行命令、观察结果和决策；必要时先使用 `research` skill。

**Exit gate**: 所有核心能力有可执行证据或明确 fallback；模型和维度已配置化；00/10/11/12/20/70 已按结果更新。未通过则不进入生产代码。

## 4. Phase 1 — Foundation and fixtures

Maps to: starts M0 and M2.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 1.1 | 创建 demo 项目结构、依赖与可重复本地配置文档 | 00, 70 | 0.5d |
| 1.2 | 先写 schema/record validation、`demo/config/predefined_entities.yaml` loader 与 version invariant tests | 10, 70 | 1–1.5d |
| 1.3 | 建立含术语别名和多版本文档的 seed corpus、questions 与 golden answers，加入完整性测试 | 10, 70 | 1–1.5d |
| 1.4 | 实现 collection/index setup 和 insert/search round trip test | 10 | 0.5–1d |
| 1.5 | 实现 grounded-response cache schema、local/Milvus store、三天 TTL、semantic candidate 与 fail-closed record validation | 10c, 70 | 1–1.5d |

**Exit criteria**: formatter/linter/tests pass；entity catalog 可校验；seed records 可插入并按 `chunk_id/doc_version` 回读；每个文档族恰有一个 current edition；fixture 引用全部解析；标准启动/测试命令已写入 demo README。

## 5. Phase 2 — Offline ingestion

Maps to: advances M2.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 2.1 | 先写 local/MinIO adapter contract tests；实现显式 mock/minio CLI mode、lazy SDK client、bounded snapshot、safe key validation 与稳定 bucket/object-key identity | 11 | 0.5d |
| 2.2 | 先写 OpenAI embedding request/config/output-validation fake-client tests | 10a, 70 | 0.5d |
| 2.3 | 在既有 `dense_vector()` seam 实现 OpenAI provider，保持 schema 与调用点不变 | 10a | 0.5d |
| 2.4 | 实现 parse → stable/version-aware id → chunk → embed → validate pipeline | 11 | 1.5–2d |
| 2.5 | 通过 `document_versions.json` 实现 idempotent full-corpus rebuild、单一 current-edition 校验与 contextual failure report | 10, 11 | 0.5–1d |
| 2.6 | 编写 `01`–`04` notebooks，复用实现模块 | 11, 70 | 1d |

**Exit criteria**: local golden path 与显式 MinIO exercise 可运行；MinIO
fake-client contract 覆盖 missing/empty/traversal/size/count/cleanup/error
路径并把稳定 `s3://bucket/key` identity 送入共享 ingestion pipeline；OpenAI
fake-client contract 与默认 no-network tests 通过；写入/查询共享 provider
与维度；重复 ingestion 保持逻辑记录集合不变；current edition 切换不产生双
current 且历史版本保留；PDF page/version citation 通过测试；notebooks 从
干净 kernel 顺序执行。

## 6. Phase 3 — Retrieval and Agent workflow

Maps to: closes M0 backend and most of M1.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 3.1 | 先写 hybrid retrieval adapter 与 normalized result tests | 12 | 0.5d |
| 3.2 | 先固定 complete-permutation/score/trace 合同和 fake-client tests；实现 OpenAI Responses strict-output reranker、整批 deterministic fallback、`rule_based/auto/openai` builder，并接入 local/LangGraph configured workflow | 12, 70 | 0.5–1d |
| 3.3 | 定义 permission/search/summarize tool contracts 与 policy-owned filters | 12, 70 | 0.5d |
| 3.4 | 先写 RuleBased classifier 兼容测试、LLM strict-output adapter、safe-action fast path、fallback/config/trace 测试，再接入 local 与 LangGraph | 12a, 12, 70 | 1–1.5d |
| 3.5 | 实现 bounded identity/rewrite/step-back/decompose query-transformation policy、strict optional provider、protected-term/scope validation、多工具执行、版本隔离、结果合并与定向补充检索 | 12, 70 | 1.5–2d |
| 3.6 | 实现 generation 前 disabled/auto/selective/summary/extraction context compressor、exact support-span validation、one-batch call bound 和 atomic original-context fallback | 12, 13, 70 | 1–1.5d |
| 3.7 | 实现 OpenAI answer generator、citation guard、terminal self-check 与 deterministic fallback | 12, 13, 70 | 0.5–1d |
| 3.8 | 加入 transform/tool/plan/coverage/compression/self-check metrics、trace 与 RAG eval runner | 12, 13, 70 | 0.5–1d |
| 3.9 | 在 permission-gated `try_grounded_cache` 中召回并校验 query/evidence/revision，命中短路并在 final 写回 | 10c, 12, 70 | 1–1.5d |

**Exit criteria**: 所有图路径有 terminal result；query classification 使用合法固定枚举，明确 memory/operation action 走 rule fast path，LLM/provider/output 失败可观察地降级且不泄漏 raw error；reranker 一次处理完整 bounded recall pool，只接受 `chunk_id` 完整排列与合法 score，配置/provider/output 失败整批降级并记录安全 reason code；术语命中可追踪且歧义不静默选择；私有检索前权限 gate 通过；初始计划不超过 3 个子查询且只调用注册工具；identity/rewrite/step-back/decompose 的词面保真、scope 继承和 background-only 拒答通过；current/exact/comparison version fixtures 无非预期串版；multi-tool/multi-hop fixture 通过；retry 不超过 3；context compression 只在 evidence-sufficient 后运行，provenance 无效时整体回退原 context 且 citation map 不变；citation subset 与 answer self-check invariant 通过；无 API key 的默认测试仍可重复。

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

**Exit criteria**: UI 不提供 metadata filters；本阶段的 Chat/Evidence/Agent Trace 共享同一 `query_id`（Phase 5.7 加入的 Memory tab 同样绑定该 `query_id`）；local/LangGraph stream 在节点完成时实时发出连续、安全的 trace events；tool call 和 retry 可动态观察；grounded answer delta 只在 self-check 成功后出现；timeline 与最终 evidence/trace 一致；raw JSON 降级到 advanced expander；fallback 和所有错误态可辨识；M0/M1 roadmap criteria 全部满足。

## 8. Phase 5 — Evaluation, rehearsal, and optional labs

Maps to: closes M2; optionally starts M3.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 5.1 | 从主要 route/terminal status 分层抽样 30–50 条 trace，只标 `overall_pass`/`review_note`，提交 `eval-error-analysis-v1` 与 failure-cluster dispositions | 70 | 0.5–1d |
| 5.2 | 实现 strict `eval-metric-registry-v1` loader/schema/checksum；按 goal/guardrail/operational 建立最小 active set，并校验 owner/action/cost/cadence/lifecycle | 70 | 0.5–1d |
| 5.3 | 将 `rag-eval-v2` 升级为 v3 decision surface，保留 trajectory/tool/outcome 首错归因，加入 registry-compatible baseline delta、latency/cost/throughput profile 与 live-provider `pass@k/pass^k` | 70 | 1–1.5d |
| 5.4 | 从干净 checkout 演练 setup、notebooks、UI workshop flow | 70, 90 | 0.5d |
| 5.5 | 实现 local/Milvus ConversationMemory store、显式 TTL、session isolation、upsert/list/delete | 10, 10b | 1–1.5d |
| 5.6 | 把 recall/persist、memory-only intent 和 bounded follow-up context 接入 local/LangGraph workflow | 10b, 12, 13 | 1–1.5d |
| 5.7 | 实现 Streamlit 多轮 history、Memory tab、degraded state 和 active-session clear | 10b, 20 | 0.5–1d |
| 5.8 | 完成 Memory adapter/workflow/UI parity、安全和生命周期测试；保持 RAG eval | 10b, 70 | 0.5–1d |
| 5.9 | 先固定 image-provider/file/fingerprint/failure contract；实现显式 deterministic/dinov3 provider、ViT-B/16 768-dim pooled L2 vector、lazy gated runtime、真实 asset ingestion 与 fake-runtime tests | 10, 11, 70, 90 | separately estimated |
| 5.10 | 实现 local/Milvus image-to-image COSINE search、caption-based text-to-image search、严格 fingerprint/filter contract，以及独立 image eval fixture/runner（Recall@K + MRR） | 10, 12, 70, 90 | separately estimated |
| 5.11 | 将现有 retrieval-only `chunking-experiment-v1` 升级为至少三配置的 `chunking-experiment-v2`，加入隔离索引、端到端 generation/citation、faithfulness/relevancy、token/cost、Pareto 选型与 reviewed recommendation artifact | 11, 70, 90 | 1–2d |
| 5.12 | 统一 workflow/classifier recall detector；实现 local/Milvus recent-user-turn chronological listing、memory-only rendering 与 honest trace，并覆盖 count/order/session/TTL/current-turn/tool-bypass parity tests | 10b, 12, 12a, 70 | 0.5–1d |

**Exit criteria**: 30–50 条 bootstrap trace 的 error-analysis/cluster artifacts 已提交；active registry 同时有 goal、guardrail、operational，且每项绑定 source/owner/grader/action/cost/cadence/retirement；`rag-eval-v3` 嵌入 registry checksum，只让 active metrics 产生 decision status，保留 trajectory/tool/outcome 归因（70 § 4.4），并在固定 profile 下报告 latency/cost/throughput；启用 live provider 的评测同时报告 pass@k/pass^k 且 gate 以 pass^k 判定（70 § 4.5）；任何 LLM-as-judge 在用于 gate 前完成 70 § 4.0b 的校准合同；`chunking-experiment-v2` 为每个配置运行隔离的端到端评测，grader 未校准时不推荐新 default，已选配置只能通过 reviewed artifact 发布；clean-run rehearsal 通过；Memory 的 local/Milvus store、local/LangGraph workflow 和 UI contract 一致；follow-up、remember/recall、recent-question chronological recall、TTL、session isolation、clear、failure degradation 均有测试；最近问题请求不调用 KB/ANN/decay 且 trace 区分 skipped/empty；Memory 不污染 citation/evidence；每个其余可选 lab 不改变 M0/M1 的稳定合同。

## 9. Phase 6 — Selective dual-speed Memory

Maps to: closes M4.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 6.1 | 先写 `MemoryEvent`/`MemoryFact`、registered reason/profile、状态迁移和 source-lineage validation tests | 10, 10d, 70 | 0.5–1d |
| 6.2 | 实现 append-only local/Milvus event store、same-session filters、bounded dual-write 与清除语义 | 10d, 70 | 1–1.5d |
| 6.3 | 实现 Rule-based Selection Gate；只为 ambiguity band 增加可选 strict-output LLM selector、bounded prompt、持久化实现 metadata 与 sanitized fallback | 10d, 70 | 0.5–1d |
| 6.4 | 实现四类 recall lane、Milvus decay/application fallback、read-only startup acceptance probe、跨 lane merge 和 anti-feedback invariant；native 模式必须在 exact request probe 后 fail closed，score-point/order 证据仍由 Phase 0 disposable exercise 提供 | 10d, 12, 70 | 1–1.5d |
| 6.5 | 实现 bounded consolidator、fact revisions、correction/conflict resolution、working-state projection，以及 exact-plan consolidation journal/outbox 与 partial-write replay | 10d, 12, 70 | 1–1.5d |
| 6.6 | 把 typed `MemoryPack` 接入 classifier/rewrite/memory-only answer，保持 permission/citation/cache 边界 | 10c, 10d, 12, 13 | 0.5–1d |
| 6.7 | 更新 Memory UI/Trace；实现 expired/tombstoned Memory 的 session/snapshot-bound keyset cleanup page、pending-outbox fence、retained-lineage protection 与 exact-id Milvus deletes；加入 strict payload-free Selective Memory eval runner/fixtures，并执行 legacy cutover rehearsal | 10d, 20, 70 | 0.5–1d |

**Exit criteria**: M4 roadmap criteria 全部满足；selection/consolidation/decay 的 local/Milvus parity 可复现；纠正、冲突、reconfirmation、expiry、clear 与跨 session isolation 全部通过；Response Cache 无行为回归；未证明 native decay 时明确运行 application fallback；legacy reader 删除需另行批准。

## 10. Phase 7 — Workflow convergence and bounded latency

Maps to: closes M5.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 7.1 | 为 evidence-state fingerprint、无进展提前终止和 query-scoped reranker sticky fallback 先固定合同与 local/LangGraph tests，再实现 | 12, 70, 99 | 0.5–1d |
| 7.2 | 将 response-cache candidate lookup 从 Memory recall 移出，形成 permission 后 fail-closed `try_grounded_cache` | 10c, 12, 70, 99 | 0.5–1d |
| 7.3 | 用 typed result 合并 `classify+route`、`tool selection+rewrite`、`grade+next action`，保持各自安全 trace | 12, 12a, 70, 99 | 1–1.5d |
| 7.4 | 建立 local/LangGraph 共用 transition contract，删除重复 terminal/retry routing | 12, 20, 70, 99 | 1–1.5d |
| 7.5 | 用 adapter capability contract 评估 ready retrieval calls 与 terminal persistence sinks 的 bounded parallelism；只有证明 client/thread safety、deterministic merge、完整状态汇总和取消语义后才启用 | 12, 70, 99 | 0.5–1d |
| 7.6 | 修复 focused atomic feature 的 single-strong-chunk grading；补充 product-associated bare version resolution、original-term-preserving retry 与 pre-execution retry-plan fingerprint 去重 | 12, 70, 99 | 0.5–1d |

**Exit criteria**: 无进展检索不重复 rerank；同 query provider failure
只尝试一次 primary；direct/Memory 请求不搜索 grounded-response cache；
cache hit 保持零 tool/rerank/generation call；local/LangGraph 消费同一
transition contract 并产生等价 terminal snapshot/event order；并发只在
显式 capability 允许时启用，默认路径保持确定性与 session/permission
隔离；focused named feature 可由一条 score≥0.80 的直接版本匹配证据回答，
弱/间接/多方面路径仍 fail closed；retry 保留原始术语且不执行重复
fingerprint；`Milvus 3.0` 路由到 exact `v3.0`；全量 tests、RAG eval 和
lint 通过。

## 11. Phase 8 — Milvus 3.0 native adapter and lifecycle

Maps to: advances M3.

**Goal**: 将已验证的 Milvus 3.0 能力落到 production adapter，同时维持 local fallback 与默认离线路径。

按依赖顺序实现：

1. 固定 [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md)，补齐 data/ingestion/eval cross-reference 和 load-bearing decisions。
2. 增加显式 `relevance|scalar` order mode；Milvus scalar mode 下发 `order_by_fields`，local adapter 实现相同 scalar-primary semantics。
3. 将 facets 改为 candidate-ID bounded Query Aggregation，保留 local exact parity。
4. 增加 `retrieval_text`、BM25 Function、inline synonym dictionary；Milvus sparse query 改为 raw text。
5. sparse index 默认移除 `DAAT_MAXSCORE` 以选择 SINDI，保留显式 compatibility mode。
6. 四个 lifecycle collection 将 `expires_at` 迁移到 `TIMESTAMPTZ ttl_field`，统一 storage codec 并保留显式 expiry predicate。
7. `doc_dedup_signatures` 切换到 server-side MINHASH Function 与 `MINHASH_LSH/MHJACCARD`，ingestion 不再生成持久化 signature。
8. 增加 named snapshot restore/pin 支持，使联网 eval 只读固定 target collection。
9. 增加 dry-run-first schema evolution/backfill 命令，支持新增 sparse/embedding 字段、显式 protobuf physical backfill、完整 schema revalidation 和 bounded partial-update embedding backfill；通过 `MILVUS_SPARSE_FIELD` 单独完成验证后的 reader cutover。
10. 运行 deterministic suite、golden eval、lint/typecheck 和可用的 real-Milvus smoke；最后由独立 reviewer 按 spec 审查并修复 findings。

**Exit criteria**: 所有 server call 由 fake-client tests 证明参数正确；local parity 测试通过；mutating CLI 默认 dry-run；真实服务不可用时只报告 skip；spec、代码、tests 与 CLI help 一致。

## 12. Phase 9 — StructArray document/passage retrieval

Maps to: closes the StructArray slice of M3.

| # | Task | Spec | Effort |
| --- | --- | --- | --- |
| 9.1 | 完成 Phase 0.10 memo，锁定 server/SDK、schema/index paths、TokenANN baseline、unsupported matrix 和 registered failure codes | 14, 70 | 0.5–1d |
| 9.2 | 先写 projection bijection/capacity/parent-invariance/fingerprint tests；实现从完整 `kb_chunks` manifest 全量构建、索引、round-trip 验证 `kb_documents.passages`，任一失败不激活 | 10, 11, 14 | 1d |
| 9.3 | 实现 `MATCH_ANY`/`element_filter` allow-list、element-level COSINE adapter、offset 解析与 stable `chunk_id` normalization；补 local/native parity 与 fail-closed tests | 12, 14, 70 | 0.5–1d |
| 9.4 | 实现受限 `struct_two_stage`；只将同 tool/scope 的 2–3 个 required aspects 组成 EmbeddingList，用 parent shortlist 约束后续 element searches，禁止 parent-only evidence | 12, 14, 70 | 0.5–1d |
| 9.5 | 实现 `struct_fused`；分别执行 flat BM25 和 StructArray element dense，按 `chunk_id` 确定性融合并保留 provenance，不走丢 offset 的 mixed-granularity server hybrid | 12, 14, 70 | 0.5d |
| 9.6 | 更新 notebook/UI trace，实现 `flat_hybrid/struct_element/struct_two_stage/struct_fused` 隔离 eval，报告 quality、latency、index/storage/build cost 与 MaxSim reference delta | 20, 70, 90 | 1d |
| 9.7 | 独立审查 schema、permission/version filters、citation identity、collapse/limit 语义和默认模式；只在 reviewed report 证明收益时改变默认 | 10–14, 70, 99 | 0.5d |

**Exit criteria**: 目标 Milvus/SDK 的 § 2.7 probe 全部通过；projection 与 flat corpus 双向对齐；same-element false-positive fixture 被拒绝；element/two-stage/fused 均只输出稳定 passage evidence；parent-only/collapsed citation 为 0；现有 citation/permission/version/abstention guardrails 全绿；profile report 固定软硬件与 corpus fingerprint，无受益则默认仍为 `disabled`；deterministic suite、golden eval、lint/typecheck 通过。

## 13. What makes this order correct

1. **先合同后消费者**：identity、citation、normalized result 先稳定，再让 graph、UI 和 notebooks 消费。
2. **先证伪高风险假设**：Milvus/model 能力失败时只改 Phase 0/spec，而不是返工所有调用点。
3. **每阶段保持可测试**：先写行为测试，再实现最小路径；组件错误显式进入 terminal state。
4. **先经历、后解释**：先固定 immutable event 和 selection metadata，再构建 consolidation/projection，避免长期事实失去来源。
5. **软遗忘不拥有有效性**：先落状态/expiry filter，再接 decay ranker，避免旧的错误或已删除事实仅因向量相似而重新出现。
6. **搜索粒度不改变证据身份**：先证明 parent/element 映射和 flat projection 对齐，再接 Agent；EmbeddingList 只缩小文档范围，最终仍由可引用 passage 完成闭环。

## 14. Cross-references

- Stakeholder milestones: [`90-roadmap.md`](./90-roadmap.md)
- Quality gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- Decisions: [`99-key-decisions.md`](./99-key-decisions.md)
