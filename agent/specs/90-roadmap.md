# 90 — Roadmap: Incremental Workshop Delivery

Status: draft v8 · Audience: stakeholders and workshop author · Last updated: 2026-08-24

## 0. Principles

- 每个里程碑都留下可运行、可解释的 Workshop 状态。
- 用户可见能力按最短学习闭环推进；未经验证的技术能力不阻塞早期教学价值。
- 估算以当前 deterministic runnable demo 为基线；新的 live-provider 查询转换、上下文压缩和端到端 chunking v2 尚未实现，相关数字仍需 Phase 0/5 校准。

## 1. Milestone map

```text
┌─────────────────────┐
│ M0 Grounded Q&A     │  ask → retrieve → OpenAI/fallback → citation
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M1 Explain the Agent│  intent + transform + tools + context + self-check
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M2 Build the Corpus │  local/MinIO + schema + end-to-end chunk eval
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M3 Explore Milvus   │  verified feature panels + optional experiments
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M4 Memory that Learns│ select → decay → consolidate → project
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M5 Predictable Flow │ short-circuit → converge → bounded latency
└─────────────────────┘
```

## 2. Milestones

### M0 — Grounded Q&A vertical slice

**User-visible unlock**: 参与者能对预置知识库提问，获得校验后分块呈现的 answer-or-abstain 结果和可打开的 chunk/page/version citation。

**Specs touched**: 00, 10, 12, 13, 20, 70.

**Exit criteria**:

- 一条 golden question 完成检索、回答和 citation round trip。
- 配置 OpenAI 时答案由 selected chunks 综合生成；provider 失败时 trace 明确显示 deterministic fallback。
- 无证据问题明确 abstain；服务失败不伪装为普通答案。
- Chat tab 和最小 Evidence 信息共享同一 `query_id`。

**Provisional calendar**: Phase 0 验证后约 2–3 focused developer days。

### M1 — Explainable Agentic RAG

**User-visible unlock**: 参与者只需自然语言提问；Agent 自动判断意图、用预定义实体理解领域术语、检查权限、选择最相关工具，并在 identity/rewrite/step-back/decompose 中选择受限查询转换策略。复杂问题在正确的文档版本范围内按证据缺口补充检索；长且含噪的 selected context 可在不改变原 citation 的前提下压缩后生成。参与者能看到完整 terminology resolution、transformation strategy、version scope、tool plan、recall/rerank、grade、compression、retry 和 answer self-check。

同一 session 内再次提出完全相同或高置信语义等价的 KB 问题时，Agent 在当前权限与 evidence freshness 校验后直接复用带 citations 的 grounded response；Trace 明确区分 exact/semantic hit 与 stale/miss。

**Specs touched**: 10, 12, 13, 20, 70.

**Exit criteria**:

- UI 无 source/doc/department controls，Chat/Evidence/Agent Trace 三个核心 tab 可用（Memory tab 属于 M3），Evidence 明确区分 tool recall 与 rerank。
- 单域问题只调用最相关工具；对比 fixture 至少调用两个工具。
- 至少一个 fixture 完成依赖前一跳证据的补充检索；另一个在 3 次后 abstain。
- 权限拒绝发生在检索前；grounded terminal answer 通过 citation/self-check。
- 主 reranker 不可用时 fallback 路径通过 workflow smoke test。
- `GO按钮`/别名 fixture 命中同一实体；无法领域消歧的问题在检索前请求澄清。
- 默认问题只使用 current edition；指定版本与版本对比 fixture 分别执行 exact/comparison scope，且不交叉拼接内容。
- identity/rewrite/step-back/decompose fixture 均产生不超过 3 个受权查询；step-back 始终保留 primary query，背景结果不能单独支撑具体结论。
- 一个长上下文 fixture 在 selective compression 后降低输入字符/token，且 required facts、selected source ids 和 citations 不变；一个非法输出 fixture 整体回退原 context。

**Provisional calendar**: M0 后约 4–6 focused developer days。

### M2 — Build and evaluate the corpus

**User-visible unlock**: 开发者可以通过 notebooks 从本地与 MinIO 导入相同类型的文档，理解 chunk、schema、embedding、hybrid search 与 RAG eval，并能从真实 trace failure 维护一组小而可行动的 eval metrics。

**Specs touched**: 10, 10a, 11, 70.

**Exit criteria**:

- `01`–`04` notebooks 按顺序在干净环境运行，并调用可复用实现而非复制核心逻辑。
- 二次导入不产生重复逻辑 chunk；PDF citation 保留页码。
- 每个 chunk 都有 `doc_version/is_current`，同一 `doc_id` 恰有一个 current edition；更新 current edition 后历史版本仍可按 exact scope 检索。
- Min-Max Chunking 对至少三组 small/medium/large 配置（初始 `max_tokens` 可取 128/256/512）分别建立隔离 chunk set/index，并在固定其余变量后运行检索、rerank、generation 与 citation self-check。
- 报告独立给出 recall、citation、required-fact coverage、faithfulness、answer relevancy、latency/token/cost；faithfulness/relevancy grader 未校准时明确标记 incomplete，不发布新默认值。
- 完成一次 30–50 条 representative traces 的 bootstrap error analysis：逐条记录 `overall_pass`/`review_note`，聚类 named failure categories，并区分一次性 prompt/schema fix、fixture-only 与 metric candidate。
- `eval-metric-registry-v1` 同时保留 goal、guardrail、operational 三类 active metric；每项有 owner、grader、dataset segment、gate/budget、决策动作、cost/cadence 与退休条件。没有动作的 report 字段只作为 diagnostic。
- `rag-eval-v3` 嵌入 registry checksum，按三类角色给出 decision status；trajectory/tool/outcome 继续负责首错归因，不产生 blended overall score。v2 baseline 不跨版本比较。
- 经审查的 recommendation artifact 记录选择与非选择配置的理由；runner 不自动改写 ingestion default。
- 显式配置 OpenAI 后，ingestion 与 query 使用同一 `text-embedding-3-small` 1024 维向量空间；默认 offline 测试不访问网络。
- golden fixture 完整性和 RAG eval 报告可重复生成。

**Provisional calendar**: M1 后约 5–7 focused developer days，其中 error-analysis/registry/report migration 约 1 day。

### M3 — Milvus 3.0 feature lab and extensions

**User-visible unlock**: 参与者可以观察经验证的 filter、排序/聚合、nullable image vector，以及 StructArray 如何在保留整体文档身份的同时命中局部 passage。单方面问题可看到 element offset 与稳定 citation 的映射，多方面问题可观察 EmbeddingList 文档 shortlist 如何再回到可引用 element。参与者也可使用 session-scoped 多轮 Memory，选择性探索去重、图片检索或 MFS。

**Specs touched**: 10, 11, 12, 13, 14, 20, 70.

**Exit criteria**:

- 每个 UI feature panel 都链接到 Phase 0 的可执行验证或明确标注为 demo-side computation。
- `kb_documents.passages` 投影与 `kb_chunks` 在 passage identity/checksum/vector fingerprint 上完整对齐，并可回读唯一权威原文；任一超容量或不一致都使整个 StructArray 模式不激活。
- same-element `MATCH_ANY`/`element_filter`、element-level repeated-parent hits 和 EmbeddingList `MAX_SIM_COSINE` 都在目标 Milvus/SDK 上有可执行证据。
- focused fixture 返回可解析 `chunk_id` 的 element evidence；multi-aspect fixture 先返回 parent shortlist，再为每个必需 aspect 解析 element。Parent-only/collapsed hit 的 citation 数为 0。
- 隔离评测对比 `flat_hybrid/struct_element/struct_two_stage/struct_fused`，独立报告 quality、latency、index/storage/build cost；无可审查收益时 StructArray 保持 opt-in lab。
- 至少 3 个图片样例展示 schema nullability；若启用图片检索，另有独立 eval。
- conversation memory 按 [`10b-conversation-memory.md`](./10b-conversation-memory.md) 独立验收并显式进入 P2 query path；MinHash dedup、MFS 仍是独立实验。
- 同 session follow-up 可观察召回和写入，跨 session/过期记录不可见，用户可清除当前 session。
- “查找下我最近的三个问题是什么”直接返回最近三个 prior user turns，顺序、数量和 trace 可解释，且不调用内部知识搜索工具。

**Provisional calendar**: 核心 feature lab 约 2–4 days；StructArray 投影、双粒度检索与隔离评测另约 3–5 focused days；每个 P2 experiment 另估。

### M4 — Selective Memory that learns and forgets

**User-visible unlock**: 参与者可以观察 Memory 为什么被保留、如何随时间软衰减、何时 consolidation 为长期事实，以及纠正如何 supersede 旧事实并保留 lineage。普通对话不再与重要纠正和任务状态获得相同的长期权重。

**Specs touched**: 10, 10b, 10c, 10d, 12, 20, 70.

**Exit criteria**:

- UI/Trace 可安全显示 selection reason counts、retention class、decay profile、事实/episode/conflict 计数，不展示内容、embedding 或 selector prompt。
- `memory_events` 与 `memory_facts` 完成 same-session local/Milvus contract；response cache 保持独立。
- 规则 selector、typed recall lanes、working-state projection、correction/supersession 与 bounded consolidation 通过 deterministic fixtures。
- Milvus Decay Ranker 已由 Phase 0 证明；若未通过，使用明确标注的 deterministic application-side decay fallback。
- expired、superseded、tombstoned、权限不兼容和跨 session records 的可见率为 0。
- recall 不刷新生命周期；explicit reconfirmation 追加 event 并可更新 projected revision。
- legacy `conversation_memory` dual-write/cutover 有 parity test 与可回退步骤。

**Provisional calendar**: Phase 0 capability proof 后约 5–8 focused developer days；cross-session identity、fork/merge 与自动 skill/prompt promotion 不包含在内。

### M5 — Predictable and convergent Agent flow

**User-visible unlock**: direct/Memory 请求走最短安全路径；没有新证据的
补充检索提前结束；一次 provider 故障不会在同一 query 的每轮重试中重复
消耗 timeout。Trace 仍保留权限、retrieval、rerank、grade、verification
和 persistence 的独立安全边界。

**Specs touched**: 10c, 12, 12a, 20, 70, 99.

**Exit criteria**:

- evidence-state fingerprint 无变化时提前 abstain 且不再次 rerank；
- reranker primary 在一个 query 中最多因 fallback 尝试一次，新 query
  不继承该降级；
- direct/Memory flow 不调用 grounded-response cache；
- local 与 LangGraph 由同一 transition contract 驱动并保持 terminal/event
  parity；
- 仅当 adapter 显式证明并发能力时启用 bounded parallelism，默认行为仍
  可重复、可取消并保持权限/session 隔离。
- focused named feature 在一条强、直接、版本匹配的 live citation 下可回答，
  comparison/exhaustive/multi-tool 与弱单证据仍拒答；
- supplementary retrieval 保留原问题产品/功能/版本术语，并在工具调用前
  拒绝重复 retry-plan fingerprint。

**Provisional calendar**: M4 后约 3–5 focused developer days。

## 3. Calendar shape and calibration

单开发者、已具备模型/API 凭据与本地 Milvus/MinIO 环境时，M0–M2 的低置信度总估算为 10–15 focused days，另加 review 与 Workshop rehearsal buffer。Phase 0 若发现核心 Milvus 能力或模型运行方式不成立，必须调整范围和估算，不以隐藏 fallback 维持原日期。

## 4. Cross-references

- Product scope: [`00-prd.md`](./00-prd.md)
- Quality gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- Dependency-ordered execution: [`91-impl-plan.md`](./91-impl-plan.md)
