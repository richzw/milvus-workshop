# 90 — Roadmap: Incremental Workshop Delivery

Status: draft v3 · Audience: stakeholders and workshop author · Last updated: 2026-07-27

## 0. Principles

- 每个里程碑都留下可运行、可解释的 Workshop 状态。
- 用户可见能力按最短学习闭环推进；未经验证的技术能力不阻塞早期教学价值。
- 估算基于当前只有文档、尚无 runnable demo 的事实，置信度低；Phase 0 后必须校准。

## 1. Milestone map

```text
┌─────────────────────┐
│ M0 Grounded Q&A     │  ask → retrieve → OpenAI/fallback → citation
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M1 Explain the Agent│  intent + tools + plan + multi-hop + self-check
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M2 Build the Corpus │  local/MinIO notebooks + schema + eval + chunking
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ M3 Explore Milvus   │  verified feature panels + optional experiments
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

**User-visible unlock**: 参与者只需自然语言提问；Agent 自动判断意图、用预定义实体理解领域术语、检查权限、选择最相关工具、拆解复杂问题，并在正确的文档版本范围内按证据缺口补充检索。参与者能看到完整 terminology resolution、version scope、tool plan、recall/rerank、grade、retry 和 answer self-check。

**Specs touched**: 10, 12, 13, 20, 70.

**Exit criteria**:

- UI 无 source/doc/department controls，三个 tab 可用，Evidence 明确区分 tool recall 与 rerank。
- 单域问题只调用最相关工具；对比 fixture 至少调用两个工具。
- 至少一个 fixture 完成依赖前一跳证据的补充检索；另一个在 3 次后 abstain。
- 权限拒绝发生在检索前；grounded terminal answer 通过 citation/self-check。
- 主 reranker 不可用时 fallback 路径通过 workflow smoke test。
- `GO按钮`/别名 fixture 命中同一实体；无法领域消歧的问题在检索前请求澄清。
- 默认问题只使用 current edition；指定版本与版本对比 fixture 分别执行 exact/comparison scope，且不交叉拼接内容。

**Provisional calendar**: M0 后约 4–6 focused developer days。

### M2 — Build and evaluate the corpus

**User-visible unlock**: 开发者可以通过 notebooks 从本地与 MinIO 导入相同类型的文档，理解 chunk、schema、embedding、hybrid search 与 RAG eval。

**Specs touched**: 10, 10a, 11, 70.

**Exit criteria**:

- `01`–`04` notebooks 按顺序在干净环境运行，并调用可复用实现而非复制核心逻辑。
- 二次导入不产生重复逻辑 chunk；PDF citation 保留页码。
- 每个 chunk 都有 `doc_version/is_current`，同一 `doc_id` 恰有一个 current edition；更新 current edition 后历史版本仍可按 exact scope 检索。
- Min-Max Chunking 至少比较两组配置并记录指标。
- 显式配置 OpenAI 后，ingestion 与 query 使用同一 `text-embedding-3-small` 1024 维向量空间；默认 offline 测试不访问网络。
- golden fixture 完整性和 RAG eval 报告可重复生成。

**Provisional calendar**: M1 后约 4–6 focused developer days。

### M3 — Milvus 3.0 feature lab and extensions

**User-visible unlock**: 参与者可以观察经验证的 filter、排序/聚合、nullable image vector，并使用 session-scoped 多轮 Memory，选择性探索去重、图片检索或 MFS。

**Specs touched**: 10, 11, 20, 70.

**Exit criteria**:

- 每个 UI feature panel 都链接到 Phase 0 的可执行验证或明确标注为 demo-side computation。
- 至少 3 个图片样例展示 schema nullability；若启用图片检索，另有独立 eval。
- conversation memory 按 [`10b-conversation-memory.md`](./10b-conversation-memory.md) 独立验收并显式进入 P2 query path；MinHash dedup、MFS 仍是独立实验。
- 同 session follow-up 可观察召回和写入，跨 session/过期记录不可见，用户可清除当前 session。

**Provisional calendar**: 核心 feature lab 约 2–4 days；每个 P2 experiment 另估。

## 3. Calendar shape and calibration

单开发者、已具备模型/API 凭据与本地 Milvus/MinIO 环境时，M0–M2 的低置信度总估算为 10–15 focused days，另加 review 与 Workshop rehearsal buffer。Phase 0 若发现核心 Milvus 能力或模型运行方式不成立，必须调整范围和估算，不以隐藏 fallback 维持原日期。

## 4. Cross-references

- Product scope: [`00-prd.md`](./00-prd.md)
- Quality gates: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- Dependency-ordered execution: [`91-impl-plan.md`](./91-impl-plan.md)
