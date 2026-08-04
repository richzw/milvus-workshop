# Agent Workshop Specs

Status: draft v6 · Last updated: 2026-07-31

本目录是 Agent Workshop 的权威设计入口。编号表示依赖与建议阅读顺序；旧稿仅作为需求来源保存在 [`archive/`](./archive/README.md)，不再定义实现契约。

## Spec map

| Spec | 类型 | 回答的问题 |
| --- | --- | --- |
| [`00-prd.md`](./00-prd.md) | PRD | 为什么做、为谁做、MVP 做到什么程度？ |
| [`10-data-model.md`](./10-data-model.md) | Foundation | 数据存在哪里，实体 catalog、文档版本、核心字段和不变量是什么？ |
| [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md) | Foundation | 如何用 OpenAI 生成真实文本向量且不迁移现有 Milvus schema？ |
| [`10b-conversation-memory.md`](./10b-conversation-memory.md) | Foundation | 多轮会话如何安全写入、按 session/TTL 语义召回或按时间列出最近问题、展示和清除？ |
| [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md) | Foundation | 相同或语义等价问题如何安全复用带 citations 的已验证回答？ |
| [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md) | Foundation | Conversation Memory 如何演进为 selective dual-speed memory，并通过 Milvus decay 实现软遗忘？ |
| [`11-ingestion.md`](./11-ingestion.md) | Component | 本地与 MinIO 文档如何离线进入知识库？ |
| [`12-agent-workflow.md`](./12-agent-workflow.md) | Component | Agent 如何理解领域术语、选择文档版本与工具、拆解问题、多跳检索、自检并回答？ |
| [`12a-query-classification.md`](./12a-query-classification.md) | Component | Rule-based 与 LLM 如何在受限枚举中分类 query，并安全 fallback？ |
| [`13-llm-answer-generation.md`](./13-llm-answer-generation.md) | Component | selected chunks 如何由 OpenAI 合成为带受控 citation 的答案？ |
| [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md) | Integration | 哪些 adapter、lifecycle、ingestion 与 eval 能力由 Milvus 3.0 原生实现，fallback 和迁移边界是什么？ |
| [`20-ui-demo.md`](./20-ui-demo.md) | Integration | Streamlit 如何呈现答案、证据、Trace 与 Milvus 特性？ |
| [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md) | Cross-cut | 如何验证正确性、可复现性、安全边界和性能？ |
| [`80-glossary.md`](./80-glossary.md) | Glossary | 容易混淆的术语在本项目中分别指什么？ |
| [`90-roadmap.md`](./90-roadmap.md) | Roadmap | 用户可见能力按什么里程碑交付？ |
| [`91-impl-plan.md`](./91-impl-plan.md) | Implementation | 工程上按什么依赖顺序实现？ |
| [`93-improvements-review.md`](./93-improvements-review.md) | Review backlog | 第一版审查发现了什么，哪些已修复或仍待验证？ |
| [`99-key-decisions.md`](./99-key-decisions.md) | Decisions | 关键选择为什么这样定？ |

## Reading order

1. 先读 PRD，确认 MVP 边界与成功标准。
2. 按 `10 → 10a → 10b → 10c → 10d → 11 → 12 → 12a → 13 → 14 → 20` 理解数据和运行链路；`10b` 是当前基线，`10d` 是下一阶段目标态，`14` 固定 Milvus 3.0 adapter 与迁移契约。
3. 用 `70` 检查每个组件的质量门槛。
4. Stakeholder 读 `90`；实现者从 `91` 的 Phase 0 开始。
5. 对设计理由有疑问时查 `99`，术语歧义查 `80`。

## Build-order graph

```text
┌──────────────────────────── Product boundary ────────────────────────────┐
│  00 PRD                                                                 │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
┌──────────────────────────── Data foundation ────────────────────────────┐
│  10 Data Model ─▶ 10a Embedding ─▶ 10b Session Memory                 │
│                                      ├─▶ 10c Response Cache            │
│                                      └─▶ 10d Selective Memory          │
│                                                │                       │
│                            10c ─────────────────┘                       │
│                                      └────────▶ 11 Ingestion revision │
└──────────────────┬───────────────────────────────┬───────────────────────┘
                   │ indexed knowledge             │ fixtures / citations
                   ▼                               ▼
┌──────────────────────────── Runtime boundary ───────────────────────────┐
│  12 Workflow ─▶ 12a Query Classification ─▶ 13 Generation             │
│       └────────────────────▶ 14 Milvus 3 Native ─────────────▶ 20 UI   │
│  tools/version   rules + OpenAI fallback      citations       trace      │
└─────────────────────────────┬────────────────────────────────────────────┘
                              ▼
┌──────────────────────────── Quality gates ──────────────────────────────┐
│  70 Evaluation + reproducibility + local-demo security boundary         │
└─────────────────────────────┬────────────────────────────────────────────┘
                              ▼
             90 User milestones ↔ 91 Engineering phases
                              │
                              ▼
                       99 Key decisions
```

## Source status

- `docs/research/` 与 `vendors/` 当前不存在，因此没有 repository-local prior-art memo 或 vendored implementation；Selective Memory spec 直接链接其外部设计来源和 Milvus 官方文档。
- Milvus 3.0 的 nullable vector、BM25/sparse、`order_by`、aggregation、TTL、Decay Ranker 和 BinaryVector/MinHash 适配性尚未全部在本仓库验证，统一列入 [`91-impl-plan.md § Phase 0`](./91-impl-plan.md#3-phase-0--risk-retirement)。
- OpenAI SDK 2.47.0 的 Responses interface、`text.format` strict JSON schema signature 与本地 fake-client 合同已经验证；真实模型调用仍需使用显式凭据执行 [`12a-query-classification.md § Tests and acceptance`](./12a-query-classification.md#7-tests-and-acceptance) 和 [`13-llm-answer-generation.md § Tests and acceptance`](./13-llm-answer-generation.md#9-tests-and-acceptance) 的 opt-in smoke test。
- OpenAI SDK 2.46.0 的 Embeddings request signature 与 fake-client 合同已经验证；真实 1024 维响应仍需执行 [`10a-openai-text-embedding.md § Tests and acceptance`](./10a-openai-text-embedding.md#7-tests-and-acceptance) 的 opt-in smoke test。
- 未通过 Phase 0 的能力不得作为 MVP 的硬依赖；必须降级、改为预计算展示，或从该里程碑移除。
