# 99 — Key Decisions

Status: draft · Last updated: 2026-07-27

新增决策时追加新的 D-id；若推翻既有决策，新增 superseding entry，不覆盖原理由。

## D1 — UI is query-only; ingestion is offline

- **Context**: MVP scope and Workshop reliability.
- **Alternatives considered**: online upload in Streamlit; background synchronization; offline notebook/CLI.
- **Decision**: MVP UI 只查询，ingestion 通过 notebook/CLI 离线执行。
- **Why**: 将数据准备与在线问答故障面分离，使 Workshop 能分别教学且现场更可恢复。
- **Pinned by**: [`00-prd.md § MVP scope`](./00-prd.md#6-mvp-scope), [`11-ingestion.md`](./11-ingestion.md), [`20-ui-demo.md`](./20-ui-demo.md)
- **Date**: 2026-07-22

## D2 — MinIO/mock S3 is the MVP object store

- **Context**: 企业资料分布在 S3 类存储，但 Workshop 需要本地可复现。
- **Alternatives considered**: 真实 S3；仅本地文件；MinIO/mock S3。
- **Decision**: 主练习使用 MinIO/mock S3，并保留本地文件 golden path。
- **Why**: 保留 object-store 语义而不要求云账号；MinIO 故障也不阻塞最小 demo。
- **Pinned by**: [`11-ingestion.md § Supported MVP inputs`](./11-ingestion.md#31-supported-mvp-inputs)
- **Date**: 2026-07-22

## D3 — LangGraph owns the explicit Agent workflow

- **Context**: 需要展示 agentic RAG 技巧和完整 trace。
- **Alternatives considered**: 单函数 pipeline；自由 tool-calling agent；显式 LangGraph state machine。
- **Decision**: 使用显式节点和 conditional edge 的 LangGraph workflow。
- **Why**: 节点职责、重试边界和 trace 可测试、可教学，不依赖隐式 agent 行为。
- **Pinned by**: [`12-agent-workflow.md`](./12-agent-workflow.md)
- **Date**: 2026-07-22

## D4 — Retrieval, reranking, and grading remain separate

- **Context**: 三步优化目标不同。
- **Alternatives considered**: 只取 Milvus top-k；rerank 与 grade 合并；三段式 pipeline。
- **Decision**: Milvus 高召回、reranker 高精排、grader 判断证据充分性。
- **Why**: 让质量问题能定位到召回、排序或覆盖判断，并支持 UI 前后对比。
- **Pinned by**: [`12-agent-workflow.md § Node contracts`](./12-agent-workflow.md#5-node-contracts), [`20-ui-demo.md § Evidence`](./20-ui-demo.md#42-evidence)
- **Date**: 2026-07-22

## D5 — Retry is capped at three and ends in abstention

- **Context**: 证据不足时需要改写重试，但不能无限循环或编造答案。
- **Alternatives considered**: 不重试；无界循环；固定 3 次上限后 abstain。
- **Decision**: 最多 3 个额外 retrieval rounds，仍不足则返回结构化 abstain。
- **Why**: 提供可预测延迟和终止保证，同时示范 grounded failure behavior。
- **Pinned by**: [`12-agent-workflow.md § Invariants`](./12-agent-workflow.md#6-invariants), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-22

## D6 — Citation identity is chunk/page/version based

- **Context**: 用户需要验证答案来源，eval 需要稳定匹配。
- **Alternatives considered**: 只显示文档标题；自由文本 URL；稳定 chunk id + page/section。
- **Decision**: citation 必须绑定 `chunk_id` 与 `doc_version`，PDF 带 `page_no`，其他文档可带 `section`。
- **Why**: 比文档级引用更可核查，并能用 fixture 自动验证。
- **Pinned by**: [`10-data-model.md § Invariants`](./10-data-model.md#32-invariants), [`12-agent-workflow.md § generate_answer`](./12-agent-workflow.md#59-generate_answer)
- **Date**: 2026-07-22

## D7 — Primary reranker has a deterministic fallback

- **Context**: 外部模型/API 可能让现场演示失败。
- **Alternatives considered**: 强依赖一个模型；完全规则排序；可替换接口 + fallback。
- **Decision**: 使用统一 reranker contract，主实现为验证后的模型，fallback 为确定性规则排序。
- **Why**: 兼顾教学质量和演示恢复能力，同时在 trace 中诚实标注实现。
- **Pinned by**: [`12-agent-workflow.md § rerank_evidence`](./12-agent-workflow.md#56-rerank_evidence), [`20-ui-demo.md`](./20-ui-demo.md)
- **Date**: 2026-07-22

## D8 — Store only authoritative domain data in Milvus

- **Context**: 旧稿提出 knowledge、memory、dedup、trace 与 eval 多类数据。
- **Alternatives considered**: 全部写 Milvus；全部落普通文件；按访问模式分界。
- **Decision**: Milvus 保存 `kb_chunks` 及 P2 memory/dedup；query trace 留在 Streamlit session state；eval fixtures 使用 JSON/YAML。
- **Why**: 每类数据使用最简单、符合生命周期与查询模式的存储，避免为 demo 过度设计。
- **Pinned by**: [`10-data-model.md § Storage architecture`](./10-data-model.md#2-storage-architecture)
- **Date**: 2026-07-22

## D9 — Unverified Milvus features are Phase 0 gates

- **Context**: 现有材料列出了多项 Milvus 3.0 能力和具体 schema/index 参数，但仓库没有可执行验证。
- **Alternatives considered**: 直接当作合同；全部删除；先验证再锁定并提供显式 fallback。
- **Decision**: hybrid/BM25、ordering、aggregation、nullable vector、TTL 和 MinHash representation 在 Phase 0 验证。
- **Why**: 遵守 AGENTS.md 的 honesty rules，避免 spec 建立在未确认 API 或语义上。
- **Pinned by**: [`70-quality-and-evaluation.md § External capability verification matrix`](./70-quality-and-evaluation.md#6-external-capability-verification-matrix), [`91-impl-plan.md § Phase 0`](./91-impl-plan.md#3-phase-0--risk-retirement)
- **Date**: 2026-07-22

## D10 — MVP is local-demo only and uses synthetic data

- **Context**: MVP 明确不做 auth/ACL，却面向“企业内部资料”场景。
- **Alternatives considered**: 默认允许真实内部文档；在 MVP 加完整 ACL；限制为合成/精选样本。
- **Decision**: MVP 只允许 Workshop 样本数据和本地使用；真实企业部署属于 P2 重新设计。
- **Why**: 没有授权、审计和 secret boundary 的 demo 不应被误用为内部生产系统。
- **Pinned by**: [`00-prd.md § Non-goals`](./00-prd.md#4-non-goals), [`70-quality-and-evaluation.md § Demo security boundary`](./70-quality-and-evaluation.md#7-demo-security-boundary)
- **Date**: 2026-07-22

## D11 — OpenAI generation has a deterministic, validated fallback

- **Context**: 第一版只拼接 chunk，无法展示大模型基于多段 evidence 综合回答；Workshop 又不能因外部 API 故障失去主链路。
- **Alternatives considered**: OpenAI 为硬依赖并直接流式输出；通用多厂商兼容层；OpenAI 主实现 + citation guard + deterministic fallback。
- **Decision**: 使用可注入 answer-generator interface；OpenAI Responses 是首个主实现，输出完整验证后再发给调用方，配置缺失或 provider/validation 失败时显式降级。
- **Why**: 把模型供应商限制在一个 seam，保持默认测试和现场演示可恢复，同时确保模型不能越过 selected-context citation 合同。
- **Pinned by**: [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`12-agent-workflow.md § generate_answer`](./12-agent-workflow.md#59-generate_answer), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-22

## D12 — OpenAI text embedding preserves the existing vector contract

- **Context**: 第一版用 deterministic hash 生成 1024 维文本向量；需要接入真实语义 embedding，同时最小化对 ingestion、query 和 Milvus schema 的改动。
- **Alternatives considered**: 改用模型原生默认维度并重建 schema；在每个调用点直接调用 OpenAI；保留统一 `dense_vector()` seam 并请求 1024 维 OpenAI embedding。
- **Decision**: `dense_vector()` 背后使用可注入 provider；OpenAI 主实现使用 `text-embedding-3-small` 和 `dimensions=1024`；默认使用 deterministic provider，只有显式选择 `openai` 或 `auto` 才允许外部请求。
- **Why**: 写入与查询天然共享一个向量空间，不需要修改现有 Milvus collection、KBChunk 或调用点；provider 失败不自动混用 fallback，避免静默破坏召回质量。
- **Pinned by**: [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`11-ingestion.md § Embeddings`](./11-ingestion.md#5-embeddings)
- **Date**: 2026-07-22

## D13 — Metadata routing is owned by tools, not UI

- **Context**: `source_type`、`doc_type`、`department` 曾作为 Streamlit controls 暴露，导致用户替 Agent 做知识源选择。
- **Alternatives considered**: 保留高级 filters；完全无 filters 的全库搜索；注册 search tools 封装知识域与 metadata policy。
- **Decision**: UI 只提交自然语言问题；Agent 选择注册工具，工具内部构造并校验 metadata filters。
- **Why**: 工具选择是 Agentic RAG 的核心可观察决策，同时集中权限交集与 filter policy，避免 UI 绕过路由。
- **Pinned by**: [`12-agent-workflow.md § Tool catalog`](./12-agent-workflow.md#4-tool-catalog), [`20-ui-demo.md`](./20-ui-demo.md), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-24

## D14 — Agent planning is bounded and explicit

- **Context**: 对比与多跳问题需要多个查询，但自由 tool-calling 难以测试、终止和教学。
- **Alternatives considered**: 单查询重写；无界 autonomous agent；显式至多三个子查询与三轮补充检索。
- **Decision**: query plan 显式记录 tool、subquery、依赖和状态；初始最多三个子查询，补充检索最多三轮，既有证据不丢弃。
- **Why**: 在覆盖复杂问题的同时提供终止保证、可复现 trace 与明确的质量测试点。
- **Pinned by**: [`12-agent-workflow.md § State contract`](./12-agent-workflow.md#3-state-contract), [`12-agent-workflow.md § Invariants`](./12-agent-workflow.md#6-invariants), [`91-impl-plan.md § Phase 3`](./91-impl-plan.md#6-phase-3--retrieval-and-agent-workflow)
- **Date**: 2026-07-24

## D15 — Predefined entities resolve domain terminology before rewrite

- **Context**: 同一词在不同行业可能含义不同，产品别名与游戏黑话也可能不在通用模型词汇中。
- **Alternatives considered**: 只依赖模型常识；检索后再猜词义；使用受版本控制的 entity catalog 在 intent 后、rewrite 前消歧。
- **Decision**: 使用带 `entity_id/entity/aliases/comment/domains` 的本地 catalog；只注入当前问题匹配的实体，无法按 domain 消歧时在检索前请求澄清。
- **Why**: 将领域语义变成可审查、可测试的输入，同时避免把整个词库塞入 prompt 或把实体解释误当作知识证据。
- **Pinned by**: [`10-data-model.md § predefined_entities.yaml`](./10-data-model.md#63-predefined_entitiesyaml), [`12-agent-workflow.md § resolve_terminology`](./12-agent-workflow.md#51a-resolve_terminology), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-27

## D16 — Retrieval is document-version-aware by default

- **Context**: 同一文档多个版本同时召回会把不同 edition 的改动交叉写入一个答案。
- **Alternatives considered**: 仅按更新时间排序；删除所有历史版本；chunk 保存显式版本 metadata 并让 tool plan 管理 version scope。
- **Decision**: 每个 chunk 必须有 `doc_version/is_current`；普通查询只查 current，指定版本使用 exact filter，只有明确版本对比才可跨版本且证据始终分组。
- **Why**: 排序不能保证 top-k 不混版，删除历史版本又无法回答历史问题；显式 scope 能在 retrieval 边界阻止污染并支持可测试的版本对比。
- **Pinned by**: [`10-data-model.md § kb_chunks`](./10-data-model.md#3-kb_chunks--authoritative-knowledge-records), [`11-ingestion.md § Chunking and identity`](./11-ingestion.md#4-chunking-and-identity), [`12-agent-workflow.md § execute_tool_plan`](./12-agent-workflow.md#55-execute_tool_plan), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-27

## D17 — Stream safe execution events before validated answer text

- **Context**: `st.write_stream` previously replayed chunks only after the complete graph and answer validation, leaving users without useful progress and presenting Agent Trace as terminal raw JSON.
- **Alternatives considered**: expose provider tokens immediately; keep all output terminal-only; stream sanitized stage/tool events while retaining validated-buffered answer release.
- **Decision**: local and LangGraph runtimes expose one ordered event contract. Sanitized operational trace events stream as nodes complete; grounded answer text remains hidden until provider citation validation and workflow self-check succeed, then bounded answer deltas and one final snapshot are emitted.
- **Why**: users see useful Agent progress without exposing chain-of-thought or unvalidated model output, and the terminal response remains authoritative and replayable.
- **Pinned by**: [`12-agent-workflow.md § Streaming event contract`](./12-agent-workflow.md#31-streaming-event-contract), [`13-llm-answer-generation.md § Prompt and output contract`](./13-llm-answer-generation.md#5-prompt-and-output-contract), [`20-ui-demo.md`](./20-ui-demo.md)
- **Date**: 2026-07-27

## D18 — Conversation Memory is session-scoped supplementary context

- **Context**: The repository had a local TTL prototype and Milvus schema, but no multi-turn Agent integration, persistence adapter or user-visible lifecycle.
- **Alternatives considered**: send the complete chat transcript on every turn; treat Memory as a citeable knowledge source; add bounded semantic Memory with explicit session/expiry filtering.
- **Decision**: every valid terminal turn is idempotently stored as bounded short-term records and a deterministic summary. The next turn recalls only live summaries/task state from the same generated session. Memory can clarify a follow-up or answer an explicit recall request, but cannot satisfy KB evidence, create `[Cn]` citations, change authorization or cross sessions.
- **Why**: bounded semantic recall demonstrates Milvus Memory without unbounded prompts, citation ambiguity or hidden identity/profile semantics. Explicit `expires_at` gives local/Milvus parity and testable deletion.
- **Pinned by**: [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`12-agent-workflow.md § Node contracts`](./12-agent-workflow.md#5-node-contracts), [`20-ui-demo.md § Memory`](./20-ui-demo.md#44-memory)
- **Date**: 2026-07-27
