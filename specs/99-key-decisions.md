# 99 — Key Decisions

Status: draft · Last updated: 2026-07-28

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
- **Implementation boundary**: `mock_s3` 和 MinIO 是显式互斥的 CLI
  source modes。MinIO 使用 lazy SDK adapter 下载到有对象数/大小限制的临时
  snapshot，再复用同一 parser/chunker；credentials 仅来自环境，object key
  在写入前执行 traversal validation。
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
- **Decision**: 使用统一 reranker run contract；首个主实现为 OpenAI
  Responses strict JSON-schema ranking，必须返回输入 `chunk_id` 的完整排列
  和 `[0, 1]` score。任何配置、provider 或本地 validation 失败都对整批
  candidates 使用确定性规则排序，不接受部分模型结果。
- **Why**: 兼顾教学质量和演示恢复能力；完整排列避免遗漏证据和混合两套
  不可比较分数，per-query run metadata 则在 trace 中诚实标注实际实现、
  model 与 fallback reason。
- **Pinned by**: [`12-agent-workflow.md § rerank_evidence`](./12-agent-workflow.md#56-rerank_evidence), [`20-ui-demo.md`](./20-ui-demo.md)
- **Date**: 2026-07-29

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

## D19 — LLM query classification is structured and rule-fallback safe

- **Context**: 内嵌关键词分类可复现但难以理解自然语言变体；让 LLM 直接输出自由 plan 又会扩大权限、工具与 filter 风险。
- **Alternatives considered**: 保持 workflow 内嵌规则；所有 query 强依赖 LLM；独立 classifier contract，LLM strict structured output + rule-based safety/fallback。
- **Decision**: 把既有规则重构为 `RuleBasedQueryClassifier`，增加只输出固定 intent/topic/retrieval-goal 枚举的 `LLMQueryClassifier`，并由 `FallbackQueryClassifier` 对显式 conversation/memory/operation action 使用 rule fast path、对配置/provider/output 失败使用带安全 reason code 的 rules fallback。LLM 不得在没有 rule baseline 支持时单独选择 `conversation` 并关闭 KB retrieval。
- **Why**: LLM 提升模糊表达理解能力，同时 rules 保证离线复现、现场恢复和安全关键 action 不被 Memory 或模型改写；权限、工具和 filters 仍由后续确定性边界拥有。
- **Pinned by**: [`12a-query-classification.md`](./12a-query-classification.md), [`12-agent-workflow.md § classify_and_route`](./12-agent-workflow.md#51-classify_and_route), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-28

## D20 — Grounded response cache is separate and fail-closed

- **Context**: 相同或语义等价的知识问题重复执行 hybrid retrieval、rerank 和 LLM generation；现有 Conversation Memory 只有摘要且按合同不可替代 KB evidence。
- **Alternatives considered**: 直接从 session summary 回答；只按三天 TTL 复用答案；独立保存完整 citations/evidence scope 并在权限后验证。
- **Decision**: 新建 same-session `grounded_response_cache`，在 recall 阶段只取私有候选，在当前 permission、query constraints、KB revision、chunk version/checksum/current 和 citation validation 全部通过后返回 `answered_from_cache`；默认 TTL 三天，任一不匹配 fail closed 到正常 RAG。
- **Why**: 保留语义命中的性能收益，同时不把旧答案、权限变化或相似但不同的问题当成权威证据；Conversation Memory 继续只承担对话上下文。
- **Pinned by**: [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md), [`12-agent-workflow.md`](./12-agent-workflow.md), [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- **Date**: 2026-07-28

## D21 — Memory uses selective dual-speed storage and projection

- **Context**: D18 的 baseline 每个 terminal turn 都写短期记录和 deterministic summary，无法区分普通对话、用户纠正、任务变化与可复用失败/成功经验，也没有 durable fact lineage 或冲突状态。
- **Alternatives considered**: 继续扩大 summary top-k；把全部 raw log 送入向量检索；每轮调用 LLM reflection；append-only typed episodes + low-cost selection + cautious versioned consolidation。
- **Decision**: 保留 D18 的 session/permission/citation 边界，把目标态拆成 append-only `memory_events`、Rule-first Selection Gate、versioned `memory_facts` 和 rebuildable working-state projection；LLM selector/consolidator 只提交受验证的 typed proposal。
- **Why**: 快速 episode capture 保留具体经历，selection 控制成本与噪声，慢速 consolidation 提供稳定事实，source lineage 允许纠正、冲突、漂移审查和未来 replay；Response Cache 继续承担已验证答案复用。
- **Pinned by**: [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md), [`10-data-model.md § memory_events`](./10-data-model.md#4b-memory_events--append-only-episode-lineage), [`12-agent-workflow.md § recall_memory`](./12-agent-workflow.md#50-recall_memory), [`91-impl-plan.md § Phase 6`](./91-impl-plan.md#9-phase-6--selective-dual-speed-memory)
- **Date**: 2026-07-28

## D22 — Milvus decay provides soft forgetting only

- **Context**: 统一 TTL 会让 Memory 在到期前权重不变、到期时突然消失；只按时间排序又会让近期无关 episode 压过旧但精确的事实。
- **Alternatives considered**: 仅使用固定 TTL；按 last-access 刷新 TTL；应用层手写统一衰减；按 Memory lane 使用 Milvus decay 并保留独立有效性状态。
- **Decision**: episode、operational experience、task state 与 durable fact 使用不同的 registered decay profiles；Milvus decay 只调整召回排名，expiry/supersession/tombstone/permission 负责逻辑可用性，cleanup 负责物理删除。单纯 recall 不刷新时间。
- **Why**: 分层遗忘允许低价值经历自然下沉，又不会复活错误、过期或已删除事实；禁止 access refresh 避免“越召回越永久”的反馈循环。目标 SDK 未验证时使用明确标注的 deterministic application fallback。
- **Pinned by**: [`10d-selective-agent-memory.md § Forgetting and Milvus decay`](./10d-selective-agent-memory.md#6-forgetting-and-milvus-decay), [`70-quality-and-evaluation.md § External capability verification matrix`](./70-quality-and-evaluation.md#6-external-capability-verification-matrix), [`90-roadmap.md § M4`](./90-roadmap.md#m4--selective-memory-that-learns-and-forgets)
- **Date**: 2026-07-28

## D23 — Image embeddings use a fingerprinted DINOv3 ViT-B vector space

- **Context**: 现有 image records 把 caption 送入 deterministic text hash，
  并不读取图片；schema 已固定 nullable 768-dim `image_vector`，而 legacy
  spec 只把 DINOv3 写成未验证候选。
- **Alternatives considered**: 继续使用 caption placeholder；改 schema 接
  DINOv3 ViT-S 384 维；使用 DINOv3 ViT-B/16 的 768 维 global pooled
  feature 并显式区分离线 fallback。
- **Decision**: 真实 provider 使用
  `facebook/dinov3-vitb16-pretrain-lvd1689m` 的 `pooler_output`，本地验证
  768 个 finite values 后 L2 normalize；默认 deterministic provider
  只 hash 已验证图片 bytes，二者 fingerprint 永不混用。DINOv3 权重、
  Pillow/Torch/Transformers 都 lazy load，必须显式配置且失败不回退。
- **Why**: ViT-B 的原生 768 维与现有 Milvus schema 对齐，避免无依据投影
  或 schema migration；读取真实像素关闭 caption placeholder 缺口，显式
  provider/fingerprint 则保护 collection 内向量空间一致性。
- **Pinned by**: [`11-ingestion.md § Embeddings`](./11-ingestion.md#5-embeddings), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-29

## D24 — Image retrieval separates visual and caption query spaces

- **Context**: DINOv3 produces image features but is not a text-image
  cross-modal encoder. Treating a text embedding as an `image_vector` query
  would be dimensionally or semantically invalid, while the corpus already has
  reviewed image titles/captions for text retrieval.
- **Alternatives considered**: cast text vectors into image space; add an
  unplanned CLIP-style model; use caption hybrid search for text queries and
  DINOv3 COSINE search for image queries.
- **Decision**: text-to-image uses the existing dense+sparse title/caption path
  with `has_image_vector == true`; image-to-image embeds real query bytes with
  the collection's configured image provider and searches `image_vector` with
  COSINE. Local and Milvus adapters share dimension, normalization,
  fingerprint, filter and public-output contracts. Image evaluation is separate
  from RAG QA evaluation and reports Recall@K/MRR for both modes.
- **Why**: this keeps each query in a valid vector space, provides a useful
  offline text path, and makes deterministic exact-image pipeline validation
  distinguishable from real DINOv3 semantic quality.
- **Pinned by**: [`10-data-model.md § Invariants`](./10-data-model.md#32-invariants),
  [`12-agent-workflow.md § Tool catalog`](./12-agent-workflow.md#4-tool-catalog),
  [`70-quality-and-evaluation.md § Image retrieval evaluation`](./70-quality-and-evaluation.md#42a-image-retrieval-evaluation)
- **Date**: 2026-07-29

## D25 — Chunking experiments use hard semantic boundaries and stable anchors

- **Context**: fixed heading/paragraph chunks cannot compare Min/Max/overlap,
  while golden `chunk_id` values change when one semantic unit splits and would
  turn the experiment metric into an identity artifact.
- **Alternatives considered**: add a model tokenizer dependency; split a flat
  document stream across headings/pages; compare against old chunk IDs; use a
  deterministic lexical tokenizer inside hard boundaries and source-term
  anchors.
- **Decision**: the offline experiment uses the versioned lexical tokenizer and
  paragraph/sentence-preferred windows only inside Markdown heading/release and
  PDF page boundaries. Evaluation anchors identify a stable `source_uri` and
  terms that must coexist in one recalled/selected chunk. At least two strict
  configs run over the same corpus/query set, and recommendation is a
  deterministic report field rather than a silent production-default change.
- **Why**: results remain dependency-free, citation-safe and comparable across
  configurations while still penalizing fragmented evidence and excessive
  overlap duplicates.
- **Pinned by**: [`11-ingestion.md § Chunking and identity`](./11-ingestion.md#4-chunking-and-identity),
  [`70-quality-and-evaluation.md § Min-Max Chunking experiment`](./70-quality-and-evaluation.md#5-min-max-chunking-experiment)
- **Date**: 2026-07-29

## D26 — Memory LLM selection is a narrow, rule-owned override

- **Context**: model reflection on every turn would add cost, nondeterminism and
  a path for assistant output or untrusted recalled context to become a durable
  user fact.
- **Alternatives considered**: model-only selection; send the full turn and
  Memory context; let the model assign salience/protected status; call a strict
  two-choice model only inside the deterministic ambiguity band.
- **Decision**: rules always run first and retain ownership of score, event type
  and explicit-marker precedence. The optional model sees only bounded user
  query, terminal outcome and rule metadata, and returns exactly
  `ephemeral | promote_candidate`. It cannot select `protected`; unavailable
  configuration, timeout, provider error or invalid output returns the exact
  rule decision with sanitized implementation metadata. A bounded registered
  future-utility phrase is the production-reachable `0.40` ambiguity signal;
  ordinary turns remain `0.20`.
- **Why**: the model can resolve a genuinely narrow uncertain case without
  becoming a general-purpose fact extractor, leaking Memory into prompts or
  weakening deterministic safety rules.
- **Pinned by**: [`10d-selective-agent-memory.md § Episode capture and Selection Gate`](./10d-selective-agent-memory.md#5-episode-capture-and-selection-gate),
  [`70-quality-and-evaluation.md § Correctness`](./70-quality-and-evaluation.md#2-correctness)
- **Date**: 2026-07-29

## D27 — Native Milvus decay is probe-gated and owns only time weighting

- **Context**: accepting `MEMORY_DECAY_MODE=milvus` without executing the target
  SDK/server request can mislabel application decay as native, while probing
  with live Memory would cross the session-content boundary or require startup
  writes.
- **Alternatives considered**: trust a configuration boolean; silently fall
  back on request failure; create and delete a collection during every startup;
  run guaranteed-empty read-only requests for every supported function and
  retain a separate disposable-collection score exercise.
- **Decision**: native mode loads both collections, then requires successful
  empty standard-search requests for `exp`, `gauss` and `linear` through the
  public PyMilvus decay `Function` API. Any failure blocks startup. Recall binds
  event decay to `event_time`, fact decay to `last_confirmed_at`, and keeps
  salience/confidence as application factors. `no_time_decay` bypasses the
  function but remains labeled as execution under the verified native adapter.
  The non-mutating startup probe proves request acceptance; exact score points,
  units and ordering remain a Phase 0 disposable-collection exercise.
- **Why**: this provides executable deployment evidence without reading or
  mutating private Memory, preserves one-field decay constraints, prevents
  silent fallback claims and keeps hard lifecycle filters independent of rank.
- **Pinned by**: [`10d-selective-agent-memory.md § Native decay adapter and startup probe`](./10d-selective-agent-memory.md#63-native-decay-adapter-and-startup-probe),
  [`70-quality-and-evaluation.md § External capability verification matrix`](./70-quality-and-evaluation.md#6-external-capability-verification-matrix)
- **Date**: 2026-07-29

## D28 — Consolidation persists an exact-plan outbox before projection

- **Context**: Milvus fact upsert, lifecycle-event append and completion marking
  are separate calls; a crash between them can lose consolidation or create a
  second interpretation when current state is recomputed.
- **Alternatives considered**: best-effort retry in process; record only the
  trigger id and recompute later; transactional external event store; persist
  the complete validated mutation plan before applying it.
- **Decision**: a session-private journal stores the exact deterministic fact
  updates and lifecycle event under a source-digest operation id. Drain replays
  those idempotent payloads and marks applied last; partial failures remain
  pending with bounded registered error metadata.
- **Why**: exact replay closes every partial-write window without inventing a
  new revision from changed current state, while remaining demonstrable with
  local and Milvus adapters.
- **Pinned by**: [`10d-selective-agent-memory.md § Consolidation journal/outbox`](./10d-selective-agent-memory.md#71-consolidation-journaloutbox),
  [`10-data-model.md § memory_consolidation_journal`](./10-data-model.md#4d-memory_consolidation_journal--recoverable-projection-outbox)

## D29 — Physical Memory cleanup uses session-bound keyset pages

- **Context**: expiry and tombstones must eventually remove sensitive payloads,
  but broad predicates, offset pagination and event cascades can cross scope,
  skip rows after deletion or break retained fact lineage.
- **Decision**: cleanup authenticates an opaque cursor with HMAC and binds it to
  one session and time snapshot, processes fact ids then event ids in ascending
  keyset pages, revalidates eligibility in exact-id deletes, fences pending
  consolidation, and preserves events referenced by retained facts. It runs
  through the affinity-routed service, never a separate online-writer CLI.
- **Consequence**: each page has bounded work and is safely resumable; newly
  eligible rows behind a cursor are collected by the next fresh run rather than
  widening the current snapshot operation.
- **Pinned by**: [`10d-selective-agent-memory.md § Bounded physical cleanup`](./10d-selective-agent-memory.md#91-bounded-physical-cleanup)
- **Date**: 2026-07-29

## D30 — Recent-question recall is a deterministic temporal query

- **Context**: Semantic recall and Selective Memory decay answer “which prior context is relevant,” but users asking “what were my last three questions” require an exact temporal/session operation. Keyword drift between the workflow gate and classifier previously routed this phrasing to `search_code_docs`.
- **Alternatives considered**: add more independent keywords at both call sites; use vector search over summaries/events and prefer recency; send the full chat transcript to the model; share one action detector and add a scalar recent-user-turn store operation.
- **Decision**: one deterministic detector owns explicit and recent-question recall actions for both workflow gating and rule classification. Recent-question recall reads only live same-session `short_term/user` records, globally orders them newest first, bounds the effective count to 20, and answers without KB tools, ANN similarity, Selective Memory decay or citations.
- **Why**: temporal requests need predictable count and ordering, while semantic ranking can omit a recent but dissimilar question or mix in summaries/facts. Reusing the existing persisted user turn preserves session/TTL/current-turn boundaries without a schema migration.
- **Pinned by**: [`10b-conversation-memory.md § Intent and grounding behavior`](./10b-conversation-memory.md#5-intent-and-grounding-behavior), [`12-agent-workflow.md § recall_memory`](./12-agent-workflow.md#50-recall_memory), [`12a-query-classification.md § RuleBasedQueryClassifier`](./12a-query-classification.md#31-rulebasedqueryclassifier), [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates)
- **Date**: 2026-07-30

## D31 — Retry requires evidence progress and provider fallback is query-scoped

- **Context**: supplementary retrieval can return an evidence pool identical
  to the preceding round, while a timed-out reranker provider was previously
  retried for every round. The numeric cap guaranteed termination but still
  allowed repeated work with no possibility of improving coverage.
- **Alternatives considered**: always execute all three retries; compare only
  rewritten query text; compare only chunk ids; fingerprint the complete
  grading-relevant evidence state and pin a provider fallback within one query.
- **Decision**: every merged pool has a deterministic fingerprint over stable
  chunk identity/version/checksum, registered tool provenance and expansion
  membership. An unchanged supplementary pool abstains before rerank/grade.
  Once a fallback wrapper reports a registered provider/output failure, later
  rounds of that query invoke deterministic fallback directly; a new query
  starts without the sticky state.
- **Why**: the workflow stops only when neither evidence nor coverage changed,
  avoids repeated provider timeouts, preserves deterministic grading and does
  not turn a transient failure into a process-wide circuit breaker.
- **Pinned by**: [`12-agent-workflow.md § execute_tool_plan`](./12-agent-workflow.md#55-execute_tool_plan),
  [`12-agent-workflow.md § rerank_evidence`](./12-agent-workflow.md#56-rerank_evidence),
  [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates),
  [`91-impl-plan.md § Phase 7`](./91-impl-plan.md#10-phase-7--workflow-convergence-and-bounded-latency)
- **Date**: 2026-07-30

## D32 — Grounded cache lookup is a permission-gated retrieval fast path

- **Context**: D20 correctly separated grounded cached answers from
  Conversation Memory, but its original placement still searched cache
  candidates inside `recall_memory` for every non-empty request. Direct,
  Memory, operation, clarification and denied requests could therefore pay for
  a vector lookup that they were not allowed or intended to consume.
- **Alternatives considered**: keep pre-permission private candidates; search in
  classification and validate later; combine same-session candidate lookup and
  fail-closed validation in one stage after routing, resolution and permission.
- **Decision**: D20's storage, grounding and fail-closed validation contract
  remains authoritative, but its lookup placement is superseded.
  `try_grounded_cache` is invoked at most once and only for an unambiguous,
  permission-allowed grounded-retrieval route. It owns both candidate lookup
  and validation. A hit terminates before authorized experience recall; a miss
  continues to `recall_authorized_experience` and retrieval planning.
- **Why**: permission now precedes every private grounded-cache read, unrelated
  fast paths perform zero wasted cache I/O, and one observable node owns the
  complete cache outcome without weakening freshness or citation checks.
- **Pinned by**: [`10c-grounded-response-cache.md § Permission-gated lookup and equivalence`](./10c-grounded-response-cache.md#5-permission-gated-lookup-and-equivalence),
  [`12-agent-workflow.md § try_grounded_cache`](./12-agent-workflow.md#52a-try_grounded_cache),
  [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates),
  [`91-impl-plan.md § Phase 7`](./91-impl-plan.md#10-phase-7--workflow-convergence-and-bounded-latency)
- **Date**: 2026-07-30

## D33 — Adjacent pure decisions collapse behind typed stage outcomes

- **Context**: classification and retrieval decision, tool selection and query
  rewrite, and evidence grading and retry planning were represented as separate
  graph nodes even though each second step consumed only the first step's
  in-memory result. This increased event noise and doubled equivalent routing
  branches in local and LangGraph runtimes.
- **Alternatives considered**: keep every helper as a workflow node; hide the
  extra nodes only in the UI; retain focused components but expose three
  composite stages with validated result types.
- **Decision**: the classifier, tool selector/rewriter and evidence grader
  remain independently testable components, while orchestration exposes
  `classify_and_route -> QueryRouteResult`,
  `plan_retrieval -> RetrievalPlanResult` and
  `evaluate_evidence -> EvidenceEvaluation`. The allowed outcome enums are
  closed and each composite stage owns its immediately dependent state
  mutation, including direct answer construction or bounded retry-plan append.
- **Why**: one stage now corresponds to one routing decision, invalid
  intermediate states cannot leak to the graph, and component-level safety
  metadata remains observable without presenting implementation plumbing as
  user-visible progress.
- **Pinned by**: [`12-agent-workflow.md § Node contracts`](./12-agent-workflow.md#5-node-contracts),
  [`12a-query-classification.md`](./12a-query-classification.md),
  [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates),
  [`91-impl-plan.md § Phase 7`](./91-impl-plan.md#10-phase-7--workflow-convergence-and-bounded-latency)
- **Date**: 2026-07-30

## D34 — One transition contract owns local and LangGraph routing

- **Context**: after composite stages were introduced, local orchestration still
  used Python `if`/`while` branches while LangGraph repeated the same conditions
  in conditional-edge callbacks. A behavior change could update one runtime,
  leave the other stale and still pass component tests.
- **Alternatives considered**: keep parity tests over duplicated branches;
  generate one runtime from the other; define a closed transition table consumed
  by both runtimes while keeping LangGraph as the explicit production graph.
- **Decision**: a shared `WorkflowTransition(next_node, reason)` function owns
  every conditional branch from classification through evidence evaluation.
  Both orchestrators invoke it directly. The local runtime dispatches the
  returned node instead of retaining a second hard-coded conditional order.
  Stage implementations establish terminal state first; the contract validates
  impossible combinations and fail-closes rather than guessing. LangGraph
  remains the explicit graph and maps the same returned node enum onto its
  registered edges. Workflow metrics, trace and failure attribution expose the
  logical node name `execute_tool_plan`, never its storage-specific helper name.
- **Why**: routing policy has one executable source of truth without hiding the
  production graph, while table-driven branch tests become smaller and more
  complete than two sets of hand-maintained conditions.
- **Pinned by**: [`12-agent-workflow.md § Shared transition contract`](./12-agent-workflow.md#31-shared-transition-contract),
  [`20-ui-demo.md`](./20-ui-demo.md),
  [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates),
  [`91-impl-plan.md § Phase 7`](./91-impl-plan.md#10-phase-7--workflow-convergence-and-bounded-latency)
- **Date**: 2026-07-30

## D35 — Parallelism is adapter-capability gated, not inferred

- **Context**: independent retrieval calls can reduce wall-clock latency, but
  PyMilvus client thread safety and three terminal persistence sinks' shared
  failure semantics are not established by the current contracts.
- **Alternatives considered**: always use a thread pool; keep all work
  sequential; parallelize only explicitly safe reads and defer writes until
  adapters return isolated results instead of mutating shared state.
- **Decision**: independent ready retrieval items use at most three workers only
  when the adapter declares `supports_parallel_search=true`, and their outputs
  are applied in plan order. The deterministic in-memory adapter opts in;
  Milvus remains sequential pending a target SDK/client exercise. Conversation
  Memory, Selective Memory and response-cache writes remain sequential because
  explicit-memory failure precedence, shared-client safety and cancellation
  have not been proven.
- **Why**: the demo gains testable parallel-read behavior without silently
  claiming production client safety or creating nondeterministic trace,
  fingerprint and terminal-write outcomes.
- **Pinned by**: [`12-agent-workflow.md § execute_tool_plan`](./12-agent-workflow.md#55-execute_tool_plan),
  [`12-agent-workflow.md § persist_turn_memory`](./12-agent-workflow.md#511-persist_turn_memory),
  [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates),
  [`91-impl-plan.md § Phase 7`](./91-impl-plan.md#10-phase-7--workflow-convergence-and-bounded-latency)
- **Date**: 2026-07-30

## D36 — Atomic focused answers may use one strong direct citation

- **Context**: the global two-relevant-chunk rule rejected an atomic
  `Milvus 3.0 Force Merge` explanation even though retrieval returned the
  exact Force Merge section at score `0.99`. The retry planner then replaced
  the named feature with a generic architecture/S3 template and executed the
  same supplementary query twice.
- **Alternatives considered**: require two chunks for every grounded answer;
  lower the global count to one; replay the prior assistant answer; introduce
  a narrow direct-section exception plus faithful, deduplicated retries.
- **Decision**: focused, non-comparison, single-tool questions may answer from
  exactly one chunk only at score `≥0.80`, with exact normalized section-name
  coverage, one requested aspect family and matching isolated version scope.
  Registered multi-aspect combinations (definition, mechanism,
  operation/configuration, constraints/risks and trade-offs) keep the
  multi-chunk/coverage rule. Grades expose registered evidence bases and
  actionable missing-aspect codes. Supplementary planning preserves original
  product/feature/version terms and rejects a duplicate
  `(tool, normalized query, version scope)` fingerprint before append or
  execution. Allow-listed product-associated bare versions initially map
  `Milvus N.N` to exact stored `vN.N`; generic decimals stay unscoped.
- **Why**: evidence sufficiency follows the question's atomicity rather than an
  arbitrary global count, without weakening comparison/exhaustive safety.
  Retry work remains relevant and convergent, and prior conversation content
  never substitutes for live citation evidence.
- **Pinned by**: [`12-agent-workflow.md § plan_retrieval`](./12-agent-workflow.md#53-plan_retrieval),
  [`12-agent-workflow.md § evaluate_evidence`](./12-agent-workflow.md#57-evaluate_evidence),
  [`70-quality-and-evaluation.md § Correctness gates`](./70-quality-and-evaluation.md#41-correctness-gates),
  [`91-impl-plan.md § Phase 7`](./91-impl-plan.md#10-phase-7--workflow-convergence-and-bounded-latency)
- **Date**: 2026-07-30

## D37 — Hybrid relevance and business ORDER BY are separate modes

- **Context**: 把 scalar ORDER BY 无条件下发到 dense/BM25 search 会把业务字段变成主排序，悄悄改变既有 relevance-first 行为。
- **Decision**: 默认 `relevance` 模式维持 hybrid score 主序；只有显式 `scalar` 模式下发 allow-listed `order_by_fields`，fusion 后再按同一 scalar key 确定性排序。Local adapter 实现同一公开语义。
- **Pinned by**: [`14-milvus-3-native-capabilities.md § ORDER BY`](./14-milvus-3-native-capabilities.md#21-order-by)
- **Date**: 2026-07-31

## D38 — Hybrid facets use bounded Query Aggregation

- **Context**: Milvus Search Aggregation 不能与 Hybrid Search 组合，而 UI facet 语义是 retained candidate pool，不是全 collection。
- **Decision**: hybrid recall 后以候选 `chunk_id` 做 bounded filter，通过一个 Query Aggregation 生成组合 groups，再 marginalize 为现有 per-field counts；local adapter 对同一集合直接计数。
- **Pinned by**: [`14-milvus-3-native-capabilities.md § Facets / aggregation`](./14-milvus-3-native-capabilities.md#22-facets--aggregation)
- **Date**: 2026-07-31

## D39 — Milvus functions own lexical and MinHash output fields

- **Context**: 客户端 token hash sparse vector 和实验性 1-bit MinHash signature 无法利用 Milvus 3.0 analyzer、BM25、DIDO 与原生索引。
- **Decision**: ingestion 只写 deterministic `retrieval_text`/`normalized_text` 输入；BM25 与 MINHASH Function 产生 output vector。BM25 sparse index 默认省略旧算法参数以选择 SINDI；DAAT 只作显式兼容开关。
- **Pinned by**: [`14-milvus-3-native-capabilities.md § BM25 Function and synonyms`](./14-milvus-3-native-capabilities.md#23-bm25-function-and-synonyms), [`14-milvus-3-native-capabilities.md § Server-side MinHash`](./14-milvus-3-native-capabilities.md#4-server-side-minhash-dido)
- **Date**: 2026-07-31

## D40 — Domain time stays epoch-ms while Milvus lifecycle uses TIMESTAMPTZ TTL

- **Context**: workflow clocks、deterministic tests 与 decay math 使用 epoch milliseconds；Milvus 3.0 native entity TTL 要求 `TIMESTAMPTZ ttl_field`。
- **Decision**: domain model 不迁移，唯一 storage codec 在 adapter boundary 转换 canonical UTC timestamp；server TTL 与显式 predicate 双层执行。已有同名 Int64 字段必须受控 recreation，不尝试原地变型。
- **Pinned by**: [`14-milvus-3-native-capabilities.md § Lifecycle contract`](./14-milvus-3-native-capabilities.md#3-lifecycle-contract)
- **Date**: 2026-07-31

## D41 — Reproducible online eval reads a named restored snapshot

- **Context**: 对 live collection 运行 golden eval 会让 concurrent ingestion 改变候选集，结果无法复现。
- **Decision**: online eval all-or-none 绑定 named snapshot 和 deterministic restored target；创建/恢复有界轮询，运行只读 target，并在报告记录非敏感 provenance。默认 local eval 保持无网络。
- **Pinned by**: [`14-milvus-3-native-capabilities.md § Evaluation snapshots`](./14-milvus-3-native-capabilities.md#5-evaluation-snapshots)
- **Date**: 2026-07-31

## D42 — Schema evolution is allow-listed, dry-run-first and revalidated

- **Context**: 任意 alter/backfill CLI 容易误改 field type/dimension，或在部分失败后把 migration 误报完成。
- **Decision**: 只支持新增 nullable sparse/embedding input/output 字段、BM25 physical backfill 与 bounded partial update；禁止 drop/rename/type change，默认 dry-run，apply 后 describe revalidation，失败只输出 registered code。
- **Pinned by**: [`14-milvus-3-native-capabilities.md § Schema evolution and backfill`](./14-milvus-3-native-capabilities.md#6-schema-evolution-and-backfill)
- **Date**: 2026-07-31
