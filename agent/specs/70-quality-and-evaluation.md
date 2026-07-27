# 70 — Quality, Evaluation, and Demo Safety

Status: draft · Owner: workshop author · Applies to: all specs

## 1. Purpose

本文把“demo 能跑”升级为可验证合同：数据质量、检索质量、回答忠实度、流程不变量、复现性和本地 demo 安全边界。性能数字在实际栈尚未验证前只记录基线，不伪造 SLA。

## 2. Test layers

| Layer | Scope | Required examples |
| --- | --- | --- |
| Contract/unit | pure transforms and state transitions | stable ids, entity/domain resolution, version-scope filters, permission gate, plan bounds, retry cap, citation subset |
| Component | adapters, generators and stores | local/MinIO read, schema validation, insert/search round trip, reranker and answer-generator adapters |
| Workflow | LangGraph terminal behavior | direct answer, permission denial, single-tool retrieval, multi-tool comparison, multi-hop supplement, retry exhausted, self-check failure |
| UI | rendered result consistency | no manual metadata controls; three tabs share `query_id`; citations match evidence; fallback/error labels |
| Offline RAG eval | seeded corpus + golden fixtures | retrieval recall, citation correctness, required-fact coverage, abstention |
| Workshop smoke | clean setup path | ingest seed corpus, ask one golden question, receive answer-or-abstain + trace |

Tests follow `AGENTS.md`: deterministic, behavior-oriented, clear names, and no disabled checks.

## 3. Golden dataset contract

`eval/questions.json` and `eval/golden_answers.yaml` are versioned with the sample corpus. Each question includes category, expected sources and optional filters; each golden answer includes required facts and citations.

Minimum fixture set covers:

- local Markdown retrieval;
- MinIO/S3-source retrieval;
- PDF page citation;
- bilingual query rewrite;
- product/game terminology resolution covering `GO按钮`, `跳转按钮` and `领取按钮`, plus one same-spelling cross-domain ambiguity;
- two editions of one logical document, with current-only, exact-version and explicit version-comparison questions;
- automatic tool routing to a policy/product/engineering domain;
- comparison requiring at least two tools;
- multi-hop retrieval where second query depends on first-hop evidence;
- permission denial before retrieval;
- low-evidence retry then success;
- retry exhaustion and abstention;
- reranker fallback;
- one nullable-image-vector record without enabling image retrieval.

## 4. Metrics and gates

### 4.1 Correctness gates

- Retry cap: 100% of workflow tests terminate with `retry_count ≤ 3`.
- Citation validity: 100% of emitted citations resolve to selected context from the same query.
- Fixture integrity: every golden citation exists in seeded `kb_chunks`.
- Error honesty: dependency failures never produce a normal grounded-answer status.
- Idempotency: ingesting the unchanged seed corpus twice leaves the same logical `(doc_id, doc_version, chunk_id)` set.
- LLM grounding: every model-generated citation marker is a subset of selected context; invalid output activates traced fallback.
- Provider isolation: deterministic tests and fallback paths make no external API calls.
- Tool authority: UI never supplies metadata filters; every search filter is produced by a registered tool and intersected with the permission decision.
- Plan bounds: at most three initial subqueries, three supplementary rounds and one registered tool per call.
- Multi-source coverage: comparison answers either cover every planned side or explicitly abstain/report uncovered sides.
- Answer self-check: 100% of grounded terminal answers have `answer_validation.valid=true`.
- Entity resolution: every configured terminology fixture records the expected `entity_id`; unresolved cross-domain collisions request clarification before retrieval.
- Version isolation: current/exact queries have zero cross-version contamination in recalled, selected and cited chunks; explicit comparisons keep citations partitioned and visibly labeled by `doc_version`.
- Streaming order: trace-event sequence is contiguous and query-local; tool/retry paths emit their corresponding events; exactly one `final` terminates the stream.
- Streaming safety: grounded answer deltas occur only after a successful verification event; trace events contain no prompts, document bodies, credentials or raw exception text.
- UI progress: the primary Agent Trace presentation is a readable timeline driven by live events, while raw JSON is available only in a collapsed advanced view.
- Memory isolation: every recall/list/delete result belongs to the active `session_id`; another session and an expired/current-turn record have zero visibility.
- Memory grounding: Memory may resolve a follow-up but never creates a KB citation or turns insufficient KB evidence into `enough_evidence=true`.
- Memory lifecycle: after all answer deltas are consumed, requesting `final` idempotently persists the valid terminal turn; incomplete/cancelled streams write nothing; explicit clear removes only the active session.
- Memory degradation: typed recall/write failures preserve an otherwise valid answer, emit only bounded status/count metadata and never expose raw dependency text.

### 4.2 RAG evaluation

Record at least:

- retrieval Recall@20 against expected sources;
- reranked Recall@8 and selected-context Recall@5;
- citation precision/coverage;
- required-fact coverage;
- abstention correctness for insufficient-evidence questions.
- entity-resolution accuracy and cross-version contamination count (target `0` outside explicit comparisons).

Numeric pass thresholds are set only after Phase 0 establishes a baseline on the curated corpus. The baseline, chosen thresholds and rationale must be committed; silently changing thresholds to make a run pass is forbidden.

### 4.3 Performance

Capture end-to-end, retrieval, rerank, time-to-first-token and generation latency for the documented local hardware profile. Phase 0 publishes median and P95 over a repeatable query set; subsequent milestones must not regress beyond an explicitly accepted budget. Legacy example latency values are illustrative, not targets.

## 5. Min-Max Chunking experiment

Compare at least two min/max/overlap configurations over the same corpus and questions. Select the simplest configuration that improves citation granularity and retrieval/answer metrics without producing excessive empty/near-duplicate chunks. Record:

- chunk-count and token-length distributions;
- retrieval Recall@20 and selected-context Recall@5;
- citation granularity for Markdown sections and PDF pages;
- ingestion time and index size where available.

## 6. External capability verification matrix

Before implementation depends on an external Milvus or OpenAI capability, Phase 0 must run a minimal test against the exact documented version:

| Capability | Evidence required | Fallback if unsupported |
| --- | --- | --- |
| dense + sparse/BM25 hybrid | executable query and stable normalized result shape | use the simplest supported hybrid composition and document it |
| scalar metadata filter | filtered query test | block P0 because source filtering is a core contract |
| ordering with hybrid results | behavior test defining interaction with score | display metadata without claiming retrieval ordering |
| aggregation/grouping | executable aggregation or grouping example | compute clearly labeled UI-only summary from returned candidates |
| nullable `image_vector` | mixed null/non-null insert and query | split image examples or defer panel to P2 |
| conversation TTL | expiration test | implemented explicit `expires_at` filter; native TTL remains an optional comparison |
| MinHash representation | similarity test on known duplicate/near/non-duplicate samples | keep checksum-only dedup or store signatures outside Milvus |
| OpenAI text embedding | fake-client contract plus opt-in 1024-dimension smoke; ingestion/query parity test | deterministic offline provider, clearly labeled and never mixed in one collection |
| OpenAI Responses generation | opt-in configured smoke test with non-empty validated citations | deterministic extractive generator with traced reason code |

## 7. Demo security boundary

MVP has no production auth or ACL, so it is restricted to synthetic/curated sample data and local Workshop use. `get_user_permission` is a deterministic teaching gate that demonstrates ordering and allowed-domain intersection; it must never be described as production authorization. Do not ingest real corporate documents or personal data into Memory. Credentials, including `OPENAI_API_KEY`, come from environment/configuration, never notebooks, traces or source URIs. Logs redact prompts, provider error bodies, document bodies and Memory content by default. Retrieved text and recalled Memory are untrusted prompt data and cannot authorize tools, alter tool filters or create sources. Production authorization, consent and retention remain separate production design work.

## 8. Definition of done

- Relevant tests, formatter and linter pass in the future runnable demo project.
- The documented clean setup and smoke test have been run in the current implementation session before success is claimed.
- No unverified Milvus capability is presented as native functionality.
- All failures include stage/source/query context and preserve the original cause where the language supports it.

## 9. Cross-references

- ← Contracts: [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md), [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`20-ui-demo.md`](./20-ui-demo.md)
- → Gates delivery in: [`90-roadmap.md`](./90-roadmap.md), [`91-impl-plan.md`](./91-impl-plan.md)
- ↔ Source notes: [`archive/Optimize.md`](./archive/Optimize.md), [`archive/UIdemo-collection.md`](./archive/UIdemo-collection.md)
