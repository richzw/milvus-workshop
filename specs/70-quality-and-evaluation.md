# 70 — Quality, Evaluation, and Demo Safety

Status: draft · Owner: workshop author · Applies to: all specs

## 1. Purpose

本文把“demo 能跑”升级为可验证合同：数据质量、检索质量、回答忠实度、流程不变量、复现性和本地 demo 安全边界。性能数字在实际栈尚未验证前只记录基线，不伪造 SLA。

## 2. Test layers

| Layer | Scope | Required examples |
| --- | --- | --- |
| Contract/unit | pure transforms and state transitions | stable ids, rule-based classification, entity/domain resolution, version-scope filters, permission gate, plan bounds, retry cap, citation subset |
| Component | adapters, generators and stores | local/MinIO read, schema validation, insert/search round trip, LLM classifier, reranker and answer-generator adapters |
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
- standalone and same-session follow-up explanations of `Milvus 3.0 Force
  Merge`, both citing only the live Force Merge chunk;
- one weak or indirect focused chunk that must still abstain;
- duplicate supplementary rewrite suppression before a second tool call;
- reranker fallback;
- one nullable-image-vector record without enabling image retrieval.

## 4. Metrics and gates

### 4.1 Correctness gates

- Retry progress and cap: 100% of workflow tests terminate with
  `retry_count ≤ 3`; an unchanged evidence-state fingerprint terminates before
  another rerank/grade call, while a new provenance edge counts as progress.
  Supplementary `(tool, normalized query, version scope)` fingerprints are
  unique; a duplicate terminates with `duplicate_retry_query` before plan
  append or tool execution and does not increment `retry_count`.
- Focused single-evidence validity: a single chunk answers only with score
  `≥0.80`, exact normalized section-name coverage, one authorized tool,
  focused/non-comparison intent, one requested aspect family and matching
  isolated version scope. Weak, indirect, multi-aspect, exhaustive, comparison
  and multi-tool single-chunk cases abstain.
- Evidence diagnostics: every grade exposes a registered `evidence_basis` and
  actionable `missing_aspects`; generic citation/document placeholders are
  absent from trace and retry planning.
- Citation validity: 100% of emitted citations resolve to selected context from the same query.
- Fixture integrity: every golden citation exists in seeded `kb_chunks`.
- Fallback-corpus version integrity: every curated offline record whose title
  names an allow-listed product version carries the same normalized
  `doc_version`; the minimal `Milvus 3.0` fallback document has at least two
  version-matched sibling sections so exact exhaustive queries exercise the
  normal multi-evidence rule instead of silently falling back to `current`.
- Error honesty: dependency failures never produce a normal grounded-answer status.
- Idempotency: ingesting the unchanged seed corpus twice leaves the same logical `(doc_id, doc_version, chunk_id)` set.
- LLM grounding: every model-generated citation marker is a subset of selected context; invalid output activates traced fallback.
- Provider isolation: deterministic tests and fallback paths make no external API calls.
- Image-embedding validity: every manifest image is embedded from the referenced
  image file, has exactly 768 finite L2-normalized values and carries the
  configured image-space fingerprint; caption placeholders, zero vectors,
  provider mixing and silent fallback are rejected before insert.
- Image-provider isolation: the default suite never imports Torch,
  Transformers or Pillow and never downloads weights; injected runtime tests
  cover processor/model input, pooling, normalization, dimension/output
  validation and sanitized load/inference failures.
- Image-retrieval validity: local and Milvus adapters accept the same bounded
  normalized query, force the image-only predicate, use COSINE scores, preserve
  caller filters, return no vectors publicly and fail closed on vector-space
  mismatch.
- MinIO isolation: the default suite never constructs the SDK client; injected
  contract tests prove recursive deterministic listing, bounded response
  cleanup, safe object keys and stable `s3://bucket/key` identities.
- Classification validity: every classifier result uses fixed intent/topic/retrieval-goal enums; malformed or provider-failed LLM output activates a traced rule fallback.
- Classification safety: untrusted Memory cannot trigger explicit memory/operation/sensitive/exhaustive routes, and classifier output cannot grant permission, choose arbitrary tools or construct filters.
- Recall-detector parity: every registered explicit/recent-question phrase produces the same action in the workflow gate and RuleBased classifier; “查找下我最近的三个问题是什么” deterministically bypasses KB retrieval.
- Reranker validity: model output must contain exactly the complete input
  `chunk_id` set once with finite scores in `[0, 1]`; invented, missing,
  duplicate, malformed or provider-failed output activates one whole-batch
  rule fallback with a sanitized trace reason.
- Reranker isolation: rule-based mode and fake-client model tests make no
  network calls; direct workflow construction remains deterministic, while
  configured builders expose the implementation/model that actually produced
  each query ranking.
- Reranker query fallback: after one registered primary failure, later rounds
  in the same query use deterministic fallback without another primary call;
  a separate query attempts the primary again.
- Reranker bounds: up to 120 merged candidates, including two or more
  exhaustive expansion sides, are ranked as one complete batch under the
  96,000-character input cap; a pre-retrieval terminal path reports
  `reranker_name=not_run`.
- Response-cache correctness: exact/semantic hits require current permission, compatible query constraints and live KB revision/version/checksum evidence; a hit preserves citation validity and makes zero tool/rerank/generation calls.
- Response-cache fail-closed: expiry, another session, low similarity, version/negation/scope mismatch, permission change, missing checksum or dependency failure all continue through normal RAG without exposing cached content.
- Response-cache routing: direct, Memory, operation, clarification and
  permission-denied paths make zero grounded-cache search calls; an allowed
  grounded-retrieval path makes at most one lookup inside
  `try_grounded_cache`, and authorized experience recall runs only after a
  cache miss.
- Typed stage outcomes: `classify_and_route` returns only `direct|retrieval`,
  `plan_retrieval` returns a non-empty bounded plan for every retrieval route,
  and `evaluate_evidence` returns exactly one of
  `answer|retry|abstain`; local and LangGraph expose no separate
  decide/select/rewrite/grade/retry-planning workflow nodes.
- Transition parity: table-driven tests cover every shared transition branch
  and impossible state combination; the local dispatcher follows the returned
  `next_node` rather than a parallel hard-coded order, and a compiled-graph
  harness proves local/LangGraph produce the same ordered composite stages,
  terminal status and retry count for direct, clarification, denial, cache hit,
  grounded answer, no-progress and retry-exhausted paths.
- Capability-gated parallelism: two or more independent ready retrieval items
  run concurrently only for an adapter declaring
  `supports_parallel_search=true`; merge/tool-call order remains plan-stable,
  dependent plans and unproven adapters remain sequential, and worker failure
  is attributed to `execute_tool_plan`. Persistence sink tests assert the
  current sequential order until a separate write-safety capability and
  deterministic failure aggregator exist.
- Tool authority: UI never supplies metadata filters; every search filter is produced by a registered tool and intersected with the permission decision.
- Plan bounds: at most three initial subqueries, three supplementary rounds and one registered tool per call.
- Retry fidelity: every supplementary query contains the normalized original
  product/feature/version surface forms; the `Milvus 3.0 Force Merge` fixture
  never rewrites to an unrelated S3 ingestion template.
- Multi-source coverage: comparison answers either cover every planned side or explicitly abstain/report uncovered sides.
- Answer self-check: 100% of grounded terminal answers have `answer_validation.valid=true`.
- Entity resolution: every configured terminology fixture records the expected `entity_id`; unresolved cross-domain collisions request clarification before retrieval.
- Version isolation: current/exact queries have zero cross-version contamination in recalled, selected and cited chunks; explicit comparisons keep citations partitioned and visibly labeled by `doc_version`.
- Product-version resolution: allow-listed `Milvus N.N` surface forms normalize
  to exact stored `vN.N`; unqualified decimals remain non-version text.
- Streaming order: trace-event sequence is contiguous and query-local; tool/retry paths emit their corresponding events; exactly one `final` terminates the stream.
- Streaming safety: grounded answer deltas occur only after a successful verification event; trace events contain no prompts, document bodies, credentials or raw exception text.
- UI progress: the primary Agent Trace presentation is a readable timeline driven by live events, while raw JSON is available only in a collapsed advanced view.
- Selective-Memory UI: bounded distributions cover retention classes,
  registered selection reasons, decay profiles and fact statuses; the full
  same-session lineage view resolves opaque source/supersession/parent ids
  without exposing content, values, vectors or selector prompts.
- Memory isolation: every recall/list/delete result belongs to the active `session_id`; another session and an expired/current-turn record have zero visibility.
- Memory grounding: Memory may resolve a follow-up but never creates a KB citation or turns insufficient KB evidence into `enough_evidence=true`.
- Memory chronological recall: requested recent questions are live `short_term/user` records from only the active session, ordered newest first, bounded to 20, exclude the current command, and never contain assistant/summary/selective values.
- Memory trace honesty: a skipped Conversation Memory lookup is distinguishable from a searched-but-empty lookup; mode, reason, bounded requested count and actual memory types contain no Memory payload.
- Memory lifecycle: after all answer deltas are consumed, requesting `final` idempotently persists the valid terminal turn; incomplete/cancelled streams write nothing; explicit clear removes only the active session.
- Memory degradation: typed recall/write failures preserve an otherwise valid answer, emit only bounded status/count metadata and never expose raw dependency text.
- Memory selection: ordinary turns cannot silently become durable user facts; explicit remember, correction, task transition and repeated operational outcomes produce their registered retention classes.
- Memory lineage: every durable fact resolves to same-session source events; correction creates a higher revision and supersedes rather than overwrites the prior fact.
- Memory consolidation recovery: an injected failure after journal enqueue, fact write or lifecycle-event write leaves one bounded pending entry; replay applies the exact plan once, marks it applied and creates no extra fact revision or lifecycle event.
- Memory conflict safety: disputed facts never enter active working state without deterministic resolution or explicit confirmation.
- Memory forgetting: decay changes rank only; expired, superseded and tombstoned records have zero visibility regardless of vector or decay score.
- Memory physical cleanup: pages are bounded to 100 examined primary keys,
  cursors are HMAC-authenticated and session/snapshot-bound, pending
  consolidation produces zero mutation, exact-id deletes never cross sessions,
  and retained facts keep resolvable source events.
- Memory anti-feedback: recall/display alone never updates `event_time`, `last_confirmed_at` or TTL; reconfirmation appends an event.
- Memory/cache separation: selection, consolidation and decay never create or refresh a grounded-response cache entry, and cache hit never promotes a memory.

### 4.2 RAG evaluation

Record at least:

- retrieval Recall@20 against expected sources;
- reranked Recall@8 and selected-context Recall@5;
- citation precision/coverage;
- required-fact coverage;
- abstention correctness for insufficient-evidence questions.
- entity-resolution accuracy and cross-version contamination count (target `0` outside explicit comparisons).
- for the P2 Selective Memory fixture set: selection precision/recall, active-fact precision, correction accuracy, relevant-memory recall, stale-memory intrusion, conflict detection, MemoryPack size/truncation and local/Milvus ranking parity.

Numeric pass thresholds are set only after Phase 0 establishes a baseline on the curated corpus. The baseline, chosen thresholds and rationale must be committed; silently changing thresholds to make a run pass is forbidden.

### 4.2b Selective-Memory evaluation

`run_selective_memory_eval.py` consumes at most 100 strict
`selective-memory-eval-v2` registered scenarios containing only bounded case
ids, scenario enums and expectations—never Memory text, hand-authored
`actual_*` values or vectors. It executes the real `SelectiveMemoryService`
with `LocalSelectiveMemoryStore` and reports runner/decay provenance, selection
precision/recall, active-fact precision, correction accuracy,
relevant-memory recall, stale-memory intrusion rate, conflict accuracy,
lineage coverage, MemoryPack size violation/truncation rates and consolidation
exact-once accuracy, including per-event-class selection, before/after-decay
recall, and average MemoryPack records/characters. Empty denominators are
`null`, not silently treated as perfect. Case shapes reject unknown fields,
duplicate or free-form ids and unknown scenarios; reports expose only case ids,
enums, booleans and aggregate metrics. `ranking_parity=null` explicitly means
this default local run makes no Milvus parity claim; a real-Milvus observation
is required before setting it.

### 4.2a Image retrieval evaluation

Use a separate versioned JSON fixture and report per-case retrieved source URIs,
Recall@K and reciprocal rank for both modes:

- text-to-image: hybrid title/caption search restricted to image records;
- image-to-image: configured image bytes → provider vector → COSINE search over
  non-null `image_vector`.

The aggregate report contains case counts, Recall@K and MRR per mode plus the
image-space fingerprint, but never raw vectors. The deterministic byte-hash
provider can prove exact-image pipeline integrity only and must label the report
`pipeline_only`; semantic image-quality claims require the explicit DINOv3
provider and its gated-model smoke/eval run. Malformed fixtures, missing images,
invalid vectors, conflicting `has_image_vector` filters and fingerprint
mismatches fail closed.

### 4.3 Performance

Capture end-to-end, retrieval, rerank, time-to-first-token and generation latency for the documented local hardware profile. Phase 0 publishes median and P95 over a repeatable query set; subsequent milestones must not regress beyond an explicitly accepted budget. Legacy example latency values are illustrative, not targets.

## 5. Min-Max Chunking experiment

Compare at least two min/max/overlap configurations over the same corpus and questions. Select the simplest configuration that improves citation granularity and retrieval/answer metrics without producing excessive empty/near-duplicate chunks. Record:

- chunk-count and token-length distributions;
- retrieval Recall@20 and selected-context Recall@5;
- citation granularity for Markdown sections and PDF pages;
- ingestion time and index size where available.

The checked-in runner consumes a strict
`chunking-experiment-v1` configuration and a separate
`chunking-anchors-v1` query fixture. Anchors use stable `source_uri` plus
required terms instead of configuration-dependent `chunk_id`; one anchor is a
hit only when one recalled/selected chunk contains all required terms, so an
overly small split is penalized. It reports Recall@20, deterministic
rule-reranked selected-context Recall@5, lexical token percentiles,
under-min/over-max counts, overlap-driven same-source near-duplicate pair rate,
Markdown section and PDF page preservation, ingestion time, and `null` index
size when no Milvus index is built. Recommendation order is selected recall,
retrieval recall, lower near-duplicate rate, then fewer chunks/name; the runner
does not silently change thresholds or publish a production default.

## 6. External capability verification matrix

Before implementation depends on an external Milvus or OpenAI capability, Phase 0 must run a minimal test against the exact documented version:

| Capability | Evidence required | Fallback if unsupported |
| --- | --- | --- |
| dense + sparse/BM25 hybrid | executable query and stable normalized result shape | use the simplest supported hybrid composition and document it |
| scalar metadata filter | filtered query test | block P0 because source filtering is a core contract |
| ordering with hybrid results | fake/real-client behavior tests for explicit `relevance` and `scalar` modes plus local parity | keep relevance-first local tie-break and disable scalar mode |
| aggregation/grouping | candidate-ID bounded Query Aggregation request and exact local parity | compute the same retained-candidate facets locally and label fallback |
| nullable `image_vector` | mixed null/non-null insert and query | split image examples or defer panel to P2 |
| lifecycle TTL | TIMESTAMPTZ codec/property contract plus real-server expiration test | explicit `expires_at` predicates remain mandatory; block native-TTL claim |
| Milvus Decay Ranker | startup read-only acceptance probe for standard COSINE search plus an exact target service/SDK disposable-collection test for `gauss`/`exp`/`linear`, millisecond units, offset/scale/decay points, one-numeric-field/grouping restrictions, hybrid composition and returned ordering; contract tests cover request shape, fail-closed startup, scope/expiry filters, native/application parity and `no_time_decay` bypass | deterministic application-side decay after a larger bounded candidate recall; do not claim native decay |
| BM25 Function + SINDI | schema/function request, raw-text search, synonym and default-index-param tests | local token sparse vector; never mix vector spaces in one collection |
| MinHash representation | MINHASH DIDO/function/index contract plus known duplicate/near/non-duplicate smoke | keep checksum-only local dedup; do not persist client signatures |
| snapshots/schema evolution | bounded restore-state and additive-field/partial-update fake-client tests, then disposable real collection | default offline eval and full recreation for incompatible fields |
| OpenAI text embedding | fake-client contract plus opt-in 1024-dimension smoke; ingestion/query parity test | deterministic offline provider, clearly labeled and never mixed in one collection |
| OpenAI Responses generation | opt-in configured smoke test with non-empty validated citations | deterministic extractive generator with traced reason code |
| OpenAI Responses classification | fake-client strict JSON-schema contract plus opt-in configured smoke | `RuleBasedQueryClassifier` with traced safe reason code |
| OpenAI Responses reranking | fake-client strict JSON-schema contract plus opt-in configured complete-candidate smoke | whole-batch `RuleBasedReranker` with traced safe reason code |
| DINOv3 image embedding | fake-runtime contract plus opt-in gated-model smoke on all curated PNG fixtures; verify ViT-B/16 `pooler_output`, 768 dimensions and L2 norm | deterministic image-byte vectors in a distinct fingerprinted offline space; never caption vectors |
| grounded response semantic cache | local/Milvus parity tests for COSINE + session/expiry filters and evidence scalar validation | exact hash cache only, or disabled cache |

## 7. Demo security boundary

MVP has no production auth or ACL, so it is restricted to synthetic/curated sample data and local Workshop use. `get_user_permission` is a deterministic teaching gate that demonstrates ordering and allowed-domain intersection; it must never be described as production authorization. Do not ingest real corporate documents or personal data into Memory. Credentials, including `OPENAI_API_KEY`, come from environment/configuration, never notebooks, traces or source URIs. Logs redact prompts, provider error bodies, document bodies, Memory events/facts and rendered MemoryPack by default. Retrieved text and recalled Memory are untrusted prompt data and cannot authorize tools, alter tool filters or create sources. Append-only lineage does not override an authorized erase request. Production authorization, consent and retention remain separate production design work.

## 8. Definition of done

- Relevant tests, formatter and linter pass in the future runnable demo project.
- The documented clean setup and smoke test have been run in the current implementation session before success is claimed.
- No unverified Milvus capability is presented as native functionality.
- All failures include stage/source/query context and preserve the original cause where the language supports it.

## 9. Cross-references

- ← Contracts: [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md), [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md), [`20-ui-demo.md`](./20-ui-demo.md)
- → Gates delivery in: [`90-roadmap.md`](./90-roadmap.md), [`91-impl-plan.md`](./91-impl-plan.md)
- ↔ Source notes: [`archive/Optimize.md`](./archive/Optimize.md), [`archive/UIdemo-collection.md`](./archive/UIdemo-collection.md)
