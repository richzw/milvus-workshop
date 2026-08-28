# 10 — Data Model and Storage Boundaries

Status: draft · Owner: workshop author · Depends on: [`00-prd.md`](./00-prd.md)

## 1. Purpose

本文定义所有下游组件共享的数据契约：八类 Milvus collection、预定义词语实体 catalog、UI session state 与评估 fixtures。它固定语义和不变量；Milvus 3.0 原生实现细节由 [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md) 约束。

## 2. Storage architecture

```text
┌──────────────────── Offline ingestion boundary ─────────────────────┐
│ local files / MinIO                                                 │
│       │ parse + chunk + embed                                       │
│       ├──────────────▶ kb_chunks                                    │
│       │                         └─▶ kb_documents StructArray        │
│       └── optional dedup ─▶ doc_dedup_signatures                    │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────── Online query boundary ────────────────────────┐
│ kb_chunks ────────────────▶ Agent workflow ─▶ Streamlit             │
│ kb_documents ─▶ document/element recall ─────────────┘              │
│ predefined_entities.yaml ─▶ terminology resolution                  │
│                                ├─ query_traces: session_state        │
│ conversation_memory ────────────┴─▶ recall / persist / clear         │
│ grounded_response_cache ──────────▶ verified answer reuse            │
│ memory_events ─▶ selection/decay ─▶ memory_facts ─▶ state projection │
└──────────────────────────────────────────────────────────────────────┘

┌────────────────────── Evaluation boundary ──────────────────────────┐
│ eval/questions.json + eval/golden_answers.yaml                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 3. `kb_chunks` — authoritative knowledge records

### 3.1 Required contract

| Group | Fields | Contract |
| --- | --- | --- |
| Identity | `id`, `doc_id`, `chunk_id`, `parent_id?` | `doc_id` identifies one logical document across editions; `chunk_id` is version-aware, stable and citation-addressable; `id` may be Milvus-generated |
| Source | `record_type`, `source_type`, `source_uri`, `bucket?`, `object_key?` | `source_type` is `local`, `s3`, or future `mfs`; secrets never appear in URI |
| Document | `doc_type`, `title`, `section?`, `page_no?`, `chunk_index` | `page_no` is set for page-addressable PDF evidence |
| Content | `text`, `retrieval_text`, `text_summary?`, `language`, `department` | `text` is the retrieval/citation payload; `retrieval_text` is the BM25 Function input; summary never replaces source text |
| Lifecycle | `updated_at`, `created_at?`, `priority`, `doc_version`, `is_current`, `checksum?` | every chunk carries an explicit document version; timestamps are UTC epoch milliseconds |
| Extension | `metadata?`, `has_image_vector` | parser- or media-specific values live in `metadata` |
| Retrieval | `text_vector`, `sparse_vector`, `image_vector?` | `sparse_vector` is the server-produced BM25 Function output; image vector is optional |

Recommended enum values are intentionally small:

- `record_type`: `text_chunk`, `pdf_page`, `image`, `table`.
- `doc_type`: `markdown`, `pdf`, `text`, `image`, `table`.
- `language`: `zh`, `en`, `mixed`.
- `department`: `engineering`, `product`, `hr`, `security`, `general`.

### 3.2 Invariants

1. `(doc_id, doc_version, chunk_id)` is unique within the demo dataset, and `chunk_id` generation includes `doc_version` so citations never collide across editions.
2. `page_no` is non-null for a citation that claims a PDF page; otherwise the citation uses `chunk_id` and optional `section`.
3. `has_image_vector == true` iff a valid `image_vector` is stored.
4. `source_uri` is display-safe: no credentials, presigned query strings, or local user secrets.
5. Every returned citation carries `chunk_id` and `doc_version` and resolves to exactly one `kb_chunks` record used by that query.
6. Vector dimensions come from the selected embedding model configuration; placeholder values from legacy notes are not binding.
7. `doc_version` is a non-empty opaque label such as `2026.07` or `v3`; code never infers recency from lexical or semantic-version ordering.
8. For each `doc_id`, exactly one ingested edition has `is_current == true`. Every chunk of one `(doc_id, doc_version)` has the same `is_current` value.
9. A normal query selects chunks from at most one `doc_version` per `doc_id`. Cross-version evidence is legal only for an explicit version-comparison plan and remains partitioned by version through generation and citation.
10. Every non-null `image_vector` has exactly 768 finite L2-normalized values
    and `metadata.image_embedding_fingerprint`; one collection never mixes
    deterministic image-byte, DINOv3 model, pooling or normalization spaces.
    Incremental image writes compare their fingerprint with an existing image
    record before any mutation; a mismatch requires collection recreation and
    a full re-ingestion.
11. Image-to-image search accepts only a validated 768-dimensional L2 query
    vector, adds `has_image_vector == true` to the caller's validated filters,
    uses COSINE similarity on `image_vector`, and rejects any returned record
    whose fingerprint differs from the query provider. Text-to-image search
    remains caption/title hybrid retrieval with the same image-only filter;
    DINOv3 is not treated as a cross-modal text encoder.

### 3.3 Search defaults

Initial teaching defaults, configurable rather than schema constants:

| Parameter | Initial value | Meaning |
| --- | ---: | --- |
| `milvus_top_k` | 20 | high-recall candidate pool |
| `reranker_top_k` | 8 | candidates retained after precision ranking |
| `answer_context_top_k` | 5 | maximum primary context chunks |
| `order_by` | `updated_at desc`, then `priority desc` | only if validated and semantically compatible with hybrid ranking |
| version scope | `is_current == true` | default tool-owned filter; an explicit requested version uses exact `doc_version` instead |

Dense + BM25 hybrid search、server-side ordering/facets 与 SINDI sparse index 的固定契约见 [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md)。

## 3a. `kb_documents` — StructArray document retrieval projection

`kb_documents` 是由同一批已验证 `kb_chunks` 构建的可重建检索投影，不是第二份 citation authority。一条 parent entity 表示一个 `(doc_id, doc_version)`，其 `passages` StructArray 按 `chunk_index` 保存该版本的局部证据。经评审的 projection manifest 显式列出适合 parent/passage 模型的 document families；被选中的 family 必须完整投影，未选中的短/简单文档保留 flat-only。投影已在 Phase 9 实现并通过目标 server/SDK probe；activation 仍由完整-build fingerprint/count 和隔离 eval gate 控制。未启用时现有 `kb_chunks` dense+BM25 路径行为不变。

```text
kb_documents entity: one document version
├─ document_key = stable(doc_id, doc_version)
├─ parent metadata / permission / version fields
└─ passages: ARRAY<STRUCT> ordered by chunk_index
   ├─ chunk_id + checksum                 stable evidence identity
   ├─ chunk_index + page_no + section    local position/provenance
   ├─ record_type + language             filter data
   ├─ embedding_list_vector              MAX_SIM* entity search
   └─ element_vector                     COSINE element search
```

### 3a.1 Parent and element schema

| Level | Fields | Contract |
| --- | --- | --- |
| Parent identity | `document_key`, `doc_id`, `doc_version` | `document_key` is the primary key and is deterministic for one document edition |
| Parent source | `source_type`, `source_uri`, `doc_type`, `title`, `department` | values must be invariant across all projected passages; permission/version filters run here before vector release |
| Parent lifecycle | `is_current`, `updated_at`, `priority` | `is_current` must be invariant; `updated_at=max(passages.updated_at)` and `priority=max(passages.priority)` are deterministic document-level rollups |
| Parent vector space | `text_embedding_fingerprint` | copied from chunk metadata and identical for every passage in the parent and the active query provider |
| Projection activation | `projection_fingerprint`, `projection_parent_count`, `projection_passage_count`, `passage_count` | one full-build identity and its expected cardinalities; every parent repeats the same full-corpus counts so a partial/stale projection cannot activate |
| Passage identity | `chunk_id`, `chunk_index`, `checksum` | `chunk_id` is the persistent citation identity; `chunk_index` controls array order; authoritative text is rehydrated from `kb_chunks` |
| Passage provenance | `page_no`, `section`, `record_type`, `language` | `page_no=-1` and `section=""` are explicit non-null sentinels because element-wise nullable values are not part of this contract |
| Passage retrieval | `embedding_list_vector`, `element_vector` | both contain the same validated 1024-d text vector but bind different metric families and indexes |

Initial Workshop schema uses `max_capacity=1024`; passage `chunk_id/checksum/section/record_type/language` use bounded `VARCHAR` maxima `512/128/2048/64/32`, and both vectors use the same dimension/fingerprint as `kb_chunks.text_vector`. Before projection creation, ingestion computes the maximum passage count and encoded scalar sizes. Any entity exceeding a schema bound disables the complete StructArray projection build and reports the offending safe document identity; it never truncates passages, silently splits one logical document into unrelated parents, or publishes a partial corpus.

Parent `document_key/doc_id/doc_version/source_type/source_uri/doc_type/title/department/text_embedding_fingerprint/projection_fingerprint` use `VARCHAR` maxima `69/128/64/32/1024/32/512/64/256/64`. `document_key` is `docv_` plus the full lowercase SHA-256 of the canonical JSON pair `[doc_id, doc_version]`; no delimiter parsing is used to recover identity. `projection_fingerprint` is the SHA-256 of the reviewed projection-manifest version, fixed schema/index recipe, embedding fingerprint, and every ordered `(document_key, chunk_id, checksum)` tuple. The counts are positive `INT32` values and must agree with actual query/read-back cardinalities before activation.

### 3a.2 Projection invariants

1. For every `kb_documents.passages[i]`, exactly one `kb_chunks` row has the same `(doc_id, doc_version, chunk_id, checksum)` and supplies its authoritative text; every manifest-selected `kb_chunks` row with a non-empty checksum appears exactly once in its parent array, and a selected family with any missing checksum is rejected in full.
2. Passage order is strictly ascending by `(chunk_index, chunk_id)`. `offset` is a zero-based execution locator for one built projection; it is not persisted in citations, cache records or eval golden labels.
3. A returned element hit is normalized only after `parent document_key + passages + offset` resolves to a non-empty `chunk_id` and that id passes permission, version and live-checksum validation. Resolution failure fails closed.
4. `embedding_list_vector` and `element_vector` are byte-for-byte equivalent before index encoding and share the `kb_chunks` text embedding fingerprint. One subfield cannot stand in for the other because one vector subfield accepts only one index.
5. StructArray does not own sparse retrieval. Sparse subfields and field functions are unsupported inside the current StructArray contract, so BM25 stays in `kb_chunks`; cross-representation fusion occurs in the application over stable passage identity.
6. Subfield schema changes, vector dimension changes, chunk-config changes and embedding-recipe changes require full `kb_documents` recreation. Additive `kb_chunks` migration does not mutate an existing StructArray subfield schema.
7. Parent permission/source/title/version fields are allowed only when every passage agrees. `updated_at` and `priority` are the documented maximum rollups because current chunk records may carry passage-level dates/priorities. A document with mixed department/permission/source identity remains on the flat `kb_chunks` path until a reviewed parent model exists.

### 3a.3 Fit assessment across current collections

| Candidate | Decision | Reason |
| --- | --- | --- |
| Long document → passages | Use in `kb_documents` | business returns/authorizes a document while relevance and citations depend on local passages |
| Visual page → patches or product → images | Future candidate | natural parent/children shape exists, but the current DINOv3 lab stores one global vector per standalone image and has no patch/multi-image corpus |
| `conversation_memory` / `memory_events` | Do not migrate | turns/events need independent TTL, append/delete identity and chronological access; TIMESTAMPTZ and functions are unavailable as StructArray subfields |
| `memory_facts` | Do not migrate | facts require independent revision, status, expiry, correction and lineage updates rather than whole-parent replacement |
| `grounded_response_cache` | Do not migrate | each answer has its own permission/freshness lifecycle and semantic query vector |
| `doc_dedup_signatures` | Do not migrate | signatures are independently queried and server MINHASH Function cannot target a StructArray subfield |
| Short/simple documents | Keep `kb_chunks` only | one chunk/vector already expresses the retrievable unit; duplicate vectors and multi-vector index cost have no demonstrated benefit |

Search modes, same-element predicates, candidate collapse and probe gates are authoritative in [`14-milvus-3-native-capabilities.md § StructArray`](./14-milvus-3-native-capabilities.md#25-structarray-search-and-filter-contract)。

## 4. `conversation_memory` — session semantic memory

This P2 collection participates in the multi-turn query path through the
bounded interface in [`10b-conversation-memory.md`](./10b-conversation-memory.md).
Its storage contract is:

| Group | Fields |
| --- | --- |
| Identity | `id`, `session_id`, `turn_id` |
| Content | `role`, `content`, `summary?`, `memory_type` |
| Lifecycle | `created_at`, `expires_at?` |
| Retrieval | `content_vector`, `metadata?` |

Invariants:

- Every query is filtered to the current `session_id` before semantic ranking.
- Expired records are never returned. Milvus stores `expires_at` as `TIMESTAMPTZ` and configures it as `ttl_field`; explicit predicates remain for deterministic parity and cleanup eligibility.
- `role` is one of `user`, `assistant`, `system`, `summary`; `memory_type` is one of `short_term`, `session_summary`, `task_state`.
- Upsert replaces one complete `(session_id, turn_id)` record set; list and delete are session-scoped.
- Memory is supplementary user context and never owns a KB citation.

`conversation_memory` is the implemented baseline. It remains readable during the bounded migration in [`10d-selective-agent-memory.md § Incremental rollout`](./10d-selective-agent-memory.md#13-incremental-rollout), but new target-state semantics must not be squeezed into its legacy `memory_type` field.

## 4a. `grounded_response_cache` — verified KB answer reuse

This collection is separate from Conversation Memory and follows the complete contract in [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md). It stores same-session exact/semantic query vectors plus the complete validated answer, citations, evidence version/checksum snapshot, permission-scope hash, `kb_revision`, workflow version and explicit `expires_at`.

The record is cacheable only when every cited `kb_chunks.checksum` is non-empty. Cache recall may find a private candidate before permission, but release requires current authorization and live evidence revalidation. Three-day expiry is an upper bound; any revision/scope/evidence mismatch fails closed to normal RAG.

## 4b. `memory_events` — append-only episode lineage

This P2 target-state collection stores immutable, same-session episode envelopes defined by [`10d-selective-agent-memory.md § MemoryEvent`](./10d-selective-agent-memory.md#41-memoryevent).

| Group | Fields |
| --- | --- |
| Identity | `event_id`, `session_id`, `query_id?`, `turn_id?`, `parent_event_id?`, `branch_id` |
| Event | `event_type`, `content`, `summary?`, `outcome?` |
| Selection | `salience_score`, `selection_reason`, `retention_class`, `decay_profile`, `selector_name`, `selector_model?`, `selector_fallback_reason?` |
| Lifecycle | `event_time`, `expires_at?`, `checksum`, `workflow_version` |
| Trust | `permission_scope_hash` |
| Retrieval | `content_vector` |

Normal correction, promotion and supersession append events; they never update a source event. Authorized erasure is stronger than append-only lineage and may physically remove content/vector while retaining only non-recoverable tombstone metadata.

## 4c. `memory_facts` — versioned consolidated projection

This P2 target-state collection stores typed current interpretations defined by [`10d-selective-agent-memory.md § MemoryFact`](./10d-selective-agent-memory.md#42-memoryfact).

| Group | Fields |
| --- | --- |
| Identity | `memory_id`, `session_id`, `memory_type`, `subject`, `predicate`, `revision` |
| Value | `value`, `confidence`, `status` |
| Lineage | `source_event_ids`, `supersedes_memory_id?` |
| Lifecycle | `valid_from`, `valid_to?`, `last_confirmed_at`, `expires_at?`, `salience_score` |
| Trust | `permission_scope_hash` |
| Retrieval | `content_vector` |

Only `active` facts enter normal Memory context. Superseded and tombstoned facts are unavailable; disputed facts are exposed only as conflicts. Every fact has at least one resolvable same-session source event. These collections never own KB citations and remain separate from `grounded_response_cache`.

Physical cleanup uses primary-key keyset pages, not offsets: eligible
`memory_facts` (`tombstoned` or expired) are processed before expired
`memory_events`. Deletes repeat the validated session, exact-id set and
eligibility predicate. An event referenced by any retained same-session fact is
skipped until that fact is independently eligible, preserving the lineage
contract.

## 4d. `memory_consolidation_journal` — recoverable projection outbox

This session-private collection stores exact validated consolidation plans
before fact/lifecycle writes. The primary key is `operation_id`; scalar fields
are `session_id`, `trigger_event_id`, `status`, `attempts`, `created_at`,
`updated_at` and nullable registered `last_error_code`. `source_event_ids` and
`plan_metadata`, `fact_update_0`, nullable `fact_update_1`, and
`lifecycle_event` are bounded JSON envelopes. Exact vectors are stored as
base64 IEEE-754 float64 in non-indexed `fact_vector_0`, `fact_vector_1`, and `lifecycle_vector` VarChar fields, so
no JSON field approaches Milvus's 65,536-byte cap. It is never a recall or UI
source. Because the target Milvus schema requires a vector field, a constant
two-dimensional `journal_anchor_vector=[1,0]` with `AUTOINDEX/COSINE` exists
only as a collection anchor and is never searched. `(session_id, status,
created_at)` supports bounded pending drains; adapter ordering uses Milvus 3.0
syntax `created_at:asc, operation_id:asc`.
Authorized session deletion removes journal rows before event/fact data.

## 5. `doc_dedup_signatures` — P2 ingestion dedup

This collection supports exact and near-duplicate experiments, not online answering.

| Group | Fields |
| --- | --- |
| Identity/source | `id`, `doc_id`, `chunk_id?`, `source_uri`, `source_type`, `record_level` |
| Signature | `normalized_text`, `checksum`, `minhash_signature` |
| Lifecycle | `created_at`, `metadata?` |

Invariants:

- SHA-256 (or the selected stable checksum) owns exact duplicate detection.
- `record_level=doc` requires null `chunk_id`; `record_level=chunk` requires non-null `chunk_id`.
- Milvus 3.0 server-side MinHash owns the stored signature and LSH index；参数和 local fallback boundary 见 [`14-milvus-3-native-capabilities.md § Server-side MinHash`](./14-milvus-3-native-capabilities.md#4-server-side-minhash-dido)。

## 6. Non-Milvus records

### 6.1 `query_traces`

Stored only in `st.session_state` for the current demo process. A trace is keyed by `query_id` and contains:

- request identity (`query_id`, `session_id`, `user_query`);
- final `answer` or abstain result and `citations`;
- recalled and reranked candidates;
- intent, permission, tool selection, query-transformation strategy/item roles,
  query plan, tool calls, rerank, grade, supplementary retrieval,
  context-compression counts/fallback and answer self-check;
- matched/ambiguous entity ids, catalog version, resolved meanings, and version scope (`current`, `exact`, or `comparison`);
- aggregations and per-stage latency/count metrics.

Trace storage is ephemeral by design and must not be presented as an audit log.

### 6.2 Evaluation fixtures

- `eval/questions.json`: question id, text, category, expected source ids and
  optional expected entity, version-scope, tool, transformation strategy/item
  role, plan and compression-path assertions.
- `eval/golden_answers.yaml`: expected facts and required citations for each question id.
- `eval/chunking_configs.json`: at least three strict Min-Max candidate
  profiles for `chunking-experiment-v2`; no candidate is implicitly the
  production default.
- `eval/chunking_anchors.json`: stable source URI/required-term anchors shared
  across configuration-dependent chunk identities.
- a committed chunking recommendation artifact records corpus/config/fixture,
  embedding, retriever/reranker/generator/grader fingerprints and per-dimension
  results; it contains no prompt, document body or provider credential.

Every `question_id` in one file must exist in the other; required citation ids must exist in the seeded `kb_chunks` dataset.

### 6.3 `predefined_entities.yaml`

The entity catalog lives at `demo/config/predefined_entities.yaml` and is a checked-in, reviewable configuration file rather than a Milvus collection. The file uses the JSON-compatible subset of YAML so the demo can parse it strictly with the Python standard library and add no runtime dependency. Its logical shape is:

```yaml
catalog_version: "1"
entities:
  - entity_id: ui.go_button
    entity: GO按钮
    aliases: [跳转按钮, 领取按钮]
    comment: 表示触发页面跳转或领取动作的按钮
    domains: [product, game]
```

`entity` is the canonical display term and `comment` is the concise meaning supplied to prompts. `aliases` contains equivalent surface forms; `domains` disambiguates the same spelling across industries or query topics. `entity_id` is stable across wording edits. Matching is deterministic and case-normalized where appropriate. If one surface form maps to multiple entries and topic/domain context cannot select exactly one, the resolver records ambiguity and must not silently choose a meaning.

The catalog contains at most 500 entities. `entity_id`, `entity`, every alias, comment and domain are non-empty bounded strings; ids are unique and unknown fields are rejected. Workflow construction loads and validates the catalog once. Missing/malformed files, duplicate ids or malformed entries fail construction with path/context; tests may inject a validated catalog. At most 20 matched entries enter one prompt.

### 6.4 Compatibility and migration

`doc_version/is_current` supersede the first demo schema's nullable `version` field. This is an intentional incompatible demo-schema migration: `demo/scripts/cleanup_milvus.py --confirm-drop-demo-data` drops and verifies removal of only the eight fixed demo collections, collection setup recreates them, then ingestion repopulates every record before version-aware queries run. Running the cleanup script without the confirmation flag is a connection-free preview. Readers reject legacy records missing either field; the demo never mixes old and new record shapes in one collection. `expires_at Int64 → TIMESTAMPTZ ttl_field` follows the same recreation rule；additive sparse/embedding migrations use the bounded flow in spec 14。`kb_documents` is always recreated from the validated `kb_chunks` manifest; it is never incrementally inferred from a partially migrated collection.

Offline fixtures may declare editions in `demo/sample_data/document_versions.json`, keyed by logical source URI with `doc_id`, `doc_version` and `is_current`. A source absent from that manifest receives `doc_version=unversioned` and `is_current=true`; ingestion rejects this fallback if more than one source resolves to the same `doc_id`.

## 7. Error handling and engineering norms

Per `AGENTS.md § Error Handling`, ingestion and retrieval fail fast with contextual errors and never silently drop malformed records. Validation occurs before insert and immediately after retrieval deserialization. Interfaces are explicit and injectable; tests assert behavior and invariants rather than private implementation.

## 8. Cross-references

- ← Depends on: [`00-prd.md § MVP scope`](./00-prd.md#6-mvp-scope)
- → Consumed by: [`11-ingestion.md`](./11-ingestion.md), [`12-agent-workflow.md`](./12-agent-workflow.md), [`14-milvus-3-native-capabilities.md`](./14-milvus-3-native-capabilities.md), [`20-ui-demo.md`](./20-ui-demo.md)
- ↔ Validated by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Source notes: [`archive/UIdemo-collection.md`](./archive/UIdemo-collection.md)
- ↔ Decision: [`99-key-decisions.md § D48`](./99-key-decisions.md#d48--structarray-is-a-derived-document-projection-chunk-identity-remains-authoritative)
