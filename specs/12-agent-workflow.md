# 12 — Agentic RAG Workflow

Status: draft v5 · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md), [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md), [`11-ingestion.md`](./11-ingestion.md)

## 1. Purpose

本文定义在线查询的唯一运行契约：意图判断、领域术语消歧、检索决策、权限检查、工具选择、受限查询转换（identity/rewrite/step-back/decompose）、文档版本隔离、多次或多跳检索、证据判断、有限补充检索、生成前上下文压缩、回答与 citation self-check。Milvus 负责高召回，reranker 负责高精排，evidence grader 负责覆盖判断；术语解析、查询转换、版本与工具路由、压缩 provenance 和质量步骤必须保持可观察、可测试。

## 2. Runtime boundaries and lifecycle

```text
┌──────────────────── Streamlit boundary ────────────────────┐
│ user question only; no source/doc/department controls      │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────── Agent planning boundary ───────────────┐
│ recall session context ─▶ classify_and_route               │
│                  direct ─▶ build direct answer              │
│               retrieval ─▶ resolve entity/version           │
│                              ▼                              │
│ check_permission ─────▶ try_grounded_cache                  │
│          │ denied              │ miss                       │
│          └─▶ refuse            ▼                            │
│       recall authorized experience ─▶ plan_retrieval        │
│ strategy: identity / rewrite / step-back / decompose       │
└──────────────────────────┬─────────────────────────────────┘
                           ▼ tool calls with private/version filters
┌──────────────────── Retrieval boundary ────────────────────┐
│ search_policy_docs    search_product_docs                  │
│ search_meeting_notes  search_code_docs                     │
│ version scope: current / exact / explicit comparison       │
│ flat: dense+BM25   struct: element / EmbeddingList→element │
│          └──────▶ normalized passage evidence ◀────┘       │
│                       │ merge/dedupe                        │
│                       ▼                                    │
│                    reranker                                │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────── Evidence loop ─────────────────────────┐
│ evaluate_evidence: grade + choose typed next action         │
│   ├─ focused + one strong/direct chunk ─┐                  │
│   ├─ complete multi-chunk coverage ─────┴─▶ prepare context│
│   ├─ retry(unique next_plan) ─────────────▶ execute plan   │
│   ├─ duplicate retry fingerprint ─────────▶ abstain        │
│   └─ exhausted / insufficient ────────────▶ abstain        │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────── Answer boundary ───────────────────────┐
│ selected context ─▶ optional compression                   │
│                         └─▶ generation + citation check     │
│ memory-only answer ─────────────▶ memory-grounding check   │
│ valid answer ─▶ persist completed turn ─▶ terminal result  │
└────────────────────────────────────────────────────────────┘
                           │ sanitized progress events
                           ▼
┌──────────────────── Presentation boundary ─────────────────┐
│ ordered trace_event stream ─▶ validated answer_delta       │
│                          └─▶ one immutable final snapshot   │
└────────────────────────────────────────────────────────────┘
```

`retry_count` counts supplementary retrieval rounds after the initial plan and never exceeds `max_retry=3`. One round may contain multiple bounded tool calls. All tool calls share the current `query_id`.

## 3. State contract

```python
class AgentState(TypedDict):
    query_id: str
    session_id: str
    user_query: str
    recalled_memories: list[dict]
    memory_context: str
    memory_pack: dict
    memory_status: str
    memory_written_count: int
    memory_conflict_count: int
    memory_decay_profiles: list[str]
    response_cache_status: str
    response_cache_candidate_count: int
    response_cache_match_type: str | None
    response_cache_similarity: float | None
    response_cache_source_query_id: str | None
    response_cache_fallback_reason: str | None
    response_cache_expires_at: int | None
    intent: str
    query_type: str
    retrieval_goal: str
    classifier_name: str
    classifier_model: str | None
    classification_confidence: float | None
    classification_fallback_reason: str | None
    entity_catalog_version: str
    matched_entities: list[dict]
    ambiguous_entities: list[dict]
    need_retrieval: bool
    retrieval_decision: dict
    permission_decision: dict
    selected_tools: list[str]
    tool_calls: list[dict]
    query_plan: list[dict]
    query_transformation: dict
    rewritten_queries: list[str]
    version_scope: dict
    search_filters: dict
    retrieval_profile: str
    structarray_status: str
    document_candidates: list[dict]
    retrieved_chunks: list[dict]
    reranker_name: str
    reranker_model: str | None
    reranker_fallback_reason: str | None
    reranked_chunks: list[dict]
    enough_evidence: bool
    evidence_grade: dict
    generation_contexts: list[dict]
    context_compression: dict
    retry_count: int
    max_retry: int
    answer: str
    citations: list[dict]
    answer_validation: dict
    metrics: dict
    trace: dict
```

This is a logical shape, not a verified code symbol. `intent` describes the requested action (`conversation`, `private_knowledge`, `comparison`, `operation`, `permission_sensitive`, `memory_write`, `memory_recall`); `query_type` describes the topic (`architecture`, `policy`, `product`, `general`, `unknown`). They are separate because the same topic can require different execution plans.

### 3.1 Shared transition contract

Local generator orchestration and LangGraph conditional edges consume the same
pure, allow-listed `next_transition(completed_node, state,
evidence_action=None) -> WorkflowTransition` contract. A transition contains a
closed `next_node` enum and registered `reason`; arbitrary node names are
invalid.

The local runtime is a node dispatcher: after every conditional node it assigns
the returned `next_node` and dispatches that node. It must not call
`next_transition()` only as a guard and then continue through a separately
hard-coded Python order. LangGraph maps the same returned enum to a registered
edge. Unconditional preparation edges may remain explicit in each runtime, but
neither runtime may duplicate conditional route, terminal or retry policy.

| Completed node | Condition | Next node | Registered reason |
| --- | --- | --- | --- |
| `classify_and_route` | direct | `output_gate` | `direct_route` |
| `classify_and_route` | retrieval | `resolve_terminology` | `retrieval_route` |
| `resolve_terminology` | ambiguous | `output_gate` | `clarification_required` |
| `resolve_terminology` | resolved | `check_permission` | `entities_resolved` |
| `check_permission` | denied | `output_gate` | `permission_denied` |
| `check_permission` | allowed | `try_grounded_cache` | `permission_allowed` |
| `try_grounded_cache` | hit | `output_gate` | `cache_hit` |
| `try_grounded_cache` | miss/stale/error | `recall_authorized_experience` | `cache_miss` |
| `execute_tool_plan` | fingerprint unchanged | `generate_candidate_answer` | `no_progress` |
| `execute_tool_plan` | changed | `rerank_evidence` | `evidence_progress` |
| `evaluate_evidence` | retry | `execute_tool_plan` | `supplementary_retry` |
| `evaluate_evidence` | answer | `prepare_generation_context` | `evidence_ready_for_context` |
| `evaluate_evidence` | abstain | `generate_candidate_answer` | `evidence_terminal` |

`evaluate_evidence` is the last conditional node. `prepare_generation_context`
and `generate_candidate_answer` have no conditional edge: their successors are
the unconditional preparation edges each runtime keeps explicitly, so requesting
a transition for them is a contract error, not a route.

The node that establishes a terminal condition owns the corresponding terminal
state before requesting its transition. The shared contract fail-closes on an
impossible combination such as a direct route with `terminal_status=running`,
a denied permission without `permission_denied`, or an evaluation edge without
an `EvidenceAction`. It does not execute tools or make permission decisions.

### 3.2 Streaming event contract

`stream()` is an ordered execution-event interface, not a post-hoc answer chunker. It yields zero or more sanitized `trace_event` envelopes while nodes complete, then validated `answer_delta` envelopes, and exactly one terminal `final` envelope:

```python
{
    "type": "trace_event",
    "event": {
        "query_id": str,
        "sequence": int,          # starts at 1, strictly increases
        "kind": "stage_completed | tool_completed | retry_scheduled",
        "stage": str,
        "title": str,             # bounded user-facing label
        "summary": str,           # bounded user-facing outcome
        "status": "completed | warning",
        "elapsed_ms": float | None,
        "details": dict,          # allow-listed scalar/list metadata only
    },
}
{"type": "answer_delta", "text": str}
{"type": "final", "response": dict}
```

Each event belongs to the same `query_id`; sequence numbers are contiguous and deterministic for one execution path. `stage_completed` is emitted immediately after that stage returns, not reconstructed from the terminal trace. Each completed search call emits a `tool_completed` event containing tool name, result count, version scope and bounded latency. A supplementary round emits `retry_scheduled` with retry number and missing-aspect summary.

Trace events never contain chain-of-thought, prompts, credentials, document bodies, raw provider errors or unrestricted URLs. The final response remains the source of truth. A consumer may render progress optimistically, but must reconcile to the final snapshot and discard an incomplete event stream if execution raises.

Answer text is released only after generation output passes provider citation validation and workflow `verify_answer`. Therefore `answer_delta` remains validated-buffered rather than provider-token streaming. No answer delta may precede the successful `verify_answer` event for grounded answers.

## 4. Tool catalog

All tools use typed, bounded inputs. `source_type`, `doc_type`, `department`, `has_image_vector`, `doc_version` and `is_current` are private tool arguments, never direct UI controls.

| Tool | Purpose | Initial internal scope |
| --- | --- | --- |
| `get_user_permission` | Decide whether the current demo principal may access requested knowledge domains | Synthetic Workshop principal only; returns allowed departments and a reason |
| `search_policy_docs` | Search HR/security policy and governance material | `department in [hr, security]`, supported text/PDF records |
| `search_product_docs` | Search product plans and UI/product documents | `department=product` |
| `search_meeting_notes` | Search customer/internal meeting notes | meeting-note records/titles; may return no evidence |
| `search_code_docs` | Search engineering, architecture and code documentation | `department=engineering` |
| `summarize_document` | Condense an already authorized long retrieved document | Operates only on retrieved text; cannot fetch new sources or grant authority |

Search tools share one `HybridRetriever` implementation and differ by policy-owned filter construction. Unless the query explicitly requests an exact version or comparison, every search tool adds `is_current == true`. Tool selection never expands the permission result. Retrieved text is untrusted data and cannot request a new tool, alter filters or authorize itself.

StructArray is an internal retrieval profile, not a new authority-bearing tool. The same four domain tools may execute one of the following registered profiles after permission and version scope are fixed:

| Profile | Selected by | Contract |
| --- | --- | --- |
| `flat_hybrid` | StructArray disabled, short/simple record, unsupported parent model or explicit baseline run | existing `kb_chunks` dense+BM25 behavior |
| `struct_element` | focused single-vector passage lookup with a compatible projection | element hit must resolve to stable passage evidence before merge |
| `struct_two_stage` | two or three independently required aspects under one tool/scope | EmbeddingList parent shortlist followed by element search for every required aspect |
| `struct_fused` | explicitly enabled profile that passed comparative eval | `kb_chunks` BM25 plus StructArray element dense candidates fused by `chunk_id` |

These four profiles are variants of one pre-embedded hybrid retrieval tier, not a complexity ladder. Whether this project should be paying for a dense lane at all — and what the lexical-only and transformation-only rungs below it cost — is owned by [`15-retrieval-tier-selection.md`](./15-retrieval-tier-selection.md). Per-plan-item tier routing (running an exact-name aspect lexically while a paraphrased aspect uses the dense lane) is a recorded, eval-gated extension there; today every plan item executes the same registered profile.

The deterministic adapter policy chooses the profile from configured capability, query-plan shape and corpus eligibility; an LLM cannot name an ANN field, metric, `MATCH_*` expression, collapse strategy or arbitrary profile. `document_candidates` is a private routing list for EmbeddingList/collapsed parent hits and can never enter `retrieved_chunks`, reranking, grading or generation until an element resolves to a citeable passage.

The adapter surface is `search_profile(queries, top_k, filters, order_by, element_predicates=()) -> ProfileSearchRun`. `queries` contains one normalized string except for an eligible two-stage group. The immutable result declares configured/effective profile, capability status, one passage-result tuple per input query, and bounded parent-only `DocumentCandidate` records. Each `SearchResult` also declares `retrieval_profile`, `result_granularity="passage"`, `document_key`, `struct_field`, optional `element_offset/parent_rank`, and registered retrieval paths. No mutable “last search” state is allowed because ready tool calls may execute concurrently.

`struct_two_stage` groups only two or three ready calls whose tool, normalized parent filters, version scope and query role (`primary|aspect`) match. Background/hop items, mixed tools, mixed versions and a single query execute effective `struct_element` independently. Every parent filter is attached to each native request and every rehydrated chunk is checked again. This is intentional defense in depth because a top-level native hybrid filter is not the authorization boundary for its individual requests.

The optional image-retrieval lab extends that adapter with
`search_image_vector`. A text query can restrict normal hybrid retrieval to
captioned image records; an uploaded/local image is embedded by the configured
image provider and searched only against `image_vector`. This lab API and its
CLI eval runner do not add an image-upload route to the main Agent Chat
workflow, do not bypass tool-owned permission filters, and do not place image
bytes or vectors in trace/UI payloads.

## 5. Node contracts

Subsection numbers are stable identifiers cited from [`93-improvements-review.md`](./93-improvements-review.md)
and [`99-key-decisions.md`](./99-key-decisions.md); they are not renumbered when a node is added or merged.
Letter suffixes mark later insertions and the gap at `5.4` marks the node pair
that D33 merged into `plan_retrieval`.

### 5.0 `recall_memory`

Before classification, the shared deterministic recall detector chooses exactly one Conversation Memory mode:

- `chronological`: a recent-question request lists the effective `1..20` live `short_term/user` records for the active `session_id`, ordered by `(created_at, turn_id)` descending;
- `semantic`: another explicit recall request or bounded referential follow-up searches at most `MEMORY_TOP_K` live `session_summary`/`task_state` records;
- `none`: the Conversation Memory store is not queried.

Chronological mode is authoritative for temporal language such as “查找下我最近的三个问题是什么”; it does not run ANN search or Selective Memory episode ranking. This stage never queries the grounded-response cache. Typed store failures record sanitized component status and continue safely.

Trace reports `recall_decision`, `recall_mode`, registered `recall_reason`, bounded `requested_count` and the actual `memory_types`. A skipped lookup is distinct from a searched-but-empty result.

This paragraph is the implemented [`10b-conversation-memory.md`](./10b-conversation-memory.md) baseline. The [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md) cutover replaces summary-only recall with a Memory Router that deterministically loads same-session working state and private context. Permission-scoped reusable experiences are loaded later by `recall_authorized_experience`. Both return a typed `MemoryPack`; response-cache candidates remain a separate private state field owned only by `try_grounded_cache`. Status, session, expiry and permission predicates run before decay ranking. A current event is never visible to its own recall, and a recall hit alone never refreshes lifecycle timestamps.

### 5.1 `classify_and_route`

Delegates classification to the injected
[`QueryClassifier`](./12a-query-classification.md), records validated `intent`,
`query_type`, `retrieval_goal` and safe implementation/fallback metadata, then
returns one typed `QueryRouteResult(route=direct|retrieval, reason)`. Direct
routes build the bounded direct/Memory/refusal answer in the same workflow
stage and skip entity resolution, permission and cache. Retrieval routes
continue to entity/version resolution. `AgenticRAGWorkflow()` uses
`RuleBasedQueryClassifier`; configured builders use `LLMQueryClassifier`
wrapped by `FallbackQueryClassifier`. Explicit memory-write, memory-recall and
operation requests take the deterministic safety fast path. Private knowledge,
comparison and permission-sensitive questions retrieve by default. Bounded
recalled summaries may clarify a follow-up topic but never grant permission,
change an explicit action or establish KB evidence.

### 5.1a `resolve_terminology`

Matches the original question against [`predefined_entities.yaml`](./10-data-model.md#63-predefined_entitiesyaml), then uses `query_type` and the surrounding question context to resolve industry-specific meanings. It records `entity_id`, matched surface form, canonical `entity`, `comment`, domains and resolution status. No match leaves the query unchanged. An unresolved collision between domains terminates with a structured clarification request before retrieval rather than guessing.

Only matched, bounded entries are added to the terminology-resolution/rewrite prompt:

```text
Here are some word entity definitions to help interpret and rewrite the query.
<entity_info>
{entities}
</entity_info>
```

`{entities}` is serialized structured data from the trusted catalog, for example `{"entity": "GO按钮", "comment": "表示触发页面跳转或领取动作的按钮"}`. Entity comments guide query interpretation; they are not retrieved evidence and cannot by themselves support an answer or citation.

### 5.2 `check_permission`

Runs before any private search tool. The default Workshop implementation authorizes only the synthetic corpus and returns allowed departments. A denied decision terminates without retrieval or generation. This node demonstrates placement and data flow; it is not production authentication or ACL.

### 5.2a `try_grounded_cache`

Runs only for a grounded-retrieval route after current permission succeeds. It
performs one bounded same-session exact/semantic candidate lookup and validates
classification/entity/version constraints, permission-scope hash, TTL,
KB/workflow revisions, and every cited chunk's live
version/checksum/current status in the same stage. A hit restores the complete
answer and citations, sets `answered_from_cache/cached_grounded`, then
terminates without experience recall, tool, retrieval, rerank or generation
calls. Any miss, stale candidate or typed cache failure continues to
`recall_authorized_experience`. Direct, Memory, operation, clarification and
permission-denied routes never invoke this stage.

### 5.2b `recall_authorized_experience`

Runs only after an allowed grounded-retrieval request misses
`try_grounded_cache`. It recalls bounded reusable episodes/facts under the
current permission-scope hash and merges them into private planning context.
Failure is sanitized and degrades to retrieval without experience context. This
stage cannot answer, grant permission, alter filters or become citation
evidence.

### 5.3 `plan_retrieval`

Combines tool selection and bounded query transformation into one typed
`RetrievalPlanResult(selected_tools, transformation, plan_count)`. It chooses
the smallest relevant set of tools from topic and intent, then selects exactly
one primary strategy from the closed enum `identity | rewrite | step_back |
decompose` and produces one to three executable plan items. A simple question
normally selects one search tool; a comparison selects two or more. Selection
records tool name, reason and intended knowledge domain. The Agent never
searches every domain by default, and no transformation may add a tool that
selection did not authorize.

The initial rule-owned strategy policy is:

| Strategy | Use when | Output contract |
| --- | --- | --- |
| `identity` | exact named feature, quoted term or already search-shaped query | one item retaining the normalized original query |
| `rewrite` | colloquial/vague wording, resolved alias or bounded referential follow-up | one clearer query preserving every named product, feature, version, negation and constraint |
| `step_back` | a mechanism/why question needs broader principle or architecture background | one `background` query plus one original-retaining `primary` query under the same authorized tool/version scope |
| `decompose` | explicit comparison, multiple independently answerable aspects or multi-hop dependency | two or three items; every item declares parallel execution or dependency ids |

The user-proposed two-to-four decomposition is deliberately narrowed to two or
three because D14's end-to-end plan cap is already a termination and cost
contract. Transformation is not recursive: a plan cannot silently step back,
then decompose again. Terminology normalization may be applied inside the
chosen strategy but does not count as another strategy.

The transformer produces plan items containing `subquery_id`,
`query_role=primary|background|aspect|hop`, transformed query, selected tool,
version scope, dependency ids, registered retrieval profile and status. Terminology expansion preserves
original intent and incorporates only the resolver's matched entities; each
query retains the original surface form or canonical term so exact product
vocabulary is not lost. Comparison questions produce parallel subqueries;
multi-hop questions may leave a dependent subquery whose text is refined from
first-hop evidence.

A step-back background item can add explanatory context, but it cannot alone
establish coverage for the concrete named feature, requested version or
operation. `evaluate_evidence` must still find primary evidence matching the
original question; background-only recall ends in retry or abstention. All
derived items inherit the current permission decision, selected-tool allow
list, entity ids, explicit numeric/version constraints and version scope.

Every supplementary rewrite starts with the bounded original user query and
preserves its named product, feature and version surface forms. It may append
registered missing-aspect codes, resolved canonical terms and the highest
relevant chunk's bounded title/section as retrieval hints. Topic-wide templates
that replace the original subject—such as rewriting every architecture query
to an S3 ingestion query—are forbidden.

Before a supplementary item is appended, planning computes a deterministic
`retry_plan_fingerprint` over its registered tool, NFKC/case-folded
whitespace-normalized query and canonical JSON version scope. A fingerprint
already present in a completed or pending supplementary item is
`duplicate_retry_query`: no plan item or tool call is created, `retry_count`
does not advance, and `evaluate_evidence` returns a terminal abstention. The
fingerprint includes tool and version scope so the same terms remain legal
across an intentional cross-tool or cross-version plan.

For a normal follow-up, the rewritten query may include bounded recalled summaries in the baseline or the target `MemoryPack.rendered_context` to resolve pronouns or omitted topic words. The raw rewrite remains private trace data; Memory does not add a new source or permission domain.

An optional strict-output transformer may propose only the strategy enum and
bounded query strings. It performs at most one provider request per planning
invocation; local validation rejects unknown tools, too many items, empty or
duplicate queries, lost protected surface forms, widened version/permission
scope and invalid dependency graphs. Provider/configuration/output failure uses
the deterministic `identity` or rule rewrite result with a sanitized reason.
The default deterministic workflow performs no external transformation call.

Configuration is explicit:

| Variable | Values / default | Contract |
| --- | --- | --- |
| `QUERY_TRANSFORMER` | `rule_based` (default), `auto`, `openai` | `rule_based` performs no network I/O; `auto` uses the model only when key/model exist; explicit `openai` fails startup when they do not |
| `OPENAI_QUERY_TRANSFORMER_MODEL` | empty | may reuse a non-empty `OPENAI_MODEL` |
| `OPENAI_QUERY_TRANSFORMER_TIMEOUT_SECONDS` | `10` | finite and positive; SDK retries remain disabled at this seam |

### 5.5 `execute_tool_plan`

Executes ready search plan items, potentially multiple calls in one round, then merges candidates by `chunk_id`. The highest-scoring occurrence owns ranking fields while tool/query provenance is retained in `tool_calls`. Calls are bounded by three initial subqueries and `milvus_top_k` per call.

Every adapter result declares `result_granularity=passage|document`. Flat chunks and successfully resolved StructArray elements are `passage`; EmbeddingList or native-collapse hits are `document`. `document` hits are stored only in `document_candidates`. In `struct_two_stage`, the parent shortlist must be followed by element searches constrained to the same authorized `document_key` set, and only those element results join the passage pool. A missing second-stage hit leaves that aspect uncovered.

`struct_fused` keeps BM25 in `kb_chunks`, performs dense element ANN in `kb_documents`, normalizes both lanes to `(doc_id, doc_version, chunk_id)`, rejects identity/checksum disagreement, and applies the configured deterministic fusion before the existing reranker. An identical passage from two lanes is one evidence candidate with both provenance edges. Element offsets and parent ranks remain trace diagnostics; they do not participate in merge identity or citation.

`execute_tool_plan` is also the canonical observable stage name in latency
metrics, streaming trace events and dependency-failure attribution. Storage
engine or retrieval implementation names such as
`milvus_hybrid_retrieve` remain internal operation names and never appear as
workflow stages.

Ready plan items with no dependency edge may execute concurrently only when the
retrieval adapter explicitly declares `supports_parallel_search=true`.
Concurrency is bounded by both the ready-item count and the initial-plan cap of
three. Results, tool calls, expansion and provenance are always applied in
original plan order, so scheduling cannot change fingerprints, trace order or
answer selection. Adapters without the capability—including the production
Milvus adapter until its shared-client thread safety is verified—execute
sequentially. Dependent hops always wait for their prerequisite round.

After every merged round the workflow computes a deterministic
`candidate_pool_fingerprint` over the evidence state that can affect grading:
each retained chunk's `chunk_id`, `doc_version`, checksum and sorted registered
tool/retrieval-path provenance, plus the bounded expanded-document chunk ids. Parent-only document candidates, rank, offset and
provider score are deliberately excluded. If a supplementary round produces
the same fingerprint as the preceding evaluated round, it has added no new
evidence or coverage. The workflow records `no_progress`, skips reranking and
grading that unchanged pool, and terminates as an abstention even when the
numeric retry cap has not been reached. The initial round can never be
classified as no-progress.

Queries that explicitly request an exhaustive list, such as “有哪些” or “list all”, use bounded document expansion after the initial semantic seed search. The expansion performs an exact scalar lookup for the seed's `(doc_id, doc_version)`, reapplies the originating tool's source, document-type, department and version filters, and returns siblings in `chunk_index` order. Expansion is capped at `milvus_top_k`, recorded separately from ANN recall in trace, and cannot broaden permission or version scope.

Version scope is part of every plan item and tool call:

- `current` (default): filter `is_current == true`;
- `exact`: when the user names a version, filter exact `doc_version` and never fall back to current if it is absent;
- `comparison`: only when the user explicitly asks to compare versions; execute one exact/current scope per side and keep candidates partitioned by `(doc_id, doc_version)`.

Normal merge, rerank and selection reject multiple versions of the same `doc_id`. A comparison plan may retain them, but version labels and provenance must remain attached through answer generation.

The deterministic MVP recognizes exact version tokens shaped as `vN`/`vN.N` or
`YYYY.MM`, case-insensitively. It also recognizes allow-listed
product-associated bare semantic versions; initially `Milvus 3.0` normalizes to
the stored `v3.0`. A free-standing decimal such as `3.0` is not a version
because it may be a metric or value. `current`/`latest`/`当前` select the current
edition. One explicit token selects `exact`; two explicit sides combined with
comparison intent select `comparison`. Unknown exact versions return no
evidence and never fall back. More than two distinct requested versions, or
comparison wording without two resolvable sides, returns a clarification
request instead of broadening scope.

For multi-hop retrieval, a later plan item may depend on facts extracted from earlier evidence. Example:

```text
customer meeting notes ─▶ extract frequent concerns
                         └▶ query product roadmap for those concerns
                             └▶ compare covered vs uncovered
```

### 5.6 `rerank_evidence`

Reranks the complete merged candidate set against the original user question before applying `reranker_top_k`. The configured main implementation sends one bounded request to the OpenAI Responses API using strict JSON-schema output. The request contains the bounded question and, for every candidate, only its stable `chunk_id`, title, section and truncated text.

Candidates from the initial plan, every supplementary round and document expansion
are merged by `chunk_id` before ranking. The reranker additionally validates a
configured hard input bound of 120 unique candidates and fails closed above it
rather than ranking the pool in parts; that bound is a separate safety limit, not
a sum derived from the plan limits. The serialized model input is additionally
capped at 96,000 characters by dividing the remaining text budget across the
complete batch. Candidate text is untrusted data and cannot change the output contract.

The model must return exactly one `(chunk_id, relevance_score)` item for every input candidate. Local validation rejects a missing, duplicated or unknown id, non-finite score, score outside `[0, 1]`, malformed JSON or provider failure. A valid batch is ordered by descending score with original recall rank as the deterministic tie-breaker, then converted to stable `old_rank`, `rerank_score` and selection status. The workflow never mixes a partial model ranking with rule scores.

`RERANKER=rule_based|auto|openai` controls the configured builder and defaults to `auto`. `auto` uses OpenAI only when both an API key and `OPENAI_RERANKER_MODEL` (or the shared `OPENAI_MODEL`) are configured; otherwise it executes the rule fallback with `fallback_reason=not_configured`. Explicit `openai` mode fails configuration when required values are absent. Timeout, connection, authentication, rate-limit, provider and invalid-output failures execute the deterministic rule reranker once with a bounded reason code. Direct `AgenticRAGWorkflow()` construction remains rule-based for offline reproducibility.

The per-query trace records the implementation that actually produced the ranking, configured model when applicable, whether fallback was active and its sanitized reason. Queries that terminate before ranking use `reranker_name=not_run`, never the configured wrapper name. It never contains provider error text, prompt text or candidate bodies. Exhaustive document expansion reserves output capacity for every bounded sibling; focused questions retain the normal output cap.

Provider fallback is sticky only inside one query execution. Once the
configured fallback wrapper reports a registered fallback reason, later
supplementary rounds for that query invoke its deterministic whole-batch
fallback directly instead of retrying the same unavailable provider. A new
query starts with a fresh primary attempt. Trace records bounded
`primary_attempt_count`, `fallback_only_count` and the registered sticky reason;
it never exposes the provider error.

### 5.7 `evaluate_evidence`

Combines grading and retry planning into one typed
`EvidenceEvaluation(action=answer|retry|abstain, reason)`. The grader still
records `enough_evidence`, covered/missing aspects, contradictions and
suggested tool/query. For comparisons, evidence must cover every required side;
one-side-only evidence is insufficient. When evidence is insufficient and
budget remains, the same stage first proves the next plan fingerprint is
unique, then appends only that needed item, increments the retry count and
preserves prior evidence. At the cap or on a duplicate it returns `abstain`;
sufficient evidence returns `answer`.

The normal evidence rule requires at least two relevant chunks and complete
tool/version coverage. One exception supports atomic feature explanations:
exactly one relevant chunk is sufficient only when all of these hold:

- `retrieval_goal=focused`, intent is not `comparison`, and exactly one
  authorized tool is selected;
- the question requests only one registered aspect family; two or more of
  definition, mechanism, operation/configuration, constraints/risks and
  trade-offs make it a multi-aspect question and disable this exception;
- its rerank score is at least the threshold declared by the reranker that
  actually produced the ranking. Rerank scores are not comparable across
  implementations, so the gate is a per-implementation
  `strong_single_evidence_threshold` rather than one shared constant; both
  shipped rerankers declare `0.80` and a reranker that declares no finite
  value in `[0, 1]` fails closed;
- the question contains the chunk's non-empty section name after
  case-insensitive NFKC/whitespace normalization, establishing direct named
  feature coverage;
- the chunk satisfies the resolved current/exact version scope and normal
  version isolation found no conflicting edition.

This path records `evidence_basis=single_strong_chunk` and selects exactly that
chunk, so generation and verification still require a live citation.
Exhaustive, comparison, multi-tool and indirect/weak single-chunk questions
never use this exception.

Retrieval granularity does not weaken this rule. A StructArray offset is useful provenance but is not evidence identity; an EmbeddingList parent score, MATCH-qualified parent or collapsed document score contributes zero covered aspects until at least one live element is resolved to `chunk_id/text/checksum` and passes the normal rerank/grade contract. A multi-aspect `struct_two_stage` plan must resolve citeable elements for every required aspect, not merely return one parent with a high MaxSim score.

`missing_aspects` contains only registered, actionable codes derived from the
actual state: `no_relevant_evidence`, `single_weak_chunk`,
`single_indirect_chunk`, `multi_aspect_requires_coverage`,
`incomplete_multi_evidence`,
`incomplete_exhaustive_coverage`, `tool:<name>` or
`version:<scope>`. Generic placeholders such as `specific document terms` and
`additional citations` are forbidden. `evidence_basis` is one of
`single_strong_chunk`, `multi_chunk_coverage` or
`insufficient_evidence`.

`max_retry=3` is an upper bound, not a required number of attempts. A retry
must be both bounded and capable of changing the evidence-state fingerprint.
No-progress or duplicate-retry detection may terminate earlier, and the trace
distinguishes `retry_exhausted`, `no_progress` and
`duplicate_retry_query`.

Conversation Memory recall gating is unchanged. A prior assistant answer,
Memory value or non-equivalent response-cache candidate cannot become citation
evidence. Future planners may use previously validated citation lineage only
as a permission-checked retrieval hint; the current question must still fetch,
rerank, select and verify live KB chunks. Therefore a self-contained follow-up
such as “介绍下 Milvus 3.0 Force Merge” must also succeed in a fresh session.

### 5.8 `prepare_generation_context`

This stage runs only after `evaluate_evidence` returns `answer`. It treats the
selected original chunks and their citation map as immutable evidence, then
either passes them through or creates a smaller generation projection. It
never runs before rerank/grading, cannot change `enough_evidence`, and is not a
search tool.

`CONTEXT_COMPRESSION_MODE` accepts `disabled | auto | selective | summary |
extraction` and defaults to `disabled`. `auto` selects `selective` only when the
selected original text exceeds the configured character trigger and an OpenAI
key/model exist; otherwise it is an identity pass-through with a registered
`below_trigger` or `not_configured` reason. Explicit model-backed modes require
configured provider/model/timeout values and fail startup when these are
missing. The default offline workflow and tests perform no network I/O.

| Variable | Default | Validation |
| --- | ---: | --- |
| `CONTEXT_COMPRESSION_MODE` | `disabled` | closed enum above |
| `CONTEXT_COMPRESSION_TRIGGER_CHARS` | `12000` | integer `1000..20000` |
| `CONTEXT_COMPRESSION_MAX_OUTPUT_CHARS` | `12000` | integer `1000..20000`, no greater than the generation input cap |
| `OPENAI_CONTEXT_COMPRESSOR_MODEL` | empty | may reuse a non-empty `OPENAI_MODEL` |
| `OPENAI_CONTEXT_COMPRESSOR_TIMEOUT_SECONDS` | `15` | finite and positive; SDK retries disabled at this seam |

The logical output for each retained source is:

```python
class GenerationContextProjection(TypedDict):
    chunk_id: str
    compression_mode: str
    prompt_text: str
    support_spans: list[dict]  # start, end, exact quote in original text
    source_text_checksum: str
```

The modes have different trust contracts:

- `selective` returns exact sentences/paragraphs from the original chunk in
  original order. Every span is verified by offset and exact substring match.
- `summary` proposes a query-focused summary plus exact supporting spans. The
  derived summary is discarded after validation and serves only to select
  support; generation receives the ordered exact support spans.
- `extraction` proposes bounded facts linked to exact support spans. Derived
  fact text is likewise discarded; a unit without exact support is rejected.

One query performs at most one bounded compression provider request over the
complete selected set, not one unbounded call per chunk. Output uses strict
structured ids and is validated against original chunks, required coverage
aspects and every comparison/version side. Unknown/dropped required ids,
invented text, reordered selective spans, empty support, oversized output,
provider failure or validation failure falls back to the original selected
contexts for the whole query. Partial model output is never mixed with source
contexts. Trace records mode, implementation, before/after character counts,
retained source count and a sanitized fallback reason, never text or spans.
The deterministic boundary derives bounded technical terms and CJK n-grams
from the original question plus resolved entity names, keeps only terms that
actually occur in the selected originals, and requires their combined presence
in the exact retained spans and in each corresponding source projection where
they originally occur. This per-source rule prevents one comparison/version
side from compensating for evidence lost on another side. Losing one term is a
whole-query `invalid_model_output` fallback.

### 5.9 `generate_candidate_answer`

Delegates at most five generation-context projections for focused questions,
or at most sixteen bounded siblings for an exhaustive document query, plus
their immutable source citation map, resolved entity info and validated
version scope to the answer generator defined in
[`13-llm-answer-generation.md`](./13-llm-answer-generation.md).
`generate_candidate_answer` is the registered transition-target name; its
observable stage, latency metric and trace event use
`generate_answer_streaming`, because the stage releases validated deltas rather
than returning one buffered string. Both names refer to this single node; no
other alias is registered. The shared
character cap still applies. The answer must distinguish supported conclusions,
uncovered comparison items and missing evidence; explicit version comparisons
label each conclusion with its source version.

### 5.10 `verify_answer`

Runs after generation and before answer chunks become terminal output. It verifies that every structured citation belongs to selected context, every inline marker resolves, version scope obeys the current/exact/comparison policy, at least one citation supports a grounded answer, and an abstention does not claim unsupported specifics. It records a structured `answer_validation` result without chain-of-thought.

For `answered_from_memory`, verification requires at least one recalled live record, no KB citations and an answer constructed only from bounded Memory values. In chronological mode, every answer item must correspond to a recalled `short_term/user` record and preserve most-recent-first order; assistant content, summaries and selective facts are forbidden. For `memory_write`, verification requires a non-empty remembered statement and no citation.

### 5.11 `persist_turn_memory`

After answer verification (or another valid direct terminal outcome), after all answer deltas are consumed and while producing `final`, write the bounded user turn, assistant turn and deterministic per-turn summary under the active `(session_id, query_id)`. Explicit remember intent adds one `task_state`. Idempotent upsert prevents retries/reruns from duplicating a turn. A typed write failure sets `write_failed`; the final snapshot drives a safe UI warning and preserves an otherwise valid answer. For explicit `memory_write`, the response remains non-committal until this final status and uses `memory_write_failed` when persistence fails.

Conversation Memory, Selective Memory and grounded-response cache are three
logical sinks, but the current implementation persists them sequentially.
Conversation Memory failure may change the explicit-memory terminal outcome,
and deployed sinks may share one Milvus client; no current adapter contract
proves thread-safe writes, deterministic failure precedence and cancellation.
Parallel sink writes remain disabled until all three capabilities are explicit
and a result-aggregation contract replaces direct shared-state mutation.

## 6. Invariants

1. The graph reaches direct answer, grounded answer, clarification request, permission denial, safe operation refusal or abstain in bounded steps.
2. No private search occurs before an allowed `permission_decision`.
3. `1 ≤ initial subqueries ≤ 3`, `0 ≤ retry_count ≤ max_retry == 3`.
4. A supplementary round whose evidence-state fingerprint is unchanged skips
   rerank/grade and terminates safely; provenance changes prevent a false
   no-progress result.
5. Every search call names one registered tool and uses only that tool's policy-owned filters intersected with allowed departments.
6. Comparison answers require evidence for every planned side or explicitly identify uncovered sides.
7. Supplementary retrieval preserves earlier evidence and tool provenance.
8. Every citation points to selected context from the current `query_id`.
9. The answer is not terminal until `answer_validation.valid == true`.
10. Errors include stage and query context; secrets, raw prompts and document bodies stay out of trace.
11. Every rewrite is attributable to zero or more catalog entity ids; an ambiguous entity is never silently resolved.
12. A non-comparison answer contains at most one `doc_version` for each `doc_id`; an explicit comparison preserves and displays the version of every selected chunk and citation.
13. Streaming emits strictly ordered, query-local sanitized events; grounded `answer_delta` events occur only after successful answer verification and precede exactly one `final`.
14. Memory recall/write/list/delete never crosses `session_id`; expired records and the current turn cannot enter recall.
15. Memory context may affect classification and query rewrite but cannot satisfy KB evidence grading, create a citation or bypass permission.
16. Validated answer deltas are streamed before persistence. `persist_turn_memory`
    runs only while the consumer requests the terminal envelope; it either
    completes or records a visible degraded status, and `final` exposes that
    status. A cancelled or incomplete stream never writes the current turn.
17. Query classification always returns validated fixed enums; provider/output failure activates a traced rule fallback, and classifier output cannot grant permission or construct filters. LLM-only `conversation` cannot disable retrieval without deterministic rule support.
18. A response-cache candidate never becomes observable before current permission and evidence freshness validation; a cache hit preserves citations and skips all expensive RAG stages.
19. Reranking consumes the complete bounded recall pool. A model result is accepted only as a complete permutation of input chunk ids with finite `[0, 1]` scores; every provider/output failure falls back for the whole batch and is labeled in trace.
20. Recent-question recall uses a deterministic chronological user-turn listing, never KB tool selection, ANN similarity or Selective Memory decay; its current command is not self-recalled.
21. A reranker provider failure is sticky only for later rounds of the same
    query; it never suppresses the primary attempt of a new query.
22. Query transformation chooses one registered primary strategy, produces at
    most three executable items, preserves protected original terms and cannot
    widen tool, permission, entity or version scope.
23. Step-back background evidence cannot by itself satisfy a concrete primary
    aspect; a plan always retains an original-question primary item.
24. Context compression occurs only after evidence sufficiency, preserves the
    selected source/citation identities and required version sides, and falls
    back atomically to original contexts on any provider or provenance failure.
25. StructArray parent, EmbeddingList and collapsed hits never enter evidence grading or citation directly; only permission/version/checksum-validated element hits normalized to stable `chunk_id` may do so.
26. Element offset is query-local provenance. Merge, cache, fixture and citation identities remain `(doc_id, doc_version, chunk_id)`.
27. Sparse/BM25 retrieval remains on `kb_chunks`; a StructArray profile cannot claim element-level BM25 or silently omit the lexical lane when the selected profile requires it.

## 7. Observability

Trace shows, in order:

- memory recall status/count/decision/mode/reason/requested-count/actual-types without Memory content;
- intent/topic/retrieval-goal classification, classifier/model/fallback metadata, matched/ambiguous entities, catalog version and retrieval decision;
- permission decision without credentials or identity secrets;
- selected tools and reasons;
- query plan, dependencies and each rewrite/retry round;
- query-transformation strategy, item roles and implementation/fallback metadata;
- each tool call's safe filters, version scope, result count and latency;
- retrieval profile, StructArray capability status, query/result granularity, parent-shortlist count, resolved-element count, safe StructArray field name and bounded offsets; no vectors or full nested payloads;
- merged recall, actual reranker/model/fallback metadata and evidence coverage/missing aspects;
- supplementary retrieval decisions;
- context-compression mode, implementation, retained-source and before/after
  character counts, and fallback reason without compressed text;
- generation implementation/fallback;
- citation/self-check result and terminal status.
- memory write status/count and configured TTL without user/assistant content;
- the same safe stage/tool/retry summaries incrementally emitted by `stream()`, with sequence and bounded elapsed time.

The trace contains summaries and identifiers, not chain-of-thought. Tool filters are visible for teaching but are produced by the Agent, not accepted from UI controls.

## 8. Cross-references

- ← Depends on: [`10-data-model.md`](./10-data-model.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`11-ingestion.md`](./11-ingestion.md)
- → Consumed by: [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`20-ui-demo.md`](./20-ui-demo.md)
- ↔ Tested by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Retrieval tier and cost boundary: [`15-retrieval-tier-selection.md`](./15-retrieval-tier-selection.md)
- ↔ Decisions: [`99-key-decisions.md § D13`](./99-key-decisions.md#d13--metadata-routing-is-owned-by-tools-not-ui), [`99-key-decisions.md § D14`](./99-key-decisions.md#d14--agent-planning-is-bounded-and-explicit), [`99-key-decisions.md § D15`](./99-key-decisions.md#d15--predefined-entities-resolve-domain-terminology-before-rewrite), [`99-key-decisions.md § D16`](./99-key-decisions.md#d16--retrieval-is-document-version-aware-by-default), [`99-key-decisions.md § D45`](./99-key-decisions.md#d45--query-transformation-is-bounded-and-scope-preserving), [`99-key-decisions.md § D46`](./99-key-decisions.md#d46--context-compression-is-a-provenance-preserving-generation-projection), [`99-key-decisions.md § D49`](./99-key-decisions.md#d49--search-granularity-is-explicit-and-entity-hits-are-not-citation-evidence), [`99-key-decisions.md § D50`](./99-key-decisions.md#d50--retrieval-complexity-is-a-measured-ladder-not-a-default), [`99-key-decisions.md § D52`](./99-key-decisions.md#d52--sub-query-tier-routing-is-planner-owned-and-eval-gated)
