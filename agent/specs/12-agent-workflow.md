# 12 — Agentic RAG Workflow

Status: draft v3 · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`11-ingestion.md`](./11-ingestion.md)

## 1. Purpose

本文定义在线查询的唯一运行契约：意图判断、领域术语消歧、检索决策、权限检查、工具选择、查询改写/拆解、文档版本隔离、多次或多跳检索、证据判断、有限补充检索、回答与 citation self-check。Milvus 负责高召回，reranker 负责高精排，evidence grader 负责覆盖判断；术语解析、版本路由、工具路由和这些质量步骤必须保持可观察、可测试。

## 2. Runtime boundaries and lifecycle

```text
┌──────────────────── Streamlit boundary ────────────────────┐
│ user question only; no source/doc/department controls      │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────── Agent planning boundary ───────────────┐
│ recall session memory ─▶ classify intent/topic ─▶ resolve  │
│                              │ matched entity info          │
│                              ▼                              │
│                         decide retrieval                    │
│          │ private knowledge                               │
│          ▼                                                 │
│ get_user_permission ─▶ select tools ─▶ rewrite/decompose   │
│          │ denied                    │ 1..3 subqueries      │
│          └──────────────▶ refuse     ▼                     │
└──────────────────────────┬─────────────────────────────────┘
                           ▼ tool calls with private/version filters
┌──────────────────── Retrieval boundary ────────────────────┐
│ search_policy_docs    search_product_docs                  │
│ search_meeting_notes  search_code_docs                     │
│ version scope: current / exact / explicit comparison       │
│          └──────▶ Milvus dense+sparse search ◀──────┘      │
│                       │ merge/dedupe                        │
│                       ▼                                    │
│                    reranker                                │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────── Evidence loop ─────────────────────────┐
│ grade coverage / contradictions / missing aspects          │
│   ├─ enough ─────────────▶ generate answer                 │
│   ├─ missing dependency ─▶ choose next tool/query          │
│   └─ retry exhausted ────▶ structured abstain              │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────── Answer boundary ───────────────────────┐
│ generate from selected context ─▶ citation/self-check      │
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
    memory_status: str
    memory_written_count: int
    intent: str
    query_type: str
    entity_catalog_version: str
    matched_entities: list[dict]
    ambiguous_entities: list[dict]
    need_retrieval: bool
    retrieval_decision: dict
    permission_decision: dict
    selected_tools: list[str]
    tool_calls: list[dict]
    query_plan: list[dict]
    rewritten_queries: list[str]
    version_scope: dict
    search_filters: dict
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    enough_evidence: bool
    evidence_grade: dict
    retry_count: int
    max_retry: int
    answer: str
    citations: list[dict]
    answer_validation: dict
    metrics: dict
    trace: dict
```

This is a logical shape, not a verified code symbol. `intent` describes the requested action (`conversation`, `private_knowledge`, `comparison`, `operation`, `permission_sensitive`, `memory_write`, `memory_recall`); `query_type` describes the topic (`architecture`, `policy`, `product`, `general`, `unknown`). They are separate because the same topic can require different execution plans.

### 3.1 Streaming event contract

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

## 5. Node contracts

### 5.0 `recall_memory`

Before classification, an explicit recall request or bounded referential follow-up searches at most `MEMORY_TOP_K` live `session_summary`/`task_state` records for the active `session_id`. An unrelated standalone question does not inject old Memory. Build a bounded context string without trace content or vectors. Empty recall is normal. A typed store failure records `recall_failed` and continues without memory; it never silently switches persistence backends.

### 5.1 `classify_intent`

Classifies both `intent` and `query_type`, records a reason, and sets `need_retrieval`. Greetings and generic capability explanations may answer directly. Explicit remember/recall language selects `memory_write`/`memory_recall` and skips KB retrieval. Private knowledge, comparison and permission-sensitive questions retrieve by default. Operation requests are classified and traced but may only execute tools explicitly present in the catalog; unsupported mutations return a safe refusal. Bounded recalled summaries may clarify a follow-up topic but never grant permission or establish KB evidence.

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

### 5.3 `select_tools`

Chooses the smallest relevant set of tools from topic and intent. A simple question normally selects one search tool; a comparison selects two or more. Selection records tool name, reason and intended knowledge domain. The Agent never searches every domain by default.

### 5.4 `rewrite_and_decompose`

Produces one to three retrieval subqueries. Each plan item contains `subquery_id`, rewritten query, selected tool, version scope, dependency ids and status. Terminology expansion preserves original intent and incorporates only the resolver's matched entities; each rewrite retains the original surface form or canonical term so exact product vocabulary is not lost. Comparison questions produce parallel subqueries; multi-hop questions may leave a dependent subquery whose text is refined from first-hop evidence.

For a normal follow-up, the rewritten query may include bounded recalled summaries to resolve pronouns or omitted topic words. The raw rewrite remains private trace data; Memory does not add a new source or permission domain.

### 5.5 `execute_tool_plan`

Executes ready search plan items, potentially multiple calls in one round, then merges candidates by `chunk_id`. The highest-scoring occurrence owns ranking fields while tool/query provenance is retained in `tool_calls`. Calls are bounded by three initial subqueries and `milvus_top_k` per call.

Version scope is part of every plan item and tool call:

- `current` (default): filter `is_current == true`;
- `exact`: when the user names a version, filter exact `doc_version` and never fall back to current if it is absent;
- `comparison`: only when the user explicitly asks to compare versions; execute one exact/current scope per side and keep candidates partitioned by `(doc_id, doc_version)`.

Normal merge, rerank and selection reject multiple versions of the same `doc_id`. A comparison plan may retain them, but version labels and provenance must remain attached through answer generation.

The deterministic MVP recognizes exact version tokens shaped as `vN`/`vN.N` or `YYYY.MM`, case-insensitively. `current`/`latest`/`当前` select the current edition. One explicit token selects `exact`; two explicit sides combined with comparison intent select `comparison`. Unknown exact versions return no evidence and never fall back. More than two distinct requested versions, or comparison wording without two resolvable sides, returns a clarification request instead of broadening scope.

For multi-hop retrieval, a later plan item may depend on facts extracted from earlier evidence. Example:

```text
customer meeting notes ─▶ extract frequent concerns
                         └▶ query product roadmap for those concerns
                             └▶ compare covered vs uncovered
```

### 5.6 `rerank_evidence`

Reranks the merged candidate set against the original user question. It returns stable `old_rank`, `rerank_score` and selection status. A deterministic rule fallback remains explicit in trace.

### 5.7 `grade_evidence`

Returns `enough_evidence`, reason, covered aspects, missing aspects, contradictions and an optional supplementary plan. For comparisons, evidence must cover every required side; one-side-only evidence is insufficient. Missing referenced artifacts such as a “城市级别表” trigger a targeted tool/query rather than a generic repeat.

### 5.8 `prepare_supplementary_retrieval`

If evidence is insufficient and retry budget remains, selects only the tool/query needed for missing aspects, appends it to the plan and preserves prior evidence. It never discards successful earlier hops. At the cap, the workflow abstains and reports the unresolved aspects.

### 5.9 `generate_answer`

Delegates at most five selected chunks, resolved entity info and the validated version scope to the answer generator defined in [`13-llm-answer-generation.md`](./13-llm-answer-generation.md). The answer must distinguish supported conclusions, uncovered comparison items and missing evidence; explicit version comparisons label each conclusion with its source version.

### 5.10 `verify_answer`

Runs after generation and before answer chunks become terminal output. It verifies that every structured citation belongs to selected context, every inline marker resolves, version scope obeys the current/exact/comparison policy, at least one citation supports a grounded answer, and an abstention does not claim unsupported specifics. It records a structured `answer_validation` result without chain-of-thought.

For `answered_from_memory`, verification requires at least one recalled live record, no KB citations and an answer constructed only from bounded Memory summaries. For `memory_write`, verification requires a non-empty remembered statement and no citation.

### 5.11 `persist_turn_memory`

After answer verification (or another valid direct terminal outcome), after all answer deltas are consumed and while producing `final`, write the bounded user turn, assistant turn and deterministic per-turn summary under the active `(session_id, query_id)`. Explicit remember intent adds one `task_state`. Idempotent upsert prevents retries/reruns from duplicating a turn. A typed write failure sets `write_failed`; the final snapshot drives a safe UI warning and preserves an otherwise valid answer. For explicit `memory_write`, the response remains non-committal until this final status and uses `memory_write_failed` when persistence fails.

## 6. Invariants

1. The graph reaches direct answer, grounded answer, clarification request, permission denial, safe operation refusal or abstain in bounded steps.
2. No private search occurs before an allowed `permission_decision`.
3. `1 ≤ initial subqueries ≤ 3`, `0 ≤ retry_count ≤ max_retry == 3`.
4. Every search call names one registered tool and uses only that tool's policy-owned filters intersected with allowed departments.
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
16. validated answer deltas are streamed before persistence;
17. only when the consumer requests the terminal envelope, `persist_turn_memory` completes or records a visible degraded status and `final` exposes that status. A cancelled/incomplete stream never writes the current turn.

## 7. Observability

Trace shows, in order:

- memory recall status/count without Memory content;
- intent/topic classification, matched/ambiguous entities, catalog version and retrieval decision;
- permission decision without credentials or identity secrets;
- selected tools and reasons;
- query plan, dependencies and each rewrite/retry round;
- each tool call's safe filters, version scope, result count and latency;
- merged recall, rerank and evidence coverage/missing aspects;
- supplementary retrieval decisions;
- generation implementation/fallback;
- citation/self-check result and terminal status.
- memory write status/count and configured TTL without user/assistant content;
- the same safe stage/tool/retry summaries incrementally emitted by `stream()`, with sequence and bounded elapsed time.

The trace contains summaries and identifiers, not chain-of-thought. Tool filters are visible for teaching but are produced by the Agent, not accepted from UI controls.

## 8. Cross-references

- ← Depends on: [`10-data-model.md`](./10-data-model.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`11-ingestion.md`](./11-ingestion.md)
- → Consumed by: [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`20-ui-demo.md`](./20-ui-demo.md)
- ↔ Tested by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decisions: [`99-key-decisions.md § D13`](./99-key-decisions.md#d13--metadata-routing-is-owned-by-tools-not-ui), [`99-key-decisions.md § D14`](./99-key-decisions.md#d14--agent-planning-is-bounded-and-explicit), [`99-key-decisions.md § D15`](./99-key-decisions.md#d15--predefined-entities-resolve-domain-terminology-before-rewrite), [`99-key-decisions.md § D16`](./99-key-decisions.md#d16--retrieval-is-document-version-aware-by-default)
