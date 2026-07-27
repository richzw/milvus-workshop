# 20 — Streamlit UI Demo

Status: draft v2 · Owner: workshop author · Depends on: [`12-agent-workflow.md`](./12-agent-workflow.md)

## 1. Purpose

Streamlit UI 是 Workshop 的自然语言入口和解释层。用户只负责提问；知识源、文档类型和部门由 Agent 通过工具选择。UI 展示答案、证据与决策，但不承担在线 ingestion、工具授权、检索路由或生产级权限控制。

## 2. Page architecture

```text
┌──────────────────────────── Streamlit app ────────────────────────────┐
│ Chat input: natural-language question only                           │
│ no source_type / doc_type / department / image filter controls       │
├───────────────────────────────────────────────────────────────────────┤
│ Tab 1 Chat             │ Tab 2 Evidence      │ Tab 3 Agent Trace      │
│ answer / abstain       │ merged tool recall  │ intent + permission    │
│ citations/source cards │ rerank + coverage   │ tools + plan + retries │
│                        │                      │ answer self-check      │
├────────────────────────┴──────────────────────┴────────────────────────┤
│ Tab 4 Memory: recalled summaries / live session records / clear action │
└───────────────────────────────────────────────────────────────────────┘
            │ one query_id                      ▲
            └──────── Agent workflow result ────┘
```

## 3. Interaction lifecycle

1. User submits one natural-language question; UI sends no metadata filters.
2. UI consumes ordered `trace_event` envelopes and updates one in-message progress timeline while the Agent classifies intent, resolves terminology, checks permission, selects tools, plans, retrieves, reranks and grades evidence.
3. When evidence is sufficient, citation-validated answer chunks render only after the verification event; otherwise Chat shows permission denial, unsupported-operation refusal, abstain or execution failure.
4. Evidence and the persisted Agent Trace reconcile to the same terminal snapshot and `query_id`.
5. A new query appends user/assistant messages to visible session history, creates a new event list, and uses the same stable `session_id`.
6. A successful terminal turn refreshes the Memory tab. “Clear conversation & memory” deletes only the active session, then clears its UI messages/events.

## 4. Tab contracts

### 4.1 Chat

- All user/assistant turns for the current Streamlit session in chronological order.
- Inline `[C1]`, `[C2]` markers and source cards.
- Clear permission-denied, unsupported-operation, missing-evidence and execution-error states.
- No source/doc/department controls or “reset filters” recovery language.

### 4.2 Evidence

- **Tool Recall Results**: tool, subquery, rank, hybrid score, source, document version, chunk/page, snippet.
- **Reranked Results**: new rank, rerank score, old rank, source, selected flag.
- **Coverage Summary**: covered/missing aspects and contradictions.

Selected context is visually identifiable. Tool filters may be displayed as trace metadata, but never as editable controls.

### 4.3 Agent Trace

During execution, Chat shows a compact, softly styled timeline in the assistant message:

- completed items use a quiet check mark and one-line summary;
- the overall Agent container remains in a running state until `final`;
- warnings use amber copy without flashing or alarm styling;
- completion collapses the container to a one-line stage/count/latency summary;
- technical details remain optional and collapsed.

The Agent Trace tab replays the same event list in execution order and shows:

- intent/topic, matched or ambiguous entities, catalog version and `need_retrieval`;
- permission decision;
- selected tools and routing reasons;
- decomposed subqueries and dependency edges;
- every tool call, safe filters, version scope, count and latency;
- merged recall, reranker and evidence grade;
- supplementary retrieval rounds and unresolved aspects;
- generator/fallback and citation/self-check;
- per-stage latency and terminal status.

Raw trace JSON remains available only inside an explicitly labeled “Advanced trace JSON” expander. It is never the primary presentation.

### 4.4 Memory

- Current Memory status, configured TTL, recalled/written counts.
- Bounded recalled summaries used by the latest query.
- Up to 200 live records for the active session, showing role, memory type, bounded summary/content preview, created/expiry time.
- No vector or arbitrary metadata rendering.
- One explicit “Clear conversation & memory” control scoped to the generated session id.
- `recall_failed`/`write_failed` render as a non-blocking warning distinct from an empty Memory result.

## 5. UI invariants

1. All tabs render from one immutable result keyed by `query_id`.
2. UI never constructs or overrides search filters.
3. Citation markers, source cards and selected evidence agree on `chunk_id` and `doc_version`.
4. Permission denial and execution failure are distinct from insufficient evidence.
5. Sanitized trace progress is displayed as events arrive; persisted timeline/evidence render after `final` and reconcile to its immutable response.
6. Raw secrets, full document bodies and unsanitized URLs never render.
7. Deterministic rerank/generation fallback is clearly labeled.
8. Source cards and evidence always show `doc_version`; multiple versions of one `doc_id` render only for an explicit comparison and are visually grouped by version.
9. The UI ignores duplicate/out-of-order trace events, never renders unsafe arbitrary fields, and treats a stream ending without `final` as an error.
10. Grounded answer text never appears before successful citation/self-check; progress copy calls this “生成与校验”, not token streaming.
11. Chat history and Memory records use one generated session id; neither the question nor a UI control may select another session.
12. An incomplete/failed workflow stream appends no assistant turn and persists no current-turn Memory.
13. Clearing deletes only active-session Memory and resets messages/events/last response; it does not drop collections.

## 6. Error states

| State | UI response |
| --- | --- |
| No tool results | explain which tools ran and what evidence is missing |
| Permission denied | show a safe denial reason; no retrieval evidence |
| Unsupported operation | explain that the query-only demo cannot mutate systems |
| Retry exhausted | show abstain reason, attempted tools and unresolved aspects |
| Milvus unavailable | show setup error; do not generate an ungrounded answer |
| Generator/citation validation failure | show sanitized fallback or contextual execution error |
| Memory empty | continue normally and label that no prior session context matched |
| Memory recall/write unavailable | keep a valid answer, show a sanitized degraded warning, never show raw dependency text |

## 7. Cross-references

- ← Depends on: [`12-agent-workflow.md`](./12-agent-workflow.md), [`13-llm-answer-generation.md`](./13-llm-answer-generation.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`10-data-model.md § query_traces`](./10-data-model.md#61-query_traces)
- ↔ Acceptance: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decision: [`99-key-decisions.md § D13`](./99-key-decisions.md#d13--metadata-routing-is-owned-by-tools-not-ui)
