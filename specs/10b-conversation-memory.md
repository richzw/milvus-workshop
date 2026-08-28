# 10b — Session-Scoped Conversation Memory

Status: stable baseline v2 · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)

## 1. Purpose and scope

本文把 `conversation_memory` 从独立 prototype 提升为 Agentic RAG 的 P2 多轮会话能力。Memory 用于延续当前 Streamlit session 的用户上下文、回答省略主语的 follow-up，并让用户观察、清除自己的会话记忆。它不是权威知识库、citation source、生产审计日志或跨用户 profile。

本文固定当前已实现的 baseline contract。下一阶段的 typed episode、Selection Gate、durable fact、working-state projection、selective consolidation 与 Milvus decay 由 [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md) 定义；迁移完成前不得把目标态描述成当前能力。

本阶段交付：

- local deterministic store 与 Milvus-backed store 使用同一接口；
- 每个成功 terminal turn 自动写入 user、assistant 和 bounded session summary；
- 显式 recall 或带指代词的 follow-up 在分类/改写前召回同 session、未过期的历史；
- “最近 N 个问题”类显式 recall 按时间确定性读取同 session、未过期的 user turns，而不是使用语义相似度；
- 显式“请记住…”内容额外写为 `task_state`，显式回忆问题可从 Memory 直接回答；
- Streamlit 展示完整多轮 chat history、Memory 召回/写入状态和清除操作。

## 2. Runtime architecture

```text
┌──────────────────────────── Streamlit session ───────────────────────────┐
│ stable session_id │ chat history │ Clear conversation & memory          │
└───────────────┬───────────────────────────────────────────────┬──────────┘
                │ question                                      │ delete
                ▼                                               ▼
┌──────────────────────── Agent workflow ──────────────────────────────────┐
│ recall_memory ─▶ classify/resolve ─▶ KB planning/retrieval or memory-only│
│      │                 ▲                                                   │
│      └─ bounded context ┘                                                  │
│ verify answer ─▶ answer deltas ─▶ persist_turn_memory + final snapshot    │
└───────────────┬───────────────────────────────────────────────────────────┘
                │ MemoryStore protocol
        ┌───────┴────────────────────┐
        ▼                            ▼
┌───────────────────┐      ┌────────────────────────────────────┐
│ Local list store  │      │ Milvus conversation_memory         │
│ deterministic CI  │      │ HNSW/COSINE + session/expiry filter│
└───────────────────┘      └────────────────────────────────────┘
```

Memory retrieval is supplementary context. `kb_chunks` remains the only source of KB citations. A memory-only answer has no `[Cn]` marker and uses `answer_validation.mode=memory_grounded`.

## 3. Data and public interfaces

```python
@dataclass(frozen=True)
class MemoryRecord:
    session_id: str
    turn_id: str
    role: Literal["user", "assistant", "system", "summary"]
    content: str
    summary: str | None
    memory_type: Literal["short_term", "session_summary", "task_state"]
    created_at: int
    expires_at: int | None
    metadata: dict
    content_vector: list[float]

class ConversationMemory(Protocol):
    def upsert_turn(self, records: list[MemoryRecord]) -> int: ...
    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int,
        memory_types: tuple[str, ...],
    ) -> list[MemoryRecord]: ...
    def list_session(
        self, session_id: str, *, now_ms: int, limit: int
    ) -> list[MemoryRecord]: ...
    def list_recent_user_questions(
        self, session_id: str, *, now_ms: int, limit: int
    ) -> list[MemoryRecord]: ...
    def delete_session(self, session_id: str) -> int: ...
```

All methods validate identifiers, bounds and enums before storage calls. `content` is non-empty and at most 8,192 characters; `summary` is at most 2,048; metadata is bounded JSON. `top_k` is `1..20`; UI list limit is `1..200`. The configured text embedding provider owns the shared 1,024-dimensional vector space.

Milvus upsert is idempotent per `(session_id, turn_id)`: delete that turn, insert its complete record set, verify mutation count where the SDK reports it, then flush. Search always combines:

```text
session_id == active_session
AND memory_type in requested_types
AND (expires_at is null OR expires_at > now_ms)
```

Local search applies the same predicate before cosine ranking. `list_session` returns only live records ordered by `(created_at, turn_id, role)`. The Milvus implementation scans the matching session through a bounded-batch query iterator, validates every row, and retains only the requested ordered window in process; it does not apply an arbitrary server-side limit before global ordering.

`list_recent_user_questions` is a distinct scalar-history operation. It applies:

```text
session_id == active_session
AND memory_type == "short_term"
AND role == "user"
AND (expires_at is null OR expires_at > now_ms)
ORDER BY created_at DESC, turn_id DESC
LIMIT effective_count
```

The local and Milvus adapters must produce the same most-recent-first order. The Milvus adapter may use a query iterator, but must retain the global newest window across all pages before applying the limit. It does not embed the recall request, invoke ANN search, inspect `session_summary`, or consult Selective Memory decay.

## 4. Turn lifecycle

The workflow uses the following order:

```text
question received
  │
  ├─ 1a. semantic follow-up/recall: search prior summary/task state
  ├─ 1b. recent-question recall: list prior user short_term by time
  ├─ 2. classify and plan with bounded memory context
  ├─ 3a. normal RAG: retrieve KB → generate → verify citations
  ├─ 3b. memory command: confirm write or answer from recalled memory
  ├─ 4. stream the complete validated answer
  ├─ 5. persist the completed turn only when the consumer requests final
│      ├─ user short_term
│      ├─ assistant short_term
│      ├─ deterministic session_summary
│      └─ optional task_state for explicit remember intent
  └─ 6. expose persistence status in the final envelope
```

Defaults:

- `MEMORY_TOP_K=3`;
- `MEMORY_TTL_SECONDS=86400` (24 hours);
- recalled prompt context is capped at three records and 2,000 characters;
- deterministic per-turn summary is capped at 2,048 characters;
- raw user/assistant turn content is capped at the schema limit.

`MEMORY_TTL_SECONDS` must be a positive integer. The workflow receives an injectable UTC wall clock for deterministic tests. A record whose `expires_at <= now_ms` is unavailable everywhere, including UI listing.

## 5. Intent and grounding behavior

- Explicit remember markers (`请记住`, `记住`, `remember`) set `intent=memory_write`, skip KB retrieval and return a confirmation. The remembered statement is stored as `task_state`.
- One shared deterministic action detector is authoritative for both the pre-classification recall gate and `RuleBasedQueryClassifier`. The two call sites must not maintain divergent recall marker lists.
- Explicit semantic recall markers (`你还记得`, `我之前`, `what did I`, `do you remember`, `我叫什么`) set `intent=memory_recall`. If live memory exists, answer only from the highest-ranked bounded records; otherwise say that no matching session memory exists.
- Recent-question requests combine a self/session-history cue with a question-history cue, for example `我最近的三个问题`, `我之前问过什么`, `我的历史提问` or `my last 3 questions`. They also set `intent=memory_recall`, but use `recall_mode=chronological`.
- `chronological` mode parses an Arabic or supported Chinese count, defaults to `3`, and bounds the effective count to `1..20`. It returns only prior `short_term/user` content, most recent first. Fewer live turns return the available subset. The current recall command is absent because persistence occurs after the answer is validated and streamed.
- A recent-question answer is constructed directly from those user records, contains no assistant answer/session summary/Selective Memory fact, has no KB citation, and never selects a search tool.
- Other private-knowledge follow-ups keep their original RAG path. Only explicit recall language or bounded referential markers activate Memory injection, preventing an unrelated old turn from changing a new question. Recalled summaries enrich classification and query rewrite, but cannot make weak KB evidence sufficient and cannot be cited as `[Cn]`.
- Security-sensitive intent (`operation`, permission-sensitive routing), tool authorization and metadata filters are derived only from the current user query. Untrusted Memory cannot select mutation capabilities, authorize tools, skip permission checks, or alter filters.
- A current question is written only after a valid terminal answer; it is never visible to its own recall stage.

## 6. Failure, privacy, and trust boundaries

Memory is optional context, so a recall/write dependency failure does not turn a citation-valid KB answer into a failure. It sets a bounded status (`recall_failed` or `write_failed`) and shows a safe UI warning; recall failure is also visible in its live stage event, while write failure is first exposed by the final snapshot. Raw Milvus/provider errors, memory content and vectors never enter trace events.

Session isolation is mandatory:

1. every read, write, list and delete is scoped to one validated `session_id`;
2. the UI never accepts an arbitrary session id from a text field;
3. clearing the conversation deletes only the active session and clears its local chat/trace state;
4. no memory content is logged or placed in presentation-safe trace details;
5. Memory is local Workshop functionality over synthetic data, not production identity, consent, retention or right-to-be-forgotten compliance.

Milvus collection absence fails UI workflow construction because the UI promises Memory functionality. Runtime memory failures degrade explicitly. Local fallback is used by deterministic/unit workflows, not silently substituted for a failed Milvus store. Every Milvus result is revalidated client-side against the active `session_id`, requested memory types and `expires_at > now_ms`; invalid backend rows fail closed.

## 7. Observability and UI contract

The workflow adds `recall_memory` and `persist_turn_memory` trace stages. `recall_memory` is streamed live. To make cancellation safe, `persist_turn_memory` runs only after all answer deltas have been consumed and while producing the requested terminal envelope; its status is exposed by the terminal snapshot rather than by an earlier standalone event. Safe details contain counts, status and memory types only. The terminal snapshot adds:

```python
"memory": {
    "status": "empty | recalled | saved | recall_failed | write_failed",
    "recalled_count": int,
    "recalled": [
        {
            "turn_id": str,
            "role": str,
            "memory_type": str,
            "summary": str,
            "created_at": int,
            "expires_at": int | None,
        }
    ],
    "written_count": int,
    "ttl_seconds": int,
}
```

The recall stage and terminal Memory snapshot additionally expose content-free routing metadata:

```python
{
    "recall_decision": "skipped | searched",
    "recall_mode": "none | semantic | chronological",
    "recall_reason": "not_applicable | contextual_followup | explicit_recall | recent_questions",
    "requested_count": int | None,
    "memory_types": list[str],
}
```

`empty` means a search ran and returned no eligible records; `skipped` is reported separately and must not be presented as a zero-result search. `memory_types` reflects the operation actually attempted: `["short_term"]` for chronological recent questions and `["session_summary", "task_state"]` for semantic recall.

Unlike live trace events, the Memory tab may display bounded memory summaries because it is an explicit session-private surface. It never displays vectors or arbitrary metadata.

## 8. Performance and limits

- One query performs at most one Memory vector search or one chronological user-turn listing, plus one bounded turn upsert.
- Memory search top-k never exceeds 20; default is 3.
- One persisted turn contains at most four records.
- UI session listing is capped at 200 live records; Milvus pagination uses batches of at most 200 and retains at most the requested limit while scanning.
- Memory latency is measured separately as `memory_recall_latency_ms` and `memory_write_latency_ms`; it is not counted as KB retrieval latency.

No hard latency target is claimed until a real Milvus baseline is recorded. Deterministic tests assert call bounds and result bounds.

## 9. Tests and exit criteria

- Local store: validation, upsert idempotency, semantic ranking, session isolation, TTL boundary, list and delete.
- Milvus adapter: exact semantic and chronological filter expressions, record serialization, upsert/delete/flush, search decoding, global newest-window ordering, client-side session/role/type/TTL fail-closed checks, list and session-scoped deletion.
- Workflow: prior turn influences a follow-up; explicit remember/recall works; “查找下我最近的三个问题是什么” returns the three newest prior user turns in order without KB tools; another session cannot recall them; current turn is not self-recalled; expired memory is ignored; memory cannot become a KB citation.
- Classification: all supported recent-question phrasings take the same deterministic `memory_recall` fast path in rule-only and LLM-configured workflows; the primary LLM is not invoked for this action.
- Trace: skipped versus searched is distinguishable; chronological mode reports the bounded requested count and actual `short_term` type without Memory content.
- Failure: recall/write failures produce safe degraded status without exposing raw dependency text.
- Runtime parity: local and LangGraph recall before planning, stream validated answer deltas, and persist only when final is requested; both return the same terminal Memory contract.
- UI: historical turns survive reruns, Memory tab is visible, clear affects only active session, incomplete streams do not persist a turn.
- Full deterministic tests, RAG eval and lint remain green.

## 10. Cross-references

- ← Depends on: [`10-data-model.md § conversation_memory`](./10-data-model.md#4-conversation_memory--session-semantic-memory), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)
- → Consumed by: [`12-agent-workflow.md`](./12-agent-workflow.md), [`20-ui-demo.md`](./20-ui-demo.md)
- ↔ Validated by: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decisions: [`99-key-decisions.md § D18`](./99-key-decisions.md#d18--conversation-memory-is-session-scoped-supplementary-context), [`99-key-decisions.md § D30`](./99-key-decisions.md#d30--recent-question-recall-is-a-deterministic-temporal-query)
- → Evolves into: [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md)
