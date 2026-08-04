# 10c — Grounded Response Cache

Status: draft v1 · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md)

## 1. Purpose

本文定义同一 session 内 exact 或高置信语义等价问题的回答复用合同。它保存经过 citation/self-check 的完整 KB answer、citations、evidence identity 和安全/freshness scope，避免重复执行 hybrid retrieval、rerank 与 answer generation。

Grounded response cache 不是 Conversation Memory。Conversation Memory 解释多轮上下文；response cache 复用一个仍然有效的 grounded terminal response。Cache hit 仍然是 KB-grounded answer，必须保留 citations，不能伪装成 `answered_from_memory`。未来的 Selection Gate、consolidation 和 decay 也不得写入、刷新或替代 cache validity，完整边界见 [`10d-selective-agent-memory.md`](./10d-selective-agent-memory.md)。

## 2. Runtime architecture

```text
┌──────────────────── Session request boundary ─────────────────────┐
│ current query + session_id                                        │
└──────────────────────────────┬────────────────────────────────────┘
                               ▼
┌──────────────────── Conversation routing ─────────────────────────┐
│ recall session context → classify/route → resolve entity/version  │
│ direct / Memory / operation / clarification requests exit here    │
└──────────────────────────────┬────────────────────────────────────┘
                               ▼
┌──────────────────── Current authorization ────────────────────────┐
│ retrieval intent → check current permission                       │
│ denied requests terminate before any cache lookup                 │
└──────────────────────────────┬────────────────────────────────────┘
                               ▼
┌──────────────────── try_grounded_cache ───────────────────────────┐
│ same-session exact hash, else bounded vector candidate lookup     │
│ query equivalence + scope + TTL + KB revision                     │
│ permission-scope hash + live chunk version/checksum/current flag  │
│ citation marker/self-check                                        │
└───────────────┬───────────────────────────────────┬───────────────┘
                │ valid                             │ miss/stale/error
                ▼                                   ▼
┌──────────────────────────┐        ┌───────────────────────────────┐
│ answered_from_cache      │        │ tools → retrieve → rerank     │
│ cached_grounded          │        │ → generate → verify           │
│ original citations kept  │        └──────────────┬────────────────┘
└──────────────┬───────────┘                       │ verified answer
               └──────────────────────┬────────────┘
                                      ▼
                         persist turn + cache record
```

Cache candidate lookup and validation are one permission-gated workflow stage.
Direct answers, Conversation/Selective Memory actions, unsupported operations,
clarification paths and denied requests perform zero grounded-cache searches.
Candidates remain private workflow state and no cached answer or citation becomes
observable before all current-request checks succeed.

## 3. Record and store contract

```python
@dataclass(frozen=True)
class CachedEvidence:
    chunk_id: str
    doc_id: str
    doc_version: str
    checksum: str
    is_current: bool

@dataclass(frozen=True)
class GroundedResponseCacheRecord:
    cache_id: str
    session_id: str
    source_query_id: str
    normalized_query: str
    query_hash: str
    query_vector: list[float]
    embedding_fingerprint: str
    intent: str
    query_type: str
    retrieval_goal: str
    version_scope: dict
    entity_ids: list[str]
    query_constraints: list[str]
    permission_scope_hash: str
    kb_revision: str
    workflow_version: str
    answer: str
    citations: list[dict]
    evidence: list[CachedEvidence]
    created_at: int
    expires_at: int

class GroundedResponseCache(Protocol):
    def search(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int,
        now_ms: int,
    ) -> list[ResponseCacheCandidate]: ...
    def upsert(self, record: GroundedResponseCacheRecord) -> int: ...
    def delete_session(self, session_id: str) -> int: ...
```

`ResponseCacheCandidate` adds `similarity` and `match_type=exact|semantic` without changing the stored record.

The Milvus collection is named `grounded_response_cache`. Required scalar fields are query/session identity, classification scope, permission/KB/workflow revisions and lifecycle timestamps. `answer` is bounded to 12,000 characters. Citations, evidence, version scope, entity ids and query constraints are bounded JSON. `query_vector` uses the configured 1,024-dimensional text embedding space.

One `(session_id, normalized query hash)` owns one current cache record. Upsert replaces that identity idempotently. Cache records never cross sessions.

## 4. Configuration

| Variable | Default | Contract |
| --- | ---: | --- |
| `RESPONSE_CACHE_ENABLED` | `true` | strict boolean |
| `RESPONSE_CACHE_TTL_SECONDS` | `259200` | positive; three days |
| `RESPONSE_CACHE_TOP_K` | `3` | `1..20` |
| `RESPONSE_CACHE_SIMILARITY_THRESHOLD` | `0.92` | finite `[0,1]`; calibrated by eval |
| `KB_REVISION` | `demo-v1` | non-empty operator/ingestion publication id |
| `MILVUS_RESPONSE_CACHE_COLLECTION_NAME` | `grounded_response_cache` | non-empty |

TTL is an upper bound, not a freshness proof. `KB_REVISION` must change whenever a corpus publication can alter an answer, including adding a new feature chunk to an exhaustive document. The cache additionally validates cited chunks so a targeted content/version change fails closed.

`RESPONSE_CACHE_ENABLED=false` performs no cache search/write and preserves the existing workflow. It does not disable Conversation Memory.

## 5. Permission-gated lookup and equivalence

Only a request classified and routed to grounded retrieval, resolved without
ambiguity and allowed by `check_permission` invokes `try_grounded_cache`. That
stage:

1. normalizes Unicode/whitespace/case and computes a stable SHA-256 query hash;
2. returns a live same-session exact-hash record first;
3. otherwise runs COSINE search over live same-session records and returns at most `top_k`;
4. immediately applies the validation contract in §6 to the private candidates;
5. records only candidate count/status in presentation-safe trace.

There is at most one cache search per eligible request. A miss, stale candidate
or typed dependency failure continues to authorized experience recall and normal
retrieval planning. A hit skips both of those stages.

Exact hash match does not need a similarity threshold. Semantic match requires all of:

- similarity at or above the configured threshold;
- same validated `intent`, `query_type`, `retrieval_goal`;
- identical resolved `version_scope` and entity ids;
- identical extracted query constraints for explicit version numbers and negation markers;
- same embedding fingerprint, permission scope, KB revision and workflow version.

Threshold alone never establishes equivalence. The constraint checks prevent reuse across `3.0`/`2.6`, positive/negative questions, focused/exhaustive requests and normal/comparison plans. A semantic candidate that cannot prove compatibility is a normal miss.

## 6. Freshness, permission, and grounding validation

Inside `try_grounded_cache`, after the current request passes
`check_permission`, compute a stable permission-scope hash from the checker
identity and sorted allowed departments. It must equal the cached value.

For every cached evidence item, perform an authorized scalar lookup by `chunk_id` and require:

- exactly one live chunk exists;
- `doc_id`, `doc_version` and non-empty `checksum` equal the snapshot;
- current scope still has `is_current=true`;
- the chunk still satisfies the current permission/tool filter domain.

The cache is stale if any item fails. A current `KB_REVISION` mismatch causes an immediate miss even if cited checksums are unchanged, because an exhaustive answer may become incomplete when new chunks are added.

Before release, every inline `[Cn]` marker must resolve to the cached citations, every citation must map to validated evidence, and at least one citation must remain. A hit sets:

```text
terminal_status = answered_from_cache
answer_validation.mode = cached_grounded
```

Cache hit skips tools, hybrid retrieval, rerank, grader and answer generation. It never skips current permission.

## 7. Write and failure lifecycle

Only `terminal_status=answered` with `answer_validation.valid=true`, non-empty citations and non-empty checksums is cacheable. Abstentions, direct answers, permission denials, Memory answers, fallback refusals and cache hits are not written/refreshed.

The cache record is written only while producing `final`, alongside the completed-turn Memory write. A cancelled stream writes neither current turn nor cache. Cache write/search/validation dependency failures are sanitized and degrade to the normal RAG path; they never convert a valid answer into failure.

`Clear conversation & memory` deletes both same-session Conversation Memory and grounded response cache. No other session is affected.

## 8. State, trace, and limits

Private `response_cache_candidates` are removed from serialized state. Public state/trace contains only:

```python
response_cache_status: str
response_cache_candidate_count: int
response_cache_match_type: str | None
response_cache_similarity: float | None
response_cache_source_query_id: str | None
response_cache_fallback_reason: str | None
response_cache_expires_at: int | None
```

Allowed statuses are `disabled`, `miss`, `candidate`, `hit`, `stale`, `recall_failed`, `validation_failed`, `saved`, `write_failed`. Trace contains no cached answer body, raw query, vectors, permission details or provider/storage error.

Bounds:

- candidate top-k `≤20`, default `3`;
- citations/evidence `1..16`;
- query/answer follow existing 8,192/12,000 character limits;
- cache JSON fields are locally size-validated before insert;
- one cache vector search and one bounded evidence scalar lookup per query.

## 9. Tests and acceptance

- exact repeated question hits cache and performs zero search-tool/rerank/generation calls;
- direct, Memory, operation, clarification and permission-denied paths perform
  zero grounded-cache searches;
- an eligible cache miss performs exactly one candidate lookup, then continues
  to authorized experience recall and retrieval planning;
- a high-similarity paraphrase with identical classification/version/entities/constraints hits;
- below-threshold, different version, negation, comparison or exhaustive mismatch misses;
- expiry boundary at exactly three days misses;
- KB revision, checksum, current-version, workflow-version or permission-scope mismatch misses;
- missing/nullable checksum prevents cache write/hit;
- cached answer preserves valid inline and structured citations;
- another session cannot observe or hit the record;
- cache dependency failures continue through normal RAG with sanitized trace;
- cancelled stream writes no record; clear removes only the active session;
- local and Milvus stores have equivalent filters, bounds and fail-closed result validation;
- local and LangGraph runtimes expose the same hit/miss terminal contract;
- full unit tests, offline RAG eval and lint remain green.

## 10. Cross-references

- ← Data: [`10-data-model.md`](./10-data-model.md)
- ← Embeddings: [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md)
- ↔ Conversation context: [`10b-conversation-memory.md`](./10b-conversation-memory.md)
- → Workflow: [`12-agent-workflow.md`](./12-agent-workflow.md)
- → UI: [`20-ui-demo.md`](./20-ui-demo.md)
- ↔ Quality: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decision: [`99-key-decisions.md § D20`](./99-key-decisions.md#d20--grounded-response-cache-is-separate-and-fail-closed)
