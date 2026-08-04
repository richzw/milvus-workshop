# 10d — Selective Dual-Speed Agent Memory

Status: draft v1 · Owner: workshop author · Depends on: [`10-data-model.md`](./10-data-model.md), [`10a-openai-text-embedding.md`](./10a-openai-text-embedding.md), [`10b-conversation-memory.md`](./10b-conversation-memory.md), [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md)

## 1. Purpose and migration scope

本文定义当前 session-scoped Conversation Memory 的下一阶段目标：从“每轮写入并按语义召回历史文本”升级为 selective dual-speed memory。系统快速记录可追溯 episode，通过低成本 Selection Gate 选择值得保留的经历，再把重复、确认或高价值经历缓慢 consolidation 为带来源和版本的 durable facts。Milvus Decay Ranker 负责软遗忘；状态、TTL 和清理任务分别负责逻辑失效与物理删除。

本 spec 扩展而不推翻 [`10b-conversation-memory.md`](./10b-conversation-memory.md) 的既有安全边界：

- 首个实现仍只允许当前生成的 `session_id`，不引入隐式跨 session 用户画像；
- Memory 不能成为 KB citation、授权工具、修改 filters 或让不足的 KB evidence 变充分；
- [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md) 继续独立保存和校验 grounded answer，不并入 experiential memory；
- current implementation 的 `short_term/session_summary/task_state` 在迁移完成前仍是兼容输入，但不再是目标数据模型。

受 [What the Brain Knows About Long-Running Agents](https://x.com/yoheinakajima/article/2081741659260477666) 与 [The Log Is the Agent](https://arxiv.org/abs/2605.21997) 启发，本设计采用快慢两种学习速度、append-only event lineage、selective consolidation 和 projection；完整 replay/fork/merge 只保留接口前向兼容性，不进入首个实现。

## 2. Goals and non-goals

### 2.1 Goals

1. 普通 turn 不再和用户纠正、任务状态变化、严重失败获得相同的保留优先级。
2. 每项 durable memory 都能追溯到不可变 source events，并能表达 active、superseded、disputed 与 tombstoned。
3. `recall_memory` 返回 typed、bounded、可解释的 `MemoryPack`，而不是无结构文本拼接。
4. 时间、语义、显著性、任务相关性、来源置信度和冲突共同影响召回。
5. 低价值 episode 会随时间下沉并最终清理；重新确认或成功复用可延长有用记忆，但单纯“被召回”不能刷新生命周期。
6. local deterministic store 与 Milvus-backed store 在 selection、projection、expiry 和排序语义上保持可测试 parity。

### 2.2 Non-goals

- 不把 Memory 变成生产审计日志、生产身份/profile 系统或合规归档。
- 不允许 Memory 自动修改 system prompt、tool implementation、权限 policy 或外部系统状态。
- 不承诺 fork 能撤销已经发生的外部副作用。
- 不在每轮对话调用 LLM 做 reflection；LLM selector 只处理规则无法确定的窄区间。
- 不用 decay 代替显式 expiry、supersession、权限检查或删除请求。

## 3. Runtime architecture

```text
┌────────────────────────────── Query runtime ──────────────────────────────┐
│ current query + current permission scope                                  │
│         │                                                                 │
│         ├──────────────▶ grounded response-cache candidates               │
│         │                 private until fail-closed validation             │
│         │                                                                 │
│         ▼                                                                 │
│  Memory Router                                                            │
│   ├─ working-state projection ───────────────┐                             │
│   ├─ durable-fact recall ────────────────────┤                             │
│   └─ recent-episode hybrid + decay recall ───┤                             │
│                                              ▼                             │
│                                  bounded typed MemoryPack                  │
│                                              │                             │
│                         classify / rewrite / direct memory recall          │
└──────────────────────────────────────────────┬─────────────────────────────┘
                                               │ validated terminal outcome
                                               ▼
┌────────────────────────────── Memory runtime ─────────────────────────────┐
│ Episode capture ─▶ Selection Gate ─┬─ discard recallable payload          │
│       │                            ├─ ephemeral episode                   │
│       │                            ├─ promotion candidate                 │
│       │                            └─ protected episode                   │
│       │                                      │                            │
│       ▼                                      ▼                            │
│ append-only memory_events              Consolidator                       │
│ source of episode lineage              cluster / reconcile / promote       │
│       │                                      │                            │
│       └──────────────▶ Projector ◀──────── memory_facts                    │
│                        active tasks / preferences / decisions / conflicts  │
└────────────────────────────────────────────────────────────────────────────┘
```

The event log owns what happened. `memory_facts` owns the current consolidated interpretation. The working state is a rebuildable projection, not mutable truth. The response cache is a sibling performance mechanism, not a memory tier.

## 4. Data contracts

### 4.1 `MemoryEvent`

```python
@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    session_id: str
    query_id: str | None
    turn_id: str | None
    parent_event_id: str | None
    branch_id: str
    event_type: Literal[
        "user_statement",
        "user_preference",
        "user_correction",
        "assistant_answer",
        "task_opened",
        "task_updated",
        "task_completed",
        "decision_made",
        "retrieval_failure",
        "answer_rejected",
        "strategy_succeeded",
        "memory_reconfirmed",
        "memory_promoted",
        "memory_superseded",
        "memory_tombstoned",
    ]
    content: str
    summary: str | None
    outcome: str | None
    event_time: int
    expires_at: int | None
    salience_score: float
    selection_reason: tuple[str, ...]
    retention_class: Literal["ephemeral", "candidate", "protected"]
    decay_profile: str
    permission_scope_hash: str
    workflow_version: str
    checksum: str
    content_vector: list[float]
```

Invariants:

- `event_id` is immutable and globally unique; correction and promotion append new events instead of overwriting source events.
- `event_time` and all lifecycle values use UTC epoch milliseconds.
- `salience_score` is finite and within `0..1`; `selection_reason` contains only registered reason codes.
- `content` is bounded and may be erased by an authorized clear operation; integrity metadata must not retain recoverable personal content.
- `branch_id="main"` is the only executable branch in the first implementation.

### 4.2 `MemoryFact`

```python
@dataclass(frozen=True)
class MemoryFact:
    memory_id: str
    session_id: str
    memory_type: Literal[
        "user_preference",
        "user_fact",
        "task_state",
        "decision",
        "failure_pattern",
        "successful_strategy",
    ]
    subject: str
    predicate: str
    value: str
    status: Literal["active", "superseded", "disputed", "tombstoned"]
    confidence: float
    revision: int
    source_event_ids: tuple[str, ...]
    supersedes_memory_id: str | None
    valid_from: int
    valid_to: int | None
    last_confirmed_at: int
    expires_at: int | None
    salience_score: float
    permission_scope_hash: str
    content_vector: list[float]
```

Invariants:

- `source_event_ids` is non-empty and resolves to same-session events.
- A new interpretation appends a higher revision and marks the previous projection `superseded`; it does not rewrite event history.
- Conflicting high-confidence facts become `disputed` until deterministic policy or explicit confirmation resolves them.
- `user_preference` and `user_fact` require explicit user evidence; agent inference alone cannot silently create them.
- Only `active` facts can enter normal `MemoryPack`; disputed items appear only in its `conflicts` section.

### 4.3 `MemoryPack`

```python
@dataclass(frozen=True)
class MemoryPack:
    working_state: tuple[MemoryFact, ...]
    durable_facts: tuple[MemoryFact, ...]
    recent_episodes: tuple[MemoryEvent, ...]
    conflicts: tuple[MemoryFact, ...]
    provenance_event_ids: tuple[str, ...]
    rendered_context: str
    truncated_count: int
```

`rendered_context` is a deterministic projection of the typed sections. It is capped at `MEMORY_CONTEXT_MAX_CHARS`; consumers must not reconstruct extra context from arbitrary metadata. Grounded response-cache candidates remain a separate workflow field and are never embedded in `MemoryPack`.

### 4.4 Store and service interfaces

```python
class SelectiveMemoryStore(Protocol):
    def append_events(self, events: Sequence[MemoryEvent]) -> int: ...
    def upsert_facts(self, facts: Sequence[MemoryFact]) -> int: ...
    def list_events(
        self, session_id: str, *, now_ms: int, limit: int
    ) -> list[MemoryEvent]: ...
    def list_facts(
        self,
        session_id: str,
        *,
        now_ms: int,
        statuses: tuple[str, ...],
        limit: int,
    ) -> list[MemoryFact]: ...
    def search_events(
        self,
        query: str,
        *,
        session_id: str,
        now_ms: int,
        decay_profile: str,
        top_k: int,
    ) -> list[MemoryEventMatch]: ...
    def search_facts(
        self,
        query: str,
        *,
        session_id: str,
        now_ms: int,
        decay_profile: str,
        top_k: int,
    ) -> list[MemoryFactMatch]: ...
    def delete_session(self, session_id: str) -> int: ...
```

`append_events` is idempotent by `event_id`: an identical replay returns without mutation; a different payload using an existing id fails closed. `upsert_facts` accepts only a same-session, monotonic revision update. Milvus and local adapters return the same scored-match shape; native decay and application fallback must expose `decay_mode` so traces and parity tests can distinguish them.

`SelectiveMemoryService` owns capture, selection, consolidation, projection and recall. The workflow must not duplicate scoring or fact-transition logic.

## 5. Episode capture and Selection Gate

Every validated terminal turn appends a bounded episode envelope. Raw tool bodies and KB chunks are not duplicated into Memory; only safe outcome summaries and stable references may be captured. A cancelled or incomplete stream appends no terminal episode.

The first selector is deterministic and uses registered signals:

| Signal | Example | Effect |
| --- | --- | --- |
| `explicit_remember` | “请记住以后用中文” | protected |
| `user_correction` | “不是 2.6，是 3.0” | protected + consolidation candidate |
| `task_transition` | open/completed/blocker changed | candidate |
| `failure_severity` | permission-safe retrieval repeatedly fails | candidate |
| `successful_reuse` | strategy succeeds on repeated task | raises salience |
| `novelty` | not represented by active facts | raises salience |
| `recurrence` | equivalent event occurs repeatedly | raises salience |
| `future_utility` | likely needed by current open task | raises salience |
| `ordinary_turn` | no durable state or learning signal | ephemeral |

The selector returns:

```text
discard-recallable-payload | ephemeral | promote_candidate | protected
```

Initial rule scores and precedence are binding:

| Rule | Score | Result |
| --- | ---: | --- |
| explicit remember or explicit user correction | `1.00` | `protected` |
| task opened/updated/completed | `0.85` | `promote_candidate` |
| answer rejected or retrieval failure | `0.70` | `promote_candidate` |
| successful strategy observed at least twice | `0.65` | `promote_candidate` |
| compatible event observed at least twice | `+0.10`, capped at `0.90` | preserves or raises candidate |
| registered ambiguous future-utility wording with no higher signal | `0.40` | `ephemeral`; eligible for optional LLM review |
| ordinary validated turn | `0.20` | `ephemeral` |

Higher-precedence rules win; additive recurrence never downgrades a result. Scores below `0.45` are ephemeral, scores in `0.45..<0.80` are candidates, and scores at or above `0.80` are candidates unless an explicit rule makes them protected. The optional LLM ambiguity band is `0.40..0.60`; it may choose only `ephemeral` or `promote_candidate`, never `protected`.

Explicit marker parsing is deliberately bounded:

- remember markers reuse the classifier's registered `请记住`/`记住`/`remember` set;
- correction markers are `不是…是…`, `更正`, `纠正`, `actually`, or `correction`;
- task transitions use registered open/update/complete markers;
- ambiguous future-utility matching requires paired semantics: Chinese `以后|下次|将来` + `可能|也许|或许` must be followed within 24 non-terminal characters by `复用|用到|需要|再用|还会用`; English requires `might|may|could` plus `reuse` or bounded `need|use … later|again`. A future phrase alone does not qualify and the signal does not mean explicit remember;
- no free-form model inference creates a user preference or correction.

`discard-recallable-payload` may retain a minimal redacted event envelope for trace lineage but stores no vector-searchable conversational payload. An optional `LLMMemorySelector` may review only scores in a configured ambiguity band; invalid output, timeout or missing configuration falls back to the rule result. The model receives only the bounded user query, terminal outcome code and deterministic rule decision, never the assistant answer, recalled Memory, KB chunks or tool bodies. Its strict schema contains exactly one `decision` enum with `ephemeral | promote_candidate`; local validation rejects unknown fields, explanations and `protected`. The rule score, event type and explicit-marker precedence remain authoritative.

Every decision stores `salience_score`, `selection_reason`, `retention_class` and bounded selector implementation metadata (`selector_name`, optional model, sanitized fallback reason). A successful LLM choice appends only the registered `llm_ephemeral` or `llm_promote_candidate` reason code; no rationale or hidden chain-of-thought is requested, persisted or traced. At most one provider call is allowed per terminal turn.

## 6. Forgetting and Milvus decay

### 6.1 Three forgetting layers

```text
soft forgetting      Milvus decay lowers recall rank
        │
        ▼
logical forgetting   expiry / supersession / dispute / tombstone blocks use
        │
        ▼
physical forgetting  bounded cleanup erases payload/vector after retention rule
```

Decay is relevance adjustment, not an availability decision. Search must first apply:

```text
session_id == active_session
AND status/retention_class is recallable
AND (expires_at is null OR expires_at > now_ms)
AND permission_scope_hash is compatible
```

Milvus documents decay reranking as:

```text
final_score = normalized_similarity_score * decay_score
```

The target SDK/service must prove this behavior, supported numeric types, hybrid-search composition and parameter units in Phase 0 before it becomes a hard dependency. The current official contract also limits one decay ranker to one numeric field and excludes grouping search, so salience/task/confidence remain application-rerank factors rather than additional decay inputs. See the official [Decay Ranker overview](https://milvus.io/docs/decay-ranker-overview.md) and [time-based ranking tutorial](https://milvus.io/docs/tutorial-implement-a-time-based-ranking-in-milvus.md).

### 6.2 Initial decay profiles

| Profile | Applies to | Function | Offset | Scale | Decay | Hard lifecycle |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `episode_fast` | ordinary recent episodes | `exp` | 1 day | 7 days | 0.5 | 7-day default TTL |
| `experience_balanced` | failure/success experiences | `gauss` | 3 days | 30 days | 0.5 | 90-day maximum unless promoted |
| `task_deadline` | bounded task state | `linear` | configured task window | configured deadline window | 0.5 | unavailable when task completes/expires |
| `durable_gentle` | confirmed facts where time matters | `gauss` on `last_confirmed_at` | 30 days | 180 days | 0.8 | version/status governed |
| `no_time_decay` | protected correction or durable preference | none | — | — | — | until consolidated, superseded or deleted |

These are initial evaluation values, not production defaults. All time values use milliseconds because the project-wide contract uses UTC epoch milliseconds. Profiles live in validated configuration; records store only the registered profile name.

Configuration defaults:

| Setting | Default | Validation |
| --- | ---: | --- |
| `SELECTIVE_MEMORY_ENABLED` | `true` | strict boolean |
| `MEMORY_CONTEXT_MAX_CHARS` | `4000` | `512..8192` |
| `MEMORY_LANE_TOP_K` | `3` | `1..20` |
| `MEMORY_PACK_MAX_RECORDS` | `12` | `1..20` |
| `MEMORY_CONSOLIDATION_BATCH_SIZE` | `20` | `2..100` |
| `MEMORY_RECURRENCE_THRESHOLD` | `2` | `2..10` |
| `MEMORY_DECAY_MODE` | `application` | `application` or verified `milvus` |
| `MEMORY_SELECTOR` | `rule_based` | `rule_based`, `auto` or `openai` |
| `MEMORY_SELECTOR_AMBIGUITY_MIN` | `0.40` | finite and inside `0.40..0.60` |
| `MEMORY_SELECTOR_AMBIGUITY_MAX` | `0.60` | finite, `min ≤ max`, inside `0.40..0.60` |
| `OPENAI_MEMORY_SELECTOR_MODEL` | empty | falls back to `OPENAI_MODEL` |
| `OPENAI_MEMORY_SELECTOR_TIMEOUT_SECONDS` | `5` | finite and `> 0` |

Selecting `milvus` without a successful capability probe is a configuration error; no silent fallback may be labeled native.
`MEMORY_SELECTOR=rule_based` performs no model initialization. `auto` and
`openai` still fall back to the rule result when credentials/model/client are
unavailable, as required by the optional-selector contract; the provider is
called only if the rule score is inside the validated band.

### 6.3 Native decay adapter and startup probe

`MEMORY_DECAY_MODE=milvus` is an explicit fail-closed deployment choice. After
both Memory collections are loaded and before the workflow is returned, the
adapter must issue three read-only, guaranteed-empty searches against
`memory_events`: one each for `exp`, `gauss` and `linear`. Every request uses
the public PyMilvus `Function(function_type=RERANK, params={"reranker":
"decay", ...})` contract, the numeric `event_time` field, a deterministic
1024-dimensional probe vector, millisecond `origin`/`offset`/`scale`, COSINE,
no grouping, and a session filter that cannot match a valid identifier. The
probe never embeds text, reads Memory payload, inserts data or creates/drops a
collection. It proves that the configured SDK/server accepts the exact
standard-search request shape; the Phase 0 external matrix remains responsible
for a disposable-collection score-point/order exercise.

The adapter becomes `native_decay_verified` only after all three requests
succeed and return the ordinary nested search-result envelope. An unsupported
SDK symbol, malformed response or server error blocks startup with a sanitized
`SelectiveMemoryError`; it never changes the requested mode to `application`.
Application mode neither imports PyMilvus decay symbols nor runs this probe.

For real recall, episode lanes bind the registered profile to `event_time` and
fact lanes bind it to `last_confirmed_at`, with `origin=now_ms`.
`no_time_decay` intentionally supplies no ranker and uses the base Milvus
COSINE score. Native returned score is the semantic×decay component; the
adapter then applies the same bounded application factors as the fallback:
event `salience_score`, and fact `salience_score × confidence`. Candidate
pooling remains `min(20, top_k × 4)`, scalar scope/expiry/status filters run in
Milvus before ranking, and the adapter validates every returned entity again
before merging. `MemoryEventMatch`, `MemoryFactMatch`, `MemoryPack` and trace
metadata report `decay_mode="milvus"` even for an empty result or the
`no_time_decay` lane, because the mode describes the verified store execution
contract rather than whether one particular profile has a time function.

```text
startup                                  recall
  load event/fact collections             validate session/profile/time
              │                                      │
              ▼                                      ▼
  empty exp/gauss/linear searches          scalar scope + expiry/status filter
              │                                      │
       all accepted? ── no ──► fail                  ▼
              │ yes                         native decay/base COSINE
              ▼                                      │
  native_decay_verified=true                         ▼
                                          salience/confidence merge
                                                     │
                                                     ▼
                                          scope recheck + bounded top-k
```

Milvus applies decay within separate recall lanes:

```text
recent episodes       semantic/hybrid × episode_fast
operational experience semantic/hybrid × experience_balanced
active task state      semantic/hybrid × task_deadline
durable facts          semantic/hybrid × durable_gentle or no_time_decay
```

Each lane takes a bounded candidate set. The application reranker merges lanes using:

```text
final_memory_score =
    milvus_relevance_after_decay
  * salience_factor
  * task_relevance_factor
  * source_confidence
  * confirmation_factor
  - contradiction_penalty
```

The exact factor calibration is fixed by evaluation fixtures, not guessed at runtime. A very recent irrelevant event must not outrank an older exact task fact solely because of recency.

### 6.3 Reconfirmation and anti-feedback invariant

Simply retrieving or displaying a memory never refreshes `event_time`, `last_confirmed_at` or TTL. Otherwise frequently recalled items become permanently self-reinforcing.

Only these events may extend lifecycle:

- explicit user reconfirmation;
- a task outcome proves the recalled strategy useful;
- consolidation verifies repeated compatible evidence.

Reconfirmation appends `memory_reconfirmed` and updates the projected fact revision. It never mutates the original event timestamp.

## 7. Consolidation and projection

Consolidation runs after a bounded number of selected candidates, explicit session close, or a manual workshop action. It:

1. loads only eligible same-session candidate/protected events;
2. clusters by typed subject/predicate/task, not only embedding similarity;
3. detects compatible repetition, correction and contradiction;
4. emits a new `MemoryFact` revision with complete `source_event_ids`;
5. appends `memory_promoted` and, when necessary, `memory_superseded`;
6. rebuilds the working-state projection;
7. marks processed candidates without deleting their source lineage.

Consolidation is lossy and therefore cannot delete original evidence immediately. An LLM consolidator, if enabled, must return a strict typed proposal. Deterministic validation owns enum, scope, source resolution, conflict, length and confidence constraints. User facts/preferences inferred without explicit evidence are rejected.

Initial working-state projections are:

- active tasks and blockers;
- confirmed user preferences;
- current decisions;
- recent corrections;
- recurring failure patterns;
- validated successful strategies;
- unresolved conflicts.

Projection must be deterministic from events and fact revisions. Replay equivalence is tested even though arbitrary historical forks are deferred.

The demo performs a bounded consolidation pass after each protected event and whenever at least `MEMORY_RECURRENCE_THRESHOLD` compatible candidate events exist. The pass reads at most `MEMORY_CONSOLIDATION_BATCH_SIZE` newest eligible events. Its idempotency key is the ordered source-event-id digest; replaying the same batch creates no new fact revision or promotion event.

The deterministic first implementation promotes only shapes it can validate:

- explicit language preference → `user_preference/user/preferred_language`;
- other explicit remembered statement → `user_fact/user/remembered:<content-hash>`;
- parsed `不是 <old>，是 <new>` correction → `user_fact/user/correction:<old-normalized>`;
- task transition → `task_state/task/<task-hash>`;
- repeated equivalent failure/success events → `failure_pattern` or `successful_strategy`.

An unsupported protected statement remains a protected episode and can still be recalled; it is not coerced into an invented subject/predicate.

### 7.1 Consolidation journal/outbox

Consolidation crosses three durable mutations—fact revisions, lifecycle event,
and completion state—so it must not rely on an in-process `try` block. Before
the first projection mutation, the service persists one
`ConsolidationJournalEntry` in `memory_consolidation_journal`. The entry stores
the deterministic operation id, session/trigger/source ids, the exact validated
fact-update and lifecycle-event outbox payload, `pending|applied` status,
bounded attempt count, timestamps and a registered sanitized error code.
Journal payload is sensitive Memory data: it is session-scoped, never traced or
shown in UI, and is deleted by the same authorized session erase. To stay below
Milvus's per-JSON-field limit, plan metadata, up to two fact envelopes and the
lifecycle-event envelope are separate JSON fields; their three 1024-d vectors
live in dedicated non-indexed base64 IEEE-754 float64 VarChar fields. Each non-vector envelope uses
the registered `zlib-json-v1` codec, is capped at 40,000 base64 characters and
decodes to at most 128,000 bytes. Malformed, oversized or unknown-codec payloads
fail closed before replay. The Milvus collection also carries a constant
two-dimensional `journal_anchor_vector=[1,0]`, indexed with
`AUTOINDEX/COSINE`, solely to satisfy the collection schema contract; it is
never searched and has no semantic meaning. Before enqueue, fact and
lifecycle-event vectors are canonicalized to IEEE-754 float32, matching Milvus
`FloatVector` readback. Therefore a retry after an insert succeeded but its
flush/RPC response failed compares equal to the exact journal plan instead of
producing a false identity collision.

The operation id is a stable digest of session id plus ordered source event ids.
An identical enqueue is idempotent; any different payload under the same id
fails closed. Draining reads at most `MEMORY_CONSOLIDATION_BATCH_SIZE` oldest
pending entries for one validated session. It replays the exact payload in
order: idempotent fact upserts, idempotent lifecycle-event append, then journal
status `applied`. The oldest-first scalar query pushes down a bounded `limit`
and Milvus 3.0 order syntax `created_at:asc, operation_id:asc`. A failure before
the final marker leaves the entry pending,
increments attempts and records only `fact_write_failed` or
`event_write_failed`; the next terminal turn or a manual drain retries it.
Failure to update the journal itself surfaces as a sanitized dependency error
and leaves the previous pending state intact. There is no silent abandonment
and no raw dependency message in journal or trace. Replay after either partial
write produces no new revision because all ids and payloads were fixed before
enqueue.

Only one projection chain may advance per session. If the oldest pending entry
still fails, draining stops and the current turn reports `deferred_pending`
without deriving another revision from stale projection state. The event
remains append-only evidence; a later equivalent consolidation includes all
bounded equivalent source events, so deferred evidence rejoins lineage after
the fence clears. Capture, drain and authorized erasure share one re-entrant
per-session lock in the service, so an erasure waits for an in-flight projection
and then removes its journal/fact/event writes. This is a process-local workshop
contract: deployments with multiple service processes must provide session
affinity or replace it with a distributed per-session lock before scaling out.

```text
validated plan
     │
     ▼
enqueue pending ── failure ──► fail closed (no projection write)
     │
     ▼
idempotent fact updates
     │
     ▼
idempotent lifecycle event
     │
     ▼
mark applied

next turn/manual drain ──► oldest pending ──► same replay path
```

## 8. Recall protocol

`recall_memory` becomes a Memory Router:

1. Load bounded response-cache candidates into the separate private cache field.
2. Decide whether the current question needs working state, durable facts, recent episodes or an explicit memory-only answer.
3. Apply session, permission, status and expiry filters before vector/hybrid ranking.
4. Search at most four typed lanes with the configured Milvus decay profile or deterministic local equivalent.
5. Merge with the application score, deduplicate by event/fact identity and expose conflicts.
6. Render a typed `MemoryPack` under record and character budgets.
7. Pass only the required sections to classifier/rewrite/generator.

Exact active task state and explicit corrections are retrieved deterministically before semantic search. Semantic similarity is an optimization for relevance, not the sole source of current state.

Before the current KB permission decision exists, only `session_private` preferences, user facts and task state may enter rendered context. Permission-scoped operational experiences remain private candidates until their stored `permission_scope_hash` equals the current decision. The Workshop's synthetic permission scope is deterministic, but the check remains explicit in code.

The current query remains authoritative. Recalled Memory:

- may resolve pronouns, task continuity and explicit personal recall;
- may suggest a previously successful non-mutating strategy;
- cannot grant permission, select a mutation tool, modify metadata filters or override explicit current instructions;
- cannot become `[Cn]` evidence or satisfy KB evidence grading;
- must expose source ids and confidence internally for memory-only validation.

## 9. Trust, privacy, and deletion

- All stores remain scoped to the generated session in this milestone. Cross-session durable memory requires explicit identity, consent and permission design and is deferred.
- Event content, facts, embeddings and rendered MemoryPack are sensitive session data and never enter presentation-safe traces or ordinary logs.
- `Clear conversation & memory` physically removes same-session event/fact payloads and vectors plus response-cache entries. A non-recoverable tombstone may retain only deletion time, opaque ids and non-content integrity metadata when required for local consistency.
- Append-only means normal correction/consolidation does not rewrite history; it does not override an authorized erasure request.
- Permission-scope mismatch fails closed. A memory written under a broader scope cannot be released under a narrower or unrelated scope.
- Raw KB chunks, prompts, credentials, provider errors and unrestricted tool results are never copied into Memory.

### 9.1 Bounded physical cleanup

Physical cleanup is an explicit session-scoped maintenance operation, separate
from recall. One request binds `session_id`, an immutable `now_ms` snapshot, a
`page_size` in `1..100`, and an optional opaque cursor. The cursor is
versioned, bounded, HMAC-SHA256 authenticated, bound to the session digest and
snapshot, and carries only the current `facts|events` stage plus the last
validated primary key. Unknown, malformed, tampered, cross-session or
cross-snapshot cursors fail before any mutation. The service accepts an
injected/deployment `MEMORY_CLEANUP_CURSOR_SECRET` of at least 32 bytes; a
process-random fallback exists only for same-process local use.

The facts stage selects only `status == tombstoned` or
`expires_at <= now_ms`; the events stage selects only
`expires_at <= now_ms`. Each stage uses primary-key ascending keyset pagination,
pushes `limit=remaining+1` and Milvus 3.0 `field:asc` ordering into the query,
and deletes only the returned same-session ids while repeating the eligibility
predicate. It never issues a session-wide or eligibility-only delete. One page
examines at most `page_size` rows across both stages and reports separate fact,
event and protected-event counts plus an opaque next cursor.

An expired event remains physically present while any non-expired,
non-tombstoned same-session fact names it in `source_event_ids`; this preserves
lineage for every retained projection. A tombstoned fact is itself eligible,
but its source episode follows the independent event expiry rule rather than an
implicit cascade. Any pending consolidation journal entry blocks the cleanup
page with zero mutation. Capture, replay, cleanup and authorized erasure share
the same per-session service lock. Cleanup is intentionally a service method,
not a standalone Milvus-writing CLI: an independent process could not share
that lock. Multi-process deployments must affinity-route all operations for one
session or replace the lock with a distributed equivalent. Rerunning the same
cursor is safe: exact-id deletes are idempotent, and a fresh run eventually
sees eligible records inserted behind an older cursor.

## 10. Observability

Safe trace details may include:

```json
{
  "memory_status": "empty | recalled | degraded",
  "working_state_count": 0,
  "durable_fact_count": 0,
  "episode_candidate_count": 0,
  "selected_episode_count": 0,
  "conflict_count": 0,
  "selection_reason_counts": {},
  "decay_profiles": [],
  "consolidation_status": "not_run | completed | degraded",
  "truncated_count": 0
}
```

Trace must not expose Memory content, embeddings, source payloads, selector prompts or model reasoning. Metrics distinguish capture, selection, recall, projection and consolidation latency. Decay profile and version are recorded so ranking changes are reproducible.

The Memory tab derives a bounded, presentation-safe session dashboard from at
most 200 live rows. It shows retention-class, selection-reason, decay-profile
and fact-status distributions. Its lineage table includes only opaque fact and
event ids, revision, `source_event_ids`, `supersedes_memory_id`, parent/branch
ids and selector implementation name. Every source id is marked resolved or
missing against the same returned session page. It never includes content,
fact values, summaries, vectors, selector prompts or model rationale.
Selection reasons are validated against the registered code allowlist at rule,
event and storage-decode boundaries. `MemoryFact.source_event_ids` is capped at
100 and the UI renders at most 500 lineage edges while reporting omitted row
and edge counts. Lineage rows include branch and selector implementation
metadata for event nodes.

## 11. Performance and operational limits

- At most four recall lanes per query; each lane has `top_k <= 20`.
- The merged `MemoryPack` contains at most 12 records and `MEMORY_CONTEXT_MAX_CHARS=4000` by default.
- Selection uses deterministic rules first; at most one optional LLM selection call per completed turn.
- Consolidation is outside answer critical path and processes a bounded batch with an idempotency key.
- A failed lane degrades independently; a valid KB answer is not failed because optional Memory is unavailable.
- Physical cleanup follows §9.1: at most 100 examined rows per page, opaque
  keyset cursor, pending-outbox fence, exact-id revalidation and no unvalidated
  broad delete expression.

No latency SLA is claimed until local and real Milvus baselines include no-decay, decay, multi-lane merge and consolidation measurements.

## 12. Tests and acceptance criteria

### 12.1 Deterministic behavior

- ordinary turns remain ephemeral and do not create durable user facts;
- explicit remember, correction and task transition select the correct retention class;
- out-of-band rule decisions make zero LLM calls; in-band valid output may only keep ephemeral or promote candidate;
- timeout, unavailable configuration, provider failure, extra fields and invalid enums reproduce the rule decision with a sanitized fallback reason;
- selector prompts and traces contain no assistant answer, Memory payload, model rationale or hidden chain-of-thought;
- correction supersedes an older fact while retaining source lineage;
- repeated compatible episodes consolidate into one versioned fact;
- conflicting facts become disputed and do not silently enter active working state;
- current turn is invisible to its own recall.

### 12.2 Forgetting

- records inside `offset` retain full decay weight;
- records at configured `scale` match the expected decay score within tolerance;
- exponential, Gaussian and linear local test doubles preserve target Milvus ordering;
- low-salience old episodes fall below relevant recent items;
- an older exact durable fact remains retrievable when a recent irrelevant episode exists;
- expiry, supersession and tombstone block retrieval regardless of decay score;
- retrieval alone never refreshes time; explicit reconfirmation appends lineage and extends the projected lifecycle;
- paged cleanup removes only eligible same-session payloads, preserves expired
  events referenced by retained facts, rejects cursor substitution, never
  exceeds the page bound, and resumes without duplicate side effects.

### 12.3 Safety and separation

- another session has zero event/fact/cache visibility;
- Memory cannot produce KB citation, grant permission or change tool filters;
- response-cache hit/miss behavior remains independent of Memory promotion and decay;
- clear erases active-session Memory and response cache without affecting another session;
- failure traces contain bounded reason codes and counts only.

### 12.4 Quality measures

The fixture set reports:

- selection precision/recall by event class;
- active-fact precision after consolidation;
- correction/supersession accuracy;
- relevant-memory recall before and after decay;
- stale-memory intrusion rate;
- conflict detection rate;
- average MemoryPack records/characters and truncation rate;
- local/Milvus ranking parity.

Phase acceptance requires zero permission/citation/session-isolation violations. Quality thresholds for selection and decay are set only after the baseline is recorded.

## 13. Incremental rollout

1. **Typed capture and selection**: add event envelope, registered reason codes and rule selector; dual-write current `conversation_memory` for compatibility.
2. **Decay-aware recall**: add numeric lifecycle fields, Phase 0 capability proof, local decay parity and typed recall lanes.
3. **Consolidation and projection**: add fact revisions, correction/conflict handling and `MemoryPack`.
4. **Cutover and cleanup**: stop creating legacy summaries after parity tests pass; retain a bounded migration reader, then remove it in a separately approved change.
5. **Deferred research**: evaluate cross-session identity/consent, durable event-store technology, replay/fork/merge and promoted strategy evaluation.

Each step must leave the current RAG and response-cache contracts green.

## 14. Cross-references

- ← Baseline: [`10b-conversation-memory.md`](./10b-conversation-memory.md)
- ← Storage and identity: [`10-data-model.md`](./10-data-model.md)
- ↔ Response reuse boundary: [`10c-grounded-response-cache.md`](./10c-grounded-response-cache.md)
- → Runtime consumer: [`12-agent-workflow.md`](./12-agent-workflow.md)
- → UI consumer: [`20-ui-demo.md`](./20-ui-demo.md)
- ↔ Validation: [`70-quality-and-evaluation.md`](./70-quality-and-evaluation.md)
- ↔ Decisions: [`99-key-decisions.md § D21`](./99-key-decisions.md#d21--memory-uses-selective-dual-speed-storage-and-projection), [`99-key-decisions.md § D22`](./99-key-decisions.md#d22--milvus-decay-provides-soft-forgetting-only)
