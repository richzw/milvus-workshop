# 93 — First-version Review Findings

Status: active · Last updated: 2026-07-24

This is the canonical backlog for findings discovered while reviewing the first runnable demo against the numbered specs. Entries are closed only after the named test or evidence exists.

## In-scope fixes

| ID | Severity | Finding | Spec contract | Fix shape | Status |
| --- | --- | --- | --- | --- | --- |
| R1 | P1 | Local `run`/`stream` overwrite the non-retrieval response with an abstention | `12 § 4.1`, `12 § 5` | Preserve direct terminal response; pinned by `demo/tests/test_workflow.py:48` | Fixed |
| R2 | P1 | Preferred LangGraph adapter has no streaming surface, so installed UI loses answer streaming | `12 § 6`, `20 § 3` | Add adapter streaming; pinned by `demo/tests/test_workflow.py:103` | Fixed |
| R3 | P1 | Responses lack `query_id`/`session_id`; tabs cannot prove snapshot consistency | `10 § 6.1`, `12 § 3`, `20 § 5` | Add validated identity; pinned by `demo/tests/test_workflow.py:60` | Fixed |
| R4 | P1 | Stage latency values are fabricated from total elapsed time | `12 § 6`, `70 § 4.3` | Measure actual stages in `demo/src/agent_workshop_demo/workflow.py:446` | Fixed |
| R5 | P1 | Ingestion IDs depend on absolute checkout path and missing roots are silently accepted | `11 § 4`, `11 § 6` | Logical URI IDs; pinned by `demo/tests/test_ingestion_eval_memory.py:122` and `:146` | Fixed |
| R6 | P1 | Query filters and blank questions are not validated at the workflow boundary | `10 § 3.1`, `12 § 5`, `20 § 6` | Central validation; pinned by `demo/tests/test_workflow.py:92` | Fixed |
| R7 | P2 | Eval calls hit-rate `recall_at_k` and ignores per-question recall denominator | `70 § 4.2` | Per-case ratios; pinned by `demo/tests/test_ingestion_eval_memory.py:167` | Fixed |
| R8 | P2 | README points to archived requirements and overstates native Milvus/model behavior | `index.md § Source status`, `70 § 6` | Link numbered specs and distinguish deterministic fallback from verified native features | Fixed |
| R10 | P1 | Dependency failures surfaced without stage or query context | `12 § 5`, `20 § 6` | Wrap causes in `WorkflowStageError`; pinned by `demo/tests/test_workflow.py` | Fixed |
| R11 | P1 | The first version concatenates selected chunks instead of synthesizing a grounded answer with an LLM | `13 § 3–8` | Add an injected OpenAI Responses generator, citation guard, deterministic fallback and trace; pinned by `demo/tests/test_generation.py` and `demo/tests/test_workflow.py` | Fixed |
| R12 | P1 | Dense chunk/query vectors use a deterministic hash placeholder rather than a semantic embedding model | `10a`, `11 § 5` | Add an injected OpenAI Embeddings provider behind `dense_vector()` with fixed 1024 dimensions and no implicit network default; pinned by `demo/tests/test_embedding.py` | Fixed |
| R13 | P1 | `unknown` query classification vetoes relevant retrieved evidence | `12 § 4.1`, `12 § 4.5` | Grade query-to-chunk relevance independently of the teaching classifier; pinned by `demo/tests/test_workflow.py` | Fixed |
| R14 | P1 | The local streaming fallback leaks raw generation failures without stage/query context | `12 § 5`, `20 § 6` | Buffer validated chunks inside the measured stage and wrap failures consistently; pinned by `demo/tests/test_workflow.py` | Fixed |
| R15 | P1 | Offline eval ignores golden facts, reranked/selected recall and abstention correctness | `70 § 3–4` | Load both fixtures and report the complete metric set; pinned by `demo/tests/test_ingestion_eval_memory.py` | Fixed |
| R16 | P1 | Offline ingestion silently skips PDFs and unsupported document types | `11 § 3.1`, `11 § 6` | Parse unmanifested PDFs page-by-page and reject unhandled types explicitly; pinned by `demo/tests/test_ingestion_eval_memory.py` | Fixed |
| R17 | P1 | Streamlit exposed source/doc/department filters, so users performed knowledge routing instead of the Agent | `12 § 4`, `20 § 2–5` | Remove Search Controls and make registered tools own validated metadata filters; pinned by `demo/tests/test_agentic_tools.py` | Fixed |
| R18 | P1 | Workflow lacked permission gating, explicit tool selection, bounded decomposition, multi-hop provenance and terminal answer self-check | `12 § 2–7`, `70 § 4.1` | Add explicit LangGraph/local nodes, bounded tool plan and comparison fixture; pinned by `demo/tests/test_agentic_tools.py` and `demo/eval/questions.json:q004` | Fixed |

## Deferred findings

| ID | Severity | Finding | Why deferred | Required next step |
| --- | --- | --- | --- | --- |
| D1 | P1 | Real Milvus collection/index/insert/hybrid query round trip is not tested | Insert/query adapter has deterministic fake-client coverage, but exact-version server evidence still requires a running service | Execute `91 § Phase 0` capability matrix and update schema/index parameters from observed APIs |
| D2 | P2 | Binary-vector MinHash representation is experimental and semantic validity is unproven | P2 feature; spec explicitly gates it | Validate near-duplicate fixtures or keep checksum-only dedup |
| D3 | P2 | DINOv3 and model reranker are labels/fallbacks, not integrated models | P2/model selection remains unverified | Benchmark selected models and record dimensions/runtime in `docs/research/` |
| D4 | P3 | The installed experimental `ruff format` corrupts Python source into placeholder AST names | Tool version is external to this repo; all affected source was recovered and verified | Do not use this formatter version; upgrade/pin Ruff before enabling formatting automation |
| D5 | P3 | Full strict mypy is blocked by pymilvus lacking `py.typed` and the installed mypy rejecting a Python 3.14 NumPy stub | Core 15 modules/tests pass strict mypy; failure is inside third-party typing metadata | Pin a supported Python/mypy toolchain and add a narrow optional-adapter typing policy without weakening core checks |
| D6 | P1 | A real configured OpenAI Responses round trip has not been executed | No API credential/model entitlement was available in this session; SDK 2.46.0 signatures and fake-client behavior are verified | Run an opt-in smoke query with `OPENAI_API_KEY` and `OPENAI_MODEL`, then record model, latency and citation-valid output without storing prompt or response bodies |
| D7 | P1 | A real configured OpenAI Embeddings round trip has not been executed | SDK 2.46.0 request signature and fake-client behavior are verified; the task does not authorize spending API quota with ambient credentials | Run an opt-in smoke with `EMBEDDING_PROVIDER=openai`, confirm a finite 1024-value response, then re-ingest the target collection before semantic queries |
| D8 | P2 | Selective-Memory consolidation commits the fact projection before its deterministic lifecycle event | The Phase 6 review fixed correctness/security P1s first; crash-safe multi-collection commit needs a journal/outbox contract rather than a partial retry patch | Add a replayable consolidation outbox and a fault-injection test spanning `selective_memory.py:_consolidate` fact/event writes |
| D9 | P2 | Selective-Memory has same-session clear but no bounded resumable physical cleanup for expired/tombstoned rows | Logical expiry/status already blocks recall; safe physical deletion requires an operator-facing cursor/progress contract | Specify and implement validated session/status/expiry cleanup batches with cross-session non-deletion tests |
| D10 | P2 | Milvus selective-memory listing bounds results only after iterator reads and fake-client parity does not prove full server ordering/filter semantics | Real Milvus remains Phase 0-gated and the current adapter preserves correctness for the bounded workshop corpus | Push limit/order to Milvus, share a local/Milvus contract suite, and execute an opt-in correction/conflict/expiry/isolation/clear/decay round trip |
