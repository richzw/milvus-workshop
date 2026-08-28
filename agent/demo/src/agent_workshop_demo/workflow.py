"""Deterministic Agentic RAG workflow matching the workshop MVP."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
import uuid
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from typing import Any, cast

from agent_workshop_demo.classification import (
    ClassificationRequest,
    QueryClassifier,
    RuleBasedQueryClassifier,
    detect_memory_recall,
    validate_classification_result,
)
from agent_workshop_demo.config import (
    DEFAULT_SEARCH_PARAMS,
    MAX_EXHAUSTIVE_CONTEXTS,
)
from agent_workshop_demo.context_compression import (
    ContextCompressor,
    DisabledContextCompressor,
    validate_compression_run,
)
from agent_workshop_demo.embedding import (
    TextEmbeddingError,
    text_embedding_fingerprint,
    tokenize,
)
from agent_workshop_demo.entities import EntityCatalog, load_entity_catalog
from agent_workshop_demo.events import (
    EventKind,
    EventStatus,
    WorkflowEventEmitter,
)
from agent_workshop_demo.generation import (
    ANSWER_CHUNK_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXTS,
    AnswerGenerator,
    DeterministicAnswerGenerator,
    GenerationContext,
    GenerationRequest,
    validate_generation_result,
)
from agent_workshop_demo.knowledge_tools import (
    SEARCH_TOOLS,
    DemoPermissionChecker,
    KnowledgeSearchTool,
    PermissionChecker,
)
from agent_workshop_demo.memory import (
    MAX_SESSION_RECORDS,
    ConversationMemory,
    ConversationMemoryStore,
    MemoryRecord,
    MemoryStoreError,
    build_turn_records,
    utc_now_ms,
)
from agent_workshop_demo.models import (
    AgentState,
    EvidenceAction,
    EvidenceEvaluation,
    KBChunk,
    QueryRoute,
    QueryRouteResult,
    RerankedResult,
    RetrievalPlanResult,
    SearchResult,
)
from agent_workshop_demo.reranker import (
    FallbackReranker,
    Reranker,
    RuleBasedReranker,
    validate_rerank_run,
)
from agent_workshop_demo.query_transform import (
    IdentityQueryTransformer,
    QueryTransformRequest,
    QueryTransformer,
    RuleBasedQueryTransformer,
    validate_transformation,
)
from agent_workshop_demo.response_cache import (
    DEFAULT_KB_REVISION,
    DEFAULT_RESPONSE_CACHE_SIMILARITY_THRESHOLD,
    DEFAULT_RESPONSE_CACHE_TOP_K,
    DEFAULT_RESPONSE_CACHE_TTL_SECONDS,
    RESPONSE_CACHE_WORKFLOW_VERSION,
    CachedEvidence,
    GroundedResponseCache,
    GroundedResponseCacheStore,
    ResponseCacheCandidate,
    ResponseCacheError,
    build_cache_record,
    permission_scope_hash,
    query_constraints,
)
from agent_workshop_demo.retrieval import HybridRetriever, InMemoryHybridRetriever
from agent_workshop_demo.retrieval_tier import (
    LEXICAL_ONLY_CATALOG_VERSION,
    LexicalOnlyRetriever,
    RetrievalTier,
    RetrievalTierConfig,
    SparseSearchRetriever,
    parse_tier,
)
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.selective_memory import (
    MemoryPack,
    SESSION_PRIVATE_SCOPE_HASH,
    SelectiveMemoryError,
    SelectiveMemoryService,
)
from agent_workshop_demo.transitions import (
    WorkflowNode,
    mark_no_progress_abstention,
    next_transition,
)
from agent_workshop_demo.validation import (
    normalize_filters,
    validate_identifier,
    validate_question,
)

DIRECT_RESPONSE = (
    "这个问题不需要检索内部知识库。请问一个和 workshop、RAG 或内部资料相关的问题。"
)
UNSUPPORTED_OPERATION_RESPONSE = (
    "这个 Workshop 是只读查询 demo，不能执行修改、删除、审批或提交操作。"
)
PERMISSION_DENIED_RESPONSE = "当前演示身份没有访问所需知识域的权限。"
CLARIFICATION_REQUIRED_RESPONSE = (
    "问题中的领域术语或文档版本存在歧义，请补充具体行业场景或两个版本。"
)
RELEVANCE_THRESHOLD = 0.22
MAX_INITIAL_SUBQUERIES = 3
MAX_DOCUMENT_EXPANSION_CHUNKS = 20
DEFAULT_MEMORY_TOP_K = 3
DEFAULT_MEMORY_TTL_SECONDS = 86_400
MAX_MEMORY_CONTEXT_CHARS = 2_000
VERSION_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:v\d+(?:\.\d+)*|\d{4}\.\d{1,2})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
PRODUCT_BARE_VERSION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])milvus\s+"
    r"(?P<version>\d+(?:\.\d+)+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
MULTI_ASPECT_MARKER_FAMILIES = (
    ("是什么", "定义", "what is", "definition"),
    ("原理", "怎么工作", "如何工作", "how it works", "mechanism"),
    (
        "怎么用",
        "如何使用",
        "使用方法",
        "操作步骤",
        "配置",
        "how to use",
        "configuration",
    ),
    (
        "限制",
        "约束",
        "风险",
        "注意事项",
        "limitations",
        "constraints",
        "risks",
    ),
    (
        "优缺点",
        "优点",
        "缺点",
        "优势",
        "劣势",
        "trade-offs",
        "pros and cons",
    ),
)


@dataclass(frozen=True)
class _PreparedToolCall:
    """One immutable, permission-filtered search invocation."""

    plan_item: dict[str, Any]
    tool: KnowledgeSearchTool
    filters: dict[str, Any]
    version_scope: dict[str, Any]


@dataclass(frozen=True)
class _ToolSearchOutcome:
    """One isolated search result applied later in deterministic plan order."""

    prepared: _PreparedToolCall
    results: tuple[SearchResult, ...]
    latency_ms: float
    retrieval_profile: str = "flat_hybrid"
    capability_status: str = "ready"
    document_candidates: tuple[dict[str, Any], ...] = ()


UNSERIALIZED_STATE_FIELDS = frozenset(
    {
        "retrieved_chunks",
        "reranked_chunks",
        "response_cache_candidates",
        "selective_memory_private_values",
        "generation_contexts",
        "generation_citation_map",
    }
)

CURRENT_VERSION_MARKERS = ("current", "latest", "当前", "最新版")
VERSION_COMPARISON_MARKERS = ("版本", "edition")
CONTEXTUAL_MEMORY_MARKERS = (
    "刚才",
    "前面",
    "上述",
    "继续",
    "基于此",
    "它",
    "该流程",
    "what about",
    "how about",
    "that ",
    " it ",
)


class WorkflowStageError(RuntimeError):
    """Wrap a dependency failure with its stage and query identity."""

    def __init__(self, stage: str, query_id: str, cause: Exception) -> None:
        self.stage = stage
        self.query_id = query_id
        super().__init__(
            f"Workflow stage {stage!r} failed for query {query_id!r}: {cause}"
        )


class AgenticRAGWorkflow:
    """Local node sequence compatible with the LangGraph adapter."""

    node_sequence = [
        "recall_memory",
        "classify_and_route",
        "resolve_terminology",
        "check_permission",
        "try_grounded_cache",
        "recall_authorized_experience",
        "plan_retrieval",
        "execute_tool_plan",
        "rerank_evidence",
        "evaluate_evidence",
        "prepare_generation_context",
        "generate_answer_streaming",
        "verify_answer",
        "persist_turn_memory",
    ]

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        query_classifier: QueryClassifier | None = None,
        query_transformer: QueryTransformer | None = None,
        answer_generator: AnswerGenerator | None = None,
        context_compressor: ContextCompressor | None = None,
        permission_checker: PermissionChecker | None = None,
        entity_catalog: EntityCatalog | None = None,
        memory_store: ConversationMemory | None = None,
        memory_top_k: int = DEFAULT_MEMORY_TOP_K,
        memory_ttl_seconds: int = DEFAULT_MEMORY_TTL_SECONDS,
        selective_memory: SelectiveMemoryService | None = None,
        selective_memory_enabled: bool = True,
        response_cache: GroundedResponseCache | None = None,
        response_cache_enabled: bool = True,
        response_cache_top_k: int = DEFAULT_RESPONSE_CACHE_TOP_K,
        response_cache_ttl_seconds: int = (DEFAULT_RESPONSE_CACHE_TTL_SECONDS),
        response_cache_similarity_threshold: float = (
            DEFAULT_RESPONSE_CACHE_SIMILARITY_THRESHOLD
        ),
        kb_revision: str = DEFAULT_KB_REVISION,
        retrieval_tier: RetrievalTier | str = RetrievalTier.HYBRID_DENSE,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock_ms: Callable[[], int] = utc_now_ms,
    ) -> None:
        if not 1 <= memory_top_k <= 20:
            raise ValueError("memory_top_k must be between 1 and 20")
        if memory_ttl_seconds <= 0:
            raise ValueError("memory_ttl_seconds must be positive")
        if not 1 <= response_cache_top_k <= 20:
            raise ValueError("response_cache_top_k must be between 1 and 20")
        if response_cache_ttl_seconds <= 0:
            raise ValueError("response_cache_ttl_seconds must be positive")
        if not 0 <= response_cache_similarity_threshold <= 1:
            raise ValueError("response_cache_similarity_threshold must be in [0, 1]")
        if not kb_revision.strip() or len(kb_revision) > 128:
            raise ValueError("kb_revision must contain 1..128 characters")
        self.retrieval_tier = RetrievalTierConfig(parse_tier(retrieval_tier))
        configured_retriever = retriever or InMemoryHybridRetriever(load_kb_chunks())
        self.retriever = self._tier_retriever(configured_retriever)
        self.reranker = reranker or RuleBasedReranker()
        self.query_classifier = query_classifier or RuleBasedQueryClassifier()
        self.query_transformer = (
            query_transformer or RuleBasedQueryTransformer()
            if self.retrieval_tier.uses_query_transformation
            else IdentityQueryTransformer()
        )
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.context_compressor = context_compressor or DisabledContextCompressor()
        self.permission_checker = permission_checker or DemoPermissionChecker()
        self.entity_catalog = (
            entity_catalog or load_entity_catalog()
            if self.retrieval_tier.uses_query_transformation
            else EntityCatalog(LEXICAL_ONLY_CATALOG_VERSION, ())
        )
        self.memory_store = memory_store or ConversationMemoryStore(
            now_ms=wall_clock_ms()
        )
        self.selective_memory = selective_memory or SelectiveMemoryService()
        self.selective_memory_enabled = selective_memory_enabled
        self.response_cache = response_cache or GroundedResponseCacheStore(
            now_ms=wall_clock_ms()
        )
        self.memory_top_k = memory_top_k
        self.memory_ttl_seconds = memory_ttl_seconds
        self.response_cache_enabled = response_cache_enabled
        self.response_cache_top_k = response_cache_top_k
        self.response_cache_ttl_seconds = response_cache_ttl_seconds
        self.response_cache_similarity_threshold = response_cache_similarity_threshold
        self.kb_revision = kb_revision
        self._clock = clock
        self._wall_clock_ms = wall_clock_ms

    def _tier_retriever(self, source: HybridRetriever) -> HybridRetriever:
        """Restrict the adapter to the lanes the selected tier may use."""

        if self.retrieval_tier.uses_dense_lane or isinstance(
            source,
            LexicalOnlyRetriever,
        ):
            return source
        return cast(
            HybridRetriever,
            LexicalOnlyRetriever(cast(SparseSearchRetriever, source)),
        )

    def run(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one query and return its immutable terminal snapshot."""

        state, started = self._prepare_answer_state(
            user_query,
            filters,
            session_id,
            query_id,
        )
        if state.terminal_status in {"answered", "abstained"}:
            chunks: list[str] = []
            self._measure_stage(
                state,
                "generate_answer_streaming",
                lambda: chunks.extend(self.generate_answer_streaming(state)),
            )
            state.answer = "".join(chunks)
            self._measure_stage(
                state,
                "verify_answer",
                lambda: self.verify_answer(state),
            )
        self._measure_stage(
            state,
            "persist_turn_memory",
            lambda: self.persist_turn_memory(state),
        )
        self._finalize(state, started)
        return self._serialize(state)

    def stream(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        """Yield safe trace events, validated answer deltas, then a snapshot."""

        state, started, emitter = yield from self._prepare_answer_state_stream(
            user_query,
            filters,
            session_id,
            query_id,
        )
        yield from self._stream_with_emitter(state, started, emitter)

    def _stream_with_emitter(
        self,
        state: AgentState,
        started: float,
        emitter: WorkflowEventEmitter,
    ) -> Iterable[dict[str, Any]]:
        """Complete generation while preserving validation-before-answer."""

        if state.terminal_status not in {"answered", "abstained"}:
            yield {"type": "answer_delta", "text": state.answer}
            self._measure_stage_delta(
                state,
                "persist_turn_memory",
                lambda: self.persist_turn_memory(state),
            )
            self._finalize(state, started)
            yield {"type": "final", "response": self._serialize(state)}
            return

        chunks: list[str] = []
        elapsed = self._measure_stage_delta(
            state,
            "generate_answer_streaming",
            lambda: chunks.extend(self.generate_answer_streaming(state)),
        )
        state.answer = "".join(chunks)
        yield self._stage_event(
            emitter,
            state,
            "generate_answer_streaming",
            elapsed,
        )
        elapsed = self._measure_stage_delta(
            state,
            "verify_answer",
            lambda: self.verify_answer(state),
        )
        yield self._stage_event(
            emitter,
            state,
            "verify_answer",
            elapsed,
        )
        for chunk_text in chunks:
            yield {"type": "answer_delta", "text": chunk_text}
        self._measure_stage_delta(
            state,
            "persist_turn_memory",
            lambda: self.persist_turn_memory(state),
        )
        self._finalize(state, started)
        yield {"type": "final", "response": self._serialize(state)}

    def create_state(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> AgentState:
        """Create validated state shared by local and graph orchestration."""

        normalized_filters = (
            normalize_filters(DEFAULT_SEARCH_PARAMS["filters"])
            if filters is None
            else normalize_filters(filters)
        )
        return AgentState(
            user_query=validate_question(user_query),
            query_id=validate_identifier(
                query_id or f"query_{uuid.uuid4().hex}",
                field_name="query_id",
            ),
            session_id=validate_identifier(
                session_id or "session_local",
                field_name="session_id",
            ),
            search_filters=normalized_filters,
            search_order_by=list(DEFAULT_SEARCH_PARAMS["order_by"]),
            milvus_top_k=DEFAULT_SEARCH_PARAMS["milvus_top_k"],
            reranker_top_k=DEFAULT_SEARCH_PARAMS["reranker_top_k"],
            reranker_name="not_run",
            max_retry=DEFAULT_SEARCH_PARAMS["max_retry"],
            search_mode=(
                DEFAULT_SEARCH_PARAMS["search_mode"]
                if self.retrieval_tier.uses_dense_lane
                else "lexical"
            ),
            retrieval_tier=self.retrieval_tier.tier.value,
            memory_ttl_seconds=self.memory_ttl_seconds,
            response_cache_status=(
                "miss" if self.response_cache_enabled else "disabled"
            ),
        )

    def _prepare_answer_state(
        self,
        user_query: str,
        filters: dict[str, Any] | None,
        session_id: str | None,
        query_id: str | None,
    ) -> tuple[AgentState, float]:
        stream = self._prepare_answer_state_stream(
            user_query,
            filters,
            session_id,
            query_id,
        )
        while True:
            try:
                next(stream)
            except StopIteration as completed:
                state, started, _emitter = completed.value
                return state, started

    def _prepare_answer_state_stream(
        self,
        user_query: str,
        filters: dict[str, Any] | None,
        session_id: str | None,
        query_id: str | None,
    ) -> Generator[
        dict[str, Any],
        None,
        tuple[AgentState, float, WorkflowEventEmitter],
    ]:
        started = self._clock()
        state = self.create_state(
            user_query,
            filters,
            session_id=session_id,
            query_id=query_id,
        )
        emitter = WorkflowEventEmitter(state.query_id)
        elapsed = self._measure_stage_delta(
            state,
            "recall_memory",
            lambda: self.recall_memory(state),
        )
        yield self._stage_event(emitter, state, "recall_memory", elapsed)
        current_node = WorkflowNode.CLASSIFY_AND_ROUTE
        while current_node not in {
            WorkflowNode.GENERATE_CANDIDATE_ANSWER,
            WorkflowNode.OUTPUT_GATE,
        }:
            if current_node is WorkflowNode.CLASSIFY_AND_ROUTE:
                elapsed = self._measure_stage_delta(
                    state,
                    "classify_and_route",
                    lambda: self.classify_and_route(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "classify_and_route",
                    elapsed,
                )
                current_node = next_transition(
                    WorkflowNode.CLASSIFY_AND_ROUTE,
                    state,
                ).next_node
                continue

            if current_node is WorkflowNode.RESOLVE_TERMINOLOGY:
                elapsed = self._measure_stage_delta(
                    state,
                    "resolve_terminology",
                    lambda: self.resolve_terminology(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "resolve_terminology",
                    elapsed,
                )
                current_node = next_transition(
                    WorkflowNode.RESOLVE_TERMINOLOGY,
                    state,
                ).next_node
                continue

            if current_node is WorkflowNode.CHECK_PERMISSION:
                elapsed = self._measure_stage_delta(
                    state,
                    "check_permission",
                    lambda: self.check_permission(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "check_permission",
                    elapsed,
                )
                current_node = next_transition(
                    WorkflowNode.CHECK_PERMISSION,
                    state,
                ).next_node
                continue

            if current_node is WorkflowNode.TRY_GROUNDED_CACHE:
                elapsed = self._measure_stage_delta(
                    state,
                    "try_grounded_cache",
                    lambda: self.try_grounded_cache(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "try_grounded_cache",
                    elapsed,
                )
                current_node = next_transition(
                    WorkflowNode.TRY_GROUNDED_CACHE,
                    state,
                ).next_node
                continue

            if current_node is WorkflowNode.RECALL_AUTHORIZED_EXPERIENCE:
                elapsed = self._measure_stage_delta(
                    state,
                    "recall_authorized_experience",
                    lambda: self.recall_authorized_experience(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "recall_authorized_experience",
                    elapsed,
                )
                elapsed = self._measure_stage_delta(
                    state,
                    "plan_retrieval",
                    lambda: self.plan_retrieval(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "plan_retrieval",
                    elapsed,
                )
                current_node = WorkflowNode.EXECUTE_TOOL_PLAN
                continue

            if current_node is WorkflowNode.RERANK_EVIDENCE:
                elapsed = self._measure_stage_delta(
                    state,
                    "rerank_evidence",
                    lambda: self.rerank_evidence(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "rerank_evidence",
                    elapsed,
                )
                current_node = WorkflowNode.EVALUATE_EVIDENCE
                continue

            if current_node is WorkflowNode.EVALUATE_EVIDENCE:
                evaluation, elapsed = self._measure_stage_result_delta(
                    state,
                    "evaluate_evidence",
                    lambda: self.evaluate_evidence(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "evaluate_evidence",
                    elapsed,
                    kind=(
                        "retry_scheduled"
                        if evaluation.action is EvidenceAction.RETRY
                        else "stage_completed"
                    ),
                )
                current_node = next_transition(
                    WorkflowNode.EVALUATE_EVIDENCE,
                    state,
                    evidence_action=evaluation.action,
                ).next_node
                continue

            if current_node is WorkflowNode.PREPARE_GENERATION_CONTEXT:
                elapsed = self._measure_stage_delta(
                    state,
                    "prepare_generation_context",
                    lambda: self.prepare_generation_context(state),
                )
                yield self._stage_event(
                    emitter,
                    state,
                    "prepare_generation_context",
                    elapsed,
                )
                current_node = WorkflowNode.GENERATE_CANDIDATE_ANSWER
                continue

            if current_node is not WorkflowNode.EXECUTE_TOOL_PLAN:
                raise RuntimeError(f"Unsupported workflow node: {current_node.value}")
            prior_tool_calls = len(state.tool_calls)
            elapsed = self._measure_stage_delta(
                state,
                "execute_tool_plan",
                lambda: self.milvus_hybrid_retrieve(state),
            )
            yield self._stage_event(
                emitter,
                state,
                "execute_tool_plan",
                elapsed,
            )
            for tool_call in state.tool_calls[prior_tool_calls:]:
                yield self._tool_event(emitter, tool_call)
            current_node = next_transition(
                WorkflowNode.EXECUTE_TOOL_PLAN,
                state,
            ).next_node

        return state, started, emitter

    def recall_memory(self, state: AgentState) -> None:
        """Recall bounded, live context from only the active session."""

        directive = detect_memory_recall(state.user_query)
        if directive is not None and directive.mode == "chronological":
            state.memory_recall_decision = "searched"
            state.memory_recall_mode = "chronological"
            state.memory_recall_reason = directive.reason
            state.memory_requested_count = directive.requested_count
            state.memory_recall_types = ["short_term"]
            try:
                records = self.memory_store.list_recent_user_questions(
                    state.session_id,
                    now_ms=self._wall_clock_ms(),
                    limit=directive.requested_count or DEFAULT_MEMORY_TOP_K,
                )
            except (MemoryStoreError, TextEmbeddingError):
                state.memory_status = "recall_failed"
                state.recalled_memories = []
                state.memory_context = ""
            else:
                state.recalled_memories = [
                    self._memory_presentation(record) for record in records
                ]
                state.memory_context = ""
                state.memory_status = "recalled" if records else "empty"
            state.selective_memory_status = "skipped"
            state.selective_memory_pack = {}
            state.selective_memory_private_values = []
        elif not self._should_recall_memory(state.user_query):
            state.memory_recall_decision = "skipped"
            state.memory_recall_mode = "none"
            state.memory_recall_reason = "not_applicable"
            state.memory_requested_count = None
            state.memory_recall_types = []
            state.memory_status = "empty"
            state.recalled_memories = []
            state.memory_context = ""
            self._recall_selective_memory(state)
        else:
            state.memory_recall_decision = "searched"
            state.memory_recall_mode = "semantic"
            state.memory_recall_reason = (
                directive.reason if directive is not None else "contextual_followup"
            )
            state.memory_requested_count = None
            state.memory_recall_types = ["session_summary", "task_state"]
            try:
                records = self.memory_store.search(
                    state.user_query,
                    session_id=state.session_id,
                    top_k=self.memory_top_k,
                    now_ms=self._wall_clock_ms(),
                )
            except (MemoryStoreError, TextEmbeddingError):
                state.memory_status = "recall_failed"
                state.recalled_memories = []
                state.memory_context = ""
            else:
                state.recalled_memories = [
                    self._memory_presentation(record) for record in records
                ]
                context_parts: list[str] = []
                remaining = MAX_MEMORY_CONTEXT_CHARS
                for record in records:
                    value = record.presentation_summary()[:remaining]
                    if not value:
                        continue
                    context_parts.append(value)
                    remaining -= len(value)
                    if remaining <= 0:
                        break
                state.memory_context = "\n".join(context_parts)
                state.memory_status = "recalled" if records else "empty"
            self._recall_selective_memory(state)

    def persist_turn_memory(self, state: AgentState) -> None:
        """Persist one verified terminal turn without invalidating its answer."""

        if not state.answer_validation.get("valid", False):
            raise ValueError("Cannot persist an unvalidated answer")
        now_ms = self._wall_clock_ms()
        expires_at = now_ms + self.memory_ttl_seconds * 1000
        try:
            records = build_turn_records(
                session_id=state.session_id,
                turn_id=state.query_id,
                user_content=state.user_query,
                assistant_content=state.answer,
                created_at=now_ms,
                expires_at=expires_at,
                remembered_statement=state.remembered_statement,
            )
            state.memory_written_count = self.memory_store.upsert_turn(records)
        except (MemoryStoreError, TextEmbeddingError):
            state.memory_written_count = 0
            state.memory_status = "write_failed"
            if state.intent == "memory_write":
                state.terminal_status = "memory_write_failed"
                state.answer_validation = {
                    "valid": True,
                    "mode": "memory_write_failed",
                    "reason": "The Memory write failed safely.",
                }
        else:
            if state.memory_status != "recall_failed":
                state.memory_status = "saved"
        self._persist_selective_memory(state, now_ms=now_ms)
        self._persist_response_cache(state, now_ms=now_ms)

    def _recall_selective_memory(self, state: AgentState) -> None:
        """Recall typed working state and decay-aware episodes."""

        if not self.selective_memory_enabled:
            state.selective_memory_status = "disabled"
            return
        try:
            pack = self.selective_memory.recall(
                state.user_query,
                session_id=state.session_id,
                now_ms=self._wall_clock_ms(),
                include_episodes=self._should_recall_memory(state.user_query),
            )
        except (SelectiveMemoryError, TextEmbeddingError, ValueError):
            state.selective_memory_status = "recall_failed"
            state.selective_memory_pack = {}
            state.selective_memory_private_values = []
            return
        state.selective_memory_pack = pack.trace_summary()
        self._apply_selective_pack(state, pack)

    def _apply_selective_pack(
        self,
        state: AgentState,
        pack: MemoryPack,
        *,
        append: bool = False,
    ) -> None:
        """Apply private MemoryPack payload without exposing it in trace."""

        state.selective_memory_pack = pack.trace_summary()
        private_values = pack.private_values()
        if append:
            private_values = list(
                dict.fromkeys((*state.selective_memory_private_values, *private_values))
            )
        state.selective_memory_private_values = private_values
        state.selective_memory_status = "recalled" if pack.rendered_context else "empty"
        if pack.rendered_context:
            combined = "\n".join(
                value
                for value in (
                    pack.rendered_context,
                    state.memory_context,
                )
                if value
            )
            state.memory_context = combined[:MAX_MEMORY_CONTEXT_CHARS]
            recalled = [
                {
                    "memory_id": fact.memory_id,
                    "memory_type": fact.memory_type,
                    "summary": fact.value,
                    "created_at": fact.valid_from,
                    "expires_at": fact.expires_at,
                    "source_event_ids": list(fact.source_event_ids),
                }
                for fact in (
                    *pack.working_state,
                    *pack.durable_facts,
                )
            ]
            recalled.extend(
                {
                    "event_id": event.event_id,
                    "memory_type": event.event_type,
                    "summary": event.content,
                    "created_at": event.event_time,
                    "expires_at": event.expires_at,
                }
                for event in pack.recent_episodes
            )
            if append:
                state.recalled_memories.extend(recalled)
            else:
                state.recalled_memories = recalled

    def _persist_selective_memory(
        self,
        state: AgentState,
        *,
        now_ms: int,
    ) -> None:
        """Capture and consolidate one validated terminal episode."""

        if not self.selective_memory_enabled:
            state.selective_memory_status = "disabled"
            return
        scope_hash = (
            permission_scope_hash(state.permission_decision)
            if state.terminal_status == "abstained" and state.permission_decision
            else SESSION_PRIVATE_SCOPE_HASH
        )
        try:
            result = self.selective_memory.persist_turn(
                session_id=state.session_id,
                query_id=state.query_id,
                query=state.user_query,
                answer=state.answer,
                terminal_status=state.terminal_status,
                remembered_statement=state.remembered_statement,
                now_ms=now_ms,
                permission_scope_hash_value=scope_hash,
            )
        except (SelectiveMemoryError, TextEmbeddingError, ValueError):
            state.selective_memory_status = "write_failed"
            state.selective_memory_written_count = 0
            return
        state.selective_memory_written_count = result.event_count + result.fact_count
        state.selective_memory_retention_class = result.retention_class
        state.selective_memory_selection_reasons = list(result.selection_reasons)
        state.selective_memory_consolidation_status = result.consolidation_status
        state.selective_memory_selector_name = result.selector_name
        state.selective_memory_selector_model = result.selector_model
        state.selective_memory_selector_fallback_reason = (
            result.selector_fallback_reason
        )
        if state.selective_memory_status != "recall_failed":
            state.selective_memory_status = "saved"

    def list_memories(
        self,
        session_id: str,
        *,
        limit: int = MAX_SESSION_RECORDS,
    ) -> list[dict[str, Any]]:
        """Return live, presentation-bounded Memory for one session."""

        records = self.memory_store.list_session(
            session_id,
            now_ms=self._wall_clock_ms(),
            limit=limit,
        )
        return [self._memory_presentation(record) for record in records]

    def list_selective_memories(
        self,
        session_id: str,
        *,
        limit: int = MAX_SESSION_RECORDS,
    ) -> list[dict[str, Any]]:
        """Return session-private selective events and facts."""

        if not self.selective_memory_enabled:
            return []
        return self.selective_memory.list_session(
            session_id,
            now_ms=self._wall_clock_ms(),
            limit=limit,
        )

    def clear_memory(self, session_id: str) -> int:
        """Delete only the selected session's Memory and response cache."""

        memory_count = self.memory_store.delete_session(session_id)
        selective_count = (
            self.selective_memory.delete_session(session_id)
            if self.selective_memory_enabled
            else 0
        )
        cache_count = (
            self.response_cache.delete_session(session_id)
            if self.response_cache_enabled
            else 0
        )
        return memory_count + selective_count + cache_count

    def _recall_response_cache(self, state: AgentState) -> None:
        """Recall private cache candidates without making them authoritative."""

        if not self.response_cache_enabled:
            state.response_cache_status = "disabled"
            return
        try:
            candidates = self.response_cache.search(
                state.user_query,
                session_id=state.session_id,
                top_k=self.response_cache_top_k,
                now_ms=self._wall_clock_ms(),
            )
        except (ResponseCacheError, TextEmbeddingError):
            state.response_cache_status = "recall_failed"
            state.response_cache_fallback_reason = "cache_unavailable"
            state.response_cache_candidates = []
            state.response_cache_candidate_count = 0
            return
        state.response_cache_candidates = list(candidates)
        state.response_cache_candidate_count = len(candidates)
        state.response_cache_status = "candidate" if candidates else "miss"

    def _persist_response_cache(
        self,
        state: AgentState,
        *,
        now_ms: int,
    ) -> None:
        """Persist only newly generated, verified grounded answers."""

        if (
            not self.response_cache_enabled
            or state.terminal_status != "answered"
            or not state.answer_validation.get("valid", False)
            or not state.citations
        ):
            return
        results_by_id = {
            item.chunk.chunk_id: item for item in state.reranked_chunks if item.selected
        }
        evidence: list[CachedEvidence] = []
        for citation in state.citations:
            chunk_id = str(citation.get("chunk_id", ""))
            selected_result = results_by_id.get(chunk_id)
            checksum = selected_result.chunk.checksum if selected_result else None
            if selected_result is None or not checksum:
                state.response_cache_status = "write_failed"
                state.response_cache_fallback_reason = "uncacheable_evidence"
                return
            chunk = selected_result.chunk
            evidence.append(
                CachedEvidence(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_version=chunk.doc_version,
                    checksum=checksum,
                    is_current=chunk.is_current,
                    fusion_recipe=selected_result.search_result.fusion_recipe,
                )
            )
        try:
            record = build_cache_record(
                session_id=state.session_id,
                source_query_id=state.query_id,
                user_query=state.user_query,
                intent=state.intent,
                query_type=state.query_type,
                retrieval_goal=state.retrieval_goal,
                version_scope=state.version_scope,
                entity_ids=[str(item["entity_id"]) for item in state.matched_entities],
                permission_scope_hash_value=permission_scope_hash(
                    state.permission_decision
                ),
                kb_revision=self.kb_revision,
                answer=state.answer,
                citations=state.citations,
                evidence=evidence,
                created_at=now_ms,
                expires_at=(now_ms + self.response_cache_ttl_seconds * 1000),
            )
            state.response_cache_written_count = self.response_cache.upsert(record)
        except (ResponseCacheError, TextEmbeddingError, ValueError):
            state.response_cache_status = "write_failed"
            state.response_cache_fallback_reason = "cache_unavailable"
            state.response_cache_written_count = 0
            return
        state.response_cache_status = "saved"
        state.response_cache_expires_at = record.expires_at

    def classify_query(self, state: AgentState) -> None:
        """Classify intent separately from the knowledge topic."""

        result = validate_classification_result(
            self.query_classifier.classify(
                ClassificationRequest(
                    user_query=state.user_query,
                    memory_context=state.memory_context,
                )
            )
        )
        state.intent = result.intent
        state.query_type = result.query_type
        state.retrieval_goal = result.retrieval_goal
        state.classifier_name = result.classifier_name
        state.classifier_model = result.model
        state.classification_confidence = result.confidence
        state.classification_fallback_reason = result.fallback_reason
        if result.intent == "memory_write":
            state.remembered_statement = self._remembered_statement(state.user_query)

    def classify_and_route(self, state: AgentState) -> QueryRouteResult:
        """Classify once, choose one closed route, and build direct answers."""

        self.classify_query(state)
        self.decide_retrieval(state)
        reason = str(state.retrieval_decision["reason"])
        if state.need_retrieval:
            return QueryRouteResult(QueryRoute.RETRIEVAL, reason)
        self.prepare_non_retrieval_answer(state)
        return QueryRouteResult(QueryRoute.DIRECT, reason)

    def decide_retrieval(self, state: AgentState) -> None:
        """Decide whether this intent needs private knowledge retrieval."""

        if state.intent in {"memory_write", "memory_recall"}:
            state.need_retrieval = False
            reason = "The request is handled by session Memory."
        elif state.intent == "conversation":
            state.need_retrieval = False
            reason = "Conversation intent does not require private knowledge."
        elif state.intent == "operation":
            state.need_retrieval = False
            reason = "The query-only demo cannot execute mutation tools."
        else:
            state.need_retrieval = True
            reason = "The intent requires grounded private knowledge."
        state.retrieval_decision = {
            "need_retrieval": state.need_retrieval,
            "reason": reason,
        }

    @staticmethod
    def prepare_non_retrieval_answer(state: AgentState) -> None:
        """Build one validated direct, refusal, or Memory answer."""

        if state.intent == "operation":
            state.answer = UNSUPPORTED_OPERATION_RESPONSE
            state.terminal_status = "refused_unsupported_operation"
        elif state.intent == "memory_write":
            if state.remembered_statement:
                state.answer = "我已处理这条会话记忆请求，保存结果见 Memory 状态。"
                state.terminal_status = "memory_saved"
            else:
                state.answer = "请补充希望我在当前会话中记住的具体信息。"
                state.terminal_status = "clarification_required"
        elif state.intent == "memory_recall":
            if state.recalled_memories:
                if state.memory_recall_mode == "chronological":
                    questions = [
                        str(item["summary"])
                        for item in state.recalled_memories
                        if item.get("role") == "user"
                        and item.get("memory_type") == "short_term"
                        and str(item.get("summary", "")).strip()
                    ]
                    state.answer = "你最近的问题如下（从近到远）：\n" + "\n".join(
                        f"{index}. {question}"
                        for index, question in enumerate(questions, start=1)
                    )
                else:
                    summaries = list(state.selective_memory_private_values)
                    if not summaries:
                        summaries = [
                            str(item["summary"])
                            for item in state.recalled_memories
                            if str(item.get("summary", "")).strip()
                        ]
                    summaries = list(dict.fromkeys(summaries))
                    state.answer = "根据当前会话中你之前提供的信息：\n- " + "\n- ".join(
                        summaries
                    )
                state.terminal_status = "answered_from_memory"
            else:
                state.answer = "当前会话中没有找到匹配且仍在有效期内的记忆。"
                state.terminal_status = "memory_not_found"
        else:
            state.answer = DIRECT_RESPONSE
            state.terminal_status = "answered_without_retrieval"
        state.answer_validation = {
            "valid": True,
            "mode": (
                "memory_write"
                if state.intent == "memory_write" and state.remembered_statement
                else (
                    "memory_grounded"
                    if state.intent == "memory_recall" and state.recalled_memories
                    else (
                        "memory_empty"
                        if state.intent == "memory_recall"
                        else (
                            "memory_write_empty"
                            if state.intent == "memory_write"
                            else "not_applicable"
                        )
                    )
                )
            ),
            "reason": (
                "The answer uses only live same-session Memory."
                if state.intent == "memory_recall" and state.recalled_memories
                else "No grounded KB answer was generated."
            ),
        }

    def resolve_terminology(self, state: AgentState) -> None:
        """Resolve catalog terms and the bounded document-version scope."""

        if state.intent in {"memory_write", "memory_recall"}:
            state.entity_catalog_version = self.entity_catalog.catalog_version
            state.version_scope = {
                "mode": "current",
                "doc_versions": [],
                "sides": [{"mode": "current"}],
            }
            return
        resolution = self.entity_catalog.resolve(
            state.user_query,
            query_type=state.query_type,
        )
        state.entity_catalog_version = self.entity_catalog.catalog_version
        state.matched_entities = [dict(item) for item in resolution.matched]
        state.ambiguous_entities = [dict(item) for item in resolution.ambiguous]
        version_scope, version_ambiguity = self._resolve_version_scope(
            state.user_query,
            intent=state.intent,
        )
        state.version_scope = version_scope
        if version_ambiguity is not None:
            state.ambiguous_entities.append(version_ambiguity)
        if state.ambiguous_entities:
            state.answer = CLARIFICATION_REQUIRED_RESPONSE
            state.terminal_status = "clarification_required"
            state.need_retrieval = False
            state.retrieval_decision = {
                "need_retrieval": False,
                "reason": "Terminology or version scope requires clarification.",
            }
            state.answer_validation = {
                "valid": True,
                "mode": "clarification_required",
                "reason": "No retrieval or generation was executed.",
            }

    def check_permission(self, state: AgentState) -> None:
        """Run the permission tool before any private retrieval."""

        state.permission_decision = self.permission_checker.check(
            session_id=state.session_id,
            intent=state.intent,
            query_type=state.query_type,
        ).to_dict()
        if not state.permission_decision.get("allowed", False):
            state.answer = PERMISSION_DENIED_RESPONSE
            state.terminal_status = "permission_denied"
            state.answer_validation = {
                "valid": True,
                "mode": "permission_denied",
                "reason": "Retrieval and generation were not executed.",
            }

    def recall_authorized_experience(self, state: AgentState) -> None:
        """Recall reusable experience only after permission and a cache miss."""

        if (
            not self.selective_memory_enabled
            or state.permission_decision.get("allowed") is not True
        ):
            return
        try:
            pack = self.selective_memory.recall(
                state.user_query,
                session_id=state.session_id,
                now_ms=self._wall_clock_ms(),
                include_episodes=True,
                permission_scope_hash_value=permission_scope_hash(
                    state.permission_decision
                ),
                include_session_private=False,
            )
        except (SelectiveMemoryError, TextEmbeddingError, ValueError):
            state.selective_memory_status = "recall_failed"
            return
        if pack.rendered_context:
            self._apply_selective_pack(state, pack, append=True)

    def try_grounded_cache(self, state: AgentState) -> None:
        """Look up and release one equivalent, authorized cached response."""

        self._recall_response_cache(state)

        if not self.response_cache_enabled or not state.response_cache_candidates:
            if self.response_cache_enabled and (
                state.response_cache_status == "candidate"
            ):
                state.response_cache_status = "miss"
            return
        current_entities = sorted(
            str(item["entity_id"]) for item in state.matched_entities
        )
        current_constraints = query_constraints(state.user_query)
        current_permission_hash = permission_scope_hash(state.permission_decision)
        current_fingerprint = text_embedding_fingerprint()
        stale_seen = False
        for raw_candidate in state.response_cache_candidates:
            if not isinstance(raw_candidate, ResponseCacheCandidate):
                stale_seen = True
                continue
            candidate = raw_candidate
            record = candidate.record
            if (
                candidate.match_type == "semantic"
                and candidate.similarity < self.response_cache_similarity_threshold
            ):
                continue
            if (
                record.intent != state.intent
                or record.query_type != state.query_type
                or record.retrieval_goal != state.retrieval_goal
                or record.version_scope != state.version_scope
                or record.entity_ids != current_entities
                or record.query_constraints != current_constraints
                or record.embedding_fingerprint != current_fingerprint
                or record.permission_scope_hash != current_permission_hash
                or record.kb_revision != self.kb_revision
                or record.workflow_version != RESPONSE_CACHE_WORKFLOW_VERSION
                or record.expires_at <= self._wall_clock_ms()
            ):
                stale_seen = True
                continue
            try:
                if not self._cached_evidence_is_fresh(state, candidate):
                    stale_seen = True
                    continue
            except Exception:
                state.response_cache_status = "validation_failed"
                state.response_cache_fallback_reason = "cache_unavailable"
                return
            state.answer = record.answer
            state.citations = [dict(item) for item in record.citations]
            state.terminal_status = "answered_from_cache"
            state.answer_validation = {
                "valid": True,
                "mode": "cached_grounded",
                "reason": ("Cached citations and live evidence passed validation."),
            }
            state.response_cache_status = "hit"
            state.response_cache_match_type = candidate.match_type
            state.response_cache_similarity = round(
                candidate.similarity,
                4,
            )
            state.response_cache_source_query_id = record.source_query_id
            state.response_cache_expires_at = record.expires_at
            return
        state.response_cache_status = "stale" if stale_seen else "miss"
        if stale_seen:
            state.response_cache_fallback_reason = "cache_stale"

    def _cached_evidence_is_fresh(
        self,
        state: AgentState,
        candidate: ResponseCacheCandidate,
    ) -> bool:
        """Validate cached citations against authorized live KB records."""

        record = candidate.record
        evidence_by_id = {item.chunk_id: item for item in record.evidence}
        citation_ids = {str(item.get("citation_id", "")) for item in record.citations}
        cited_chunk_ids = {str(item.get("chunk_id", "")) for item in record.citations}
        markers = {f"C{number}" for number in re.findall(r"\[C(\d+)\]", record.answer)}
        if (
            not markers
            or markers != citation_ids
            or cited_chunk_ids != set(evidence_by_id)
            or len(citation_ids) != len(record.citations)
            or len(cited_chunk_ids) != len(record.citations)
            or len(evidence_by_id) != len(record.evidence)
        ):
            return False
        for citation in record.citations:
            snapshot = evidence_by_id.get(str(citation.get("chunk_id", "")))
            if (
                snapshot is None
                or citation.get("doc_id") != snapshot.doc_id
                or citation.get("doc_version") != snapshot.doc_version
            ):
                return False
        allowed_departments = [
            str(item)
            for item in state.permission_decision.get(
                "allowed_departments",
                [],
            )
        ]
        if not allowed_departments:
            return False
        live_chunks = self.retriever.fetch_chunks_by_ids(
            chunk_ids=list(evidence_by_id),
            filters={"department": allowed_departments},
        )
        if len(live_chunks) != len(evidence_by_id):
            return False
        live_by_id = {chunk.chunk_id: chunk for chunk in live_chunks}
        if set(live_by_id) != set(evidence_by_id):
            return False
        current_scope = record.version_scope.get("mode") == "current"
        for chunk_id, snapshot in evidence_by_id.items():
            chunk = live_by_id[chunk_id]
            if (
                chunk.doc_id != snapshot.doc_id
                or chunk.doc_version != snapshot.doc_version
                or not chunk.checksum
                or chunk.checksum != snapshot.checksum
                or chunk.is_current != snapshot.is_current
                or (current_scope and not chunk.is_current)
            ):
                return False
        return True

    def select_tools(self, state: AgentState) -> None:
        """Choose the smallest useful set of registered search tools."""

        lowered = state.user_query.lower()
        entity_domains = {
            str(domain)
            for item in state.matched_entities
            for domain in item.get("domains", [])
        }
        selected: list[str] = []
        reasons: dict[str, str] = {}

        def choose(name: str, reason: str) -> None:
            if name not in selected:
                selected.append(name)
                reasons[name] = reason

        if state.query_type == "policy":
            choose("search_policy_docs", "The question targets policy.")
        if any(
            term in lowered for term in ["会议", "纪要", "客户", "meeting", "feedback"]
        ):
            choose(
                "search_meeting_notes",
                "Customer or meeting evidence is required.",
            )
        if (
            state.query_type == "product"
            or "product" in entity_domains
            or any(
                term in lowered
                for term in ["产品", "路线图", "roadmap", "feature", "ui"]
            )
        ):
            choose(
                "search_product_docs",
                "The question targets product plans or behavior.",
            )
        if state.query_type == "architecture" or any(
            term in lowered
            for term in [
                "工程",
                "代码",
                "pipeline",
                "ingestion",
                "architecture",
            ]
        ):
            choose(
                "search_code_docs",
                "The question targets engineering documentation.",
            )
        if not selected:
            fallback = (
                "search_product_docs"
                if state.query_type == "product"
                else "search_code_docs"
            )
            choose(fallback, "Fallback to the closest knowledge domain.")

        state.selected_tools = selected[:MAX_INITIAL_SUBQUERIES]
        state.tool_selection_reasons = {
            name: reasons[name] for name in state.selected_tools
        }

    def rewrite_query(self, state: AgentState) -> None:
        """Transform once, then attach only authorized tools and scopes."""

        if state.query_plan:
            return
        lowered = state.user_query.lower()
        plan: list[dict[str, Any]] = []
        resolved_terms = " ".join(
            f"{item['entity']} {item['comment']}" for item in state.matched_entities
        )
        scopes = self._plan_version_scopes(state.version_scope)
        scoped_tools: list[tuple[str, dict[str, Any]]] = [
            (tool_name, scope) for tool_name in state.selected_tools for scope in scopes
        ]
        transform_request = QueryTransformRequest(
            user_query=state.user_query,
            resolved_entities=tuple(
                str(item["entity"])
                for item in state.matched_entities
                if isinstance(item.get("entity"), str)
            ),
            memory_context=state.memory_context,
        )
        transformation = validate_transformation(
            self.query_transformer.transform(transform_request),
            transform_request,
        )
        transform_items = list(transformation.items)
        plan_slots = list(scoped_tools[:MAX_INITIAL_SUBQUERIES])
        while len(plan_slots) < len(transform_items) and scoped_tools:
            plan_slots.append(scoped_tools[0])
        plan_slots = plan_slots[:MAX_INITIAL_SUBQUERIES]
        plan_ids = [f"sq{index}" for index in range(1, len(plan_slots) + 1)]
        transform_index_by_plan: list[int] = []
        for zero_index, (tool_name, version_scope) in enumerate(plan_slots):
            transform_index = min(zero_index, len(transform_items) - 1)
            transform_index_by_plan.append(transform_index)
            transform_item = transform_items[transform_index]
            tool = SEARCH_TOOLS[tool_name]
            query = f"{transform_item.query} {tool.query_hint}"
            if resolved_terms:
                query += f" {resolved_terms}"
            if tool_name == "search_code_docs" and (
                "s3" in lowered or "同步" in state.user_query
            ):
                query += (
                    " MinIO bucket scanning change detection document parsing"
                    " chunking embedding generation Milvus insertion"
                )
            if state.retrieval_goal == "exhaustive":
                query += " release notes changelog new features capabilities"
            if tool_name == "search_product_docs" and (
                "路线图" in state.user_query or "roadmap" in lowered
            ):
                query += " product roadmap coverage planned capabilities"

            dependencies = [
                plan_ids[dependency_plan_index]
                for dependency_plan_index, candidate_transform_index in enumerate(
                    transform_index_by_plan
                )
                if (
                    candidate_transform_index in transform_item.depends_on
                    and dependency_plan_index < zero_index
                )
            ]
            subquery_id = plan_ids[zero_index]
            plan.append(
                {
                    "subquery_id": subquery_id,
                    "tool": tool_name,
                    "query": query,
                    "query_role": transform_item.query_role,
                    "depends_on": dependencies,
                    "status": "pending",
                    "round": 0,
                    "version_scope": dict(version_scope),
                }
            )

        state.query_plan = plan[:MAX_INITIAL_SUBQUERIES]
        state.query_transformation = {
            "strategy": transformation.strategy,
            "item_roles": [item.query_role for item in transformation.items],
            "item_count": len(transformation.items),
            "transformer_name": transformation.transformer_name,
            "model": transformation.model,
            "fallback_reason": transformation.fallback_reason,
        }
        state.rewritten_queries = [str(item["query"]) for item in state.query_plan]
        state.query_rewrite_rounds.append(
            {
                "round": 0,
                "queries": list(state.rewritten_queries),
                "plan_ids": [str(item["subquery_id"]) for item in state.query_plan],
            }
        )

    def plan_retrieval(self, state: AgentState) -> RetrievalPlanResult:
        """Select authorized tools and build their bounded initial plan."""

        self.select_tools(state)
        self.rewrite_query(state)
        return RetrievalPlanResult(
            selected_tools=tuple(state.selected_tools),
            transformation=dict(state.query_transformation),
            plan_count=len(state.query_plan),
        )

    def milvus_hybrid_retrieve(self, state: AgentState) -> None:
        """Execute ready tool calls and merge their results."""

        completed = {
            str(item["subquery_id"])
            for item in state.query_plan
            if item["status"] == "completed"
        }
        ready = [
            item
            for item in state.query_plan
            if item["status"] == "pending"
            and set(item["depends_on"]).issubset(completed)
        ]
        if not ready:
            raise RuntimeError("Query plan has no executable tool call")

        allowed_departments = tuple(
            state.permission_decision.get("allowed_departments", [])
        )
        prepared_calls: list[_PreparedToolCall] = []
        for item in ready:
            tool = SEARCH_TOOLS[str(item["tool"])]
            tool_filters = tool.build_filters(
                base_filters=state.search_filters,
                allowed_departments=allowed_departments,
            )
            version_scope = dict(item["version_scope"])
            tool_filters = self._apply_version_scope(
                tool_filters,
                version_scope,
            )
            prepared_calls.append(
                _PreparedToolCall(
                    plan_item=item,
                    tool=tool,
                    filters=tool_filters,
                    version_scope=version_scope,
                )
            )

        profile_group = self._profile_tool_searches(state, prepared_calls)
        parallel = (
            profile_group is None
            and len(prepared_calls) > 1
            and getattr(
                self.retriever,
                "supports_parallel_search",
                False,
            )
            is True
        )
        state.retrieval_execution_mode = "parallel" if parallel else "sequential"
        outcomes = profile_group or (
            self._parallel_tool_searches(state, prepared_calls)
            if parallel
            else [
                self._execute_tool_search(state, prepared)
                for prepared in prepared_calls
            ]
        )

        new_results: list[SearchResult] = []
        for outcome in outcomes:
            prepared = outcome.prepared
            item = prepared.plan_item
            tool = prepared.tool
            tool_filters = prepared.filters
            version_scope = prepared.version_scope
            results = list(outcome.results)
            expansion_results = self._expand_document_results(
                state,
                results,
                filters=tool_filters,
                subquery_id=str(item["subquery_id"]),
            )
            result_ids = [result.chunk.chunk_id for result in results]
            state.tool_calls.append(
                {
                    "tool": tool.name,
                    "subquery_id": item["subquery_id"],
                    "query": item["query"],
                    "query_role": item.get("query_role", "primary"),
                    "filters": tool_filters,
                    "result_count": len(results),
                    "result_chunk_ids": result_ids,
                    "latency_ms": outcome.latency_ms,
                    "round": state.retry_count,
                    "version_scope": version_scope,
                    "retrieval_profile": outcome.retrieval_profile,
                    "result_granularity": "passage",
                    "element_offsets": [
                        result.element_offset
                        for result in results
                        if result.element_offset is not None
                    ],
                    "document_candidate_count": len(outcome.document_candidates),
                    "element_predicate_count": 0,
                    "collapse_status": (
                        "parent_shortlist_then_element"
                        if outcome.document_candidates
                        else (
                            "element_identity_preserved"
                            if outcome.retrieval_profile.startswith("struct_")
                            else "not_applicable"
                        )
                    ),
                    "fusion_recipe": next(
                        (
                            result.fusion_recipe
                            for result in results
                            if result.fusion_recipe is not None
                        ),
                        None,
                    ),
                }
            )
            state.retrieval_profile = outcome.retrieval_profile
            state.structarray_status = outcome.capability_status
            resolved_by_key: dict[str, set[int]] = {}
            for result in results:
                if result.document_key and result.element_offset is not None:
                    resolved_by_key.setdefault(result.document_key, set()).add(
                        result.element_offset
                    )
            for candidate in outcome.document_candidates:
                key = str(candidate["document_key"])
                existing = next(
                    (
                        item
                        for item in state.document_candidates
                        if item.get("document_key") == key
                    ),
                    None,
                )
                if existing is None:
                    existing = dict(candidate)
                    existing["resolved_element_count"] = 0
                    existing["resolved_to_evidence"] = False
                    state.document_candidates.append(existing)
                existing["resolved_element_count"] = int(
                    existing["resolved_element_count"]
                ) + len(resolved_by_key.get(key, set()))
                existing["resolved_to_evidence"] = bool(
                    existing["resolved_element_count"]
                )
            for result in results:
                state.retrieval_provenance.setdefault(
                    result.chunk.chunk_id,
                    [],
                ).append(
                    {
                        "tool": tool.name,
                        "subquery_id": item["subquery_id"],
                        "query_role": item.get("query_role", "primary"),
                        "retrieval_profile": result.retrieval_profile,
                        "retrieval_paths": list(result.retrieval_paths),
                        "result_granularity": result.result_granularity,
                        "element_offset": result.element_offset,
                        "fusion_recipe": result.fusion_recipe,
                    }
                )
            item["status"] = "completed"
            new_results.extend(results)
            new_results.extend(expansion_results)

        self._validate_retrieved_versions(state, new_results)
        merged = self._merge_search_results([*state.retrieved_chunks, *new_results])
        required_ids = self._expanded_chunk_ids(state)
        if required_ids:
            retained_ids = set(required_ids)
            for result in merged:
                if len(retained_ids) >= state.milvus_top_k:
                    break
                retained_ids.add(result.chunk.chunk_id)
        else:
            retained_ids = self._scoped_result_ids(
                merged,
                version_scope=state.version_scope,
                limit=state.milvus_top_k,
            )
        retained = [item for item in merged if item.chunk.chunk_id in retained_ids]
        state.retrieved_chunks = [
            replace(item, rank=index) for index, item in enumerate(retained, start=1)
        ]
        state.aggregations = self.retriever.aggregations(
            state.retrieved_chunks,
            [
                "source_type",
                "doc_type",
                "department",
                "has_image_vector",
                "doc_version",
                "is_current",
            ],
        )
        fingerprint = self._candidate_pool_fingerprint(state)
        state.candidate_pool_unchanged = (
            state.retry_count > 0
            and state.candidate_pool_fingerprint is not None
            and fingerprint == state.candidate_pool_fingerprint
        )
        state.candidate_pool_fingerprint = fingerprint
        state.retrieval_stop_reason = (
            "no_progress" if state.candidate_pool_unchanged else None
        )
        if state.candidate_pool_unchanged:
            mark_no_progress_abstention(state)

    def _execute_tool_search(
        self,
        state: AgentState,
        prepared: _PreparedToolCall,
    ) -> _ToolSearchOutcome:
        """Execute one isolated raw search without mutating workflow state."""

        started = self._clock()
        profile_search = getattr(self.retriever, "search_profile", None)
        profile_name = "flat_hybrid"
        capability_status = "ready"
        candidates: tuple[dict[str, Any], ...] = ()
        if callable(profile_search):
            run = profile_search(
                [str(prepared.plan_item["query"])],
                top_k=state.milvus_top_k,
                filters=prepared.filters,
                order_by=state.search_order_by,
            )
            if len(run.results_by_query) != 1:
                raise RuntimeError(
                    "StructArray profile returned an invalid query result count"
                )
            results = list(run.results_by_query[0])
            profile_name = run.effective_profile
            capability_status = run.capability_status
            candidates = tuple(item.to_dict() for item in run.document_candidates)
        else:
            results = self.retriever.search(
                str(prepared.plan_item["query"]),
                top_k=state.milvus_top_k,
                filters=prepared.filters,
                order_by=state.search_order_by,
            )
        latency_ms = round(
            max(0.0, (self._clock() - started) * 1000),
            3,
        )
        return _ToolSearchOutcome(
            prepared=prepared,
            results=tuple(results),
            latency_ms=latency_ms,
            retrieval_profile=profile_name,
            capability_status=capability_status,
            document_candidates=candidates,
        )

    def _profile_tool_searches(
        self,
        state: AgentState,
        prepared_calls: list[_PreparedToolCall],
    ) -> list[_ToolSearchOutcome] | None:
        """Group eligible same-scope aspects into one immutable profile run."""

        search_profile = getattr(self.retriever, "search_profile", None)
        if not callable(search_profile) or not 2 <= len(prepared_calls) <= 3:
            return None
        first = prepared_calls[0]
        if any(
            prepared.tool.name != first.tool.name
            or prepared.filters != first.filters
            or prepared.version_scope != first.version_scope
            or prepared.plan_item.get("query_role", "primary")
            not in {"primary", "aspect"}
            for prepared in prepared_calls
        ):
            return None
        started = self._clock()
        run = search_profile(
            [str(item.plan_item["query"]) for item in prepared_calls],
            top_k=state.milvus_top_k,
            filters=first.filters,
            order_by=state.search_order_by,
        )
        if len(run.results_by_query) != len(prepared_calls):
            raise RuntimeError(
                "StructArray profile returned an invalid query result count"
            )
        elapsed = round(max(0.0, (self._clock() - started) * 1000), 3)
        candidates = tuple(item.to_dict() for item in run.document_candidates)
        return [
            _ToolSearchOutcome(
                prepared=prepared,
                results=tuple(results),
                latency_ms=elapsed,
                retrieval_profile=run.effective_profile,
                capability_status=run.capability_status,
                document_candidates=candidates,
            )
            for prepared, results in zip(
                prepared_calls,
                run.results_by_query,
                strict=True,
            )
        ]

    def _parallel_tool_searches(
        self,
        state: AgentState,
        prepared_calls: list[_PreparedToolCall],
    ) -> list[_ToolSearchOutcome]:
        """Run proven-safe reads concurrently and collect them in plan order."""

        executor = ThreadPoolExecutor(
            max_workers=min(MAX_INITIAL_SUBQUERIES, len(prepared_calls)),
            thread_name_prefix="agent-retrieval",
        )
        futures: list[Future[_ToolSearchOutcome]] = [
            executor.submit(self._execute_tool_search, state, prepared)
            for prepared in prepared_calls
        ]
        try:
            return [future.result() for future in futures]
        except Exception:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def rerank_evidence(self, state: AgentState) -> None:
        """Rerank recalled candidates without selecting weak context."""

        if not state.retrieved_chunks:
            state.reranked_chunks = []
            return
        query = self._effective_query(state)
        top_k = len(state.retrieved_chunks)
        if state.reranker_sticky_fallback_reason is not None and isinstance(
            self.reranker, FallbackReranker
        ):
            rerank_run = self.reranker.rerank_fallback_only(
                query,
                state.retrieved_chunks,
                top_k,
                reason_code=state.reranker_sticky_fallback_reason,
            )
            state.reranker_fallback_only_count += 1
        else:
            state.reranker_primary_attempt_count += 1
            rerank_run = validate_rerank_run(
                self.reranker.rerank(
                    query,
                    state.retrieved_chunks,
                    top_k=top_k,
                ),
                chunks=state.retrieved_chunks,
                top_k=top_k,
            )
            if (
                isinstance(self.reranker, FallbackReranker)
                and rerank_run.fallback_reason is not None
            ):
                state.reranker_sticky_fallback_reason = rerank_run.fallback_reason
        state.reranker_name = rerank_run.reranker_name
        state.reranker_model = rerank_run.model
        state.reranker_fallback_reason = rerank_run.fallback_reason
        reranked_pool = list(rerank_run.results)
        required_ids = self._expanded_chunk_ids(state)
        if required_ids:
            state.reranker_top_k = min(
                state.milvus_top_k,
                max(state.reranker_top_k, len(required_ids)),
            )
            retained_ids = set(required_ids)
            for result in reranked_pool:
                if len(retained_ids) >= state.reranker_top_k:
                    break
                retained_ids.add(result.chunk.chunk_id)
        else:
            retained_ids = self._scoped_result_ids(
                reranked_pool,
                version_scope=state.version_scope,
                limit=state.reranker_top_k,
            )
        reranked = [
            item for item in reranked_pool if item.chunk.chunk_id in retained_ids
        ]
        state.reranked_chunks = [replace(result, selected=False) for result in reranked]

    @staticmethod
    def stop_on_unchanged_candidates(state: AgentState) -> bool:
        """Finalize a supplementary round that cannot change evidence coverage."""

        return state.candidate_pool_unchanged

    def grade_evidence(self, state: AgentState) -> None:
        """Select covered evidence or produce a targeted retrieval gap."""

        expanded_ids = self._expanded_chunk_ids(state)
        relevant = [
            item
            for item in state.reranked_chunks
            if item.chunk.chunk_id in expanded_ids
            or (
                item.rerank_score >= RELEVANCE_THRESHOLD
                and self._has_query_overlap(
                    self._effective_query(state),
                    item.chunk.text,
                )
            )
        ]
        if state.query_transformation.get("strategy") == "step_back":
            primary_plan_ids = {
                str(item["subquery_id"])
                for item in state.query_plan
                if item.get("query_role") != "background"
            }
            has_primary_evidence = any(
                any(
                    provenance.get("subquery_id") in primary_plan_ids
                    for provenance in state.retrieval_provenance.get(
                        item.chunk.chunk_id,
                        [],
                    )
                )
                for item in relevant
            )
            if not has_primary_evidence:
                relevant = []
        top_score = max(
            (item.rerank_score for item in relevant),
            default=0.0,
        )
        unique_relevant = {item.chunk.chunk_id for item in relevant}
        pending_tools = {
            str(item["tool"])
            for item in state.query_plan
            if item["status"] != "completed"
        }
        relevant_ids = {item.chunk.chunk_id for item in relevant}
        covered_tools = {
            str(call["tool"])
            for call in state.tool_calls
            if relevant_ids.intersection(call["result_chunk_ids"])
        }
        missing_tools = [
            tool
            for tool in state.selected_tools
            if tool not in covered_tools or tool in pending_tools
        ]
        missing_version_scopes: list[dict[str, Any]] = []
        comparison_family_complete = True
        if state.version_scope.get("mode") == "comparison":
            comparison_scopes = self._plan_version_scopes(state.version_scope)
            missing_version_scopes = [
                scope
                for scope in comparison_scopes
                if not any(
                    self._chunk_matches_scope(item.chunk, scope) for item in relevant
                )
            ]
            doc_ids = {item.chunk.doc_id for item in relevant}
            comparison_family_complete = any(
                all(
                    any(
                        item.chunk.doc_id == doc_id
                        and self._chunk_matches_scope(item.chunk, scope)
                        for item in relevant
                    )
                    for scope in comparison_scopes
                )
                for doc_id in doc_ids
            )

        strong_single = self._is_strong_single_evidence(
            state,
            relevant,
            covered_tools=covered_tools,
        )
        enough = (
            len(relevant) >= 2
            and len(unique_relevant) >= 2
            and top_score >= 0.28
            and not missing_tools
            and not missing_version_scopes
        )
        if state.retrieval_goal == "exhaustive":
            enough = (
                enough and bool(expanded_ids) and expanded_ids.issubset(unique_relevant)
            )
        elif strong_single:
            enough = True
        if state.intent == "comparison":
            enough = (
                enough
                and not missing_tools
                and not missing_version_scopes
                and comparison_family_complete
            )
        evidence_basis = (
            "single_strong_chunk"
            if strong_single and enough
            else ("multi_chunk_coverage" if enough else "insufficient_evidence")
        )
        missing_aspects = self._missing_evidence_aspects(
            state,
            enough=enough,
            relevant=relevant,
            missing_tools=missing_tools,
            missing_version_scopes=missing_version_scopes,
            threshold=self._strong_single_threshold(),
        )

        selected: list[RerankedResult] = []
        for scope in self._plan_version_scopes(state.version_scope):
            for item in relevant:
                if (
                    self._chunk_matches_scope(item.chunk, scope)
                    and item not in selected
                ):
                    selected.append(item)
                    break
        for tool_name in state.selected_tools:
            for item in relevant:
                provenance = state.retrieval_provenance.get(
                    item.chunk.chunk_id,
                    [],
                )
                if (
                    any(entry["tool"] == tool_name for entry in provenance)
                    and item not in selected
                ):
                    selected.append(item)
                    break
        for item in relevant:
            if item not in selected:
                selected.append(item)
        answer_context_limit = (
            MAX_EXHAUSTIVE_CONTEXTS
            if state.retrieval_goal == "exhaustive"
            else DEFAULT_SEARCH_PARAMS["answer_context_top_k"]
        )
        selected_ids = {item.chunk.chunk_id for item in selected[:answer_context_limit]}
        state.reranked_chunks = [
            replace(
                item,
                selected=enough and item.chunk.chunk_id in selected_ids,
            )
            for item in state.reranked_chunks
        ]
        state.enough_evidence = enough
        reason = (
            (
                "One strong direct chunk supports this focused feature answer."
                if evidence_basis == "single_strong_chunk"
                else "Reranked evidence covers every planned knowledge aspect."
            )
            if enough
            else "Evidence does not yet satisfy the registered coverage rule."
        )
        state.evidence_grade = self._grade_payload(
            state,
            enough=enough,
            reason=reason,
            evidence_basis=evidence_basis,
            missing_aspects=missing_aspects,
            relevant_count=len(relevant),
            top_score=top_score,
            covered_tools=sorted(covered_tools),
            missing_tools=missing_tools,
            missing_version_scopes=missing_version_scopes,
        )

    def _strong_single_threshold(self) -> float:
        """Return the injected reranker's own single-strong-chunk threshold.

        Rerank scores are not comparable across implementations, so the gate is
        read from whichever reranker actually ranked rather than from a shared
        constant.
        """

        value = getattr(self.reranker, "strong_single_evidence_threshold", None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"reranker {getattr(self.reranker, 'name', '<unnamed>')!r} must "
                "declare a finite strong_single_evidence_threshold in [0, 1]"
            )
        return float(value)

    def _is_strong_single_evidence(
        self,
        state: AgentState,
        relevant: list[RerankedResult],
        *,
        covered_tools: set[str],
    ) -> bool:
        """Recognize the narrow one-citation focused feature exception."""

        if (
            len(relevant) != 1
            or state.retrieval_goal != "focused"
            or state.intent == "comparison"
            or len(state.selected_tools) != 1
            or covered_tools != set(state.selected_tools)
            or self._has_multiple_requested_aspects(state.user_query)
        ):
            return False
        result = relevant[0]
        section = self._normalize_match_text(result.chunk.section or "")
        if (
            result.rerank_score < self._strong_single_threshold()
            or not section
            or section not in self._normalize_match_text(state.user_query)
        ):
            return False
        scopes = self._plan_version_scopes(state.version_scope)
        if len(scopes) != 1 or not self._chunk_matches_scope(
            result.chunk,
            scopes[0],
        ):
            return False
        return all(
            item.chunk.doc_id != result.chunk.doc_id
            or item.chunk.doc_version == result.chunk.doc_version
            for item in state.retrieved_chunks
        )

    @staticmethod
    def _missing_evidence_aspects(
        state: AgentState,
        *,
        enough: bool,
        relevant: list[RerankedResult],
        missing_tools: list[str],
        missing_version_scopes: list[dict[str, Any]],
        threshold: float,
    ) -> list[str]:
        """Return registered diagnostic codes derived from actual gaps."""

        if enough:
            return []
        aspects = [f"tool:{tool}" for tool in missing_tools]
        aspects.extend(
            "version:" + str(scope.get("doc_version", scope.get("mode", "unknown")))
            for scope in missing_version_scopes
        )
        if state.retrieval_goal == "exhaustive":
            aspects.append("incomplete_exhaustive_coverage")
        elif not relevant:
            aspects.append("no_relevant_evidence")
        elif len(relevant) == 1:
            if AgenticRAGWorkflow._has_multiple_requested_aspects(state.user_query):
                aspects.append("multi_aspect_requires_coverage")
            if relevant[0].rerank_score < threshold:
                aspects.append("single_weak_chunk")
            else:
                section = AgenticRAGWorkflow._normalize_match_text(
                    relevant[0].chunk.section or ""
                )
                direct = bool(section) and section in (
                    AgenticRAGWorkflow._normalize_match_text(state.user_query)
                )
                if not direct:
                    aspects.append("single_indirect_chunk")
                elif not aspects:
                    aspects.append("incomplete_multi_evidence")
        else:
            aspects.append("incomplete_multi_evidence")
        return list(dict.fromkeys(aspects))

    def prepare_supplementary_retrieval(
        self,
        state: AgentState,
        *,
        retry_number: int,
    ) -> bool:
        """Activate a pending hop or append one targeted retry query."""

        pending = [item for item in state.query_plan if item["status"] == "pending"]
        if pending:
            retry_query = str(pending[0]["query"])
            state.retry_queries.append(retry_query)
            state.query_rewrite_rounds.append(
                {
                    "round": retry_number,
                    "queries": [retry_query],
                    "plan_ids": [str(pending[0]["subquery_id"])],
                    "reason": "Execute a planned dependent retrieval hop.",
                }
            )
            return True

        tool_name = str(
            state.evidence_grade.get(
                "suggested_tool",
                state.selected_tools[0],
            )
        )
        if tool_name not in SEARCH_TOOLS:
            tool_name = state.selected_tools[0]
        suggested_query = state.evidence_grade.get("suggested_retry_query")
        retry_query = (
            str(suggested_query)
            if isinstance(suggested_query, str) and suggested_query.strip()
            else self._retry_query(
                state,
                missing_aspects=[
                    str(item)
                    for item in state.evidence_grade.get(
                        "missing_aspects",
                        [],
                    )
                ],
            )
        )
        subquery_id = f"retry{retry_number}"
        retry_version_scope = self._retry_version_scope(state)
        missing_version_scopes = state.evidence_grade.get(
            "missing_version_scopes",
        )
        if (
            isinstance(missing_version_scopes, list)
            and missing_version_scopes
            and isinstance(missing_version_scopes[0], dict)
        ):
            retry_version_scope = dict(missing_version_scopes[0])
        retry_fingerprint = self._retry_plan_fingerprint(
            tool_name,
            retry_query,
            retry_version_scope,
        )
        existing_retry_fingerprints = {
            str(item.get("retry_plan_fingerprint"))
            if item.get("retry_plan_fingerprint")
            else self._retry_plan_fingerprint(
                str(item["tool"]),
                str(item["query"]),
                dict(item["version_scope"]),
            )
            for item in state.query_plan
            if int(item.get("round", 0)) > 0
        }
        if retry_fingerprint in existing_retry_fingerprints:
            state.retrieval_stop_reason = "duplicate_retry_query"
            state.evidence_grade = {
                **state.evidence_grade,
                "reason": ("Supplementary retrieval would repeat an existing plan."),
                "stop_reason": "duplicate_retry_query",
                "retry_query_status": "duplicate",
            }
            return False
        state.query_plan.append(
            {
                "subquery_id": subquery_id,
                "tool": tool_name,
                "query": retry_query,
                "depends_on": [],
                "status": "pending",
                "round": retry_number,
                "version_scope": retry_version_scope,
                "retry_plan_fingerprint": retry_fingerprint,
            }
        )
        state.retry_queries.append(retry_query)
        state.query_rewrite_rounds.append(
            {
                "round": retry_number,
                "queries": [retry_query],
                "plan_ids": [subquery_id],
                "reason": state.evidence_grade.get("reason"),
            }
        )
        return True

    def evaluate_evidence(self, state: AgentState) -> EvidenceEvaluation:
        """Grade evidence and return exactly one bounded next action."""

        self.grade_evidence(state)
        reason = str(state.evidence_grade.get("reason", "Evidence evaluated."))
        if state.enough_evidence:
            state.terminal_status = "answered"
            return EvidenceEvaluation(EvidenceAction.ANSWER, reason)
        if state.retry_count >= state.max_retry:
            state.terminal_status = "abstained"
            state.retrieval_stop_reason = "retry_exhausted"
            state.evidence_grade["stop_reason"] = "retry_exhausted"
            return EvidenceEvaluation(EvidenceAction.ABSTAIN, reason)
        retry_number = state.retry_count + 1
        if not self.prepare_supplementary_retrieval(
            state,
            retry_number=retry_number,
        ):
            state.terminal_status = "abstained"
            return EvidenceEvaluation(
                EvidenceAction.ABSTAIN,
                str(state.evidence_grade["reason"]),
            )
        state.retry_count = retry_number
        state.evidence_grade["retry_count"] = retry_number
        return EvidenceEvaluation(EvidenceAction.RETRY, reason)

    def generate_answer_streaming(
        self,
        state: AgentState,
    ) -> Iterable[str]:
        """Generate a citation-validated grounded answer from selected chunks."""

        if not state.enough_evidence:
            state.citations = []
            yield (
                "我没有找到足够可靠的内部证据来完整回答这个问题。"
                f" 已尝试 {state.retry_count} 次查询重写；"
                "可以换成更具体的系统、数据源或文档名称。"
            )
            return

        if not state.generation_contexts:
            self.prepare_generation_context(state)
        contexts = list(state.generation_contexts)
        citation_map = dict(state.generation_citation_map)
        request = GenerationRequest(
            query_id=state.query_id,
            user_query=state.user_query,
            resolved_entities=list(state.matched_entities),
            version_scope=dict(state.version_scope),
            contexts=contexts,
            memory_context=[
                str(item["summary"])
                for item in state.recalled_memories[: self.memory_top_k]
            ],
        )
        result = validate_generation_result(
            self.answer_generator.generate(request),
            contexts,
        )
        unknown = set(result.referenced_citation_ids).difference(citation_map)
        if unknown:
            raise ValueError("Answer generator returned unknown citation identifiers")
        state.citations = [
            citation_map[citation_id] for citation_id in result.referenced_citation_ids
        ]
        state.answer_generator_name = result.generator_name
        state.answer_model = result.model
        state.generation_fallback_reason = result.fallback_reason
        state.generation_context_count = len(contexts)
        state.generation_resolved_entity_count = len(state.matched_entities)
        yield from self._answer_chunks(result.text)

    def prepare_generation_context(self, state: AgentState) -> None:
        """Project selected evidence for generation without changing its identity."""

        if not state.enough_evidence:
            raise ValueError("context preparation requires sufficient evidence")
        selected = [item for item in state.reranked_chunks if item.selected]
        context_limit = (
            MAX_EXHAUSTIVE_CONTEXTS
            if state.retrieval_goal == "exhaustive"
            else MAX_CONTEXTS
        )
        selected = self._dedupe_by_chunk(selected)[:context_limit]
        originals, citation_map, truncated_count = self._generation_contexts(selected)
        original_ids = [context.chunk_id for context in originals]
        run = validate_compression_run(
            self.context_compressor.compress(state.user_query, originals),
            originals,
            query=state.user_query,
            required_terms=tuple(
                str(item["entity"])
                for item in state.matched_entities
                if isinstance(item.get("entity"), str)
            ),
        )
        projected = list(run.contexts)
        if [context.chunk_id for context in projected] != original_ids:
            raise ValueError("context compression changed selected source identity")
        before_chars = sum(len(context.prompt_text) for context in originals)
        after_chars = sum(len(context.prompt_text) for context in projected)
        state.generation_contexts = projected
        state.generation_citation_map = citation_map
        state.generation_context_truncated_count = truncated_count
        state.context_compression = {
            "configured_mode": run.configured_mode,
            "effective_mode": run.effective_mode,
            "compressor_name": run.compressor_name,
            "model": run.model,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "retained_source_count": len(projected),
            "fallback_reason": run.fallback_reason,
        }

    def verify_answer(self, state: AgentState) -> None:
        """Verify terminal citations and selected-context membership."""

        if not state.enough_evidence:
            if state.citations:
                raise ValueError("Abstention must not include citations")
            state.answer_validation = {
                "valid": True,
                "mode": "abstention",
                "citation_count": 0,
                "reason": "No unsupported grounded answer was emitted.",
            }
            return

        markers = set(re.findall(r"\[(C\d+)\]", state.answer))
        citation_ids = {str(citation["citation_id"]) for citation in state.citations}
        selected_chunk_ids = {
            item.chunk.chunk_id for item in state.reranked_chunks if item.selected
        }
        cited_chunk_ids = {str(citation["chunk_id"]) for citation in state.citations}
        if not markers or markers != citation_ids:
            raise ValueError("Answer markers do not match structured citations")
        if not cited_chunk_ids.issubset(selected_chunk_ids):
            raise ValueError("Answer citations are outside selected context")
        self._verify_version_policy(state)
        state.answer_validation = {
            "valid": True,
            "mode": "citation_self_check",
            "citation_count": len(citation_ids),
            "selected_context_count": len(selected_chunk_ids),
            "reason": "All answer citations resolve to selected context.",
        }

    @staticmethod
    def _resolve_version_scope(
        question: str,
        *,
        intent: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        lowered = question.casefold()
        version_matches = [
            (match.start(), match.group(0).casefold())
            for match in VERSION_TOKEN_PATTERN.finditer(question)
        ]
        version_matches.extend(
            (
                match.start("version"),
                f"v{match.group('version').casefold()}",
            )
            for match in PRODUCT_BARE_VERSION_PATTERN.finditer(question)
        )
        versions = list(
            dict.fromkeys(
                version
                for _position, version in sorted(
                    version_matches,
                    key=lambda item: item[0],
                )
            )
        )
        current_requested = any(marker in lowered for marker in CURRENT_VERSION_MARKERS)
        sides: list[dict[str, Any]] = [
            {"mode": "exact", "doc_version": version} for version in versions
        ]
        if current_requested:
            sides.append({"mode": "current"})
        version_comparison = intent == "comparison" and (
            bool(versions)
            or current_requested
            or any(marker in lowered for marker in VERSION_COMPARISON_MARKERS)
        )
        if (
            len(sides) > 2
            or (version_comparison and len(sides) != 2)
            or (len(versions) > 1 and not version_comparison)
        ):
            return (
                {
                    "mode": "ambiguous",
                    "doc_versions": versions,
                    "sides": sides,
                },
                {
                    "matched_surface": "document version",
                    "candidate_entity_ids": [],
                    "domains": [],
                    "status": "ambiguous_version_scope",
                },
            )
        if version_comparison:
            return (
                {
                    "mode": "comparison",
                    "doc_versions": versions,
                    "sides": sides,
                },
                None,
            )
        if versions:
            return (
                {
                    "mode": "exact",
                    "doc_versions": versions,
                    "sides": [sides[0]],
                },
                None,
            )
        return (
            {
                "mode": "current",
                "doc_versions": [],
                "sides": [{"mode": "current"}],
            },
            None,
        )

    @staticmethod
    def _effective_query(state: AgentState) -> str:
        """Combine the question with bounded session context for retrieval."""

        if not state.memory_context:
            return state.user_query
        return f"{state.user_query} {state.memory_context}"

    @staticmethod
    def _remembered_statement(question: str) -> str | None:
        """Extract the bounded payload of an explicit remember request."""

        statement = re.sub(
            r"^\s*(?:请)?记住(?:一下)?[\s:：,，]*",
            "",
            question,
            count=1,
            flags=re.IGNORECASE,
        )
        statement = re.sub(
            r"^\s*remember[\s:：,，]*",
            "",
            statement,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        return statement or None

    @staticmethod
    def _should_recall_memory(question: str) -> bool:
        """Limit Memory injection to explicit recall and referential follow-ups."""

        if detect_memory_recall(question) is not None:
            return True
        lowered = question.casefold()
        return any(marker in lowered for marker in CONTEXTUAL_MEMORY_MARKERS)

    @staticmethod
    def _memory_presentation(record: MemoryRecord) -> dict[str, Any]:
        """Serialize only session-private fields required by prompt/UI."""

        return {
            "turn_id": record.turn_id,
            "role": record.role,
            "memory_type": record.memory_type,
            "summary": record.presentation_summary(),
            "created_at": record.created_at,
            "expires_at": record.expires_at,
        }

    @staticmethod
    def _plan_version_scopes(
        version_scope: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if version_scope.get("mode") == "comparison":
            sides = version_scope.get("sides")
            if not isinstance(sides, list) or len(sides) != 2:
                raise ValueError("Comparison version scope requires two sides")
            return [dict(side) for side in sides]
        if version_scope.get("mode") == "exact":
            versions = version_scope.get("doc_versions")
            if not isinstance(versions, list) or len(versions) != 1:
                raise ValueError("Exact version scope requires one version")
            return [{"mode": "exact", "doc_version": str(versions[0])}]
        return [{"mode": "current"}]

    @staticmethod
    def _apply_version_scope(
        filters: dict[str, Any],
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        output = dict(filters)
        output.pop("doc_version", None)
        output.pop("is_current", None)
        mode = scope.get("mode")
        if mode == "current":
            output["is_current"] = True
        elif mode == "exact":
            version = scope.get("doc_version")
            if not isinstance(version, str) or not version:
                raise ValueError("Exact version scope requires doc_version")
            output["doc_version"] = version
        else:
            raise ValueError(f"Unsupported plan version scope: {mode!r}")
        return output

    @staticmethod
    def _retry_version_scope(state: AgentState) -> dict[str, Any]:
        scopes = AgenticRAGWorkflow._plan_version_scopes(state.version_scope)
        return dict(scopes[0])

    @staticmethod
    def _validate_retrieved_versions(
        state: AgentState,
        results: list[SearchResult],
    ) -> None:
        mode = state.version_scope.get("mode")
        if mode == "current" and any(not item.chunk.is_current for item in results):
            raise ValueError("Current version scope returned historical chunks")
        if mode == "exact":
            versions = state.version_scope.get("doc_versions", [])
            if len(versions) != 1 or any(
                item.chunk.doc_version != versions[0] for item in results
            ):
                raise ValueError("Exact version scope returned another edition")
        if mode == "comparison":
            scopes = AgenticRAGWorkflow._plan_version_scopes(state.version_scope)
            if any(
                not any(
                    AgenticRAGWorkflow._chunk_matches_scope(
                        item.chunk,
                        scope,
                    )
                    for scope in scopes
                )
                for item in results
            ):
                raise ValueError("Comparison retrieval returned an unrequested edition")
        if mode != "comparison":
            by_doc: dict[str, set[str]] = {}
            for item in results:
                by_doc.setdefault(item.chunk.doc_id, set()).add(item.chunk.doc_version)
            if any(len(versions) > 1 for versions in by_doc.values()):
                raise ValueError("Non-comparison retrieval mixed document editions")

    @staticmethod
    def _verify_version_policy(state: AgentState) -> None:
        mode = state.version_scope.get("mode")
        selected = [item.chunk for item in state.reranked_chunks if item.selected]
        by_doc: dict[str, set[str]] = {}
        for chunk in selected:
            by_doc.setdefault(chunk.doc_id, set()).add(chunk.doc_version)
        if mode != "comparison" and any(
            len(versions) > 1 for versions in by_doc.values()
        ):
            raise ValueError("Selected context mixes document editions")
        if mode == "current" and any(not chunk.is_current for chunk in selected):
            raise ValueError("Current answer selected a historical edition")
        if mode == "exact":
            requested = state.version_scope.get("doc_versions", [])
            if len(requested) != 1 or any(
                chunk.doc_version != requested[0] for chunk in selected
            ):
                raise ValueError("Exact answer selected another edition")
        if mode == "comparison":
            scopes = AgenticRAGWorkflow._plan_version_scopes(state.version_scope)
            if any(
                not any(
                    AgenticRAGWorkflow._chunk_matches_scope(chunk, scope)
                    for scope in scopes
                )
                for chunk in selected
            ):
                raise ValueError("Comparison answer selected an unrequested edition")
            if any(
                not any(
                    AgenticRAGWorkflow._chunk_matches_scope(chunk, scope)
                    for chunk in selected
                )
                for scope in scopes
            ):
                raise ValueError("Comparison answer is missing a requested edition")
            if not any(
                all(
                    any(
                        chunk.doc_id == doc_id
                        and AgenticRAGWorkflow._chunk_matches_scope(
                            chunk,
                            scope,
                        )
                        for chunk in selected
                    )
                    for scope in scopes
                )
                for doc_id in {chunk.doc_id for chunk in selected}
            ):
                raise ValueError("Comparison answer does not cover one document family")
        cited_versions = {str(citation["doc_version"]) for citation in state.citations}
        if mode == "comparison" and any(
            version not in state.answer for version in cited_versions
        ):
            raise ValueError("Version comparison answer must label every cited edition")

    @staticmethod
    def _generation_contexts(
        selected: list[RerankedResult],
    ) -> tuple[
        list[GenerationContext],
        dict[str, dict[str, Any]],
        int,
    ]:
        contexts: list[GenerationContext] = []
        citation_map: dict[str, dict[str, Any]] = {}
        remaining = MAX_CONTEXT_CHARS
        truncated_count = 0
        for item in selected:
            source_text = item.chunk.text
            bounded_text = source_text[:remaining]
            if not bounded_text:
                truncated_count += 1
                continue
            if len(bounded_text) < len(source_text):
                truncated_count += 1
            citation_id = f"C{len(contexts) + 1}"
            contexts.append(
                GenerationContext(
                    citation_id=citation_id,
                    chunk_id=item.chunk.chunk_id,
                    doc_id=item.chunk.doc_id,
                    doc_version=item.chunk.doc_version,
                    title=item.chunk.title,
                    page_no=item.chunk.page_no,
                    section=item.chunk.section,
                    prompt_text=bounded_text,
                )
            )
            citation_map[citation_id] = item.chunk.citation(citation_id)
            remaining -= len(bounded_text)
        return contexts, citation_map, truncated_count

    @staticmethod
    def _answer_chunks(text: str) -> Iterable[str]:
        for start in range(0, len(text), ANSWER_CHUNK_CHARS):
            yield text[start : start + ANSWER_CHUNK_CHARS]

    def _grade_payload(
        self,
        state: AgentState,
        *,
        enough: bool,
        reason: str,
        evidence_basis: str,
        missing_aspects: list[str],
        relevant_count: int,
        top_score: float,
        covered_tools: list[str],
        missing_tools: list[str],
        missing_version_scopes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        suggested_tool = missing_tools[0] if missing_tools else state.selected_tools[0]
        return {
            "enough_evidence": enough,
            "reason": reason,
            "evidence_basis": evidence_basis,
            "covered_aspects": covered_tools,
            "missing_aspects": missing_aspects,
            "missing_version_scopes": missing_version_scopes,
            "suggested_tool": suggested_tool,
            "suggested_retry_query": self._retry_query(
                state,
                missing_aspects=missing_aspects,
            ),
            "retry_count": state.retry_count,
            "max_retry": state.max_retry,
            "relevant_chunks": relevant_count,
            "top_rerank_score": round(top_score, 4),
        }

    @staticmethod
    def _chunk_matches_scope(
        chunk: KBChunk,
        scope: dict[str, Any],
    ) -> bool:
        mode = scope.get("mode")
        if mode == "current":
            return chunk.is_current
        if mode == "exact":
            return chunk.doc_version == scope.get("doc_version")
        return False

    @staticmethod
    def _has_query_overlap(query: str, text: str) -> bool:
        query_terms = set(tokenize(query))
        text_terms = set(tokenize(text))
        return bool(query_terms.intersection(text_terms))

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(normalized.split())

    @staticmethod
    def _has_multiple_requested_aspects(query: str) -> bool:
        normalized = AgenticRAGWorkflow._normalize_match_text(query)
        matched_families = sum(
            any(marker in normalized for marker in family)
            for family in MULTI_ASPECT_MARKER_FAMILIES
        )
        return matched_families >= 2

    @staticmethod
    def _retry_query(
        state: AgentState,
        *,
        missing_aspects: list[str],
    ) -> str:
        """Preserve original terms and append only bounded evidence hints."""

        parts = [state.user_query.strip()]
        parts.extend(
            str(item.get("entity", "")).strip()
            for item in state.matched_entities
            if str(item.get("entity", "")).strip()
        )
        relevant_hint = next(
            (
                item
                for item in state.reranked_chunks
                if item.rerank_score >= RELEVANCE_THRESHOLD
                and AgenticRAGWorkflow._has_query_overlap(
                    state.user_query,
                    item.chunk.text,
                )
            ),
            None,
        )
        if relevant_hint is not None:
            parts.extend(
                value.strip()
                for value in (
                    relevant_hint.chunk.title,
                    relevant_hint.chunk.section or "",
                )
                if value.strip()
            )
        aspect_hints = {
            "no_relevant_evidence": "official documentation",
            "single_weak_chunk": "definition behavior constraints",
            "single_indirect_chunk": "exact feature documentation",
            "multi_aspect_requires_coverage": (
                "definition operation constraints supporting details"
            ),
            "incomplete_multi_evidence": "complete supporting details",
            "incomplete_exhaustive_coverage": "complete feature list",
        }
        parts.extend(
            aspect_hints.get(aspect, aspect) for aspect in missing_aspects if aspect
        )
        output: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = AgenticRAGWorkflow._normalize_match_text(part)
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(part)
        return " ".join(output)

    @staticmethod
    def _retry_plan_fingerprint(
        tool_name: str,
        query: str,
        version_scope: dict[str, Any],
    ) -> str:
        """Hash the execution identity of one supplementary search."""

        payload = {
            "tool": tool_name,
            "query": AgenticRAGWorkflow._normalize_match_text(query),
            "version_scope": version_scope,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _dedupe_by_chunk(
        results: list[RerankedResult],
    ) -> list[RerankedResult]:
        seen: set[str] = set()
        output: list[RerankedResult] = []
        for result in results:
            if result.chunk.chunk_id not in seen:
                seen.add(result.chunk.chunk_id)
                output.append(result)
        return output

    def _expand_document_results(
        self,
        state: AgentState,
        results: list[SearchResult],
        *,
        filters: dict[str, Any],
        subquery_id: str,
    ) -> list[SearchResult]:
        """Expand the best seed into authorized same-edition sibling chunks."""

        if state.retrieval_goal != "exhaustive" or not results:
            return []
        fetch_chunks = getattr(self.retriever, "fetch_document_chunks", None)
        if not callable(fetch_chunks):
            return []
        expanded_families = {
            (str(item["doc_id"]), str(item["doc_version"]))
            for item in state.document_expansions
        }
        seed = next(
            (
                result
                for result in results
                if (result.chunk.doc_id, result.chunk.doc_version)
                not in expanded_families
            ),
            None,
        )
        if seed is None:
            return []
        siblings = fetch_chunks(
            doc_id=seed.chunk.doc_id,
            doc_version=seed.chunk.doc_version,
            filters=filters,
            limit=min(
                state.milvus_top_k,
                MAX_DOCUMENT_EXPANSION_CHUNKS,
            ),
        )
        if not siblings:
            return []
        existing = {result.chunk.chunk_id: result for result in results}
        expanded: list[SearchResult] = []
        for index, chunk in enumerate(siblings, start=1):
            prior = existing.get(chunk.chunk_id)
            if prior is not None:
                expanded.append(prior)
                continue
            distance = abs(chunk.chunk_index - seed.chunk.chunk_index)
            sibling_score = max(
                0.0,
                seed.hybrid_score - min(0.05, distance * 0.001),
            )
            expanded.append(
                SearchResult(
                    chunk=chunk,
                    rank=index,
                    dense_score=seed.dense_score,
                    keyword_score=seed.keyword_score,
                    recency_score=seed.recency_score,
                    priority_score=seed.priority_score,
                    hybrid_score=sibling_score,
                )
            )
        state.document_expansions.append(
            {
                "subquery_id": subquery_id,
                "doc_id": seed.chunk.doc_id,
                "doc_version": seed.chunk.doc_version,
                "seed_chunk_id": seed.chunk.chunk_id,
                "result_count": len(siblings),
                "result_chunk_ids": [chunk.chunk_id for chunk in siblings],
            }
        )
        return expanded

    @staticmethod
    def _expanded_chunk_ids(state: AgentState) -> set[str]:
        return {
            str(chunk_id)
            for expansion in state.document_expansions
            for chunk_id in expansion.get("result_chunk_ids", [])
        }

    @staticmethod
    def _candidate_pool_fingerprint(state: AgentState) -> str:
        """Hash only stable, grading-relevant evidence and provenance metadata."""

        chunks = [
            {
                "chunk_id": item.chunk.chunk_id,
                "doc_version": item.chunk.doc_version,
                "checksum": item.chunk.checksum,
                "tool_paths": sorted(
                    {
                        f"{entry['tool']}:{path}"
                        for entry in state.retrieval_provenance.get(
                            item.chunk.chunk_id,
                            [],
                        )
                        for path in entry.get(
                            "retrieval_paths",
                            ("flat_hybrid",),
                        )
                    }
                ),
            }
            for item in sorted(
                state.retrieved_chunks,
                key=lambda result: result.chunk.chunk_id,
            )
        ]
        payload = {
            "chunks": chunks,
            "expanded_chunk_ids": sorted(AgenticRAGWorkflow._expanded_chunk_ids(state)),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _merge_search_results(
        results: list[SearchResult],
    ) -> list[SearchResult]:
        best_by_chunk: dict[str, SearchResult] = {}
        for result in results:
            chunk_id = result.chunk.chunk_id
            prior = best_by_chunk.get(chunk_id)
            if prior is None or result.hybrid_score > prior.hybrid_score:
                best_by_chunk[chunk_id] = result
        ordered = sorted(
            best_by_chunk.values(),
            key=lambda item: (
                -item.hybrid_score,
                -item.chunk.updated_at,
                -item.chunk.priority,
            ),
        )
        return [
            replace(result, rank=index) for index, result in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _scoped_result_ids(
        results: list[Any],
        *,
        version_scope: dict[str, Any],
        limit: int,
    ) -> set[str]:
        if limit <= 0:
            return set()
        if version_scope.get("mode") != "comparison":
            return {item.chunk.chunk_id for item in results[:limit]}
        scopes = AgenticRAGWorkflow._plan_version_scopes(version_scope)
        per_side = max(1, limit // len(scopes))
        chosen: list[str] = []
        for scope in scopes:
            side_ids = [
                item.chunk.chunk_id
                for item in results
                if AgenticRAGWorkflow._chunk_matches_scope(
                    item.chunk,
                    scope,
                )
            ]
            for chunk_id in side_ids[:per_side]:
                if chunk_id not in chosen:
                    chosen.append(chunk_id)
        for item in results:
            if len(chosen) >= limit:
                break
            if item.chunk.chunk_id not in chosen:
                chosen.append(item.chunk.chunk_id)
        return set(chosen)

    def _finalize(self, state: AgentState, started: float) -> None:
        elapsed_ms = round((self._clock() - started) * 1000, 3)
        state.metrics = {
            "latency_ms": elapsed_ms,
            "retrieval_latency_ms": state.stage_latency_ms.get(
                "execute_tool_plan",
                0.0,
            ),
            "rerank_latency_ms": state.stage_latency_ms.get(
                "rerank_evidence",
                0.0,
            ),
            "generation_latency_ms": state.stage_latency_ms.get(
                "generate_answer_streaming",
                0.0,
            ),
            "context_compression_latency_ms": state.stage_latency_ms.get(
                "prepare_generation_context",
                0.0,
            ),
            "memory_recall_latency_ms": state.stage_latency_ms.get(
                "recall_memory",
                0.0,
            ),
            "memory_write_latency_ms": state.stage_latency_ms.get(
                "persist_turn_memory",
                0.0,
            ),
            "num_retrieved": len(state.retrieved_chunks),
            "num_reranked": len(state.reranked_chunks),
            "num_context_chunks": state.generation_context_count,
            "num_resolved_entities": len(state.matched_entities),
            "num_tool_calls": len(state.tool_calls),
            "num_recalled_memories": len(state.recalled_memories),
            "num_written_memories": state.memory_written_count,
            "num_written_selective_memories": (state.selective_memory_written_count),
            "response_cache_hit": (state.terminal_status == "answered_from_cache"),
            "response_cache_candidate_count": (state.response_cache_candidate_count),
            "response_cache_written_count": (state.response_cache_written_count),
            "response_cache_validation_latency_ms": (
                state.stage_latency_ms.get(
                    "try_grounded_cache",
                    0.0,
                )
            ),
            "stage_latency_ms": dict(state.stage_latency_ms),
        }
        state.trace = {
            "query_id": state.query_id,
            "session_id": state.session_id,
            "original_query": state.user_query,
            "terminal_status": state.terminal_status,
            "memory": {
                "status": state.memory_status,
                "recall_decision": state.memory_recall_decision,
                "recall_mode": state.memory_recall_mode,
                "recall_reason": state.memory_recall_reason,
                "requested_count": state.memory_requested_count,
                "memory_types": list(state.memory_recall_types),
                "recalled_count": len(state.recalled_memories),
                "recalled": [
                    {
                        key: value
                        for key, value in item.items()
                        if key
                        in {
                            "memory_id",
                            "event_id",
                            "turn_id",
                            "role",
                            "memory_type",
                            "created_at",
                            "expires_at",
                            "source_event_ids",
                        }
                    }
                    for item in state.recalled_memories
                ],
                "written_count": state.memory_written_count,
                "ttl_seconds": state.memory_ttl_seconds,
                "selective": {
                    "status": state.selective_memory_status,
                    **state.selective_memory_pack,
                    "written_count": (state.selective_memory_written_count),
                    "retention_class": (state.selective_memory_retention_class),
                    "selection_reasons": list(state.selective_memory_selection_reasons),
                    "selector_name": state.selective_memory_selector_name,
                    "model": state.selective_memory_selector_model,
                    "fallback_reason": (
                        state.selective_memory_selector_fallback_reason
                    ),
                    "consolidation_status": (
                        state.selective_memory_consolidation_status
                    ),
                },
            },
            "response_cache": {
                "status": state.response_cache_status,
                "candidate_count": (state.response_cache_candidate_count),
                "match_type": state.response_cache_match_type,
                "similarity": state.response_cache_similarity,
                "source_query_id": (state.response_cache_source_query_id),
                "fallback_reason": (state.response_cache_fallback_reason),
                "expires_at": state.response_cache_expires_at,
                "written_count": (state.response_cache_written_count),
                "ttl_seconds": self.response_cache_ttl_seconds,
                "kb_revision": self.kb_revision,
            },
            "classify_query": {
                "intent": state.intent,
                "query_type": state.query_type,
                "retrieval_goal": state.retrieval_goal,
                "classifier_name": state.classifier_name,
                "model": state.classifier_model,
                "confidence": state.classification_confidence,
                "fallback_reason": state.classification_fallback_reason,
            },
            "terminology_resolution": {
                "catalog_version": state.entity_catalog_version,
                "matched_entities": state.matched_entities,
                "ambiguous_entities": state.ambiguous_entities,
            },
            "version_scope": state.version_scope,
            "retrieval_decision": state.retrieval_decision,
            "permission": state.permission_decision,
            "tool_selection": {
                "selected_tools": state.selected_tools,
                "reasons": state.tool_selection_reasons,
            },
            "retrieval_goal": state.retrieval_goal,
            "query_transformation": dict(state.query_transformation),
            "rewrite_query": {"rounds": state.query_rewrite_rounds},
            "query_plan": state.query_plan,
            "tool_calls": state.tool_calls,
            "document_expansions": state.document_expansions,
            "milvus_search": {
                "mode": state.search_mode,
                "tier": state.retrieval_tier,
                "execution_mode": state.retrieval_execution_mode,
                "top_k": state.milvus_top_k,
                "order_by": state.search_order_by,
            },
            "reranker": {
                "name": state.reranker_name,
                "model": state.reranker_model,
                "fallback_active": (state.reranker_fallback_reason is not None),
                "fallback_reason": state.reranker_fallback_reason,
                "sticky_fallback_reason": (state.reranker_sticky_fallback_reason),
                "primary_attempt_count": state.reranker_primary_attempt_count,
                "fallback_only_count": state.reranker_fallback_only_count,
                "input_candidates": len(state.retrieved_chunks),
                "processed_candidates": len(state.retrieved_chunks),
                "output_top_k": state.reranker_top_k,
            },
            "evidence_grading": state.evidence_grade,
            "context_compression": dict(state.context_compression),
            "answer_generation": {
                "generator_name": state.answer_generator_name,
                "model": state.answer_model,
                "mode": state.generation_mode,
                "context_count": state.generation_context_count,
                "compressed_context_count": sum(
                    1
                    for context in state.generation_contexts
                    if getattr(context, "compression_mode", "disabled") != "disabled"
                ),
                "compression_modes": sorted(
                    {
                        str(getattr(context, "compression_mode", "disabled"))
                        for context in state.generation_contexts
                    }
                ),
                "resolved_entity_count": (state.generation_resolved_entity_count),
                "version_scope": state.version_scope.get("mode"),
                "context_truncated_count": (state.generation_context_truncated_count),
                "fallback_active": (state.generation_fallback_reason is not None),
                "fallback_reason": state.generation_fallback_reason,
            },
            "answer_validation": state.answer_validation,
            "answer": state.answer,
            "citations": state.citations,
        }

    def _measure_stage(
        self,
        state: AgentState,
        name: str,
        action: Callable[[], object],
    ) -> None:
        started = self._clock()
        try:
            action()
        except Exception as exc:
            self._record_elapsed(state, name, started)
            raise WorkflowStageError(name, state.query_id, exc) from exc
        self._record_elapsed(state, name, started)

    def _measure_stage_delta(
        self,
        state: AgentState,
        name: str,
        action: Callable[[], object],
    ) -> float:
        """Measure one invocation without exposing accumulated retry time."""

        prior = state.stage_latency_ms.get(name, 0.0)
        self._measure_stage(state, name, action)
        return round(state.stage_latency_ms[name] - prior, 3)

    def _measure_stage_result_delta(
        self,
        state: AgentState,
        name: str,
        action: Callable[[], Any],
    ) -> tuple[Any, float]:
        """Measure one typed stage invocation and return its result."""

        prior = state.stage_latency_ms.get(name, 0.0)
        started = self._clock()
        try:
            result = action()
        except Exception as exc:
            self._record_elapsed(state, name, started)
            raise WorkflowStageError(name, state.query_id, exc) from exc
        self._record_elapsed(state, name, started)
        elapsed = round(state.stage_latency_ms[name] - prior, 3)
        return result, elapsed

    def _stage_event(
        self,
        emitter: WorkflowEventEmitter,
        state: AgentState,
        stage: str,
        elapsed_ms: float,
        *,
        kind: EventKind = "stage_completed",
    ) -> dict[str, Any]:
        """Build an allow-listed, user-facing summary for one stage."""

        presentations: dict[
            str,
            tuple[str, str, dict[str, Any], EventStatus],
        ] = {
            "recall_memory": (
                "已检查会话记忆",
                (
                    "当前问题无需查询 Conversation Memory。"
                    if state.memory_recall_decision == "skipped"
                    else (
                        f"召回 {len(state.recalled_memories)} 条有效会话记忆。"
                        if state.memory_status != "recall_failed"
                        else "会话记忆暂时不可用，本轮继续执行。"
                    )
                ),
                {
                    "memory_status": state.memory_status,
                    "recall_decision": state.memory_recall_decision,
                    "recall_mode": state.memory_recall_mode,
                    "recall_reason": state.memory_recall_reason,
                    "requested_count": state.memory_requested_count,
                    "recalled_count": len(state.recalled_memories),
                    "selective_memory_status": (state.selective_memory_status),
                    "working_state_count": (
                        state.selective_memory_pack.get(
                            "working_state_count",
                            0,
                        )
                    ),
                    "durable_fact_count": (
                        state.selective_memory_pack.get(
                            "durable_fact_count",
                            0,
                        )
                    ),
                    "episode_candidate_count": (
                        state.selective_memory_pack.get(
                            "episode_candidate_count",
                            0,
                        )
                    ),
                    "conflict_count": (
                        state.selective_memory_pack.get(
                            "conflict_count",
                            0,
                        )
                    ),
                    "decay_profiles": (
                        state.selective_memory_pack.get(
                            "decay_profiles",
                            [],
                        )
                    ),
                    "decay_mode": state.selective_memory_pack.get("decay_mode"),
                    "memory_types": list(state.memory_recall_types),
                },
                ("warning" if state.memory_status == "recall_failed" else "completed"),
            ),
            "try_grounded_cache": (
                "已尝试复用已验证答案",
                (
                    "命中有效的历史答案与引用。"
                    if state.response_cache_status == "hit"
                    else "未命中可安全复用的历史答案，继续检索。"
                ),
                {
                    "cache_status": state.response_cache_status,
                    "cache_candidate_count": (state.response_cache_candidate_count),
                    "cache_match_type": (state.response_cache_match_type),
                    "cache_similarity": state.response_cache_similarity,
                    "fallback_reason": (state.response_cache_fallback_reason),
                },
                (
                    "completed"
                    if state.response_cache_status in {"hit", "miss", "stale"}
                    else "warning"
                ),
            ),
            "recall_authorized_experience": (
                "已检查授权经验",
                (
                    "已加载可复用的授权经验。"
                    if state.selective_memory_status == "recalled"
                    else (
                        "授权经验暂时不可用，本轮继续检索。"
                        if state.selective_memory_status == "recall_failed"
                        else "没有可复用的授权经验。"
                    )
                ),
                {
                    "selective_memory_status": state.selective_memory_status,
                    "working_state_count": (
                        state.selective_memory_pack.get("working_state_count", 0)
                    ),
                    "durable_fact_count": (
                        state.selective_memory_pack.get("durable_fact_count", 0)
                    ),
                    "episode_candidate_count": (
                        state.selective_memory_pack.get(
                            "episode_candidate_count",
                            0,
                        )
                    ),
                    "conflict_count": (
                        state.selective_memory_pack.get("conflict_count", 0)
                    ),
                },
                (
                    "warning"
                    if state.selective_memory_status == "recall_failed"
                    else "completed"
                ),
            ),
            "classify_and_route": (
                "已理解问题并确定路径",
                (
                    f"识别为 {state.query_type} / {state.intent}，"
                    + (
                        "需要检索内部知识库。"
                        if state.need_retrieval
                        else "将直接构建回答。"
                    )
                ),
                {
                    "intent": state.intent,
                    "query_type": state.query_type,
                    "retrieval_goal": state.retrieval_goal,
                    "route": (
                        QueryRoute.RETRIEVAL.value
                        if state.need_retrieval
                        else QueryRoute.DIRECT.value
                    ),
                    "classifier_name": state.classifier_name,
                    "model": state.classifier_model,
                    "confidence": state.classification_confidence,
                    "fallback_reason": (state.classification_fallback_reason),
                },
                "completed",
            ),
            "resolve_terminology": (
                "已解析领域术语",
                (
                    f"命中 {len(state.matched_entities)} 个预定义实体，"
                    f"版本范围为 {state.version_scope.get('mode', 'unknown')}。"
                ),
                {
                    "matched_entity_count": len(state.matched_entities),
                    "ambiguity_count": len(state.ambiguous_entities),
                    "version_mode": state.version_scope.get("mode"),
                },
                ("warning" if state.ambiguous_entities else "completed"),
            ),
            "check_permission": (
                "已检查访问权限",
                (
                    "当前身份可访问所需知识域。"
                    if state.permission_decision.get("allowed", False)
                    else "当前身份不能访问所需知识域。"
                ),
                {
                    "allowed": state.permission_decision.get(
                        "allowed",
                        False,
                    ),
                    "allowed_department_count": len(
                        state.permission_decision.get(
                            "allowed_departments",
                            [],
                        )
                    ),
                },
                (
                    "completed"
                    if state.permission_decision.get("allowed", False)
                    else "warning"
                ),
            ),
            "plan_retrieval": (
                "已规划检索",
                (
                    f"选择 {len(state.selected_tools)} 个只读工具，"
                    f"生成 {len(state.query_plan)} 个有边界的检索步骤。"
                ),
                {
                    "selected_tools": list(state.selected_tools),
                    "plan_count": len(state.query_plan),
                    "strategy": state.query_transformation.get("strategy"),
                    "item_roles": state.query_transformation.get(
                        "item_roles",
                        [],
                    ),
                    "transformer_name": state.query_transformation.get(
                        "transformer_name"
                    ),
                    "fallback_reason": state.query_transformation.get(
                        "fallback_reason"
                    ),
                },
                "completed",
            ),
            "execute_tool_plan": (
                "已完成混合检索",
                (
                    "补充检索未产生新的证据或覆盖范围。"
                    if state.candidate_pool_unchanged
                    else f"当前保留 {len(state.retrieved_chunks)} 个候选片段。"
                ),
                {
                    "candidate_count": len(state.retrieved_chunks),
                    "document_expansion_count": len(state.document_expansions),
                    "execution_mode": state.retrieval_execution_mode,
                    "retry_count": state.retry_count,
                    "candidate_pool_status": (
                        "unchanged" if state.candidate_pool_unchanged else "changed"
                    ),
                    "stop_reason": state.retrieval_stop_reason,
                },
                "warning" if state.candidate_pool_unchanged else "completed",
            ),
            "rerank_evidence": (
                "已完成证据排序",
                f"重排后保留 {len(state.reranked_chunks)} 个候选。",
                {
                    "candidate_count": len(state.reranked_chunks),
                    "primary_attempt_count": state.reranker_primary_attempt_count,
                    "fallback_only_count": state.reranker_fallback_only_count,
                    "sticky_fallback_reason": (state.reranker_sticky_fallback_reason),
                },
                "completed",
            ),
            "evaluate_evidence": (
                "已评估证据并确定下一步",
                (
                    "证据足以支撑回答。"
                    if state.enough_evidence
                    else (
                        "证据不足，已达到停止条件。"
                        if state.terminal_status == "abstained"
                        else (
                            f"证据仍有缺口，已安排第 {state.retry_count} 轮"
                            "定向补充检索。"
                        )
                    )
                ),
                {
                    "enough_evidence": state.enough_evidence,
                    "evidence_basis": state.evidence_grade.get(
                        "evidence_basis",
                        "insufficient_evidence",
                    ),
                    "next_action": (
                        EvidenceAction.ANSWER.value
                        if state.enough_evidence
                        else (
                            EvidenceAction.ABSTAIN.value
                            if state.terminal_status == "abstained"
                            else EvidenceAction.RETRY.value
                        )
                    ),
                    "retry_count": state.retry_count,
                    "relevant_count": state.evidence_grade.get(
                        "relevant_chunks",
                        0,
                    ),
                    "missing_aspects": [
                        str(item)
                        for item in state.evidence_grade.get(
                            "missing_aspects",
                            [],
                        )[:3]
                    ],
                    "stop_reason": state.evidence_grade.get("stop_reason"),
                },
                "completed" if state.enough_evidence else "warning",
            ),
            "generate_answer_streaming": (
                "已生成候选答案",
                "候选答案已生成，尚未向用户展示。",
                {
                    "generator_name": state.answer_generator_name,
                    "model": state.answer_model,
                    "context_count": state.generation_context_count,
                    "fallback_reason": state.generation_fallback_reason,
                },
                "completed",
            ),
            "prepare_generation_context": (
                "已准备生成上下文",
                (
                    f"保留 {state.context_compression.get('retained_source_count', 0)} "
                    "个原始证据来源。"
                ),
                {
                    "configured_mode": state.context_compression.get("configured_mode"),
                    "effective_mode": state.context_compression.get("effective_mode"),
                    "compressor_name": state.context_compression.get("compressor_name"),
                    "before_chars": state.context_compression.get("before_chars"),
                    "after_chars": state.context_compression.get("after_chars"),
                    "retained_source_count": state.context_compression.get(
                        "retained_source_count"
                    ),
                    "fallback_reason": state.context_compression.get("fallback_reason"),
                },
                (
                    "warning"
                    if state.context_compression.get("fallback_reason")
                    not in {None, "below_trigger", "not_configured"}
                    else "completed"
                ),
            ),
            "verify_answer": (
                "已验证答案",
                (
                    f"引用与所选证据一致，共 {len(state.citations)} 条。"
                    if state.answer_validation.get("valid", False)
                    else "答案验证未通过。"
                ),
                {
                    "valid": state.answer_validation.get("valid", False),
                    "citation_count": len(state.citations),
                    "mode": state.answer_validation.get("mode"),
                },
                (
                    "completed"
                    if state.answer_validation.get("valid", False)
                    else "warning"
                ),
            ),
            "persist_turn_memory": (
                "已更新会话记忆",
                (
                    f"保存 {state.memory_written_count} 条本轮记忆。"
                    if state.memory_status != "write_failed"
                    else "回答已完成，但本轮记忆暂未保存。"
                ),
                {
                    "memory_status": state.memory_status,
                    "written_count": state.memory_written_count,
                    "ttl_seconds": state.memory_ttl_seconds,
                    "selective_memory_status": (state.selective_memory_status),
                    "selective_written_count": (state.selective_memory_written_count),
                    "retention_class": (state.selective_memory_retention_class),
                    "selection_reasons": list(state.selective_memory_selection_reasons),
                    "selector_name": state.selective_memory_selector_name,
                    "model": state.selective_memory_selector_model,
                    "fallback_reason": (
                        state.selective_memory_selector_fallback_reason
                    ),
                    "consolidation_status": (
                        state.selective_memory_consolidation_status
                    ),
                },
                ("warning" if state.memory_status == "write_failed" else "completed"),
            ),
        }
        title, summary, details, status = presentations[stage]
        return emitter.emit(
            kind=kind,
            stage=stage,
            title=title,
            summary=summary,
            status=status,
            elapsed_ms=elapsed_ms,
            details=details,
        )

    @staticmethod
    def _tool_event(
        emitter: WorkflowEventEmitter,
        tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Summarize a completed tool without exposing query or filters."""

        tool_name = str(tool_call["tool"])
        result_count = int(tool_call["result_count"])
        version_scope = tool_call["version_scope"]
        return emitter.emit(
            kind="tool_completed",
            stage="execute_tool_plan",
            title=f"已调用 {tool_name}",
            summary=f"工具返回 {result_count} 个候选片段。",
            elapsed_ms=float(tool_call["latency_ms"]),
            details={
                "tool": tool_name,
                "result_count": result_count,
                "round": int(tool_call["round"]),
                "version_mode": version_scope.get("mode"),
                "doc_version": version_scope.get("doc_version"),
                "retrieval_profile": tool_call.get("retrieval_profile"),
                "result_granularity": tool_call.get("result_granularity"),
                "element_hit_count": len(tool_call.get("element_offsets", [])),
                "document_candidate_count": int(
                    tool_call.get("document_candidate_count", 0)
                ),
                "element_predicate_count": int(
                    tool_call.get("element_predicate_count", 0)
                ),
                "collapse_status": tool_call.get("collapse_status"),
                "fusion_recipe": tool_call.get("fusion_recipe"),
            },
        )

    def _record_elapsed(
        self,
        state: AgentState,
        name: str,
        started: float,
    ) -> None:
        elapsed_ms = max(0.0, (self._clock() - started) * 1000)
        prior = state.stage_latency_ms.get(name, 0.0)
        state.stage_latency_ms[name] = round(prior + elapsed_ms, 3)

    @staticmethod
    def _serialize(state: AgentState) -> dict[str, Any]:
        data = {
            item.name: deepcopy(getattr(state, item.name))
            for item in fields(state)
            if item.name not in UNSERIALIZED_STATE_FIELDS
        }
        data["milvus_recalled"] = [item.to_dict() for item in state.retrieved_chunks]
        data["reranked"] = [item.to_dict() for item in state.reranked_chunks]
        return data
