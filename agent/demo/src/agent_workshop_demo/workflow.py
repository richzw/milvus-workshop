"""Deterministic Agentic RAG workflow matching the workshop MVP."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Generator, Iterable
from dataclasses import asdict, replace
from typing import Any

from agent_workshop_demo.config import DEFAULT_SEARCH_PARAMS
from agent_workshop_demo.embedding import TextEmbeddingError, tokenize
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
    KBChunk,
    RerankedResult,
    SearchResult,
)
from agent_workshop_demo.reranker import Reranker, RuleBasedReranker
from agent_workshop_demo.retrieval import HybridRetriever, InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.validation import (
    normalize_filters,
    validate_identifier,
    validate_question,
)

DIRECT_RESPONSE = (
    "这个问题不需要检索内部知识库。"
    "请问一个和 workshop、RAG 或内部资料相关的问题。"
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
DEFAULT_MEMORY_TOP_K = 3
DEFAULT_MEMORY_TTL_SECONDS = 86_400
MAX_MEMORY_CONTEXT_CHARS = 2_000
VERSION_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:v\d+(?:\.\d+)*|\d{4}\.\d{1,2})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
CURRENT_VERSION_MARKERS = ("current", "latest", "当前", "最新版")
VERSION_COMPARISON_MARKERS = ("版本", "edition")


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
        "classify_query",
        "resolve_terminology",
        "decide_retrieval",
        "check_permission",
        "select_tools",
        "rewrite_query",
        "milvus_hybrid_retrieve",
        "rerank_evidence",
        "grade_evidence",
        "generate_answer_streaming",
        "verify_answer",
        "persist_turn_memory",
    ]

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        answer_generator: AnswerGenerator | None = None,
        permission_checker: PermissionChecker | None = None,
        entity_catalog: EntityCatalog | None = None,
        memory_store: ConversationMemory | None = None,
        memory_top_k: int = DEFAULT_MEMORY_TOP_K,
        memory_ttl_seconds: int = DEFAULT_MEMORY_TTL_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock_ms: Callable[[], int] = utc_now_ms,
    ) -> None:
        if not 1 <= memory_top_k <= 20:
            raise ValueError("memory_top_k must be between 1 and 20")
        if memory_ttl_seconds <= 0:
            raise ValueError("memory_ttl_seconds must be positive")
        self.retriever = retriever or InMemoryHybridRetriever(load_kb_chunks())
        self.reranker = reranker or RuleBasedReranker()
        self.answer_generator = (
            answer_generator or DeterministicAnswerGenerator()
        )
        self.permission_checker = (
            permission_checker or DemoPermissionChecker()
        )
        self.entity_catalog = entity_catalog or load_entity_catalog()
        self.memory_store = memory_store or ConversationMemoryStore(
            now_ms=wall_clock_ms()
        )
        self.memory_top_k = memory_top_k
        self.memory_ttl_seconds = memory_ttl_seconds
        self._clock = clock
        self._wall_clock_ms = wall_clock_ms

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
            reranker_name=self.reranker.name,
            max_retry=DEFAULT_SEARCH_PARAMS["max_retry"],
            search_mode=DEFAULT_SEARCH_PARAMS["search_mode"],
            memory_ttl_seconds=self.memory_ttl_seconds,
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
        elapsed = self._measure_stage_delta(
            state,
            "classify_query",
            lambda: self.classify_query(state),
        )
        yield self._stage_event(emitter, state, "classify_query", elapsed)
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
        if state.terminal_status == "clarification_required":
            return state, started, emitter
        elapsed = self._measure_stage_delta(
            state,
            "decide_retrieval",
            lambda: self.decide_retrieval(state),
        )
        yield self._stage_event(
            emitter,
            state,
            "decide_retrieval",
            elapsed,
        )
        if not state.need_retrieval:
            self.prepare_non_retrieval_answer(state)
            return state, started, emitter

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
        if not state.permission_decision.get("allowed", False):
            state.answer = PERMISSION_DENIED_RESPONSE
            state.terminal_status = "permission_denied"
            state.answer_validation = {
                "valid": True,
                "mode": "permission_denied",
                "reason": "Retrieval and generation were not executed.",
            }
            return state, started, emitter

        elapsed = self._measure_stage_delta(
            state,
            "select_tools",
            lambda: self.select_tools(state),
        )
        yield self._stage_event(emitter, state, "select_tools", elapsed)
        elapsed = self._measure_stage_delta(
            state,
            "rewrite_query",
            lambda: self.rewrite_query(state),
        )
        yield self._stage_event(emitter, state, "rewrite_query", elapsed)
        while True:
            prior_tool_calls = len(state.tool_calls)
            elapsed = self._measure_stage_delta(
                state,
                "milvus_hybrid_retrieve",
                lambda: self.milvus_hybrid_retrieve(state),
            )
            yield self._stage_event(
                emitter,
                state,
                "milvus_hybrid_retrieve",
                elapsed,
            )
            for tool_call in state.tool_calls[prior_tool_calls:]:
                yield self._tool_event(emitter, tool_call)
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
            elapsed = self._measure_stage_delta(
                state,
                "grade_evidence",
                lambda: self.grade_evidence(state),
            )
            yield self._stage_event(
                emitter,
                state,
                "grade_evidence",
                elapsed,
            )
            if state.enough_evidence or state.retry_count >= state.max_retry:
                break
            state.retry_count += 1
            elapsed = self._measure_stage_delta(
                state,
                "prepare_supplementary_retrieval",
                lambda: self.prepare_supplementary_retrieval(state),
            )
            yield self._stage_event(
                emitter,
                state,
                "prepare_supplementary_retrieval",
                elapsed,
                kind="retry_scheduled",
            )

        state.terminal_status = (
            "answered" if state.enough_evidence else "abstained"
        )
        return state, started, emitter

    def recall_memory(self, state: AgentState) -> None:
        """Recall bounded, live context from only the active session."""

        if not self._should_recall_memory(state.user_query):
            state.memory_status = "empty"
            state.recalled_memories = []
            state.memory_context = ""
            return
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
            return
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
            return
        if state.memory_status != "recall_failed":
            state.memory_status = "saved"

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

    def clear_memory(self, session_id: str) -> int:
        """Delete only the selected session's Memory records."""

        return self.memory_store.delete_session(session_id)

    def classify_query(self, state: AgentState) -> None:
        """Classify intent separately from the knowledge topic."""

        lowered_query = state.user_query.lower()
        lowered = f"{state.user_query} {state.memory_context}".lower()
        if any(
            term in lowered_query
            for term in ["请记住", "记住", "remember "]
        ):
            state.intent = "memory_write"
            state.query_type = "general"
            state.remembered_statement = self._remembered_statement(
                state.user_query
            )
            return
        if any(
            term in lowered_query
            for term in [
                "你还记得",
                "我之前",
                "我叫什么",
                "do you remember",
                "what did i",
            ]
        ):
            state.intent = "memory_recall"
            state.query_type = "general"
            return
        if any(
            term in lowered_query
            for term in [
                "帮我删除",
                "帮我修改",
                "帮我创建",
                "帮我提交",
                "执行命令",
                "approve this",
                "delete ",
            ]
        ):
            state.intent = "operation"
        elif any(
            term in lowered_query
            for term in ["对比", "比较", "覆盖", "vs", "versus", "有没有被"]
        ):
            state.intent = "comparison"
        elif any(
            term in lowered_query
            for term in ["权限", "敏感", "机密", "salary", "薪资", "acl"]
        ):
            state.intent = "permission_sensitive"
        else:
            state.intent = "private_knowledge"

        architecture_terms = [
            "s3",
            "milvus",
            "rag",
            "架构",
            "同步",
            "检索",
            "embedding",
        ]
        if any(term in lowered for term in architecture_terms):
            state.query_type = "architecture"
        elif any(
            term in lowered
            for term in ["pto", "policy", "hr", "假期", "制度"]
        ):
            state.query_type = "policy"
        elif any(
            term in lowered
            for term in [
                "ui",
                "streamlit",
                "界面",
                "产品",
                "路线图",
                "roadmap",
                "客户",
            ]
        ):
            state.query_type = "product"
        elif any(term in lowered_query for term in ["hello", "hi", "你好"]):
            state.query_type = "general"
            state.intent = "conversation"
        else:
            state.query_type = "unknown"

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
                state.answer = (
                    "我已处理这条会话记忆请求，保存结果见 Memory 状态。"
                )
                state.terminal_status = "memory_saved"
            else:
                state.answer = "请补充希望我在当前会话中记住的具体信息。"
                state.terminal_status = "clarification_required"
        elif state.intent == "memory_recall":
            if state.recalled_memories:
                summaries = [
                    str(item["summary"]) for item in state.recalled_memories
                ]
                state.answer = (
                    "根据当前会话中你之前提供的信息：\n- "
                    + "\n- ".join(summaries)
                )
                state.terminal_status = "answered_from_memory"
            else:
                state.answer = (
                    "当前会话中没有找到匹配且仍在有效期内的记忆。"
                )
                state.terminal_status = "memory_not_found"
        else:
            state.answer = DIRECT_RESPONSE
            state.terminal_status = "answered_without_retrieval"
        state.answer_validation = {
            "valid": True,
            "mode": (
                "memory_write"
                if state.intent == "memory_write"
                and state.remembered_statement
                else (
                    "memory_grounded"
                    if state.intent == "memory_recall"
                    and state.recalled_memories
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
                if state.intent == "memory_recall"
                and state.recalled_memories
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
        state.ambiguous_entities = [
            dict(item) for item in resolution.ambiguous
        ]
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
            term in lowered
            for term in ["会议", "纪要", "客户", "meeting", "feedback"]
        ):
            choose(
                "search_meeting_notes",
                "Customer or meeting evidence is required.",
            )
        if state.query_type == "product" or "product" in entity_domains or any(
            term in lowered
            for term in ["产品", "路线图", "roadmap", "feature", "ui"]
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
        """Create a bounded tool-aware query plan."""

        if state.query_plan:
            return
        lowered = state.user_query.lower()
        plan: list[dict[str, Any]] = []
        resolved_terms = " ".join(
            f"{item['entity']} {item['comment']}"
            for item in state.matched_entities
        )
        scopes = self._plan_version_scopes(state.version_scope)
        scoped_tools = [
            (tool_name, scope)
            for tool_name in state.selected_tools
            for scope in scopes
        ]
        prior_id: str | None = None
        for index, (tool_name, version_scope) in enumerate(
            scoped_tools[:MAX_INITIAL_SUBQUERIES],
            start=1,
        ):
            tool = SEARCH_TOOLS[tool_name]
            query = f"{state.user_query} {tool.query_hint}"
            if state.memory_context:
                query += f" prior session context {state.memory_context}"
            if resolved_terms:
                query += f" {resolved_terms}"
            if tool_name == "search_code_docs" and (
                "s3" in lowered or "同步" in state.user_query
            ):
                query += (
                    " MinIO bucket scanning change detection document parsing"
                    " chunking embedding generation Milvus insertion"
                )
            if tool_name == "search_product_docs" and (
                "路线图" in state.user_query or "roadmap" in lowered
            ):
                query += " product roadmap coverage planned capabilities"

            dependencies: list[str] = []
            if (
                state.intent == "comparison"
                and prior_id is not None
                and tool_name == "search_product_docs"
            ):
                dependencies = [prior_id]
            subquery_id = f"sq{index}"
            plan.append(
                {
                    "subquery_id": subquery_id,
                    "tool": tool_name,
                    "query": query,
                    "depends_on": dependencies,
                    "status": "pending",
                    "round": 0,
                    "version_scope": dict(version_scope),
                }
            )
            prior_id = subquery_id

        state.query_plan = plan[:MAX_INITIAL_SUBQUERIES]
        state.rewritten_queries = [
            str(item["query"]) for item in state.query_plan
        ]
        state.query_rewrite_rounds.append(
            {
                "round": 0,
                "queries": list(state.rewritten_queries),
                "plan_ids": [
                    str(item["subquery_id"]) for item in state.query_plan
                ],
            }
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
        new_results: list[SearchResult] = []
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
            call_started = self._clock()
            results = self.retriever.search(
                str(item["query"]),
                top_k=state.milvus_top_k,
                filters=tool_filters,
                order_by=state.search_order_by,
            )
            call_latency_ms = round(
                max(0.0, (self._clock() - call_started) * 1000),
                3,
            )
            result_ids = [result.chunk.chunk_id for result in results]
            state.tool_calls.append(
                {
                    "tool": tool.name,
                    "subquery_id": item["subquery_id"],
                    "query": item["query"],
                    "filters": tool_filters,
                    "result_count": len(results),
                    "result_chunk_ids": result_ids,
                    "latency_ms": call_latency_ms,
                    "round": state.retry_count,
                    "version_scope": version_scope,
                }
            )
            for result in results:
                state.retrieval_provenance.setdefault(
                    result.chunk.chunk_id,
                    [],
                ).append(
                    {
                        "tool": tool.name,
                        "subquery_id": item["subquery_id"],
                    }
                )
            item["status"] = "completed"
            new_results.extend(results)

        self._validate_retrieved_versions(state, new_results)
        merged = self._merge_search_results(
            [*state.retrieved_chunks, *new_results]
        )
        retained_ids = self._scoped_result_ids(
            merged,
            version_scope=state.version_scope,
            limit=state.milvus_top_k,
        )
        retained = [
            item for item in merged if item.chunk.chunk_id in retained_ids
        ]
        state.retrieved_chunks = [
            replace(item, rank=index)
            for index, item in enumerate(retained, start=1)
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

    def rerank_evidence(self, state: AgentState) -> None:
        """Rerank recalled candidates without selecting weak context."""

        rerank_pool_size = (
            state.milvus_top_k
            if state.version_scope.get("mode") == "comparison"
            else state.reranker_top_k
        )
        reranked_pool = self.reranker.rerank(
            self._effective_query(state),
            state.retrieved_chunks,
            top_k=rerank_pool_size,
        )
        retained_ids = self._scoped_result_ids(
            reranked_pool,
            version_scope=state.version_scope,
            limit=state.reranker_top_k,
        )
        reranked = [
            item
            for item in reranked_pool
            if item.chunk.chunk_id in retained_ids
        ]
        state.reranked_chunks = [
            replace(result, selected=False) for result in reranked
        ]

    def grade_evidence(self, state: AgentState) -> None:
        """Select covered evidence or produce a targeted retrieval gap."""

        relevant = [
            item
            for item in state.reranked_chunks
            if item.rerank_score >= RELEVANCE_THRESHOLD
            and self._has_query_overlap(
                self._effective_query(state),
                item.chunk.text,
            )
        ]
        top_score = (
            state.reranked_chunks[0].rerank_score
            if state.reranked_chunks
            else 0.0
        )
        unique_relevant = {item.chunk.chunk_id for item in relevant}
        enough = len(relevant) >= 2 and top_score >= 0.28
        enough = enough and len(unique_relevant) >= 2
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
            comparison_scopes = self._plan_version_scopes(
                state.version_scope
            )
            missing_version_scopes = [
                scope
                for scope in comparison_scopes
                if not any(
                    self._chunk_matches_scope(item.chunk, scope)
                    for item in relevant
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
        if state.intent == "comparison":
            enough = (
                enough
                and not missing_tools
                and not missing_version_scopes
                and comparison_family_complete
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
                if any(
                    entry["tool"] == tool_name for entry in provenance
                ) and item not in selected:
                    selected.append(item)
                    break
        for item in relevant:
            if item not in selected:
                selected.append(item)
        selected_ids = {
            item.chunk.chunk_id
            for item in selected[
                : DEFAULT_SEARCH_PARAMS["answer_context_top_k"]
            ]
        }
        state.reranked_chunks = [
            replace(
                item,
                selected=enough and item.chunk.chunk_id in selected_ids,
            )
            for item in state.reranked_chunks
        ]
        state.enough_evidence = enough
        reason = (
            "Reranked evidence covers every planned knowledge aspect."
            if enough
            else "Evidence is missing planned knowledge aspects or citations."
        )
        state.evidence_grade = self._grade_payload(
            state,
            enough=enough,
            reason=reason,
            relevant_count=len(relevant),
            top_score=top_score,
            covered_tools=sorted(covered_tools),
            missing_tools=missing_tools,
            missing_version_scopes=missing_version_scopes,
        )

    def prepare_supplementary_retrieval(self, state: AgentState) -> None:
        """Activate a pending hop or append one targeted retry query."""

        pending = [
            item for item in state.query_plan if item["status"] == "pending"
        ]
        if pending:
            retry_query = str(pending[0]["query"])
            state.retry_queries.append(retry_query)
            state.query_rewrite_rounds.append(
                {
                    "round": state.retry_count,
                    "queries": [retry_query],
                    "plan_ids": [str(pending[0]["subquery_id"])],
                    "reason": "Execute a planned dependent retrieval hop.",
                }
            )
            return

        tool_name = str(
            state.evidence_grade.get(
                "suggested_tool",
                state.selected_tools[0],
            )
        )
        if tool_name not in SEARCH_TOOLS:
            tool_name = state.selected_tools[0]
        retry_query = str(
            state.evidence_grade.get(
                "suggested_retry_query",
                self._retry_query(state),
            )
        )
        subquery_id = f"retry{state.retry_count}"
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
        state.query_plan.append(
            {
                "subquery_id": subquery_id,
                "tool": tool_name,
                "query": retry_query,
                "depends_on": [],
                "status": "pending",
                "round": state.retry_count,
                "version_scope": retry_version_scope,
            }
        )
        state.retry_queries.append(retry_query)
        state.query_rewrite_rounds.append(
            {
                "round": state.retry_count,
                "queries": [retry_query],
                "plan_ids": [subquery_id],
                "reason": state.evidence_grade.get("reason"),
            }
        )

    def generate_answer_streaming(
        self,
        state: AgentState,
    ) -> Iterable[str]:
        """Generate a citation-validated grounded answer from selected chunks."""

        selected = [item for item in state.reranked_chunks if item.selected]
        if not state.enough_evidence:
            state.citations = []
            yield (
                "我没有找到足够可靠的内部证据来完整回答这个问题。"
                f" 已尝试 {state.retry_count} 次查询重写；"
                "可以换成更具体的系统、数据源或文档名称。"
            )
            return

        selected = self._dedupe_by_chunk(selected)[:MAX_CONTEXTS]
        contexts, citation_map, truncated_count = self._generation_contexts(
            selected
        )
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
            raise ValueError(
                "Answer generator returned unknown citation identifiers"
            )
        state.citations = [
            citation_map[citation_id]
            for citation_id in result.referenced_citation_ids
        ]
        state.answer_generator_name = result.generator_name
        state.answer_model = result.model
        state.generation_fallback_reason = result.fallback_reason
        state.generation_context_count = len(contexts)
        state.generation_resolved_entity_count = len(
            state.matched_entities
        )
        state.generation_context_truncated_count = truncated_count
        yield from self._answer_chunks(result.text)

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
        citation_ids = {
            str(citation["citation_id"]) for citation in state.citations
        }
        selected_chunk_ids = {
            item.chunk.chunk_id
            for item in state.reranked_chunks
            if item.selected
        }
        cited_chunk_ids = {
            str(citation["chunk_id"]) for citation in state.citations
        }
        if not markers or markers != citation_ids:
            raise ValueError(
                "Answer markers do not match structured citations"
            )
        if not cited_chunk_ids.issubset(selected_chunk_ids):
            raise ValueError(
                "Answer citations are outside selected context"
            )
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
        versions = list(
            dict.fromkeys(
                match.group(0).casefold()
                for match in VERSION_TOKEN_PATTERN.finditer(question)
            )
        )
        current_requested = any(
            marker in lowered for marker in CURRENT_VERSION_MARKERS
        )
        sides: list[dict[str, Any]] = [
            {"mode": "exact", "doc_version": version}
            for version in versions
        ]
        if current_requested:
            sides.append({"mode": "current"})
        version_comparison = intent == "comparison" and (
            bool(versions)
            or current_requested
            or any(
                marker in lowered for marker in VERSION_COMPARISON_MARKERS
            )
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

        lowered = question.casefold()
        return any(
            marker in lowered
            for marker in [
                "你还记得",
                "我之前",
                "我叫什么",
                "刚才",
                "前面",
                "上述",
                "继续",
                "基于此",
                "它",
                "该流程",
                "do you remember",
                "what did i",
                "what about",
                "how about",
                "that ",
                " it ",
            ]
        )

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
        scopes = AgenticRAGWorkflow._plan_version_scopes(
            state.version_scope
        )
        return dict(scopes[0])

    @staticmethod
    def _validate_retrieved_versions(
        state: AgentState,
        results: list[SearchResult],
    ) -> None:
        mode = state.version_scope.get("mode")
        if mode == "current" and any(
            not item.chunk.is_current for item in results
        ):
            raise ValueError("Current version scope returned historical chunks")
        if mode == "exact":
            versions = state.version_scope.get("doc_versions", [])
            if len(versions) != 1 or any(
                item.chunk.doc_version != versions[0] for item in results
            ):
                raise ValueError("Exact version scope returned another edition")
        if mode == "comparison":
            scopes = AgenticRAGWorkflow._plan_version_scopes(
                state.version_scope
            )
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
                raise ValueError(
                    "Comparison retrieval returned an unrequested edition"
                )
        if mode != "comparison":
            by_doc: dict[str, set[str]] = {}
            for item in results:
                by_doc.setdefault(item.chunk.doc_id, set()).add(
                    item.chunk.doc_version
                )
            if any(len(versions) > 1 for versions in by_doc.values()):
                raise ValueError(
                    "Non-comparison retrieval mixed document editions"
                )

    @staticmethod
    def _verify_version_policy(state: AgentState) -> None:
        mode = state.version_scope.get("mode")
        selected = [
            item.chunk for item in state.reranked_chunks if item.selected
        ]
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
            scopes = AgenticRAGWorkflow._plan_version_scopes(
                state.version_scope
            )
            if any(
                not any(
                    AgenticRAGWorkflow._chunk_matches_scope(chunk, scope)
                    for scope in scopes
                )
                for chunk in selected
            ):
                raise ValueError(
                    "Comparison answer selected an unrequested edition"
                )
            if any(
                not any(
                    AgenticRAGWorkflow._chunk_matches_scope(chunk, scope)
                    for chunk in selected
                )
                for scope in scopes
            ):
                raise ValueError(
                    "Comparison answer is missing a requested edition"
                )
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
                raise ValueError(
                    "Comparison answer does not cover one document family"
                )
        cited_versions = {
            str(citation["doc_version"]) for citation in state.citations
        }
        if mode == "comparison" and any(
            version not in state.answer for version in cited_versions
        ):
            raise ValueError(
                "Version comparison answer must label every cited edition"
            )

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
                    text=bounded_text,
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
        relevant_count: int,
        top_score: float,
        covered_tools: list[str],
        missing_tools: list[str],
        missing_version_scopes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        suggested_tool = (
            missing_tools[0] if missing_tools else state.selected_tools[0]
        )
        return {
            "enough_evidence": enough,
            "reason": reason,
            "covered_aspects": covered_tools,
            "missing_aspects": (
                []
                if enough
                else (
                    [
                        *missing_tools,
                        *[
                            "version:"
                            + str(
                                scope.get(
                                    "doc_version",
                                    scope.get("mode", "unknown"),
                                )
                            )
                            for scope in missing_version_scopes
                        ],
                    ]
                    or ["specific document terms", "additional citations"]
                )
            ),
            "missing_version_scopes": missing_version_scopes,
            "suggested_tool": suggested_tool,
            "suggested_retry_query": self._retry_query(state),
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
    def _retry_query(state: AgentState) -> str:
        if state.query_type == "architecture":
            return (
                "S3 ingestion pipeline scheduler metadata extraction "
                "Milvus indexing"
            )
        return f"{state.user_query} internal docs citation evidence"

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
            replace(result, rank=index)
            for index, result in enumerate(ordered, start=1)
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
            return {
                item.chunk.chunk_id for item in results[:limit]
            }
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
                "milvus_hybrid_retrieve",
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
            "stage_latency_ms": dict(state.stage_latency_ms),
        }
        state.trace = {
            "query_id": state.query_id,
            "session_id": state.session_id,
            "original_query": state.user_query,
            "terminal_status": state.terminal_status,
            "memory": {
                "status": state.memory_status,
                "recalled_count": len(state.recalled_memories),
                "recalled": list(state.recalled_memories),
                "written_count": state.memory_written_count,
                "ttl_seconds": state.memory_ttl_seconds,
            },
            "classify_query": {
                "intent": state.intent,
                "query_type": state.query_type,
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
            "rewrite_query": {"rounds": state.query_rewrite_rounds},
            "query_plan": state.query_plan,
            "tool_calls": state.tool_calls,
            "milvus_search": {
                "mode": state.search_mode,
                "top_k": state.milvus_top_k,
                "order_by": state.search_order_by,
            },
            "reranker": {
                "name": self.reranker.name,
                "input_candidates": len(state.retrieved_chunks),
                "output_top_k": state.reranker_top_k,
            },
            "evidence_grading": state.evidence_grade,
            "answer_generation": {
                "generator_name": state.answer_generator_name,
                "model": state.answer_model,
                "mode": state.generation_mode,
                "context_count": state.generation_context_count,
                "resolved_entity_count": (
                    state.generation_resolved_entity_count
                ),
                "version_scope": state.version_scope.get("mode"),
                "context_truncated_count": (
                    state.generation_context_truncated_count
                ),
                "fallback_active": (
                    state.generation_fallback_reason is not None
                ),
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
        action: Callable[[], None],
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
        action: Callable[[], None],
    ) -> float:
        """Measure one invocation without exposing accumulated retry time."""

        prior = state.stage_latency_ms.get(name, 0.0)
        self._measure_stage(state, name, action)
        return round(state.stage_latency_ms[name] - prior, 3)

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
                    f"召回 {len(state.recalled_memories)} 条有效会话记忆。"
                    if state.memory_status != "recall_failed"
                    else "会话记忆暂时不可用，本轮继续执行。"
                ),
                {
                    "memory_status": state.memory_status,
                    "recalled_count": len(state.recalled_memories),
                    "memory_types": [
                        "session_summary",
                        "task_state",
                    ],
                },
                (
                    "warning"
                    if state.memory_status == "recall_failed"
                    else "completed"
                ),
            ),
            "classify_query": (
                "已理解问题",
                f"识别为 {state.query_type} / {state.intent}。",
                {
                    "intent": state.intent,
                    "query_type": state.query_type,
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
                (
                    "warning"
                    if state.ambiguous_entities
                    else "completed"
                ),
            ),
            "decide_retrieval": (
                "已确定回答路径",
                (
                    "需要检索内部知识库。"
                    if state.need_retrieval
                    else "无需检索内部知识库。"
                ),
                {"need_retrieval": state.need_retrieval},
                "completed",
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
            "select_tools": (
                "已选择知识工具",
                f"将调用 {len(state.selected_tools)} 个只读搜索工具。",
                {"selected_tools": list(state.selected_tools)},
                "completed",
            ),
            "rewrite_query": (
                "已规划检索",
                f"生成 {len(state.query_plan)} 个有边界的检索步骤。",
                {"plan_count": len(state.query_plan)},
                "completed",
            ),
            "milvus_hybrid_retrieve": (
                "已完成混合检索",
                f"当前保留 {len(state.retrieved_chunks)} 个候选片段。",
                {
                    "candidate_count": len(state.retrieved_chunks),
                    "retry_count": state.retry_count,
                },
                "completed",
            ),
            "rerank_evidence": (
                "已完成证据排序",
                f"重排后保留 {len(state.reranked_chunks)} 个候选。",
                {"candidate_count": len(state.reranked_chunks)},
                "completed",
            ),
            "grade_evidence": (
                "已检查证据覆盖",
                (
                    "证据足以支撑回答。"
                    if state.enough_evidence
                    else "证据仍有缺口，正在评估是否补充检索。"
                ),
                {
                    "enough_evidence": state.enough_evidence,
                    "retry_count": state.retry_count,
                    "relevant_count": state.evidence_grade.get(
                        "relevant_chunks",
                        0,
                    ),
                },
                "completed" if state.enough_evidence else "warning",
            ),
            "prepare_supplementary_retrieval": (
                "正在补充检索",
                (
                    f"已安排第 {state.retry_count} 轮定向补充检索；"
                    "待补充："
                    + "、".join(
                        str(item)
                        for item in state.evidence_grade.get(
                            "missing_aspects",
                            [],
                        )[:3]
                    )
                    + "。"
                ),
                {
                    "retry_count": state.retry_count,
                    "missing_aspects": [
                        str(item)
                        for item in state.evidence_grade.get(
                            "missing_aspects",
                            [],
                        )[:3]
                    ],
                },
                "warning",
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
                },
                (
                    "warning"
                    if state.memory_status == "write_failed"
                    else "completed"
                ),
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
            stage="milvus_hybrid_retrieve",
            title=f"已调用 {tool_name}",
            summary=f"工具返回 {result_count} 个候选片段。",
            elapsed_ms=float(tool_call["latency_ms"]),
            details={
                "tool": tool_name,
                "result_count": result_count,
                "round": int(tool_call["round"]),
                "version_mode": version_scope.get("mode"),
                "doc_version": version_scope.get("doc_version"),
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
        data = asdict(state)
        data.pop("retrieved_chunks", None)
        data.pop("reranked_chunks", None)
        data["milvus_recalled"] = [
            item.to_dict() for item in state.retrieved_chunks
        ]
        data["reranked"] = [
            item.to_dict() for item in state.reranked_chunks
        ]
        return data
