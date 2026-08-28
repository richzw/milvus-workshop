"""Optional LangGraph adapter for the deterministic workflow nodes."""

from __future__ import annotations

import os
from collections.abc import Iterable
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Protocol, cast

from agent_workshop_demo.classification import build_query_classifier
from agent_workshop_demo.context_compression import (
    ContextCompressor,
    build_context_compressor,
)
from agent_workshop_demo.generation import build_answer_generator
from agent_workshop_demo.knowledge_tools import PermissionChecker
from agent_workshop_demo.query_transform import (
    QueryTransformer,
    build_query_transformer,
)
from agent_workshop_demo.reranker import build_reranker
from agent_workshop_demo.retrieval_tier import (
    RetrievalTierConfig,
    runtime_config_from_mapping as retrieval_tier_config_from_mapping,
)
from agent_workshop_demo.events import WorkflowEventEmitter
from agent_workshop_demo.memory import (
    ConversationMemory,
    MilvusConversationMemoryStore,
)
from agent_workshop_demo.models import EvidenceAction
from agent_workshop_demo.response_cache import (
    DEFAULT_KB_REVISION,
    DEFAULT_RESPONSE_CACHE_SIMILARITY_THRESHOLD,
    DEFAULT_RESPONSE_CACHE_TOP_K,
    DEFAULT_RESPONSE_CACHE_TTL_SECONDS,
    GroundedResponseCache,
    MilvusGroundedResponseCacheStore,
)
from agent_workshop_demo.selective_memory import (
    DecayMode,
    MilvusSelectiveMemoryStore,
    SelectiveMemoryService,
    build_memory_selector,
)
from agent_workshop_demo.retrieval import HybridRetriever
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.struct_array import (
    MilvusStructArrayRetriever,
    StructArrayProfile,
    runtime_config_from_mapping,
)
from agent_workshop_demo.transitions import (
    WorkflowNode,
    next_transition,
)
from agent_workshop_demo.workflow import (
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_MEMORY_TTL_SECONDS,
    AgenticRAGWorkflow,
)


class LangGraphUnavailableError(RuntimeError):
    """Raised only when the optional LangGraph package is unavailable."""


class CompiledGraph(Protocol):
    """Minimal compiled graph surface used by the adapter."""

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def stream(
        self,
        payload: dict[str, Any],
        *,
        stream_mode: str,
    ) -> Iterable[dict[str, Any]]: ...


class LangGraphAgenticRAGWorkflow:
    """Run the same node contracts through LangGraph when installed."""

    def __init__(self, workflow: AgenticRAGWorkflow | None = None) -> None:
        self.workflow = workflow or AgenticRAGWorkflow()
        self.app = self._build_app()

    def run(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the compiled graph and return its terminal snapshot."""

        result = self._invoke(
            user_query,
            filters,
            session_id=session_id,
            query_id=query_id,
        )
        return cast(dict[str, Any], result["response"])

    def list_memories(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Delegate session Memory listing to the underlying workflow."""

        return self.workflow.list_memories(session_id, limit=limit)

    def list_selective_memories(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Delegate selective-Memory listing to the underlying workflow."""

        return self.workflow.list_selective_memories(
            session_id,
            limit=limit,
        )

    def clear_memory(self, session_id: str) -> int:
        """Delete only the requested session's Memory."""

        return self.workflow.clear_memory(session_id)

    def stream(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        query_id: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        """Stream graph-node progress before the validated answer."""

        payload = self._create_payload(
            user_query,
            filters,
            session_id=session_id,
            query_id=query_id,
        )
        state = payload["state"]
        emitter = WorkflowEventEmitter(state.query_id)
        tool_call_count = 0
        answer_emitted = False
        final_emitted = False
        observed_stage_latency: dict[str, float] = {}
        for update in self.app.stream(payload, stream_mode="updates"):
            if not isinstance(update, dict) or len(update) != 1:
                raise RuntimeError("LangGraph returned an invalid update")
            node_name, node_payload = next(iter(update.items()))
            if not isinstance(node_payload, dict):
                raise RuntimeError("LangGraph node update must be a mapping")
            state = node_payload["state"]
            if node_name == "finalize":
                if not answer_emitted:
                    for chunk in node_payload["answer_chunks"]:
                        yield {"type": "answer_delta", "text": chunk}
                response = node_payload["response"]
                if not isinstance(response, dict):
                    raise RuntimeError(
                        "LangGraph stream ended without a terminal response"
                    )
                yield {"type": "final", "response": response}
                final_emitted = True
                continue

            if node_name == "answer_ready":
                for chunk in node_payload["answer_chunks"]:
                    yield {"type": "answer_delta", "text": chunk}
                answer_emitted = True
                continue
            if node_name in {"output_gate", "persist_turn_memory"}:
                continue

            stage = node_name
            accumulated = state.stage_latency_ms.get(stage, 0.0)
            elapsed = round(
                accumulated - observed_stage_latency.get(stage, 0.0),
                3,
            )
            observed_stage_latency[stage] = accumulated
            yield self.workflow._stage_event(
                emitter,
                state,
                stage,
                elapsed,
                kind=(
                    "retry_scheduled"
                    if (
                        node_name == "evaluate_evidence"
                        and node_payload.get("evidence_action") == "retry"
                    )
                    else "stage_completed"
                ),
            )
            if node_name == "execute_tool_plan":
                for tool_call in state.tool_calls[tool_call_count:]:
                    yield self.workflow._tool_event(emitter, tool_call)
                tool_call_count = len(state.tool_calls)
        if not final_emitted:
            raise RuntimeError("LangGraph stream ended without a terminal response")

    def _invoke(
        self,
        user_query: str,
        filters: dict[str, Any] | None,
        *,
        session_id: str | None,
        query_id: str | None,
    ) -> dict[str, Any]:
        return self.app.invoke(
            self._create_payload(
                user_query,
                filters,
                session_id=session_id,
                query_id=query_id,
            )
        )

    def _create_payload(
        self,
        user_query: str,
        filters: dict[str, Any] | None,
        *,
        session_id: str | None,
        query_id: str | None,
    ) -> dict[str, Any]:
        return {
            "started": self.workflow._clock(),
            "state": self.workflow.create_state(
                user_query,
                filters,
                session_id=session_id,
                query_id=query_id,
            ),
            "response": None,
            "answer_chunks": [],
        }

    def _build_app(self) -> CompiledGraph:
        try:
            graph_module = import_module("langgraph.graph")
        except ImportError as exc:
            raise LangGraphUnavailableError(
                "Install LangGraph with `pip install -r demo/requirements.txt`."
            ) from exc

        end = graph_module.END
        state_graph = graph_module.StateGraph

        graph = state_graph(dict)

        def recall_memory(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "recall_memory",
                lambda: self.workflow.recall_memory(state),
            )
            return payload

        def classify_and_route(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            result, _elapsed = self.workflow._measure_stage_result_delta(
                state,
                "classify_and_route",
                lambda: self.workflow.classify_and_route(state),
            )
            payload["query_route"] = result.route.value
            if result.route.value == "direct":
                payload["answer_chunks"] = [state.answer]
            return payload

        def resolve_terminology(
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "resolve_terminology",
                lambda: self.workflow.resolve_terminology(state),
            )
            if state.terminal_status == "clarification_required":
                payload["answer_chunks"] = [state.answer]
            return payload

        def check_permission(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "check_permission",
                lambda: self.workflow.check_permission(state),
            )
            if not state.permission_decision.get("allowed", False):
                payload["answer_chunks"] = [state.answer]
            return payload

        def try_grounded_cache(
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "try_grounded_cache",
                lambda: self.workflow.try_grounded_cache(state),
            )
            if state.terminal_status == "answered_from_cache":
                payload["answer_chunks"] = [state.answer]
            return payload

        def recall_authorized_experience(
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "recall_authorized_experience",
                lambda: self.workflow.recall_authorized_experience(state),
            )
            return payload

        def plan_retrieval(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            result, _elapsed = self.workflow._measure_stage_result_delta(
                state,
                "plan_retrieval",
                lambda: self.workflow.plan_retrieval(state),
            )
            payload["retrieval_plan_count"] = result.plan_count
            return payload

        def retrieve(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "execute_tool_plan",
                lambda: self.workflow.milvus_hybrid_retrieve(state),
            )
            return payload

        def rerank(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "rerank_evidence",
                lambda: self.workflow.rerank_evidence(state),
            )
            return payload

        def evaluate_evidence(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            result, _elapsed = self.workflow._measure_stage_result_delta(
                state,
                "evaluate_evidence",
                lambda: self.workflow.evaluate_evidence(state),
            )
            payload["evidence_action"] = result.action.value
            return payload

        def answer(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            chunks: list[str] = []
            self.workflow._measure_stage(
                state,
                "generate_answer_streaming",
                lambda: chunks.extend(self.workflow.generate_answer_streaming(state)),
            )
            state.answer = "".join(chunks)
            payload["answer_chunks"] = chunks
            return payload

        def prepare_generation_context(
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "prepare_generation_context",
                lambda: self.workflow.prepare_generation_context(state),
            )
            return payload

        def verify(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "verify_answer",
                lambda: self.workflow.verify_answer(state),
            )
            return payload

        def persist_memory(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "persist_turn_memory",
                lambda: self.workflow.persist_turn_memory(state),
            )
            return payload

        def answer_ready(payload: dict[str, Any]) -> dict[str, Any]:
            """Expose validated chunks before the final-triggered write."""

            return payload

        def finalize(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._finalize(state, payload["started"])
            payload["response"] = self.workflow._serialize(state)
            return payload

        def after_classify_and_route(payload: dict[str, Any]) -> str:
            return next_transition(
                WorkflowNode.CLASSIFY_AND_ROUTE,
                payload["state"],
            ).next_node.value

        def after_terminology_resolution(
            payload: dict[str, Any],
        ) -> str:
            return next_transition(
                WorkflowNode.RESOLVE_TERMINOLOGY,
                payload["state"],
            ).next_node.value

        def after_permission(payload: dict[str, Any]) -> str:
            return next_transition(
                WorkflowNode.CHECK_PERMISSION,
                payload["state"],
            ).next_node.value

        def after_cache_validation(payload: dict[str, Any]) -> str:
            return next_transition(
                WorkflowNode.TRY_GROUNDED_CACHE,
                payload["state"],
            ).next_node.value

        def after_evaluate_evidence(payload: dict[str, Any]) -> str:
            action = payload.get("evidence_action")
            if not isinstance(action, str):
                raise ValueError("evidence_action is missing")

            return next_transition(
                WorkflowNode.EVALUATE_EVIDENCE,
                payload["state"],
                evidence_action=EvidenceAction(action),
            ).next_node.value

        def after_retrieve(payload: dict[str, Any]) -> str:
            return next_transition(
                WorkflowNode.EXECUTE_TOOL_PLAN,
                payload["state"],
            ).next_node.value

        graph.add_node("recall_memory", recall_memory)
        graph.add_node("classify_and_route", classify_and_route)
        graph.add_node("resolve_terminology", resolve_terminology)
        graph.add_node("check_permission", check_permission)
        graph.add_node(
            "try_grounded_cache",
            try_grounded_cache,
        )
        graph.add_node(
            "recall_authorized_experience",
            recall_authorized_experience,
        )
        graph.add_node("plan_retrieval", plan_retrieval)
        graph.add_node("execute_tool_plan", retrieve)
        graph.add_node("rerank_evidence", rerank)
        graph.add_node("evaluate_evidence", evaluate_evidence)
        graph.add_node(
            "prepare_generation_context",
            prepare_generation_context,
        )
        graph.add_node("generate_answer_streaming", answer)
        graph.add_node("verify_answer", verify)
        graph.add_node("answer_ready", answer_ready)
        graph.add_node("persist_turn_memory", persist_memory)
        graph.add_node("finalize", finalize)
        graph.set_entry_point("recall_memory")
        graph.add_edge("recall_memory", "classify_and_route")
        graph.add_conditional_edges(
            "classify_and_route",
            after_classify_and_route,
            {
                "resolve_terminology": "resolve_terminology",
                "output_gate": "answer_ready",
            },
        )
        graph.add_conditional_edges(
            "resolve_terminology",
            after_terminology_resolution,
            {
                "check_permission": "check_permission",
                "output_gate": "answer_ready",
            },
        )
        graph.add_conditional_edges(
            "check_permission",
            after_permission,
            {
                "try_grounded_cache": "try_grounded_cache",
                "output_gate": "answer_ready",
            },
        )
        graph.add_conditional_edges(
            "try_grounded_cache",
            after_cache_validation,
            {
                "recall_authorized_experience": ("recall_authorized_experience"),
                "output_gate": "answer_ready",
            },
        )
        graph.add_edge("recall_authorized_experience", "plan_retrieval")
        graph.add_edge("plan_retrieval", "execute_tool_plan")
        graph.add_conditional_edges(
            "execute_tool_plan",
            after_retrieve,
            {
                "rerank_evidence": "rerank_evidence",
                "generate_candidate_answer": "generate_answer_streaming",
            },
        )
        graph.add_edge("rerank_evidence", "evaluate_evidence")
        graph.add_conditional_edges(
            "evaluate_evidence",
            after_evaluate_evidence,
            {
                "execute_tool_plan": "execute_tool_plan",
                "prepare_generation_context": "prepare_generation_context",
                "generate_candidate_answer": "generate_answer_streaming",
            },
        )
        graph.add_edge(
            "prepare_generation_context",
            "generate_answer_streaming",
        )
        graph.add_edge("generate_answer_streaming", "verify_answer")
        graph.add_edge("verify_answer", "answer_ready")
        graph.add_edge("answer_ready", "persist_turn_memory")
        graph.add_edge("persist_turn_memory", "finalize")
        graph.add_edge("finalize", end)
        return cast(CompiledGraph, graph.compile())


def _configured_retrieval_tier(
    environ: Mapping[str, str] | None,
) -> RetrievalTierConfig:
    """Resolve the spec 15 retrieval tier from explicit configuration."""

    return retrieval_tier_config_from_mapping(
        os.environ if environ is None else environ
    )


def build_default_workflow(
    *,
    environ: Mapping[str, str] | None = None,
    retriever: HybridRetriever | None = None,
    query_transformer: QueryTransformer | None = None,
    context_compressor: ContextCompressor | None = None,
    permission_checker: PermissionChecker | None = None,
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
) -> AgenticRAGWorkflow | LangGraphAgenticRAGWorkflow:
    """Build configured generation and prefer LangGraph orchestration."""

    tier = _configured_retrieval_tier(environ)
    configured_selective_memory = (
        selective_memory
        if selective_memory is not None
        else SelectiveMemoryService(
            selector=(build_memory_selector() if selective_memory_enabled else None)
        )
    )
    workflow = AgenticRAGWorkflow(
        retriever=retriever,
        reranker=build_reranker(environ),
        query_classifier=build_query_classifier(environ),
        query_transformer=(
            None
            if not tier.uses_query_transformation
            else (
                query_transformer
                if query_transformer is not None
                else build_query_transformer(environ)
            )
        ),
        answer_generator=build_answer_generator(environ),
        context_compressor=(
            context_compressor
            if context_compressor is not None
            else build_context_compressor(environ)
        ),
        permission_checker=permission_checker,
        memory_store=memory_store,
        memory_top_k=memory_top_k,
        memory_ttl_seconds=memory_ttl_seconds,
        selective_memory=configured_selective_memory,
        selective_memory_enabled=selective_memory_enabled,
        response_cache=response_cache,
        response_cache_enabled=response_cache_enabled,
        response_cache_top_k=response_cache_top_k,
        response_cache_ttl_seconds=response_cache_ttl_seconds,
        response_cache_similarity_threshold=(response_cache_similarity_threshold),
        kb_revision=kb_revision,
        retrieval_tier=tier.tier,
    )
    try:
        return LangGraphAgenticRAGWorkflow(workflow)
    except LangGraphUnavailableError:
        return workflow


def build_milvus_workflow(
    environ: Mapping[str, str] | None = None,
) -> AgenticRAGWorkflow | LangGraphAgenticRAGWorkflow:
    """Build the Streamlit workflow backed by a loaded Milvus collection."""

    values = os.environ if environ is None else environ
    struct_array_config = runtime_config_from_mapping(values)
    uri = values.get("MILVUS_URI", "http://localhost:19530").strip()
    if not uri:
        raise ValueError("MILVUS_URI must be non-empty")
    token = values.get("MILVUS_TOKEN", "").strip() or None
    collection_name = values.get(
        "MILVUS_COLLECTION_NAME",
        "kb_chunks",
    ).strip()
    if not collection_name:
        raise ValueError("MILVUS_COLLECTION_NAME must be non-empty")
    sparse_field = values.get(
        "MILVUS_SPARSE_FIELD",
        "sparse_vector",
    ).strip()
    if not sparse_field:
        raise ValueError("MILVUS_SPARSE_FIELD must be non-empty")
    memory_collection_name = values.get(
        "MILVUS_MEMORY_COLLECTION_NAME",
        "conversation_memory",
    ).strip()
    if not memory_collection_name:
        raise ValueError("MILVUS_MEMORY_COLLECTION_NAME must be non-empty")
    memory_top_k = _positive_int(
        values.get("MEMORY_TOP_K", str(DEFAULT_MEMORY_TOP_K)),
        name="MEMORY_TOP_K",
        maximum=20,
    )
    memory_ttl_seconds = _positive_int(
        values.get(
            "MEMORY_TTL_SECONDS",
            str(DEFAULT_MEMORY_TTL_SECONDS),
        ),
        name="MEMORY_TTL_SECONDS",
    )
    selective_memory_enabled = _boolean(
        values.get("SELECTIVE_MEMORY_ENABLED", "true"),
        name="SELECTIVE_MEMORY_ENABLED",
    )
    memory_events_collection_name = values.get(
        "MILVUS_MEMORY_EVENTS_COLLECTION_NAME",
        "memory_events",
    ).strip()
    if not memory_events_collection_name:
        raise ValueError("MILVUS_MEMORY_EVENTS_COLLECTION_NAME must be non-empty")
    memory_facts_collection_name = values.get(
        "MILVUS_MEMORY_FACTS_COLLECTION_NAME",
        "memory_facts",
    ).strip()
    if not memory_facts_collection_name:
        raise ValueError("MILVUS_MEMORY_FACTS_COLLECTION_NAME must be non-empty")
    memory_journal_collection_name = values.get(
        "MILVUS_MEMORY_CONSOLIDATION_JOURNAL_COLLECTION_NAME",
        "memory_consolidation_journal",
    ).strip()
    if not memory_journal_collection_name:
        raise ValueError(
            "MILVUS_MEMORY_CONSOLIDATION_JOURNAL_COLLECTION_NAME must be non-empty"
        )
    memory_lane_top_k = _positive_int(
        values.get("MEMORY_LANE_TOP_K", "3"),
        name="MEMORY_LANE_TOP_K",
        maximum=20,
    )
    memory_pack_max_records = _positive_int(
        values.get("MEMORY_PACK_MAX_RECORDS", "12"),
        name="MEMORY_PACK_MAX_RECORDS",
        maximum=20,
    )
    memory_context_max_chars = _bounded_int(
        values.get("MEMORY_CONTEXT_MAX_CHARS", "4000"),
        name="MEMORY_CONTEXT_MAX_CHARS",
        minimum=512,
        maximum=8192,
    )
    memory_consolidation_batch_size = _bounded_int(
        values.get("MEMORY_CONSOLIDATION_BATCH_SIZE", "20"),
        name="MEMORY_CONSOLIDATION_BATCH_SIZE",
        minimum=2,
        maximum=100,
    )
    memory_recurrence_threshold = _bounded_int(
        values.get("MEMORY_RECURRENCE_THRESHOLD", "2"),
        name="MEMORY_RECURRENCE_THRESHOLD",
        minimum=2,
        maximum=10,
    )
    memory_decay_mode = values.get(
        "MEMORY_DECAY_MODE",
        "application",
    ).strip()
    if memory_decay_mode not in {"application", "milvus"}:
        raise ValueError("MEMORY_DECAY_MODE must be application or milvus")
    memory_selector = (
        build_memory_selector(values) if selective_memory_enabled else None
    )
    response_cache_enabled = _boolean(
        values.get("RESPONSE_CACHE_ENABLED", "true"),
        name="RESPONSE_CACHE_ENABLED",
    )
    response_cache_collection_name = values.get(
        "MILVUS_RESPONSE_CACHE_COLLECTION_NAME",
        "grounded_response_cache",
    ).strip()
    if not response_cache_collection_name:
        raise ValueError("MILVUS_RESPONSE_CACHE_COLLECTION_NAME must be non-empty")
    response_cache_top_k = _positive_int(
        values.get(
            "RESPONSE_CACHE_TOP_K",
            str(DEFAULT_RESPONSE_CACHE_TOP_K),
        ),
        name="RESPONSE_CACHE_TOP_K",
        maximum=20,
    )
    response_cache_ttl_seconds = _positive_int(
        values.get(
            "RESPONSE_CACHE_TTL_SECONDS",
            str(DEFAULT_RESPONSE_CACHE_TTL_SECONDS),
        ),
        name="RESPONSE_CACHE_TTL_SECONDS",
    )
    response_cache_similarity_threshold = _unit_float(
        values.get(
            "RESPONSE_CACHE_SIMILARITY_THRESHOLD",
            str(DEFAULT_RESPONSE_CACHE_SIMILARITY_THRESHOLD),
        ),
        name="RESPONSE_CACHE_SIMILARITY_THRESHOLD",
    )
    kb_revision = values.get(
        "KB_REVISION",
        DEFAULT_KB_REVISION,
    ).strip()
    if not kb_revision or len(kb_revision) > 128:
        raise ValueError("KB_REVISION must contain 1..128 characters")

    try:
        flat_retriever = MilvusHybridRetriever.connect(
            uri,
            token,
            collection_name=collection_name,
            sparse_field=sparse_field,
        )
        flat_retriever.ensure_collection_ready()
        flat_retriever.ensure_embedding_space_ready()
        if struct_array_config.profile is StructArrayProfile.DISABLED:
            retriever: HybridRetriever = flat_retriever
        else:
            struct_retriever = MilvusStructArrayRetriever(
                flat_retriever.client,
                flat_retriever,
                struct_array_config,
            )
            struct_retriever.ensure_ready()
            retriever = struct_retriever
        memory_store = MilvusConversationMemoryStore(
            flat_retriever.client,
            collection_name=memory_collection_name,
        )
        memory_store.ensure_collection_ready()
        selective_memory_store = MilvusSelectiveMemoryStore(
            flat_retriever.client,
            events_collection_name=memory_events_collection_name,
            facts_collection_name=memory_facts_collection_name,
            journal_collection_name=memory_journal_collection_name,
            decay_mode=cast(DecayMode, memory_decay_mode),
        )
        if selective_memory_enabled:
            selective_memory_store.ensure_collections_ready()
        selective_memory = SelectiveMemoryService(
            selective_memory_store,
            selector=memory_selector,
            lane_top_k=memory_lane_top_k,
            pack_max_records=memory_pack_max_records,
            context_max_chars=memory_context_max_chars,
            consolidation_batch_size=memory_consolidation_batch_size,
            recurrence_threshold=memory_recurrence_threshold,
        )
        response_cache = MilvusGroundedResponseCacheStore(
            flat_retriever.client,
            collection_name=response_cache_collection_name,
        )
        if response_cache_enabled:
            response_cache.ensure_collection_ready()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize the Milvus retriever for collection "
            f"{collection_name!r} from MILVUS_URI"
        ) from exc
    return build_default_workflow(
        environ=values,
        retriever=retriever,
        memory_store=memory_store,
        memory_top_k=memory_top_k,
        memory_ttl_seconds=memory_ttl_seconds,
        selective_memory=selective_memory,
        selective_memory_enabled=selective_memory_enabled,
        response_cache=response_cache,
        response_cache_enabled=response_cache_enabled,
        response_cache_top_k=response_cache_top_k,
        response_cache_ttl_seconds=response_cache_ttl_seconds,
        response_cache_similarity_threshold=(response_cache_similarity_threshold),
        kb_revision=kb_revision,
    )


def _positive_int(
    raw_value: str,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be positive{suffix}")
    return value


def _boolean(raw_value: str, *, name: str) -> bool:
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _bounded_int(
    raw_value: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _unit_float(raw_value: str, *, name: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number in [0, 1]") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number in [0, 1]")
    return value
