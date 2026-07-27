"""Optional LangGraph adapter for the deterministic workflow nodes."""

from __future__ import annotations

import os
from collections.abc import Iterable
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Protocol, cast

from agent_workshop_demo.generation import build_answer_generator
from agent_workshop_demo.events import WorkflowEventEmitter
from agent_workshop_demo.memory import (
    ConversationMemory,
    MilvusConversationMemoryStore,
)
from agent_workshop_demo.retrieval import HybridRetriever
from agent_workshop_demo.schema.pymilvus_adapter import MilvusHybridRetriever
from agent_workshop_demo.workflow import (
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_MEMORY_TTL_SECONDS,
    PERMISSION_DENIED_RESPONSE,
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
            if node_name == "persist_turn_memory":
                continue

            stage = (
                "prepare_supplementary_retrieval"
                if node_name == "prepare_retry"
                else node_name
            )
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
                    if node_name == "prepare_retry"
                    else "stage_completed"
                ),
            )
            if node_name == "milvus_hybrid_retrieve":
                for tool_call in state.tool_calls[tool_call_count:]:
                    yield self.workflow._tool_event(emitter, tool_call)
                tool_call_count = len(state.tool_calls)
        if not final_emitted:
            raise RuntimeError(
                "LangGraph stream ended without a terminal response"
            )

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

        def classify(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "classify_query",
                lambda: self.workflow.classify_query(state),
            )
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

        def decide_retrieval(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "decide_retrieval",
                lambda: self.workflow.decide_retrieval(state),
            )
            if not state.need_retrieval:
                self.workflow.prepare_non_retrieval_answer(state)
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
                state.answer = PERMISSION_DENIED_RESPONSE
                state.terminal_status = "permission_denied"
                state.answer_validation = {
                    "valid": True,
                    "mode": "permission_denied",
                    "reason": "Retrieval and generation were not executed.",
                }
                payload["answer_chunks"] = [state.answer]
            return payload

        def select_tools(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "select_tools",
                lambda: self.workflow.select_tools(state),
            )
            return payload

        def rewrite(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "rewrite_query",
                lambda: self.workflow.rewrite_query(state),
            )
            return payload

        def retrieve(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "milvus_hybrid_retrieve",
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

        def grade(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            self.workflow._measure_stage(
                state,
                "grade_evidence",
                lambda: self.workflow.grade_evidence(state),
            )
            return payload

        def prepare_retry(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            state.retry_count += 1
            self.workflow._measure_stage(
                state,
                "prepare_supplementary_retrieval",
                lambda: self.workflow.prepare_supplementary_retrieval(state),
            )
            return payload

        def answer(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload["state"]
            chunks: list[str] = []
            self.workflow._measure_stage(
                state,
                "generate_answer_streaming",
                lambda: chunks.extend(
                    self.workflow.generate_answer_streaming(state)
                ),
            )
            state.answer = "".join(chunks)
            payload["answer_chunks"] = chunks
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

        def after_retrieval_decision(payload: dict[str, Any]) -> str:
            return (
                "finalize"
                if not payload["state"].need_retrieval
                else "check_permission"
            )

        def after_terminology_resolution(
            payload: dict[str, Any],
        ) -> str:
            return (
                "finalize"
                if payload["state"].terminal_status
                == "clarification_required"
                else "decide_retrieval"
            )

        def after_permission(payload: dict[str, Any]) -> str:
            return (
                "select_tools"
                if payload["state"].permission_decision.get("allowed", False)
                else "finalize"
            )

        def after_grade(payload: dict[str, Any]) -> str:
            state = payload["state"]
            if state.enough_evidence or state.retry_count >= state.max_retry:
                state.terminal_status = (
                    "answered" if state.enough_evidence else "abstained"
                )
                return "answer"
            return "prepare_retry"

        graph.add_node("recall_memory", recall_memory)
        graph.add_node("classify_query", classify)
        graph.add_node("resolve_terminology", resolve_terminology)
        graph.add_node("decide_retrieval", decide_retrieval)
        graph.add_node("check_permission", check_permission)
        graph.add_node("select_tools", select_tools)
        graph.add_node("rewrite_query", rewrite)
        graph.add_node("milvus_hybrid_retrieve", retrieve)
        graph.add_node("rerank_evidence", rerank)
        graph.add_node("grade_evidence", grade)
        graph.add_node("prepare_retry", prepare_retry)
        graph.add_node("generate_answer_streaming", answer)
        graph.add_node("verify_answer", verify)
        graph.add_node("answer_ready", answer_ready)
        graph.add_node("persist_turn_memory", persist_memory)
        graph.add_node("finalize", finalize)
        graph.set_entry_point("recall_memory")
        graph.add_edge("recall_memory", "classify_query")
        graph.add_edge("classify_query", "resolve_terminology")
        graph.add_conditional_edges(
            "resolve_terminology",
            after_terminology_resolution,
            {
                "decide_retrieval": "decide_retrieval",
                "finalize": "answer_ready",
            },
        )
        graph.add_conditional_edges(
            "decide_retrieval",
            after_retrieval_decision,
            {
                "check_permission": "check_permission",
                "finalize": "answer_ready",
            },
        )
        graph.add_conditional_edges(
            "check_permission",
            after_permission,
            {
                "select_tools": "select_tools",
                "finalize": "answer_ready",
            },
        )
        graph.add_edge("select_tools", "rewrite_query")
        graph.add_edge("rewrite_query", "milvus_hybrid_retrieve")
        graph.add_edge("milvus_hybrid_retrieve", "rerank_evidence")
        graph.add_edge("rerank_evidence", "grade_evidence")
        graph.add_conditional_edges(
            "grade_evidence",
            after_grade,
            {
                "prepare_retry": "prepare_retry",
                "answer": "generate_answer_streaming",
            },
        )
        graph.add_edge("prepare_retry", "milvus_hybrid_retrieve")
        graph.add_edge("generate_answer_streaming", "verify_answer")
        graph.add_edge("verify_answer", "answer_ready")
        graph.add_edge("answer_ready", "persist_turn_memory")
        graph.add_edge("persist_turn_memory", "finalize")
        graph.add_edge("finalize", end)
        return cast(CompiledGraph, graph.compile())


def build_default_workflow(
    *,
    retriever: HybridRetriever | None = None,
    memory_store: ConversationMemory | None = None,
    memory_top_k: int = DEFAULT_MEMORY_TOP_K,
    memory_ttl_seconds: int = DEFAULT_MEMORY_TTL_SECONDS,
) -> AgenticRAGWorkflow | LangGraphAgenticRAGWorkflow:
    """Build configured generation and prefer LangGraph orchestration."""

    workflow = AgenticRAGWorkflow(
        retriever=retriever,
        answer_generator=build_answer_generator(),
        memory_store=memory_store,
        memory_top_k=memory_top_k,
        memory_ttl_seconds=memory_ttl_seconds,
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

    try:
        retriever = MilvusHybridRetriever.connect(
            uri,
            token,
            collection_name=collection_name,
        )
        retriever.ensure_collection_ready()
        memory_store = MilvusConversationMemoryStore(
            retriever.client,
            collection_name=memory_collection_name,
        )
        memory_store.ensure_collection_ready()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize the Milvus retriever for collection "
            f"{collection_name!r} from MILVUS_URI"
        ) from exc
    return build_default_workflow(
        retriever=retriever,
        memory_store=memory_store,
        memory_top_k=memory_top_k,
        memory_ttl_seconds=memory_ttl_seconds,
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
