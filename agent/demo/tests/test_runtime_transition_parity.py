from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent_workshop_demo.embedding import sparse_vector
from agent_workshop_demo.knowledge_tools import PermissionDecision
from agent_workshop_demo.langgraph_workflow import (
    LangGraphAgenticRAGWorkflow,
)
from agent_workshop_demo.models import SearchResult
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.workflow import AgenticRAGWorkflow


class _CompiledGraph:
    def __init__(
        self,
        *,
        nodes: dict[str, Any],
        edges: dict[str, str],
        conditional_edges: dict[str, tuple[Any, dict[str, str]]],
        entry_point: str,
        end: str,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.end = end

    def stream(
        self,
        payload: dict[str, Any],
        *,
        stream_mode: str,
    ) -> Any:
        if stream_mode != "updates":
            raise ValueError("fake graph supports only update streaming")
        current = self.entry_point
        while current != self.end:
            payload = self.nodes[current](payload)
            yield {current: payload}
            conditional = self.conditional_edges.get(current)
            if conditional is None:
                current = self.edges[current]
                continue
            route, mapping = conditional
            current = mapping[route(payload)]

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        for update in self.stream(payload, stream_mode="updates"):
            payload = next(iter(update.values()))
        return payload


class _StateGraph:
    def __init__(self, state_type: type[dict[str, Any]]) -> None:
        del state_type
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[Any, dict[str, str]]] = {}
        self.entry_point = ""
        self.end = "__end__"

    def add_node(self, name: str, node: Any) -> None:
        self.nodes[name] = node

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def add_conditional_edges(
        self,
        source: str,
        route: Any,
        mapping: dict[str, str],
    ) -> None:
        self.conditional_edges[source] = (route, mapping)

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def compile(self) -> _CompiledGraph:
        return _CompiledGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
            end=self.end,
        )


class _DenyPermissionChecker:
    def check(
        self,
        *,
        session_id: str,
        intent: str,
        query_type: str,
    ) -> PermissionDecision:
        del session_id, intent, query_type
        return PermissionDecision(
            allowed=False,
            allowed_departments=(),
            reason="Denied by parity fixture.",
            checker_name="parity-deny",
        )


class _UnrelatedRetriever(InMemoryHybridRetriever):
    def __init__(self, *, growing: bool) -> None:
        super().__init__(load_kb_chunks())
        self.growing = growing
        self.calls = 0

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        order_mode: Any = "relevance",
    ) -> list[SearchResult]:
        del query, top_k, filters, order_by, order_mode
        self.calls += 1
        identity = self.calls if self.growing else 1
        chunk = replace(
            self.chunks[0],
            doc_id=f"doc_unrelated_{identity}",
            chunk_id=f"chunk_unrelated_{identity}",
            title="Unrelated fixture",
            text="alpha beta",
            text_summary=None,
            checksum=f"checksum-{identity}",
            sparse_vector=sparse_vector("alpha beta"),
        )
        return [
            SearchResult(
                chunk=chunk,
                rank=1,
                dense_score=0.05,
                keyword_score=0.0,
                recency_score=0.0,
                priority_score=0.0,
                hybrid_score=0.05,
            )
        ]


def _graph(workflow: AgenticRAGWorkflow) -> LangGraphAgenticRAGWorkflow:
    graph_module = SimpleNamespace(END="__end__", StateGraph=_StateGraph)
    with patch(
        "agent_workshop_demo.langgraph_workflow.import_module",
        return_value=graph_module,
    ):
        return LangGraphAgenticRAGWorkflow(workflow)


def _observable_result(
    runtime: AgenticRAGWorkflow | LangGraphAgenticRAGWorkflow,
    question: str,
    *,
    session_id: str,
    query_id: str,
) -> tuple[list[str], dict[str, Any]]:
    events = list(
        runtime.stream(
            question,
            session_id=session_id,
            query_id=query_id,
        )
    )
    stages = [
        str(event["event"]["stage"])
        for event in events
        if event["type"] == "trace_event"
        and event["event"]["kind"] != "tool_completed"
    ]
    response = events[-1]["response"]
    projection = {
        "terminal_status": response["terminal_status"],
        "retry_count": response["retry_count"],
        "enough_evidence": response["enough_evidence"],
        "citation_chunks": [
            citation["chunk_id"] for citation in response["citations"]
        ],
        "tool_names": [call["tool"] for call in response["tool_calls"]],
        "cache_status": response["response_cache_status"],
    }
    return stages, projection


class RuntimeTransitionParityTests(unittest.TestCase):
    def test_local_and_compiled_graph_paths_match(self) -> None:
        fixtures = [
            (
                "direct",
                "你好",
                lambda: AgenticRAGWorkflow(),
            ),
            (
                "clarification",
                "段位是什么意思？",
                lambda: AgenticRAGWorkflow(),
            ),
            (
                "denial",
                "请查看内部产品路线图",
                lambda: AgenticRAGWorkflow(
                    permission_checker=_DenyPermissionChecker()
                ),
            ),
            (
                "grounded",
                "RAG 架构里 Milvus 负责哪一层？",
                lambda: AgenticRAGWorkflow(),
            ),
            (
                "no_progress",
                "不存在的采购审批宇宙飞船编号是什么？",
                lambda: AgenticRAGWorkflow(
                    retriever=_UnrelatedRetriever(growing=False)
                ),
            ),
            (
                "retry_exhausted",
                "不存在的采购审批宇宙飞船编号是什么？",
                lambda: AgenticRAGWorkflow(
                    retriever=_UnrelatedRetriever(growing=True)
                ),
            ),
        ]

        for name, question, build in fixtures:
            with self.subTest(path=name):
                local = build()
                graph = _graph(build())
                local_result = _observable_result(
                    local,
                    question,
                    session_id=f"session_parity_{name}",
                    query_id=f"query_parity_{name}",
                )
                graph_result = _observable_result(
                    graph,
                    question,
                    session_id=f"session_parity_{name}",
                    query_id=f"query_parity_{name}",
                )
                self.assertEqual(graph_result, local_result)

    def test_local_and_compiled_graph_cache_hit_paths_match(self) -> None:
        local = AgenticRAGWorkflow()
        graph = _graph(AgenticRAGWorkflow())
        question = "Milvus 3.0 有哪些新功能"
        session_id = "session_parity_cache"
        local.run(
            question,
            session_id=session_id,
            query_id="query_parity_cache_local_source",
        )
        graph.run(
            question,
            session_id=session_id,
            query_id="query_parity_cache_graph_source",
        )

        local_result = _observable_result(
            local,
            question,
            session_id=session_id,
            query_id="query_parity_cache_hit",
        )
        graph_result = _observable_result(
            graph,
            question,
            session_id=session_id,
            query_id="query_parity_cache_hit",
        )

        self.assertEqual(graph_result, local_result)
        self.assertEqual(
            local_result[1]["terminal_status"],
            "answered_from_cache",
        )


if __name__ == "__main__":
    unittest.main()
