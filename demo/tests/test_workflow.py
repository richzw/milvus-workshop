from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from typing import Any

from agent_workshop_demo.cli import main
from agent_workshop_demo.generation import (
    GenerationRequest,
    GenerationResult,
)
from agent_workshop_demo.langgraph_workflow import build_default_workflow
from agent_workshop_demo.models import SearchResult
from agent_workshop_demo.retrieval import InMemoryHybridRetriever
from agent_workshop_demo.sample_data import load_kb_chunks
from agent_workshop_demo.workflow import (
    AgenticRAGWorkflow,
    WorkflowStageError,
)


class RecordingAnswerGenerator:
    name = "recording"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text="这是综合后的回答。[C1]",
            generator_name="openai",
            model="configured-model",
            referenced_citation_ids=["C1"],
        )


class InvalidCitationGenerator:
    name = "invalid"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text="错误引用。[C9]",
            generator_name="invalid",
            model=None,
            referenced_citation_ids=["C1"],
        )


class StaticRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[SearchResult]:
        return self.results[:top_k]

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        return {}


class WorkflowTests(unittest.TestCase):
    def test_workflow_injects_selected_context_into_answer_generator(
        self,
    ) -> None:
        generator = RecordingAnswerGenerator()
        response = AgenticRAGWorkflow(answer_generator=generator).run(
            "我们 S3 文档同步流程是怎么设计的？"
        )

        self.assertEqual(len(generator.requests), 1)
        request = generator.requests[0]
        self.assertEqual(request.user_query, response["user_query"])
        self.assertLessEqual(len(request.contexts), 5)
        selected_ids = {
            item["chunk_id"]
            for item in response["reranked"]
            if item["selected"]
        }
        self.assertEqual(
            {context.chunk_id for context in request.contexts},
            selected_ids,
        )
        self.assertEqual(response["answer"], "这是综合后的回答。[C1]")
        self.assertEqual(len(response["citations"]), 1)
        self.assertEqual(
            response["citations"][0]["chunk_id"],
            request.contexts[0].chunk_id,
        )
        self.assertEqual(response["answer_generator_name"], "openai")
        self.assertEqual(response["answer_model"], "configured-model")

    def test_terminal_branches_without_evidence_do_not_call_generator(
        self,
    ) -> None:
        generator = RecordingAnswerGenerator()
        workflow = AgenticRAGWorkflow(answer_generator=generator)

        direct = workflow.run("你好")
        abstained = workflow.run("不存在的采购宇宙飞船编号是什么？")

        self.assertEqual(generator.requests, [])
        self.assertEqual(
            direct["terminal_status"],
            "answered_without_retrieval",
        )
        self.assertEqual(direct["answer_generator_name"], "not_invoked")
        self.assertEqual(abstained["terminal_status"], "abstained")
        self.assertEqual(abstained["answer_generator_name"], "not_invoked")

    def test_workflow_rejects_generator_citation_contract_violation(
        self,
    ) -> None:
        workflow = AgenticRAGWorkflow(
            answer_generator=InvalidCitationGenerator()
        )

        with self.assertRaisesRegex(
            WorkflowStageError,
            "generate_answer_streaming",
        ):
            workflow.run("我们 S3 文档同步流程是怎么设计的？")

    def test_stream_wraps_generator_failure_with_stage_and_query(self) -> None:
        workflow = AgenticRAGWorkflow(
            answer_generator=InvalidCitationGenerator()
        )

        with self.assertRaisesRegex(
            WorkflowStageError,
            "generate_answer_streaming.*query_stream_failure",
        ) as captured:
            list(
                workflow.stream(
                    "我们 S3 文档同步流程是怎么设计的？",
                    query_id="query_stream_failure",
                )
            )
        self.assertEqual(
            captured.exception.stage,
            "generate_answer_streaming",
        )

    def test_unknown_classification_does_not_veto_relevant_evidence(
        self,
    ) -> None:
        response = AgenticRAGWorkflow().run(
            "What does the ingestion pipeline do?"
        )

        self.assertEqual(response["query_type"], "unknown")
        self.assertTrue(response["enough_evidence"])
        self.assertEqual(response["terminal_status"], "answered")
        self.assertTrue(response["citations"])

    def test_generation_context_is_bounded_and_traced(self) -> None:
        source_chunks = load_kb_chunks()[:2]
        long_chunks = [
            replace(
                source_chunks[0],
                text=source_chunks[0].text + "A" * 15_000,
            ),
            replace(
                source_chunks[1],
                text=source_chunks[1].text + "B" * 15_000,
            ),
        ]
        recalled = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.9,
                keyword_score=0.9,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.9,
            )
            for index, chunk in enumerate(long_chunks, start=1)
        ]
        generator = RecordingAnswerGenerator()
        workflow = AgenticRAGWorkflow(
            retriever=StaticRetriever(recalled),
            answer_generator=generator,
        )

        response = workflow.run("Milvus 如何检索？")

        contexts = generator.requests[0].contexts
        self.assertEqual(sum(len(item.text) for item in contexts), 20_000)
        self.assertEqual(response["generation_context_count"], 2)
        self.assertEqual(response["generation_context_truncated_count"], 1)

    def test_s3_question_returns_answer_citations_and_trace(self) -> None:
        response = AgenticRAGWorkflow().run(
            "我们 S3 文档同步流程是怎么设计的？"
        )

        self.assertTrue(response["enough_evidence"])
        self.assertIn("Milvus", response["answer"])
        self.assertGreaterEqual(len(response["citations"]), 2)
        self.assertEqual(
            response["trace"]["classify_query"]["query_type"],
            "architecture",
        )
        self.assertEqual(
            response["trace"]["reranker"]["name"],
            "rule-based-reranker",
        )
        cited = {item["chunk_id"] for item in response["citations"]}
        self.assertIn("doc_s3_sync_design_c003", cited)
        self.assertNotIn("doc_ttl_memory_policy_c001", cited)
        self.assertLessEqual(len(cited), 5)

    def test_unknown_question_stops_at_retry_cap(self) -> None:
        response = AgenticRAGWorkflow().run(
            "这个完全不存在的采购审批宇宙飞船编号是什么？"
        )

        self.assertFalse(response["enough_evidence"])
        self.assertEqual(response["retry_count"], 3)
        self.assertIn("没有找到足够可靠", response["answer"])
        self.assertEqual(response["citations"], [])
        self.assertEqual(response["terminal_status"], "abstained")
        self.assertEqual(response["trace"]["terminal_status"], "abstained")
        grade = response["trace"]["evidence_grading"]
        self.assertEqual(grade["retry_count"], 3)

    def test_general_question_preserves_direct_response(self) -> None:
        response = AgenticRAGWorkflow().run("你好")

        self.assertEqual(
            response["terminal_status"],
            "answered_without_retrieval",
        )
        self.assertIn("不需要检索", response["answer"])
        self.assertEqual(response["milvus_recalled"], [])
        self.assertEqual(response["reranked"], [])
        self.assertEqual(response["citations"], [])

    def test_response_has_identity_and_real_stage_metrics(self) -> None:
        response = AgenticRAGWorkflow().run(
            "RAG 架构里 Milvus 负责哪一层？",
            session_id="session_test",
            query_id="query_test",
        )

        self.assertEqual(response["session_id"], "session_test")
        self.assertEqual(response["query_id"], "query_test")
        self.assertEqual(response["trace"]["session_id"], "session_test")
        self.assertEqual(response["trace"]["query_id"], "query_test")
        stages = response["metrics"]["stage_latency_ms"]
        self.assertIn("milvus_hybrid_retrieve", stages)
        self.assertGreaterEqual(response["metrics"]["latency_ms"], 0)

    def test_citations_are_inline_and_selected(self) -> None:
        response = AgenticRAGWorkflow().run(
            "我们 S3 文档同步流程是怎么设计的？"
        )

        selected = {
            item["chunk_id"]
            for item in response["reranked"]
            if item["selected"]
        }
        cited = {item["chunk_id"] for item in response["citations"]}
        self.assertTrue(cited)
        self.assertTrue(cited.issubset(selected))
        for citation in response["citations"]:
            marker = f"[{citation['citation_id']}]"
            self.assertIn(marker, response["answer"])

    def test_blank_query_and_unknown_filter_are_rejected(self) -> None:
        workflow = AgenticRAGWorkflow()

        with self.assertRaisesRegex(ValueError, "question"):
            workflow.run("   ")
        with self.assertRaisesRegex(ValueError, "Unsupported search filter"):
            workflow.run(
                "Milvus",
                filters={"secret_department": "engineering"},
            )

    def test_dependency_failure_has_stage_and_query_context(self) -> None:
        class FailingRetriever(InMemoryHybridRetriever):
            def search(
                self,
                query: str,
                *,
                top_k: int,
                filters: dict[str, Any] | None = None,
                order_by: list[str] | None = None,
            ) -> list[SearchResult]:
                raise OSError("Milvus unavailable")

        workflow = AgenticRAGWorkflow(FailingRetriever(load_kb_chunks()))

        with self.assertRaisesRegex(
            WorkflowStageError,
            "milvus_hybrid_retrieve.*query_failure",
        ) as captured:
            workflow.run("Milvus 架构", query_id="query_failure")
        self.assertIsInstance(captured.exception.__cause__, OSError)

    def test_default_adapter_streams_then_returns_same_snapshot(self) -> None:
        events = list(
            build_default_workflow().stream(
                "我们 S3 文档同步流程是怎么设计的？",
                session_id="session_graph",
                query_id="query_graph",
            )
        )

        deltas = [event for event in events if event["type"] == "answer_delta"]
        trace_events = [
            event["event"]
            for event in events
            if event["type"] == "trace_event"
        ]
        self.assertGreaterEqual(len(deltas), 2)
        self.assertTrue(trace_events)
        self.assertEqual(
            [event["sequence"] for event in trace_events],
            list(range(1, len(trace_events) + 1)),
        )
        self.assertEqual(events[-1]["type"], "final")
        response = events[-1]["response"]
        self.assertEqual(response["query_id"], "query_graph")
        self.assertEqual(response["session_id"], "session_graph")
        self.assertEqual(response["answer_generator_name"], "deterministic")
        self.assertEqual(
            response["generation_fallback_reason"],
            "not_configured",
        )
        self.assertTrue(
            response["trace"]["answer_generation"]["fallback_active"]
        )

    def test_cli_main_runs_successfully(self) -> None:
        with redirect_stdout(StringIO()):
            exit_code = main(["RAG 架构里 Milvus 负责哪一层？"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
