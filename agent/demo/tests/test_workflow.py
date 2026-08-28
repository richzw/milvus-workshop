from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from threading import Barrier
from typing import Any

from agent_workshop_demo.cli import main
from agent_workshop_demo.embedding import tokenize
from agent_workshop_demo.generation import (
    GenerationRequest,
    GenerationResult,
)
from agent_workshop_demo.langgraph_workflow import build_default_workflow
from agent_workshop_demo.ingestion import ingest_demo_sources
from agent_workshop_demo.memory import ConversationMemoryStore, MemoryRecord
from agent_workshop_demo.models import (
    AgentState,
    KBChunk,
    QueryRoute,
    QueryRouteResult,
    RerankedResult,
    RetrievalPlanResult,
    SearchResult,
)
from agent_workshop_demo.reranker import Reranker, RerankRun, RuleBasedReranker
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
        order_mode: Any = "relevance",
    ) -> list[SearchResult]:
        return self.results[:top_k]

    def aggregations(
        self,
        results: list[SearchResult],
        fields: list[str],
    ) -> dict[str, dict[str, int]]:
        return {}

    def fetch_document_chunks(
        self,
        *,
        doc_id: str,
        doc_version: str,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> list[KBChunk]:
        return [
            result.chunk
            for result in self.results
            if result.chunk.doc_id == doc_id
            and result.chunk.doc_version == doc_version
        ][:limit]

    def fetch_chunks_by_ids(
        self,
        *,
        chunk_ids: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[KBChunk]:
        requested = set(chunk_ids)
        return [
            result.chunk
            for result in self.results
            if result.chunk.chunk_id in requested
        ]


class RecordingReranker(RuleBasedReranker):
    def __init__(self) -> None:
        self.input_counts: list[int] = []

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> RerankRun:
        self.input_counts.append(len(chunks))
        return super().rerank(query, chunks, top_k)


class SectionScoreReranker(Reranker):
    name = "section-score"
    # An injected scorer declares its own scale rather than inheriting one.
    strong_single_evidence_threshold = 0.80

    def __init__(
        self,
        *,
        section_scores: dict[str, float] | None = None,
        default_score: float = 0.05,
    ) -> None:
        self.section_scores = section_scores or {}
        self.default_score = default_score

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int,
    ) -> RerankRun:
        del query
        scored = [
            (
                self.section_scores.get(
                    item.chunk.section or "",
                    self.default_score,
                ),
                item,
            )
            for item in chunks
        ]
        ordered = sorted(scored, key=lambda pair: (-pair[0], pair[1].rank))
        return RerankRun(
            results=tuple(
                RerankedResult(
                    search_result=item,
                    rerank=index,
                    old_rank=item.rank,
                    rerank_score=score,
                )
                for index, (score, item) in enumerate(
                    ordered[:top_k],
                    start=1,
                )
            ),
            reranker_name=self.name,
        )


class GrowingUnrelatedRetriever(InMemoryHybridRetriever):
    def __init__(self, chunks: list[KBChunk]) -> None:
        super().__init__(chunks)
        self.search_count = 0

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
        self.search_count += 1
        source = next(
            chunk
            for chunk in self.chunks
            if chunk.doc_id == "doc_milvus_release_notes"
            and chunk.doc_version == "v3.0"
        )
        chunk = replace(
            source,
            doc_id=f"doc_unrelated_{self.search_count}",
            chunk_id=f"chunk_unrelated_{self.search_count}",
            section=f"Unrelated {self.search_count}",
            text="alpha beta",
            text_summary=None,
            checksum=f"checksum-{self.search_count}",
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


class WorkflowTests(unittest.TestCase):
    @staticmethod
    def _release_chunks() -> list[KBChunk]:
        return ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
        ).kb_chunks

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

    def test_reranker_processes_the_complete_recall_pool(self) -> None:
        chunks = [
            chunk for chunk in load_kb_chunks() if chunk.is_current
        ][:12]
        recalled = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.9,
                keyword_score=0.9,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.9 - index / 1000,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        reranker = RecordingReranker()
        recalled_count = len(recalled)

        response = AgenticRAGWorkflow(
            retriever=StaticRetriever(recalled),
            reranker=reranker,
        ).run("Milvus 如何检索？")

        self.assertGreater(recalled_count, 8)
        self.assertEqual(reranker.input_counts, [recalled_count])
        self.assertEqual(
            response["trace"]["reranker"]["processed_candidates"],
            recalled_count,
        )
        self.assertEqual(len(response["reranked"]), 8)

    def test_exhaustive_release_query_expands_and_keeps_all_siblings(
        self,
    ) -> None:
        ingestion = ingest_demo_sources(
            Path("demo/sample_data/local_docs"),
            Path("demo/sample_data/mock_s3"),
        )
        release_ids = {
            chunk.chunk_id
            for chunk in ingestion.kb_chunks
            if chunk.doc_id == "doc_milvus_release_notes"
            and chunk.doc_version == "v3.0"
        }

        response = AgenticRAGWorkflow(
            retriever=InMemoryHybridRetriever(ingestion.kb_chunks)
        ).run("Milvus 3.0 有哪些新功能？")

        recalled = {
            item["chunk_id"] for item in response["milvus_recalled"]
        }
        reranked = {item["chunk_id"] for item in response["reranked"]}
        selected = {
            item["chunk_id"]
            for item in response["reranked"]
            if item["selected"]
        }
        self.assertEqual(len(release_ids), 12)
        self.assertTrue(release_ids.issubset(recalled))
        self.assertTrue(release_ids.issubset(reranked))
        self.assertTrue(release_ids.issubset(selected))
        self.assertEqual(response["retrieval_goal"], "exhaustive")
        self.assertEqual(response["retry_count"], 0)
        self.assertEqual(
            response["document_expansions"][0]["result_count"],
            12,
        )

    def test_focused_force_merge_accepts_one_strong_direct_chunk(
        self,
    ) -> None:
        chunks = self._release_chunks()
        response = AgenticRAGWorkflow(
            retriever=InMemoryHybridRetriever(chunks),
            reranker=SectionScoreReranker(
                section_scores={"Force Merge": 0.95}
            ),
        ).run("介绍下 Milvus 3.0 Force Merge 功能是什么？")

        self.assertEqual(response["terminal_status"], "answered")
        self.assertEqual(response["retry_count"], 0)
        self.assertEqual(response["version_scope"]["mode"], "exact")
        self.assertEqual(response["version_scope"]["doc_versions"], ["v3.0"])
        self.assertEqual(response["evidence_grade"]["relevant_chunks"], 1)
        self.assertEqual(
            response["evidence_grade"]["evidence_basis"],
            "single_strong_chunk",
        )
        self.assertEqual(len(response["citations"]), 1)
        self.assertEqual(response["citations"][0]["section"], "Force Merge")
        self.assertIn(
            "doc_milvus_release_notes_v3_0",
            response["citations"][0]["chunk_id"],
        )

    def test_force_merge_followup_uses_live_evidence_without_memory(
        self,
    ) -> None:
        chunks = self._release_chunks()

        def build() -> AgenticRAGWorkflow:
            return AgenticRAGWorkflow(
                retriever=InMemoryHybridRetriever(chunks),
                reranker=SectionScoreReranker(
                    section_scores={"Force Merge": 0.95}
                ),
            )

        standalone = build().run(
            "介绍下 Milvus 3.0 Force Merge 功能是什么？",
            session_id="session_force_standalone",
        )
        followup_workflow = build()
        followup_workflow.run(
            "Milvus 3.0 有哪些新功能？",
            session_id="session_force_followup",
        )
        followup = followup_workflow.run(
            "介绍下 Milvus 3.0 Force Merge 功能是什么？",
            session_id="session_force_followup",
        )

        self.assertEqual(followup["terminal_status"], "answered")
        self.assertEqual(followup["memory_recall_decision"], "skipped")
        self.assertTrue(followup["tool_calls"])
        self.assertEqual(
            [item["chunk_id"] for item in followup["citations"]],
            [item["chunk_id"] for item in standalone["citations"]],
        )
        self.assertEqual(followup["answer"], standalone["answer"])

    def test_weak_single_chunk_and_single_chunk_exhaustive_still_abstain(
        self,
    ) -> None:
        force_merge = next(
            chunk
            for chunk in self._release_chunks()
            if chunk.section == "Force Merge"
        )
        result = SearchResult(
            chunk=force_merge,
            rank=1,
            dense_score=0.9,
            keyword_score=0.9,
            recency_score=1.0,
            priority_score=1.0,
            hybrid_score=0.9,
        )

        weak = AgenticRAGWorkflow(
            retriever=StaticRetriever([result]),
            reranker=SectionScoreReranker(
                section_scores={"Force Merge": 0.79}
            ),
        ).run("介绍下 Milvus 3.0 Force Merge 功能是什么？")
        exhaustive = AgenticRAGWorkflow(
            retriever=StaticRetriever([result]),
            reranker=SectionScoreReranker(
                section_scores={"Force Merge": 0.95}
            ),
        ).run("Milvus 3.0 有哪些新功能？")

        self.assertEqual(weak["terminal_status"], "abstained")
        self.assertIn(
            "single_weak_chunk",
            weak["evidence_grade"]["missing_aspects"],
        )
        self.assertEqual(exhaustive["terminal_status"], "abstained")
        self.assertIn(
            "incomplete_exhaustive_coverage",
            exhaustive["evidence_grade"]["missing_aspects"],
        )

    def test_step_back_background_only_evidence_cannot_answer(self) -> None:
        chunks = [
            chunk
            for chunk in load_kb_chunks()
            if chunk.doc_id == "doc_milvus_feature_map"
        ][:2]
        results = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.95,
                keyword_score=0.95,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.95,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]

        class BackgroundOnlyRetriever(StaticRetriever):
            def search(
                self,
                query: str,
                *,
                top_k: int,
                filters: dict[str, Any] | None = None,
                order_by: list[str] | None = None,
                order_mode: Any = "relevance",
            ) -> list[SearchResult]:
                del filters, order_by, order_mode
                return self.results[:top_k] if "背景原理" in query else []

        response = AgenticRAGWorkflow(
            retriever=BackgroundOnlyRetriever(results),
            reranker=SectionScoreReranker(default_score=0.95),
        ).run("Milvus Force Merge 为什么这样工作？")

        self.assertEqual(response["query_transformation"]["strategy"], "step_back")
        self.assertEqual(response["terminal_status"], "abstained")
        self.assertFalse(response["enough_evidence"])
        self.assertEqual(response["citations"], [])

    def test_irrelevant_top_chunk_cannot_satisfy_the_evidence_quality_gate(
        self,
    ) -> None:
        """A chunk the grader excluded must not lend its score to the gate."""

        related = [
            chunk
            for chunk in load_kb_chunks()
            if chunk.doc_id == "doc_s3_sync_design"
        ][:2]
        unrelated = next(
            chunk
            for chunk in load_kb_chunks()
            if not set(tokenize("S3 文档同步流程")).intersection(
                tokenize(chunk.text)
            )
        )

        def run(chunks: list[KBChunk]) -> dict[str, Any]:
            results = [
                SearchResult(
                    chunk=chunk,
                    rank=index,
                    dense_score=0.5,
                    keyword_score=0.5,
                    recency_score=1.0,
                    priority_score=1.0,
                    hybrid_score=0.5,
                )
                for index, chunk in enumerate(chunks, start=1)
            ]
            return AgenticRAGWorkflow(
                retriever=StaticRetriever(results),
                reranker=SectionScoreReranker(
                    section_scores={
                        (unrelated.section or ""): 0.95,
                    },
                    default_score=0.25,
                ),
            ).run("我们 S3 文档同步流程是怎么设计的？")

        without = run(related)
        with_decoy = run([unrelated, *related])

        self.assertEqual(without["terminal_status"], "abstained")
        self.assertEqual(with_decoy["terminal_status"], "abstained")
        self.assertEqual(
            with_decoy["evidence_grade"]["relevant_chunks"],
            without["evidence_grade"]["relevant_chunks"],
        )
        self.assertEqual(
            with_decoy["evidence_grade"]["top_rerank_score"],
            without["evidence_grade"]["top_rerank_score"],
        )

    def test_strong_single_threshold_comes_from_the_injected_reranker(
        self,
    ) -> None:
        """Spec 12 § 5.7's gate is per implementation: scores are not comparable.

        `RuleBasedReranker` returns a bounded composite of retrieval, overlap,
        recency and priority; a model reranker returns an assigned relevance.
        A shared constant would compare two scales.
        """

        chunks = self._release_chunks()
        strict = AgenticRAGWorkflow(
            retriever=InMemoryHybridRetriever(chunks),
            reranker=SectionScoreReranker(section_scores={"Force Merge": 0.90}),
        )
        self.assertEqual(strict._strong_single_threshold(), 0.80)
        strict_response = strict.run("介绍下 Milvus 3.0 Force Merge")
        self.assertEqual(
            strict_response["evidence_grade"]["evidence_basis"],
            "single_strong_chunk",
        )

        lenient_reranker = SectionScoreReranker(
            section_scores={"Force Merge": 0.90}
        )
        lenient_reranker.strong_single_evidence_threshold = 0.95
        lenient = AgenticRAGWorkflow(
            retriever=InMemoryHybridRetriever(chunks),
            reranker=lenient_reranker,
        )
        self.assertEqual(lenient._strong_single_threshold(), 0.95)
        lenient_response = lenient.run("介绍下 Milvus 3.0 Force Merge")
        # The same 0.90 evidence no longer clears the raised bar.
        self.assertNotEqual(
            lenient_response["evidence_grade"]["evidence_basis"],
            "single_strong_chunk",
        )
        self.assertIn(
            "single_weak_chunk",
            lenient_response["evidence_grade"]["missing_aspects"],
        )

    def test_reranker_without_a_declared_threshold_fails_closed(self) -> None:
        class _UndeclaredReranker(SectionScoreReranker):
            strong_single_evidence_threshold = None  # type: ignore[assignment]

        workflow = AgenticRAGWorkflow(
            retriever=InMemoryHybridRetriever(self._release_chunks()),
            reranker=_UndeclaredReranker(),
        )
        with self.assertRaises(Exception) as caught:
            workflow.run("介绍下 Milvus 3.0 Force Merge")
        self.assertIn("strong_single_evidence_threshold", str(caught.exception))

    def test_rule_based_reranker_declares_the_shipped_threshold(self) -> None:
        self.assertEqual(
            RuleBasedReranker().strong_single_evidence_threshold,
            0.80,
        )

    def test_single_strong_chunk_does_not_answer_multi_aspect_question(
        self,
    ) -> None:
        force_merge = next(
            chunk
            for chunk in self._release_chunks()
            if chunk.section == "Force Merge"
        )
        result = SearchResult(
            chunk=force_merge,
            rank=1,
            dense_score=0.9,
            keyword_score=0.9,
            recency_score=1.0,
            priority_score=1.0,
            hybrid_score=0.9,
        )
        response = AgenticRAGWorkflow(
            retriever=StaticRetriever([result]),
            reranker=SectionScoreReranker(
                section_scores={"Force Merge": 0.95}
            ),
        ).run(
            "Milvus 3.0 Force Merge 是什么、如何使用、有什么限制？"
        )

        self.assertEqual(response["terminal_status"], "abstained")
        self.assertIn(
            "multi_aspect_requires_coverage",
            response["evidence_grade"]["missing_aspects"],
        )

    def test_multi_chunk_evidence_must_cover_every_selected_tool(
        self,
    ) -> None:
        workflow = AgenticRAGWorkflow()
        chunks = [
            chunk
            for chunk in load_kb_chunks()
            if chunk.doc_id == "doc_milvus_feature_map"
        ]
        search_results = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.9,
                keyword_score=0.9,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.9,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        state = workflow.create_state("Milvus retrieval")
        state.selected_tools = [
            "search_code_docs",
            "search_product_docs",
        ]
        state.query_plan = [
            {
                "subquery_id": "sq1",
                "tool": "search_code_docs",
                "query": state.user_query,
                "depends_on": [],
                "status": "completed",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
            {
                "subquery_id": "sq2",
                "tool": "search_product_docs",
                "query": state.user_query,
                "depends_on": [],
                "status": "completed",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
        ]
        state.retrieved_chunks = search_results
        state.reranked_chunks = [
            RerankedResult(
                search_result=result,
                rerank=index,
                old_rank=result.rank,
                rerank_score=0.9,
            )
            for index, result in enumerate(search_results, start=1)
        ]
        result_ids = [result.chunk.chunk_id for result in search_results]
        state.tool_calls = [
            {
                "tool": "search_code_docs",
                "result_chunk_ids": result_ids,
            }
        ]
        state.retrieval_provenance = {
            chunk_id: [{"tool": "search_code_docs", "subquery_id": "sq1"}]
            for chunk_id in result_ids
        }

        workflow.grade_evidence(state)

        self.assertFalse(state.enough_evidence)
        self.assertIn(
            "tool:search_product_docs",
            state.evidence_grade["missing_aspects"],
        )
        self.assertNotIn(
            "single_indirect_chunk",
            state.evidence_grade["missing_aspects"],
        )

        direct_result = search_results[1]
        state.user_query = "Milvus Force Merge"
        state.retrieved_chunks = [direct_result]
        state.reranked_chunks = [
            RerankedResult(
                search_result=direct_result,
                rerank=1,
                old_rank=direct_result.rank,
                rerank_score=0.9,
            )
        ]
        state.tool_calls[0]["result_chunk_ids"] = [
            direct_result.chunk.chunk_id
        ]
        state.retrieval_provenance = {
            direct_result.chunk.chunk_id: [
                {"tool": "search_code_docs", "subquery_id": "sq1"}
            ]
        }

        workflow.grade_evidence(state)

        self.assertIn(
            "tool:search_product_docs",
            state.evidence_grade["missing_aspects"],
        )
        self.assertNotIn(
            "single_indirect_chunk",
            state.evidence_grade["missing_aspects"],
        )

    def test_retry_preserves_terms_and_stops_before_duplicate_plan(
        self,
    ) -> None:
        question = "介绍下 Milvus 3.0 Force Merge 功能是什么？"
        response = AgenticRAGWorkflow(
            retriever=GrowingUnrelatedRetriever(self._release_chunks()),
            reranker=SectionScoreReranker(),
        ).run(question)

        self.assertEqual(response["terminal_status"], "abstained")
        self.assertEqual(response["retry_count"], 1)
        self.assertEqual(len(response["tool_calls"]), 2)
        retry_queries = [
            item["query"]
            for item in response["query_plan"]
            if item["round"] > 0
        ]
        self.assertEqual(len(retry_queries), 1)
        self.assertIn(question, retry_queries[0])
        self.assertIn("Milvus 3.0", retry_queries[0])
        self.assertIn("Force Merge", retry_queries[0])
        self.assertNotIn("S3 ingestion pipeline", retry_queries[0])
        self.assertEqual(
            response["evidence_grade"]["stop_reason"],
            "duplicate_retry_query",
        )
        self.assertIn(
            "no_relevant_evidence",
            response["evidence_grade"]["missing_aspects"],
        )

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

    def test_unknown_question_stops_within_retry_cap(self) -> None:
        response = AgenticRAGWorkflow().run(
            "这个完全不存在的采购审批宇宙飞船编号是什么？"
        )

        self.assertFalse(response["enough_evidence"])
        self.assertLessEqual(response["retry_count"], 3)
        self.assertIn("没有找到足够可靠", response["answer"])
        self.assertEqual(response["citations"], [])
        self.assertEqual(response["terminal_status"], "abstained")
        self.assertEqual(response["trace"]["terminal_status"], "abstained")
        grade = response["trace"]["evidence_grading"]
        self.assertEqual(grade["retry_count"], response["retry_count"])

    def test_unchanged_candidate_pool_stops_before_another_rerank(self) -> None:
        unrelated = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.5,
                keyword_score=0.0,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.5 - index / 100,
            )
            for index, chunk in enumerate(load_kb_chunks()[:3], start=1)
        ]
        reranker = RecordingReranker()

        response = AgenticRAGWorkflow(
            retriever=StaticRetriever(unrelated),
            reranker=reranker,
        ).run("不存在的采购审批宇宙飞船编号是什么？")

        self.assertEqual(response["terminal_status"], "abstained")
        self.assertEqual(response["retry_count"], 1)
        self.assertEqual(reranker.input_counts, [3])
        self.assertEqual(
            response["evidence_grade"]["stop_reason"],
            "no_progress",
        )

    def test_new_tool_provenance_counts_as_candidate_progress(self) -> None:
        results = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.5,
                keyword_score=0.0,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.5 - index / 100,
            )
            for index, chunk in enumerate(load_kb_chunks()[:2], start=1)
        ]
        workflow = AgenticRAGWorkflow(retriever=StaticRetriever(results))
        state = workflow.create_state("跨域比较")
        state.permission_decision = {
            "allowed": True,
            "allowed_departments": [
                "engineering",
                "product",
            ],
        }
        state.query_plan = [
            {
                "subquery_id": "sq1",
                "tool": "search_code_docs",
                "query": "same evidence",
                "depends_on": [],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            }
        ]

        workflow.milvus_hybrid_retrieve(state)
        first = state.candidate_pool_fingerprint
        state.retry_count = 1
        state.query_plan.append(
            {
                "subquery_id": "retry1",
                "tool": "search_product_docs",
                "query": "same evidence",
                "depends_on": [],
                "status": "pending",
                "round": 1,
                "version_scope": {"mode": "current"},
            }
        )
        workflow.milvus_hybrid_retrieve(state)

        self.assertNotEqual(state.candidate_pool_fingerprint, first)
        self.assertFalse(state.candidate_pool_unchanged)

    def test_new_retrieval_path_is_progress_but_offset_alone_is_not(
        self,
    ) -> None:
        """Provenance paths guard no-progress; query-local offsets do not."""

        workflow = AgenticRAGWorkflow()
        state = workflow.create_state("同一批证据")
        state.retrieved_chunks = [
            SearchResult(
                chunk=chunk,
                rank=index,
                dense_score=0.5,
                keyword_score=0.5,
                recency_score=1.0,
                priority_score=1.0,
                hybrid_score=0.5,
            )
            for index, chunk in enumerate(load_kb_chunks()[:2], start=1)
        ]
        chunk_ids = [item.chunk.chunk_id for item in state.retrieved_chunks]
        state.retrieval_provenance = {
            chunk_id: [
                {
                    "tool": "search_code_docs",
                    "subquery_id": "sq1",
                    "retrieval_profile": "flat_hybrid",
                    "retrieval_paths": ["flat_hybrid"],
                    "result_granularity": "passage",
                    "element_offset": None,
                    "fusion_recipe": None,
                }
            ]
            for chunk_id in chunk_ids
        }
        flat_only = workflow._candidate_pool_fingerprint(state)

        for chunk_id in chunk_ids:
            state.retrieval_provenance[chunk_id].append(
                {
                    "tool": "search_code_docs",
                    "subquery_id": "retry1",
                    "retrieval_profile": "struct_element",
                    "retrieval_paths": ["struct_element"],
                    "result_granularity": "passage",
                    "element_offset": 4,
                    "fusion_recipe": None,
                }
            )
        with_struct_lane = workflow._candidate_pool_fingerprint(state)

        state.retrieval_provenance[chunk_ids[0]][-1]["element_offset"] = 9
        after_offset_change = workflow._candidate_pool_fingerprint(state)

        self.assertNotEqual(with_struct_lane, flat_only)
        self.assertEqual(after_offset_change, with_struct_lane)

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
        self.assertIn("execute_tool_plan", stages)
        self.assertIn("classify_and_route", stages)
        self.assertIn("plan_retrieval", stages)
        self.assertIn("evaluate_evidence", stages)
        self.assertIn("prepare_generation_context", stages)
        self.assertEqual(
            response["trace"]["context_compression"]["effective_mode"],
            "disabled",
        )
        self.assertEqual(
            response["trace"]["context_compression"]["before_chars"],
            response["trace"]["context_compression"]["after_chars"],
        )
        self.assertEqual(
            response["trace"]["answer_generation"]["compressed_context_count"],
            0,
        )
        self.assertEqual(
            response["trace"]["answer_generation"]["compression_modes"],
            ["disabled"],
        )
        self.assertEqual(
            response["trace"]["query_transformation"]["strategy"],
            "identity",
        )
        for removed_stage in (
            "classify_query",
            "decide_retrieval",
            "select_tools",
            "rewrite_query",
            "grade_evidence",
            "prepare_supplementary_retrieval",
        ):
            self.assertNotIn(removed_stage, stages)
        self.assertGreaterEqual(response["metrics"]["latency_ms"], 0)

    def test_composite_planning_stages_return_typed_results(self) -> None:
        workflow = AgenticRAGWorkflow()
        direct_state = workflow.create_state("你好")
        direct = workflow.classify_and_route(direct_state)

        self.assertIsInstance(direct, QueryRouteResult)
        self.assertIs(direct.route, QueryRoute.DIRECT)
        self.assertEqual(
            direct_state.terminal_status,
            "answered_without_retrieval",
        )

        retrieval_state = workflow.create_state("Milvus 3.0 有哪些新功能")
        route = workflow.classify_and_route(retrieval_state)
        workflow.resolve_terminology(retrieval_state)
        workflow.check_permission(retrieval_state)
        plan = workflow.plan_retrieval(retrieval_state)

        self.assertIs(route.route, QueryRoute.RETRIEVAL)
        self.assertIsInstance(plan, RetrievalPlanResult)
        self.assertGreaterEqual(plan.plan_count, 1)
        self.assertEqual(plan.selected_tools, tuple(retrieval_state.selected_tools))
        self.assertEqual(plan.transformation, retrieval_state.query_transformation)

    def test_independent_searches_parallelize_only_with_adapter_capability(
        self,
    ) -> None:
        class BarrierRetriever(InMemoryHybridRetriever):
            supports_parallel_search = True

            def __init__(self, chunks: list[KBChunk]) -> None:
                super().__init__(chunks)
                self.barrier = Barrier(2)

            def search(
                self,
                query: str,
                *,
                top_k: int,
                filters: dict[str, Any] | None = None,
                order_by: list[str] | None = None,
                order_mode: Any = "relevance",
            ) -> list[SearchResult]:
                self.barrier.wait(timeout=2)
                return super().search(
                    query,
                    top_k=top_k,
                    filters=filters,
                    order_by=order_by,
                    order_mode=order_mode,
                )

        workflow = AgenticRAGWorkflow(
            retriever=BarrierRetriever(load_kb_chunks())
        )
        item = workflow.create_state("比较工程架构和产品路线图")
        item.permission_decision = {
            "allowed": True,
            "allowed_departments": [
                "engineering",
                "product",
                "hr",
                "security",
                "general",
            ],
        }
        item.query_plan = [
            {
                "subquery_id": "sq1",
                "tool": "search_code_docs",
                "query": "engineering architecture",
                "depends_on": [],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
            {
                "subquery_id": "sq2",
                "tool": "search_product_docs",
                "query": "product roadmap",
                "depends_on": [],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
        ]

        workflow.milvus_hybrid_retrieve(item)

        self.assertEqual(item.retrieval_execution_mode, "parallel")
        self.assertEqual(
            [call["subquery_id"] for call in item.tool_calls],
            ["sq1", "sq2"],
        )

    def test_parallel_worker_failure_is_attributed_to_tool_plan_stage(
        self,
    ) -> None:
        class PartiallyFailingRetriever(InMemoryHybridRetriever):
            supports_parallel_search = True

            def search(
                self,
                query: str,
                *,
                top_k: int,
                filters: dict[str, Any] | None = None,
                order_by: list[str] | None = None,
                order_mode: Any = "relevance",
            ) -> list[SearchResult]:
                if query == "product roadmap":
                    raise OSError("worker unavailable")
                return super().search(
                    query,
                    top_k=top_k,
                    filters=filters,
                    order_by=order_by,
                    order_mode=order_mode,
                )

        workflow = AgenticRAGWorkflow(
            retriever=PartiallyFailingRetriever(load_kb_chunks())
        )
        item = workflow.create_state(
            "比较工程架构和产品路线图",
            query_id="query_parallel_failure",
        )
        item.permission_decision = {
            "allowed": True,
            "allowed_departments": [
                "engineering",
                "product",
                "hr",
                "security",
                "general",
            ],
        }
        item.query_plan = [
            {
                "subquery_id": "sq1",
                "tool": "search_code_docs",
                "query": "engineering architecture",
                "depends_on": [],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
            {
                "subquery_id": "sq2",
                "tool": "search_product_docs",
                "query": "product roadmap",
                "depends_on": [],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
        ]

        with self.assertRaisesRegex(
            WorkflowStageError,
            "execute_tool_plan.*query_parallel_failure",
        ) as captured:
            workflow._measure_stage(
                item,
                "execute_tool_plan",
                lambda: workflow.milvus_hybrid_retrieve(item),
            )

        self.assertEqual(item.retrieval_execution_mode, "parallel")
        self.assertIsInstance(captured.exception.__cause__, OSError)

    def test_dependent_search_plan_remains_sequential(self) -> None:
        workflow = AgenticRAGWorkflow()
        item = workflow.create_state("比较工程架构和产品路线图")
        item.permission_decision = {
            "allowed": True,
            "allowed_departments": [
                "engineering",
                "product",
                "hr",
                "security",
                "general",
            ],
        }
        item.query_plan = [
            {
                "subquery_id": "sq1",
                "tool": "search_code_docs",
                "query": "engineering architecture",
                "depends_on": [],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
            {
                "subquery_id": "sq2",
                "tool": "search_product_docs",
                "query": "product roadmap",
                "depends_on": ["sq1"],
                "status": "pending",
                "round": 0,
                "version_scope": {"mode": "current"},
            },
        ]

        workflow.milvus_hybrid_retrieve(item)

        self.assertEqual(item.retrieval_execution_mode, "sequential")
        self.assertEqual(
            [call["subquery_id"] for call in item.tool_calls],
            ["sq1"],
        )

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
                order_mode: Any = "relevance",
            ) -> list[SearchResult]:
                raise OSError("Milvus unavailable")

        workflow = AgenticRAGWorkflow(FailingRetriever(load_kb_chunks()))

        with self.assertRaisesRegex(
            WorkflowStageError,
            "execute_tool_plan.*query_failure",
        ) as captured:
            workflow.run("Milvus 架构", query_id="query_failure")
        self.assertIsInstance(captured.exception.__cause__, OSError)

    def test_terminal_persistence_sinks_remain_sequential(self) -> None:
        sink_order: list[str] = []

        class RecordingMemoryStore(ConversationMemoryStore):
            def upsert_turn(self, records: list[MemoryRecord]) -> int:
                sink_order.append("conversation_memory")
                return super().upsert_turn(records)

        class RecordingWorkflow(AgenticRAGWorkflow):
            def _persist_selective_memory(
                self,
                state: AgentState,
                *,
                now_ms: int,
            ) -> None:
                sink_order.append("selective_memory")
                super()._persist_selective_memory(state, now_ms=now_ms)

            def _persist_response_cache(
                self,
                state: AgentState,
                *,
                now_ms: int,
            ) -> None:
                sink_order.append("response_cache")
                super()._persist_response_cache(state, now_ms=now_ms)

        response = RecordingWorkflow(
            memory_store=RecordingMemoryStore()
        ).run("Milvus 3.0 有哪些新功能")

        self.assertEqual(response["terminal_status"], "answered")
        self.assertEqual(
            sink_order,
            [
                "conversation_memory",
                "selective_memory",
                "response_cache",
            ],
        )

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
